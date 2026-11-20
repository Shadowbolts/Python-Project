import argparse
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from findex.models import DocMeta, Index, InvertedIndex, PartialIndex, Posting
from findex.timing import timed
from findex.tokenize import tokenize

__all__ = [
    "DocMeta",
    "Index",
    "InvertedIndex",
    "PartialIndex",
    "Posting",
    "build_index",
    "build_partial",
    "main",
    "merge",
]


def build_partial(
    doc_specs: Sequence[tuple[int, Path]],
    *,
    record_positions: bool = False,
) -> PartialIndex:
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, path in doc_specs:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        title = path.stem.replace("_", " ").replace("-", " ").title()

        if record_positions:
            pos_map: defaultdict[str, list[int]] = defaultdict(list)
            length = 0
            for pos, token in enumerate(tokenize(text)):
                pos_map[token].append(pos)
                length += 1

            doc_lengths[doc_id] = length
            doc_meta[doc_id] = DocMeta(
                doc_id=doc_id,
                path=str(path),
                title=title,
                length=length,
            )

            for term, positions in pos_map.items():
                postings[term].append(
                    Posting(doc_id=doc_id, tf=len(positions), positions=tuple(positions))
                )
        else:
            term_counts = Counter(tokenize(text))
            length = sum(term_counts.values())

            doc_lengths[doc_id] = length
            doc_meta[doc_id] = DocMeta(
                doc_id=doc_id,
                path=str(path),
                title=title,
                length=length,
            )

            for term, count in term_counts.items():
                postings[term].append(Posting(doc_id=doc_id, tf=count))

    return PartialIndex(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
    )


def merge(partials: Iterable[PartialIndex]) -> Index:
    merged_postings: defaultdict[str, list[Posting]] = defaultdict(list)
    merged_lengths: dict[int, int] = {}
    merged_meta: dict[int, DocMeta] = {}

    for partial in partials:
        merged_lengths.update(partial.doc_lengths)
        merged_meta.update(partial.doc_meta)
        for term, plist in partial.postings.items():
            merged_postings[term].extend(plist)

    sorted_postings: dict[str, list[Posting]] = {}
    for term, plist in merged_postings.items():
        sorted_postings[term] = sorted(plist, key=lambda p: p.doc_id)

    return Index(
        postings=sorted_postings,
        doc_lengths=merged_lengths,
        doc_meta=merged_meta,
    )


def _chunk_specs(
    specs: list[tuple[int, Path]],
    n_chunks: int,
) -> list[list[tuple[int, Path]]]:
    if not specs:
        return []
    n_chunks = max(1, min(n_chunks, len(specs)))
    avg = len(specs) / n_chunks
    chunks: list[list[tuple[int, Path]]] = []
    last = 0.0
    while last < len(specs):
        end = round(last + avg)
        chunks.append(specs[round(last) : end])
        last = float(end)
    return chunks


@timed
def build_index(
    root: Path,
    *,
    record_positions: bool = False,
    limit: int | None = None,
    workers: int = 1,
    executor: Literal["serial", "threads", "processes"] = "serial",
) -> Index:
    if root.is_file():
        all_paths = [root]
    else:
        all_paths = sorted(p for p in root.rglob("*.txt") if p.is_file())

    if limit is not None:
        all_paths = all_paths[:limit]

    doc_specs = [(doc_id, path) for doc_id, path in enumerate(all_paths)]

    if executor == "serial" or workers <= 1 or len(doc_specs) <= 1:
        partial = build_partial(doc_specs, record_positions=record_positions)
        return merge([partial])

    chunks = _chunk_specs(doc_specs, workers)

    if executor == "threads":
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(build_partial, chunk, record_positions=record_positions)
                for chunk in chunks
            ]
            partials = [fut.result() for fut in futures]
        return merge(partials)

    if executor == "processes":
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(build_partial, chunk, record_positions=record_positions)
                for chunk in chunks
            ]
            partials = [fut.result() for fut in futures]
        return merge(partials)

    msg = f"Unknown executor: {executor}"
    raise ValueError(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save inverted index.")
    parser.add_argument("corpus", type=Path, help="Path to corpus directory")
    parser.add_argument("--out", type=Path, default=None, help="Output file path")
    parser.add_argument("--positions", action="store_true", help="Record token positions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of workers for parallel indexing",
    )
    parser.add_argument(
        "--executor",
        choices=["serial", "threads", "processes"],
        default="serial",
        help="Execution strategy",
    )
    parser.add_argument(
        "--format",
        choices=["pickle", "json"],
        default=None,
        help="Storage format",
    )
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    index = build_index(
        args.corpus,
        record_positions=args.positions,
        limit=args.limit,
        workers=args.workers,
        executor=args.executor,
    )

    elapsed = time.perf_counter() - start_time
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_bytes / (1024 * 1024)
    print(f"Indexed {index.num_documents} documents ({index.num_terms} unique terms)")
    print(f"Build time: {elapsed:.2f} s")
    print(f"Peak memory: {peak_mb:.2f} MB")

    if args.out:
        from findex.store import save_index

        fmt: Literal["pickle", "json"] | None = args.format
        save_index(index, args.out, format=fmt)
        print(f"Saved index to {args.out}")


if __name__ == "__main__":
    main()
