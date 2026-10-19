import argparse
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str
    length: int


@dataclass
class InvertedIndex:
    postings: dict[str, list[Posting]] = field(default_factory=dict)
    doc_lengths: dict[int, int] = field(default_factory=dict)
    doc_meta: dict[int, DocMeta] = field(default_factory=dict)

    @property
    def num_documents(self) -> int:
        return len(self.doc_meta)

    @property
    def num_terms(self) -> int:
        return len(self.postings)


def build_index(
    root: Path,
    *,
    record_positions: bool = False,
    limit: int | None = None,
) -> InvertedIndex:
    postings = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    docs_iter = iter_documents(root)
    if limit is not None:
        docs_iter = islice(docs_iter, limit)

    for doc_id, doc in enumerate(docs_iter):
        title = Path(doc.doc_id).stem.replace("_", " ").replace("-", " ").title()

        if record_positions:
            pos_map: defaultdict[str, list[int]] = defaultdict(list)
            length = 0
            for pos, token in enumerate(tokenize(doc.text)):
                pos_map[token].append(pos)
                length += 1

            doc_lengths[doc_id] = length
            doc_meta[doc_id] = DocMeta(
                doc_id=doc_id,
                path=str(doc.path),
                title=title,
                length=length,
            )

            for term, positions in pos_map.items():
                postings[term].append(
                    Posting(doc_id=doc_id, tf=len(positions), positions=tuple(positions))
                )
        else:
            term_counts = Counter(tokenize(doc.text))
            length = sum(term_counts.values())

            doc_lengths[doc_id] = length
            doc_meta[doc_id] = DocMeta(
                doc_id=doc_id,
                path=str(doc.path),
                title=title,
                length=length,
            )

            for term, count in term_counts.items():
                postings[term].append(Posting(doc_id=doc_id, tf=count))

    return InvertedIndex(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save inverted index.")
    parser.add_argument("corpus", type=Path, help="Path to corpus directory")
    parser.add_argument("--out", type=Path, default=None, help="Output file path")
    parser.add_argument("--positions", action="store_true", help="Record token positions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle", help="Storage format")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    index = build_index(args.corpus, record_positions=args.positions, limit=args.limit)

    elapsed = time.perf_counter() - start_time
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_bytes / (1024 * 1024)
    print(f"Indexed {index.num_documents} documents ({index.num_terms} unique terms)")
    print(f"Build time: {elapsed:.2f} s")
    print(f"Peak memory: {peak_mb:.2f} MB")

    if args.out:
        from findex.store import save_index

        save_index(index, args.out, format=args.format)
        print(f"Saved index to {args.out}")


if __name__ == "__main__":
    main()
