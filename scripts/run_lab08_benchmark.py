"""
Benchmark script for Lab 08:
1. Lab 3 Python-loop BM25 vs NumPy vectorized BM25 on 43 docs and 100x scaled corpus (4300 docs).
2. Memmap index load time vs Pickle load time.
3. Gap-encoding compression ratio.
"""

import gc
import json
import tempfile
import time
from pathlib import Path

from findex.models import DocMeta, Index, Posting
from findex.scoring import BM25, ranked_search
from findex.store import load_index
from findex.vector_scoring import (
    VectorIndex,
    encode_gaps,
    load_memmap_index,
    save_memmap_index,
    vectorized_search,
)


def replicate_index(base: Index, factor: int) -> Index:
    """Create a scaled corpus of size base.num_docs * factor for benchmarking."""
    n_base = base.num_docs
    new_doc_lengths: dict[int, int] = {}
    new_doc_meta: dict[int, DocMeta] = {}
    for rep in range(factor):
        offset = rep * n_base
        for doc_id, length in base.doc_lengths.items():
            new_doc_lengths[offset + doc_id] = length
        for doc_id, meta in base.doc_meta.items():
            new_doc_meta[offset + doc_id] = DocMeta(
                doc_id=offset + doc_id,
                path=meta.path,
                title=f"{meta.title} [{rep}]",
                length=meta.length,
            )

    new_postings: dict[str, list[Posting]] = {}
    for term, plist in base.postings.items():
        expanded: list[Posting] = []
        for rep in range(factor):
            offset = rep * n_base
            for p in plist:
                expanded.append(Posting(doc_id=offset + p.doc_id, tf=p.tf, positions=()))
        new_postings[term] = expanded

    return Index(postings=new_postings, doc_lengths=new_doc_lengths, doc_meta=new_doc_meta)


def measure_ms(fn, *args, rounds: int = 15, **kwargs) -> float:
    # Warm-up
    fn(*args, **kwargs)
    times: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    base_idx = load_index(Path("index.pkl"))
    base_vidx = VectorIndex.from_index(base_idx)

    # 10x and 100x corpora
    idx_10x = replicate_index(base_idx, 10)
    vidx_10x = VectorIndex.from_index(idx_10x)

    idx_100x = replicate_index(base_idx, 100)
    vidx_100x = VectorIndex.from_index(idx_100x)

    queries = [
        ("common term (the)", "the"),
        ("rare term (sherlock)", "sherlock"),
        ("3-term query (the project gutenberg)", "the project gutenberg"),
    ]

    bm25 = BM25()
    results: dict[str, object] = {"corpus_43": [], "corpus_10x_430": [], "corpus_100x_4300": []}

    for label, idx_obj, vidx_obj in [
        ("corpus_43", base_idx, base_vidx),
        ("corpus_10x_430", idx_10x, vidx_10x),
        ("corpus_100x_4300", idx_100x, vidx_100x),
    ]:
        rows = []
        for q_name, q_str in queries:
            matched_docs = sum(len(idx_obj.postings.get(t, [])) for t in q_str.split())
            py_ms = measure_ms(ranked_search, idx_obj, q_str, scorer=bm25, k=10, rounds=15)
            np_ms = measure_ms(vectorized_search, vidx_obj, q_str, scorer="bm25", k=10, rounds=15)
            speedup = py_ms / np_ms if np_ms > 0 else 0.0
            row = {
                "query": q_name,
                "postings_scanned": matched_docs,
                "lab3_ms": round(py_ms, 3),
                "numpy_ms": round(np_ms, 3),
                "speedup": round(speedup, 2),
            }
            rows.append(row)
        results[label] = rows

    # Memmap load vs Pickle load
    with tempfile.TemporaryDirectory() as tmpdir:
        mm_dir = Path(tmpdir) / "mm"
        save_memmap_index(base_vidx, mm_dir)
        t0 = time.perf_counter()
        mm_idx = load_memmap_index(mm_dir)
        mm_load_ms = (time.perf_counter() - t0) * 1000.0
        del mm_idx
        gc.collect()

    t0 = time.perf_counter()
    _ = load_index(Path("index.pkl"))
    pkl_load_ms = (time.perf_counter() - t0) * 1000.0

    # Gap-encoding ratio on 100x postings
    raw_bytes = vidx_100x.doc_ids_buf.nbytes
    # Encode per-term doc_ids
    encoded_bytes = sum(
        len(encode_gaps(vidx_100x.doc_ids_buf[s:e]))
        for s, e in vidx_100x.term_slices.values()
    )

    results["storage"] = {
        "pickle_load_ms": round(pkl_load_ms, 2),
        "memmap_load_ms": round(mm_load_ms, 2),
        "raw_doc_ids_mb": round(raw_bytes / (1024 * 1024), 2),
        "gap_encoded_mb": round(encoded_bytes / (1024 * 1024), 2),
        "compression_ratio": round(raw_bytes / max(encoded_bytes, 1), 2),
    }

    Path("lab08_benchmark_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
