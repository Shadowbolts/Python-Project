"""
Benchmark script for free-threaded Python 3.13t.
Uses pure standard library to benchmark thread scaling without GIL.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "src")

from findex.index import _chunk_specs, build_partial, merge


def benchmark_313t_threads(corpus_root: Path, workers: int) -> dict[str, float]:
    all_paths = sorted(p for p in corpus_root.rglob("*.txt") if p.is_file())
    doc_specs = [(doc_id, path) for doc_id, path in enumerate(all_paths)]

    t0_wall = time.perf_counter()
    chunks = _chunk_specs(doc_specs, workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(build_partial, chunk) for chunk in chunks]
        partials = [fut.result() for fut in futures]

    t0_merge = time.perf_counter()
    idx = merge(partials)
    merge_time = time.perf_counter() - t0_merge
    t1_wall = time.perf_counter()

    assert idx.num_docs == len(doc_specs)
    return {
        "wall": t1_wall - t0_wall,
        "merge": merge_time,
    }


def main() -> None:
    corpus = Path("data")
    print(f"Python: {sys.version}")
    print(f"GIL enabled: {getattr(sys, '_is_gil_enabled', lambda: 'N/A')()}")
    print("Running 3.13t ThreadPoolExecutor benchmarks...")

    configs = [1, 2, 4, 8, 16]
    results = []

    for w in configs:
        runs = []
        print(f"Workers {w} (3 runs)...")
        for r in range(3):
            data = benchmark_313t_threads(corpus, w)
            runs.append(data)
            print(f"  Run {r+1}: wall={data['wall']:.3f}s, merge={data['merge']:.3f}s")

        runs.sort(key=lambda x: x["wall"])
        median = runs[1]
        results.append({
            "workers": w,
            "wall_median": median["wall"],
            "merge_median": median["merge"],
            "all_runs": runs,
        })

    serial_wall = results[0]["wall_median"]
    for row in results:
        row["speedup"] = serial_wall / row["wall_median"]

    out_file = Path("benchmark_313t_results.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved 3.13t results to {out_file}")


if __name__ == "__main__":
    main()
