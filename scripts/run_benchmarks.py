"""
Benchmark script for Lab 05 concurrency and GIL experiments.
Measures wall time, CPU time, parent process peak RSS, and merge time.
"""
import json
import os
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import psutil

from findex.index import _chunk_specs, build_partial, merge
from findex.models import Index


def run_benchmark_once(
    corpus_root: Path,
    workers: int,
    executor: Literal["serial", "threads", "processes"],
) -> dict[str, float]:
    all_paths = sorted(p for p in corpus_root.rglob("*.txt") if p.is_file())
    doc_specs = [(doc_id, path) for doc_id, path in enumerate(all_paths)]

    process = psutil.Process()
    tracemalloc.start()
    t0_wall = time.perf_counter()
    t0_cpu = process.cpu_times().user + process.cpu_times().system

    merge_time = 0.0

    if executor == "serial" or workers <= 1:
        partial = build_partial(doc_specs)
        t0_merge = time.perf_counter()
        idx = merge([partial])
        merge_time = time.perf_counter() - t0_merge
    else:
        chunks = _chunk_specs(doc_specs, workers)
        if executor == "threads":
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(build_partial, chunk) for chunk in chunks]
                partials = [fut.result() for fut in futures]
        elif executor == "processes":
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(build_partial, chunk) for chunk in chunks]
                partials = [fut.result() for fut in futures]
        else:
            raise ValueError(f"Unknown executor: {executor}")

        t0_merge = time.perf_counter()
        idx = merge(partials)
        merge_time = time.perf_counter() - t0_merge

    t1_wall = time.perf_counter()
    t1_cpu = process.cpu_times().user + process.cpu_times().system
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    wall_sec = t1_wall - t0_wall
    cpu_sec = t1_cpu - t0_cpu
    peak_rss_mb = process.memory_info().rss / (1024 * 1024)

    assert idx.num_docs == len(doc_specs)

    return {
        "wall": wall_sec,
        "cpu": cpu_sec,
        "merge": merge_time,
        "peak_rss": peak_rss_mb,
    }


def main() -> None:
    corpus = Path("data")
    print(f"Starting Lab 05 benchmarks on corpus {corpus} ({len(list(corpus.glob('*.txt')))} docs)...")
    print("Warming up indexer...")
    _ = run_benchmark_once(corpus, workers=1, executor="serial")

    configs = [
        ("serial", 1),
        ("threads", 2),
        ("threads", 4),
        ("threads", 8),
        ("threads", 16),
        ("processes", 2),
        ("processes", 4),
        ("processes", 8),
        ("processes", 16),
    ]

    results = []

    for executor, workers in configs:
        runs = []
        print(f"\nRunning {executor} with {workers} worker(s) (3 runs)...")
        for r in range(3):
            data = run_benchmark_once(corpus, workers=workers, executor=executor)
            runs.append(data)
            print(f"  Run {r+1}: wall={data['wall']:.3f}s, cpu={data['cpu']:.3f}s, merge={data['merge']:.3f}s, rss={data['peak_rss']:.1f}MB")

        runs.sort(key=lambda x: x["wall"])
        median_run = runs[1]
        results.append({
            "executor": executor,
            "workers": workers,
            "wall_median": median_run["wall"],
            "cpu_median": median_run["cpu"],
            "merge_median": median_run["merge"],
            "rss_median": median_run["peak_rss"],
            "all_runs": runs,
        })

    serial_wall = results[0]["wall_median"]
    for row in results:
        row["speedup"] = serial_wall / row["wall_median"]

    out_file = Path("benchmark_results.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nAll benchmark results saved to {out_file}")


if __name__ == "__main__":
    main()
