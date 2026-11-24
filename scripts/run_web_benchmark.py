"""
Load-testing benchmark for Lab 07 FastAPI search service.
Compares 1 worker async def, 1 worker def, and 4 workers def at concurrency=50.
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import httpx
from fastapi import Depends, Query

from findex.models import Index
from findex.web.app import _execute_search, create_app
from findex.web.deps import get_index
from findex.web.schemas import SearchResponse

# Small helper app used only when benchmarking async def vs def
bench_app = create_app()


@bench_app.get("/search-async", response_model=SearchResponse)
async def search_async_endpoint(
    index: Index = Depends(get_index),  # noqa: B008
    q: str = Query(..., min_length=1, max_length=200),
    k: int = Query(10, ge=1, le=100),
    scorer: Literal["bm25", "tfidf"] = Query("bm25"),
    page: int = Query(1, ge=1, le=1000),
) -> SearchResponse:
    resp, _ = _execute_search(index, q, scorer, page, k, html_highlight=False)
    return resp


def percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(len(sorted_vals) * pct / 100.0), len(sorted_vals) - 1)
    return sorted_vals[idx]


async def wait_for_ready(url: str, timeout: float = 15.0) -> None:
    deadline = time.perf_counter() + timeout
    async with httpx.AsyncClient() as client:
        while time.perf_counter() < deadline:
            try:
                r = await client.get(f"{url}/health", timeout=1.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.2)
    raise RuntimeError(f"Server at {url} did not become ready in time")


async def run_load(base_url: str, path: str, concurrency: int = 50, duration_s: float = 5.0) -> dict[str, float]:
    latencies_ms: list[float] = []
    errors = 0
    stop_at = time.perf_counter() + duration_s

    limits = httpx.Limits(max_connections=concurrency + 10, max_keepalive_connections=concurrency + 10)
    async with httpx.AsyncClient(base_url=base_url, limits=limits, timeout=10.0) as client:
        # Warm-up
        await client.get(path, params={"q": "sherlock holmes", "k": 10})

        async def worker() -> None:
            nonlocal errors
            while time.perf_counter() < stop_at:
                t0 = time.perf_counter()
                try:
                    resp = await client.get(path, params={"q": "sherlock holmes", "k": 10})
                    if resp.status_code != 200:
                        errors += 1
                    else:
                        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                except httpx.HTTPError:
                    errors += 1

        t_start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        total_s = time.perf_counter() - t_start

    latencies_ms.sort()
    rps = len(latencies_ms) / total_s if total_s > 0 else 0.0
    return {
        "requests": len(latencies_ms),
        "errors": errors,
        "rps": round(rps, 1),
        "p50_ms": round(percentile(latencies_ms, 50), 2),
        "p95_ms": round(percentile(latencies_ms, 95), 2),
        "p99_ms": round(percentile(latencies_ms, 99), 2),
    }


def start_uvicorn(app_target: str, port: int, workers: int) -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        app_target,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--log-level",
        "warning",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def main() -> None:
    configs = [
        ("local, 1 worker, async def", "scripts.run_web_benchmark:bench_app", "/search-async", 8101, 1),
        ("local, 1 worker, def", "findex.web:app", "/search", 8102, 1),
        ("local, 4 workers, def", "findex.web:app", "/search", 8103, 4),
    ]

    results = []
    for label, target, path, port, workers in configs:
        print(f"Starting {label} on port {port}...")
        proc = start_uvicorn(target, port, workers)
        base_url = f"http://127.0.0.1:{port}"
        try:
            await wait_for_ready(base_url)
            stats = await run_load(base_url, path, concurrency=50, duration_s=5.0)
            row = {"target": label, "route": path, "concurrency": 50, **stats}
            print(f"  -> {row}")
            results.append(row)
        finally:
            proc.terminate()
            proc.wait(timeout=10)

    Path("web_benchmark_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved results to web_benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
