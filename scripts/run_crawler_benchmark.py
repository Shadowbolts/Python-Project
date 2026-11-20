"""
Benchmark script for Lab 06 async crawler.
Crawls https://books.toscrape.com at concurrency 1, 5, 20 and measures throughput & RSS.
"""

import asyncio
import json
import time
from pathlib import Path

import psutil

from findex.crawler.crawl import crawl

SEED_URL = "https://books.toscrape.com/"
MAX_PAGES = 30
CONCURRENCIES = [1, 5, 20]


async def bench_one(concurrency: int) -> dict[str, float | int]:
    process = psutil.Process()
    t0 = time.perf_counter()
    pages_count = 0

    async for _page in crawl(
        [SEED_URL],
        max_pages=MAX_PAGES,
        concurrency=concurrency,
        per_host=concurrency,
        min_delay=0.0,
        allowed_domains=["books.toscrape.com"],
    ):
        pages_count += 1

    wall = time.perf_counter() - t0
    rss_mb = process.memory_info().rss / (1024 * 1024)
    pages_per_sec = pages_count / wall if wall > 0 else 0.0

    return {
        "concurrency": concurrency,
        "pages": pages_count,
        "wall_s": round(wall, 3),
        "pages_per_sec": round(pages_per_sec, 2),
        "rss_mb": round(rss_mb, 1),
    }


async def main() -> None:
    results = []
    for c in CONCURRENCIES:
        print(f"Benchmarking concurrency={c} (max_pages={MAX_PAGES})...")
        row = await bench_one(c)
        print(f"  -> {row}")
        results.append(row)

    out_path = Path("crawler_benchmark_results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
