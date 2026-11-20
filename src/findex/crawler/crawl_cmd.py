"""
CLI runner for `findex crawl`: streams JSON lines, Rich live stats, and writes crawl.log.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

import httpx
from rich.console import Console
from rich.live import Live
from rich.table import Table

from findex.crawler.crawl import crawl


def _build_stats_table(
    pages_count: int,
    max_pages: int,
    elapsed: float,
    last_url: str,
) -> Table:
    rate = pages_count / elapsed if elapsed > 0 else 0.0
    table = Table(title="findex crawl — Live Stats")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Pages Fetched", f"{pages_count} / {max_pages}")
    table.add_row("Elapsed", f"{elapsed:.2f} s")
    table.add_row("Throughput", f"{rate:.2f} pages/s")
    table.add_row("Last URL", last_url or "-")
    return table


async def run_crawl_cli(
    seeds: list[str],
    *,
    max_pages: int = 200,
    concurrency: int = 10,
    per_host: int = 2,
    min_delay: float = 0.05,
    allowed_domains: list[str] | None = None,
    out_path: Path = Path("crawl.jsonl"),
    log_path: Path = Path("crawl.log"),
    console: Console | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[int, float]:
    """Run crawler, stream JSONL to `out_path`, log URLs to `log_path`, and show Live stats."""
    active_console = console or Console()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pages_count = 0
    last_url = ""

    with (
        out_path.open("w", encoding="utf-8") as out_f,
        log_path.open("w", encoding="utf-8") as log_f,
        Live(
            _build_stats_table(0, max_pages, 0.0, ""),
            console=active_console,
            refresh_per_second=4,
        ) as live,
    ):
        async for page in crawl(
            seeds,
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=per_host,
            min_delay=min_delay,
            allowed_domains=allowed_domains,
            client=client,
        ):
            pages_count += 1
            last_url = page.url
            out_f.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")
            out_f.flush()
            log_f.write(f"{page.fetched_at} OK {page.url} | {page.title}\n")
            log_f.flush()

            elapsed = time.perf_counter() - t0
            live.update(_build_stats_table(pages_count, max_pages, elapsed, last_url))

    total_elapsed = time.perf_counter() - t0
    return pages_count, total_elapsed
