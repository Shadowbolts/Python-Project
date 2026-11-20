"""
Core async web crawler: Queue frontier, TaskGroup workers, rate limiting, and link parsing.
"""

import asyncio
import datetime
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import httpx
from selectolax.lexbor import LexborHTMLParser

from findex.crawler.fetch import DEFAULT_USER_AGENT, fetch
from findex.crawler.robots import RobotsCache


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") if parsed.path else ""
    query_parts = parse_qsl(parsed.query, keep_blank_values=True)
    query_parts.sort()
    query = urlencode(query_parts)
    return urlunparse((scheme, netloc, path, "", query, ""))


def extract_links_and_text(base_url: str, html_text: str) -> tuple[str, str, list[str]]:
    parser = LexborHTMLParser(html_text)

    title_node = parser.css_first("title")
    title = title_node.text(strip=True) if title_node else ""

    for tag in ("script", "style", "noscript", "svg", "header", "footer", "nav"):
        for node in parser.css(tag):
            node.decompose()

    body = parser.body
    text = body.text(separator=" ", strip=True) if body else parser.text(separator=" ", strip=True)

    links: list[str] = []
    for a in parser.css("a[href]"):
        href = a.attributes.get("href")
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        full_url = urljoin(base_url, href)
        links.append(full_url)

    return title, text, links


async def crawl(
    seeds: Sequence[str],
    *,
    max_pages: int = 50,
    concurrency: int = 5,
    per_host: int = 2,
    min_delay: float = 0.05,
    allowed_domains: Sequence[str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[Page]:
    """Async generator that crawls pages from seed URLs using TaskGroup workers."""
    if max_pages <= 0:
        return

    frontier: asyncio.Queue[str] = asyncio.Queue()
    seen: set[str] = set()

    for s in seeds:
        canon = canonicalize_url(s)
        if canon not in seen:
            seen.add(canon)
            await frontier.put(canon)

    allowed: set[str] | None = {d.lower() for d in allowed_domains} if allowed_domains else None
    robots = RobotsCache(user_agent=DEFAULT_USER_AGENT)

    global_sem = asyncio.Semaphore(concurrency)
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_last_access: dict[str, float] = {}

    pages_yielded = 0
    active_tasks = 0
    stop_event = asyncio.Event()
    results_queue: asyncio.Queue[Page | None] = asyncio.Queue()

    def get_host_sem(host: str) -> asyncio.Semaphore:
        if host not in host_sems:
            host_sems[host] = asyncio.Semaphore(per_host)
        return host_sems[host]

    async def process_url(http_client: httpx.AsyncClient, url: str) -> None:
        nonlocal pages_yielded
        try:
            parsed = urlparse(url)
            host = parsed.netloc.lower()

            if allowed and host not in allowed:
                return
            if not await robots.is_allowed(http_client, url):
                return

            host_sem = get_host_sem(host)
            async with global_sem, host_sem:
                if stop_event.is_set():
                    return
                now = time.monotonic()
                last = host_last_access.get(host, 0.0)
                delay = min_delay - (now - last)
                if delay > 0:
                    await asyncio.sleep(delay)
                host_last_access[host] = time.monotonic()
                res = await fetch(http_client, url)

            if res.ok and res.text and not stop_event.is_set():
                title, text, links = extract_links_and_text(res.url, res.text)
                fetched_at = datetime.datetime.now(datetime.UTC).isoformat()
                page = Page(
                    url=canonicalize_url(res.url),
                    title=title,
                    text=text,
                    fetched_at=fetched_at,
                )
                if pages_yielded >= max_pages:
                    stop_event.set()
                    return

                pages_yielded += 1
                await results_queue.put(page)

                if pages_yielded >= max_pages:
                    stop_event.set()
                    return

                for link in links:
                    canon_link = canonicalize_url(link)
                    link_host = urlparse(canon_link).netloc.lower()
                    if allowed and link_host not in allowed:
                        continue
                    if canon_link not in seen and len(seen) < max_pages * 10:
                        seen.add(canon_link)
                        await frontier.put(canon_link)
        finally:
            frontier.task_done()

    async def worker(http_client: httpx.AsyncClient) -> None:
        nonlocal active_tasks
        while not stop_event.is_set():
            try:
                url = await asyncio.wait_for(frontier.get(), timeout=0.05)
            except TimeoutError:
                if frontier.empty() and active_tasks == 0:
                    stop_event.set()
                    break
                continue

            active_tasks += 1
            try:
                await process_url(http_client, url)
            finally:
                active_tasks -= 1

    async def run_crawler() -> None:
        manage_client = client is None
        active_client = (
            client
            if client is not None
            else httpx.AsyncClient(
                headers={"User-Agent": DEFAULT_USER_AGENT},
                follow_redirects=True,
            )
        )
        try:
            async with asyncio.TaskGroup() as tg:
                for _ in range(concurrency):
                    tg.create_task(worker(active_client))
        finally:
            stop_event.set()
            if manage_client:
                await active_client.aclose()
            await results_queue.put(None)

    crawler_task = asyncio.create_task(run_crawler())

    try:
        while True:
            item = await results_queue.get()
            if item is None:
                break
            yield item
    finally:
        stop_event.set()
        await crawler_task
