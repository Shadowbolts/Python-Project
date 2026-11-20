import json
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
from rich.console import Console

from findex.crawler.crawl import canonicalize_url, crawl, extract_links_and_text
from findex.crawler.crawl_cmd import run_crawl_cli


def test_canonicalize_url() -> None:
    raw = "HTTPS://Example.COM:443/docs/intro/?b=2&a=1#section"
    expected = "https://example.com:443/docs/intro?a=1&b=2"
    assert canonicalize_url(raw) == expected


def test_extract_links_and_text() -> None:
    html = """
    <html>
      <head><title>Test Page</title></head>
      <body>
        <h1>Hello World</h1>
        <p>This is test content.</p>
        <a href="/page2">Link 2</a>
        <a href="https://other.com/ext">Ext</a>
      </body>
    </html>
    """
    title, text, links = extract_links_and_text("https://example.com", html)
    assert title == "Test Page"
    assert "Hello World" in text
    assert "This is test content" in text
    assert "https://example.com/page2" in links
    assert "https://other.com/ext" in links


@pytest.mark.asyncio
async def test_crawl_queue_and_stop_at_max_pages() -> None:
    # Keys use canonical URLs (no trailing slash on root)
    site_pages = {
        "https://test.local": """
            <html><head><title>Home</title></head><body>
            <a href="/p1">P1</a>
            <a href="/p2">P2</a>
            </body></html>
        """,
        "https://test.local/p1": """
            <html><head><title>Page 1</title></head><body>
            <a href="/p3">P3</a>
            </body></html>
        """,
        "https://test.local/p2": "<html><head><title>Page 2</title></head><body>Content 2</body></html>",
        "https://test.local/p3": "<html><head><title>Page 3</title></head><body>Content 3</body></html>",
        "https://test.local/robots.txt": "User-agent: *\nAllow: /",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        # Strip trailing slash from root only so it matches our canonical dict keys
        from urllib.parse import urlparse

        parsed = urlparse(str(request.url))
        key = str(request.url).rstrip("/") if parsed.path in ("/", "") else str(request.url)
        content = site_pages.get(key, "<html><body>404</body></html>")
        status = 200 if key in site_pages else 404
        return httpx.Response(status, text=content, request=request)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, follow_redirects=True)

    pages = []
    async for page in crawl(
        ["https://test.local/"],
        max_pages=3,
        concurrency=2,
        min_delay=0.0,
        client=client,
    ):
        pages.append(page)

    await client.aclose()

    assert len(pages) == 3
    assert pages[0].url == "https://test.local"
    assert pages[0].title == "Home"


@pytest.mark.asyncio
async def test_run_crawl_cli_streams_jsonl_and_log(tmp_path: Path) -> None:
    site_pages = {
        "https://test.local": "<html><head><title>Root</title></head><body><a href='/a'>A</a></body></html>",
        "https://test.local/a": "<html><head><title>Page A</title></head><body>Body A</body></html>",
        "https://test.local/robots.txt": "User-agent: *\nAllow: /",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        parsed = urlparse(str(request.url))
        key = str(request.url).rstrip("/") if parsed.path in ("/", "") else str(request.url)
        content = site_pages.get(key, "<html><body>404</body></html>")
        status = 200 if key in site_pages else 404
        return httpx.Response(status, text=content, request=request)

    out_path = tmp_path / "crawl.jsonl"
    log_path = tmp_path / "crawl.log"
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)

    count, elapsed = await run_crawl_cli(
        ["https://test.local"],
        max_pages=2,
        concurrency=2,
        min_delay=0.0,
        out_path=out_path,
        log_path=log_path,
        console=Console(quiet=True),
        client=client,
    )
    await client.aclose()

    assert count == 2
    assert elapsed >= 0.0
    lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0]["title"] == "Root"
    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 2
    assert "OK https://test.local" in log_lines[0]
