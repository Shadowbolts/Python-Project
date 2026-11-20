import httpx
import pytest

from findex.crawler.crawl import canonicalize_url, crawl, extract_links_and_text


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
