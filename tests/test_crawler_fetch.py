import httpx
import pytest

from findex.crawler.fetch import FetchResult, fetch
from findex.crawler.robots import RobotsCache


@pytest.mark.asyncio
async def test_fetch_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>Hello World</body></html>", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        res = await fetch(client, "https://example.com/page")
        assert res.ok
        assert res.status_code == 200
        assert "Hello World" in res.text
        assert res.num_bytes > 0


@pytest.mark.asyncio
async def test_fetch_404_no_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, text="Not Found", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        res = await fetch(client, "https://example.com/missing", max_retries=3)
        assert not res.ok
        assert res.status_code == 404
        assert calls == 1  # 404 must not be retried


@pytest.mark.asyncio
async def test_fetch_retry_on_500() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(500, text="Internal Server Error", request=request)
        return httpx.Response(200, text="Recovered!", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        res = await fetch(client, "https://example.com/transient", max_retries=3, base_delay=0.01)
        assert res.ok
        assert res.status_code == 200
        assert calls == 3


@pytest.mark.asyncio
async def test_robots_cache() -> None:
    robots_txt = "User-agent: *\nDisallow: /private/\nAllow: /public/"

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://example.com/robots.txt":
            return httpx.Response(200, text=robots_txt, request=request)
        return httpx.Response(200, text="ok", request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        rc = RobotsCache()
        assert await rc.is_allowed(client, "https://example.com/public/doc.html")
        assert not await rc.is_allowed(client, "https://example.com/private/secret.html")
