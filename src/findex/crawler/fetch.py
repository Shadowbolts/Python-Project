"""
Polite async HTTP fetcher with retry, exponential backoff, jitter, and timeout.
"""
import asyncio
import random
import time
from dataclasses import dataclass

import httpx

DEFAULT_USER_AGENT = "findex-crawler/0.1 (+https://github.com/Shadowbolts/Python-Project)"


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status_code: int
    text: str
    num_bytes: int
    elapsed: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300 and self.error is None


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
) -> FetchResult:
    t0 = time.perf_counter()
    last_status = 0
    last_error: str | None = None

    for attempt in range(max_retries + 1):
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url, follow_redirects=True)
                elapsed = time.perf_counter() - t0
                status = resp.status_code

                if status == 429 or status >= 500:
                    last_status = status
                    last_error = f"HTTP {status}"
                    if attempt < max_retries:
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            delay = float(retry_after)
                        else:
                            delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 0.5)
                        await asyncio.sleep(delay)
                        continue

                return FetchResult(
                    url=str(resp.url),
                    status_code=status,
                    text=resp.text if 200 <= status < 300 else "",
                    num_bytes=len(resp.content),
                    elapsed=elapsed,
                    error=None if 200 <= status < 300 else f"HTTP {status}",
                )

        except TimeoutError:
            last_status = 408
            last_error = "Timeout"
            if attempt < max_retries:
                delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 0.5)
                await asyncio.sleep(delay)
                continue
        except httpx.RequestError as exc:
            last_status = 599
            last_error = str(exc)
            if attempt < max_retries:
                delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 0.5)
                await asyncio.sleep(delay)
                continue

    elapsed = time.perf_counter() - t0
    return FetchResult(
        url=url,
        status_code=last_status,
        text="",
        num_bytes=0,
        elapsed=elapsed,
        error=last_error or "Unknown fetch error",
    )
