"""
Robots.txt parsing, fetching, and caching.
"""
from dataclasses import dataclass, field
import urllib.robotparser
from urllib.parse import urlparse
import httpx


@dataclass
class RobotsCache:
    user_agent: str = "findex-crawler/0.1 (+https://github.com/Shadowbolts/Python-Project)"
    _cache: dict[str, urllib.robotparser.RobotFileParser] = field(default_factory=dict)

    async def get_parser(self, client: httpx.AsyncClient, host: str, scheme: str = "https") -> urllib.robotparser.RobotFileParser:
        if host in self._cache:
            return self._cache[host]

        robots_url = f"{scheme}://{host}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)

        try:
            resp = await client.get(robots_url, timeout=5.0)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            elif resp.status_code in (401, 403):
                rp.disallow_all = True  # type: ignore[attr-defined]
            else:
                rp.allow_all = True  # type: ignore[attr-defined]
        except Exception:
            rp.allow_all = True  # type: ignore[attr-defined]

        self._cache[host] = rp
        return rp

    async def is_allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        scheme = parsed.scheme or "https"
        rp = await self.get_parser(client, host, scheme)
        return rp.can_fetch(self.user_agent, url)
