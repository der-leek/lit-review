"""Rate-limited async Semantic Scholar Graph API client."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from rich import console as csl

from lit_review import config

if TYPE_CHECKING:
    import httpx

console = csl.Console()

_RATE_LIMITED = 429


class S2Client:
    """Thin async wrapper around the Semantic Scholar Graph API."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        rps: float = config.RATE_LIMIT_RPS,
    ):
        self.client = client
        self._sem = asyncio.Semaphore(2)
        self._last_call = 0.0
        self.rps = rps

    async def _throttle(self) -> None:
        now = time.monotonic()
        min_interval = 1.0 / self.rps
        elapsed = now - self._last_call
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_call = time.monotonic()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        max_retries: int = 30,
        graceful: bool = False,
    ) -> Any:
        async with self._sem:
            await self._throttle()
            url = f"{config.S2_BASE}{path}"
            for _ in range(max_retries):
                resp = await self.client.request(
                    method,
                    url,
                    headers=config.HEADERS,
                    params=params,
                    json=json,
                    timeout=60.0,
                )
                if resp.status_code == _RATE_LIMITED:
                    wait = 1
                    console.log(f"[yellow]S2 429 – sleeping {wait:.1f}s[/]")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            if graceful:
                console.log(
                    f"[yellow]S2 rate-limit retries exhausted for {path} – "
                    "continuing without[/]"
                )
                return None
            raise RuntimeError(f"S2 rate-limit retries exhausted for {path}")

    async def poll_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        poll_timeout: float = config.SEARCH_POLL_TIMEOUT,
        graceful: bool = False,
    ) -> Any:
        """Retry every 1s until the request succeeds or the timeout expires.

        The /paper/search endpoint returns 429s generously; rather than
        bailing at the first 429, poll once per second until it goes through.
        """
        url = f"{config.S2_BASE}{path}"
        deadline = time.monotonic() + poll_timeout
        while True:
            async with self._sem:
                await self._throttle()
                resp = await self.client.request(
                    method,
                    url,
                    headers=config.HEADERS,
                    params=params,
                    timeout=60.0,
                )
            if resp.status_code != _RATE_LIMITED:
                resp.raise_for_status()
                return resp.json()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if graceful:
                    console.log(
                        f"[yellow]S2 poll timed out after {poll_timeout:.0f}s "
                        f"for {path} – continuing without[/]"
                    )
                    return None
                message = (
                    f"S2 rate-limit poll timed out after {poll_timeout:.0f}s for {path}"
                )
                raise RuntimeError(message)
            console.log(
                f"[yellow]S2 429 – polling again in 1s ({remaining:.0f}s left)[/]"
            )
            await asyncio.sleep(min(1.0, remaining))

    async def search(
        self,
        query: str,
        *,
        limit: int = 40,
        year: str | None = None,
        fields_of_study: str | None = None,
        min_citation_count: int | None = None,
        poll_timeout: float = config.SEARCH_POLL_TIMEOUT,
        graceful: bool = False,
        fields: str = (
            "paperId,title,abstract,year,citationCount,influentialCitationCount"
        ),
    ) -> list[dict]:
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": fields,
        }
        if year and year.strip().lower() != "null":
            params["year"] = year
        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study
        if min_citation_count is not None:
            params["minCitationCount"] = min_citation_count
        data = await self.poll_request(
            "GET",
            "/paper/search",
            params=params,
            poll_timeout=poll_timeout,
            graceful=graceful,
        )
        return (data or {}).get("data") or []

    async def batch_papers(
        self,
        ids: list[str],
        fields: str = (
            "paperId,title,abstract,year,citationCount,"
            "influentialCitationCount"
        ),
        *,
        graceful: bool = False,
    ) -> list[dict]:
        results: list[dict] = []
        for i in range(0, len(ids), config.BATCH_SIZE):
            chunk = ids[i : i + config.BATCH_SIZE]
            data = await self.request(
                "POST",
                "/paper/batch",
                params={"fields": fields},
                json={"ids": chunk},
                graceful=graceful,
            )
            if data:
                results.extend(data)
        return results

    async def citations(
        self,
        paper_id: str,
        *,
        limit: int = 50,
        fields: str = (
            "paperId,title,isInfluential,citationCount,influentialCitationCount"
        ),
        graceful: bool = False,
    ) -> list[dict]:
        params = {"fields": fields, "limit": limit}
        data = await self.request(
            "GET", f"/paper/{paper_id}/citations", params=params, graceful=graceful
        )
        return (data or {}).get("data") or []

    async def references(
        self,
        paper_id: str,
        *,
        limit: int = 40,
        fields: str = "paperId,title,isInfluential",
        graceful: bool = False,
    ) -> list[dict]:
        params = {"fields": fields, "limit": limit}
        data = await self.request(
            "GET", f"/paper/{paper_id}/references", params=params, graceful=graceful
        )
        return (data or {}).get("data") or []
