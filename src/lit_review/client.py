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
        max_retries: int = 6,
    ) -> Any:
        async with self._sem:
            await self._throttle()
            url = f"{config.S2_BASE}{path}"
            for attempt in range(max_retries):
                resp = await self.client.request(
                    method,
                    url,
                    headers=config.HEADERS,
                    params=params,
                    json=json,
                    timeout=60.0,
                )
                if resp.status_code == 429:
                    wait = min(2**attempt + 1.0, 60.0)
                    console.log(f"[yellow]S2 429 – sleeping {wait:.1f}s[/]")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            raise RuntimeError(f"S2 rate-limit retries exhausted for {path}")

    async def search(
        self,
        query: str,
        *,
        limit: int = 40,
        year: str | None = None,
        fields_of_study: str | None = None,
        min_citation_count: int | None = None,
        fields: str = (
            "paperId,title,abstract,year,citationCount,influentialCitationCount"
        ),
    ) -> list[dict]:
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": fields,
        }
        if year:
            params["year"] = year
        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study
        if min_citation_count is not None:
            params["minCitationCount"] = min_citation_count
        data = await self.request("GET", "/paper/search", params=params)
        return data.get("data") or []

    async def batch_papers(
        self,
        ids: list[str],
        fields: str = (
            "paperId,title,abstract,year,citationCount,"
            "influentialCitationCount,embedding.specter_v2"
        ),
    ) -> list[dict]:
        results: list[dict] = []
        for i in range(0, len(ids), config.BATCH_SIZE):
            chunk = ids[i : i + config.BATCH_SIZE]
            data = await self.request(
                "POST",
                "/paper/batch",
                params={"fields": fields},
                json={"ids": chunk},
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
    ) -> list[dict]:
        params = {"fields": fields, "limit": limit}
        data = await self.request("GET", f"/paper/{paper_id}/citations", params=params)
        return data.get("data") or []

    async def references(
        self,
        paper_id: str,
        *,
        limit: int = 40,
        fields: str = "paperId,title,isInfluential",
    ) -> list[dict]:
        params = {"fields": fields, "limit": limit}
        data = await self.request("GET", f"/paper/{paper_id}/references", params=params)
        return data.get("data") or []
