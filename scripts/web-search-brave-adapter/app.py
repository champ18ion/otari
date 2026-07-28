"""SearXNG-compatible adapter in front of the Brave Search API.

The gateway's ``WebSearchBackend`` speaks one protocol: it issues
``GET {url}/search?q=…&format=json`` and expects

    {"results": [{"url": ..., "title": ..., "content": ...}, ...]}

Brave's API has a different shape and its own auth header, so this thin
service translates between the two. Point the gateway at it with
``OTARI_WEB_SEARCH_URL=http://brave-adapter:8080``; the Brave key lives
here, never in the gateway.

Each result's ``content`` is Brave's snippet; ``extracted_content`` is left
unset so the gateway still fetches and extracts the full page (matching the
SearXNG path). Set ``extracted_content`` here instead if you want snippet-only
behaviour and to skip the gateway's per-URL fetch.

The gateway may forward provider-specific knobs as extra query params
(``provider_options`` on the ``otari_web_search`` tool entry). This adapter
only recognizes ``time_range``, mapped onto Brave's ``freshness`` filter;
anything else is ignored, never forwarded to Brave.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
# Brave caps `count` at 20; the gateway slices to its own max_results anyway.
DEFAULT_COUNT = 10

# Same time_range vocabulary as the Tavily adapter (single vs. plural forms
# both accepted), mapped onto Brave's `freshness` filter values.
_FRESHNESS_BY_TIME_RANGE = {
    "d": "pd",
    "day": "pd",
    "w": "pw",
    "week": "pw",
    "m": "pm",
    "month": "pm",
    "y": "py",
    "year": "py",
}

app = FastAPI(title="otari web-search Brave adapter")


@app.get("/health")
async def health() -> dict[str, str]:
    # Surfaces a misconfigured deployment (missing key) without a live query.
    return {"status": "healthy" if BRAVE_API_KEY else "missing BRAVE_API_KEY"}


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    time_range: str | None = Query(default=None, pattern="^(day|week|month|year|d|w|m|y)$"),
) -> JSONResponse:
    """Translate a SearXNG-style query into a Brave Search API call."""
    if not BRAVE_API_KEY:
        return JSONResponse(status_code=503, content={"error": "BRAVE_API_KEY is not set"})

    headers = {"X-Subscription-Token": BRAVE_API_KEY, "Accept": "application/json"}
    params: dict[str, str | int] = {"q": q, "count": DEFAULT_COUNT}
    if time_range is not None:
        params["freshness"] = _FRESHNESS_BY_TIME_RANGE[time_range]
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(BRAVE_ENDPOINT, params=params, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPStatusError as exc:
        # Surface Brave's status (e.g. 401 bad key, 429 quota) so the gateway
        # logs make the cause obvious instead of a generic "unreachable".
        return JSONResponse(
            status_code=502,
            content={"error": f"brave search returned {exc.response.status_code}"},
        )
    except httpx.HTTPError as exc:
        return JSONResponse(status_code=502, content={"error": f"brave search failed: {exc}"})

    hits = (payload.get("web") or {}).get("results") or []
    results: list[dict[str, Any]] = []
    for h in hits:
        if not isinstance(h, dict) or not h.get("url"):
            continue
        result: dict[str, Any] = {
            "url": h["url"],
            "title": h.get("title", ""),
            "content": h.get("description", ""),
        }
        # Brave documents both `page_age` and `age` on a result with no
        # documented distinction between them; prefer page_age (appears to be
        # the more specific field) and fall back to age. Passed through as an
        # opaque string, whatever format Brave sent, not reformatted here.
        published_date = h.get("page_age") or h.get("age")
        if published_date:
            result["published_date"] = published_date
        results.append(result)
    return JSONResponse(content={"results": results})
