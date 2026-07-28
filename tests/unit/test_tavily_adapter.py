"""Unit tests for the Tavily web-search adapter (scripts/web-search-tavily-adapter).

The adapter is a standalone service outside the gateway package, so we load
it by path. Outbound Tavily calls are mocked via an httpx transport; the test
suite needs no network access or live key.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from _search_adapter_test_support import load_adapter, mock_async_client


def _load_adapter(monkeypatch: pytest.MonkeyPatch, *, api_key: str = "tvly-test") -> Any:
    return load_adapter(
        monkeypatch,
        adapter_dir="web-search-tavily-adapter",
        module_name="tavily_adapter_app",
        env={"TAVILY_API_KEY": api_key},
    )


@pytest.mark.asyncio
async def test_health_reports_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch, api_key="")
    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/health")
    assert resp.status_code == 503
    assert resp.json() == {"status": "missing TAVILY_API_KEY"}


@pytest.mark.asyncio
async def test_health_healthy_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch)
    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/health")
    assert resp.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_search_maps_tavily_shape_and_whitelists(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch)

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json_body(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Post A",
                        "url": "https://example.com/a",
                        "content": "snippet a",
                        "raw_content": "full page a",
                    },
                    {
                        "title": "Post B",
                        "url": "https://example.org/b",
                        "content": "snippet b",
                        "raw_content": None,
                    },
                    {"title": "no url", "content": "x"},
                ]
            },
        )

    monkeypatch.setattr(module.httpx, "AsyncClient", mock_async_client(handler))

    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get(
            "/search",
            params={
                "q": "claude code",
                "max_results": 5,
                "search_depth": "advanced",
                "topic": "news",
                # Not a whitelisted Tavily param at the body level — FastAPI
                # ignores unknown query params, so it never reaches Tavily.
                "bogus": "should-not-forward",
            },
        )

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2  # the url-less hit is dropped
    assert results[0] == {
        "url": "https://example.com/a",
        "title": "Post A",
        "content": "snippet a",
        "extracted_content": "full page a",
    }
    # raw_content None → no extracted_content key
    assert "extracted_content" not in results[1]

    assert captured["url"] == "https://api.tavily.com/search"
    assert captured["auth"] == "Bearer tvly-test"
    body = captured["body"]
    assert body["query"] == "claude code"
    assert body["include_raw_content"] is True
    assert body["max_results"] == 5
    assert body["search_depth"] == "advanced"
    assert body["topic"] == "news"
    assert "bogus" not in body


@pytest.mark.asyncio
async def test_search_surfaces_tavily_status_as_502(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    monkeypatch.setattr(module.httpx, "AsyncClient", mock_async_client(handler))

    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/search", params={"q": "x"})

    assert resp.status_code == 502
    assert resp.json() == {"error": "tavily search returned 401"}


@pytest.mark.asyncio
async def test_search_invalid_json_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>", headers={"content-type": "application/json"})

    monkeypatch.setattr(module.httpx, "AsyncClient", mock_async_client(handler))

    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/search", params={"q": "x"})

    assert resp.status_code == 502
    assert "invalid JSON" in resp.json()["error"]


@pytest.mark.asyncio
async def test_search_non_list_results_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": "not-a-list"})

    monkeypatch.setattr(module.httpx, "AsyncClient", mock_async_client(handler))

    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/search", params={"q": "x"})

    assert resp.status_code == 502
    assert "unexpected shape" in resp.json()["error"]


@pytest.mark.asyncio
async def test_search_invalid_time_range_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    # Bad time_range is rejected at the edge (Query pattern) before any Tavily call.
    module = _load_adapter(monkeypatch)
    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/search", params={"q": "x", "time_range": "bogus"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_search_503_when_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch, api_key="")
    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/search", params={"q": "x"})
    assert resp.status_code == 503


# ----- helpers -----


def json_body(request: httpx.Request) -> dict[str, Any]:
    import json

    body: dict[str, Any] = json.loads(request.content.decode())
    return body
