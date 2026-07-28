"""Unit tests for the Brave web-search adapter (scripts/web-search-brave-adapter).

The adapter is a standalone service outside the gateway package, so we load
it by path. Outbound Brave calls are mocked via an httpx transport; the test
suite needs no network access or live key.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from _search_adapter_test_support import load_adapter, mock_async_client


def _load_adapter(monkeypatch: pytest.MonkeyPatch, *, api_key: str = "brv-test") -> Any:
    return load_adapter(
        monkeypatch,
        adapter_dir="web-search-brave-adapter",
        module_name="brave_adapter_app",
        env={"BRAVE_API_KEY": api_key},
    )


@pytest.mark.asyncio
async def test_health_reports_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch, api_key="")
    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/health")
    # Fail closed: an orchestrator health check must not treat a keyless,
    # unusable adapter as healthy.
    assert resp.status_code == 503
    assert resp.json() == {"status": "missing BRAVE_API_KEY"}


@pytest.mark.asyncio
async def test_health_healthy_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch)
    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_search_maps_brave_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch)

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["params"] = dict(request.url.params)
        captured["token"] = request.headers.get("x-subscription-token")
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "url": "https://example.com/a",
                            "title": "Post A",
                            "description": "snippet a",
                            "page_age": "2026-07-20T00:00:00.000Z",
                            "age": "1 week ago",
                        },
                        {
                            "url": "https://example.org/b",
                            "title": "Post B",
                            "description": "snippet b",
                            "age": "3 days ago",
                        },
                        {"title": "no url", "description": "x"},
                    ]
                }
            },
        )

    monkeypatch.setattr(module.httpx, "AsyncClient", mock_async_client(handler))

    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/search", params={"q": "claude code"})

    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2  # the url-less hit is dropped
    assert results[0] == {
        "url": "https://example.com/a",
        "title": "Post A",
        "content": "snippet a",
        # page_age wins over age when both are present.
        "published_date": "2026-07-20T00:00:00.000Z",
    }
    # No page_age on this hit: falls back to age.
    assert results[1]["published_date"] == "3 days ago"

    assert captured["token"] == "brv-test"
    assert captured["params"]["q"] == "claude code"
    assert "freshness" not in captured["params"]


@pytest.mark.asyncio
async def test_search_forwards_time_range_as_freshness(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch)

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"web": {"results": []}})

    monkeypatch.setattr(module.httpx, "AsyncClient", mock_async_client(handler))

    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        for time_range, expected_freshness in (
            ("day", "pd"),
            ("d", "pd"),
            ("week", "pw"),
            ("w", "pw"),
            ("month", "pm"),
            ("m", "pm"),
            ("year", "py"),
            ("y", "py"),
        ):
            resp = await client.get("/search", params={"q": "x", "time_range": time_range})
            assert resp.status_code == 200, resp.text
            assert captured["params"]["freshness"] == expected_freshness


@pytest.mark.asyncio
async def test_search_invalid_time_range_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    # Bad time_range is rejected at the edge (Query pattern) before any Brave call.
    module = _load_adapter(monkeypatch)
    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/search", params={"q": "x", "time_range": "bogus"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_search_surfaces_brave_status_as_502(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    monkeypatch.setattr(module.httpx, "AsyncClient", mock_async_client(handler))

    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/search", params={"q": "x"})

    assert resp.status_code == 502
    assert resp.json() == {"error": "brave search returned 401"}


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
async def test_search_non_dict_payload_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "a", "dict"])

    monkeypatch.setattr(module.httpx, "AsyncClient", mock_async_client(handler))

    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/search", params={"q": "x"})

    assert resp.status_code == 502
    assert "unexpected shape" in resp.json()["error"]


@pytest.mark.asyncio
async def test_search_non_list_results_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"web": {"results": "not-a-list"}})

    monkeypatch.setattr(module.httpx, "AsyncClient", mock_async_client(handler))

    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/search", params={"q": "x"})

    assert resp.status_code == 502
    assert "unexpected shape" in resp.json()["error"]


@pytest.mark.asyncio
async def test_search_missing_web_key_returns_502(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    monkeypatch.setattr(module.httpx, "AsyncClient", mock_async_client(handler))

    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/search", params={"q": "x"})

    assert resp.status_code == 502
    assert "unexpected shape" in resp.json()["error"]


@pytest.mark.asyncio
async def test_search_503_when_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_adapter(monkeypatch, api_key="")
    transport = httpx.ASGITransport(app=module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://adapter") as client:
        resp = await client.get("/search", params={"q": "x"})
    assert resp.status_code == 503
