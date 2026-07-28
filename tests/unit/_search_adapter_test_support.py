"""Shared test helpers for the standalone web-search adapter scripts.

Each adapter (Brave, Tavily, ...) lives under ``scripts/`` outside the
gateway package, so its tests load it by file path rather than import it
normally. Not itself a test module (no ``test_`` prefix), pytest won't collect
it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def load_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    adapter_dir: str,
    module_name: str,
    env: dict[str, str],
) -> Any:
    """Load an adapter's ``app.py`` by path, with ``env`` set via monkeypatch.

    ``module_name`` must be unique per adapter (it's registered in
    ``sys.modules``) so loading one adapter in a test run doesn't shadow
    another's.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    adapter_path = _SCRIPTS_DIR / adapter_dir / "app.py"
    spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def mock_async_client(handler: Any) -> Any:
    """Return an httpx.AsyncClient subclass whose outbound calls use a mock
    transport, so an adapter's own outbound call to its upstream API is
    intercepted."""

    real_async_client = httpx.AsyncClient

    class _MockAsyncClient(real_async_client):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # The test drives the ASGI app through its own AsyncClient with an
            # explicit transport; leave that one untouched. Only the adapter's
            # own outbound call (which passes no transport) gets the mock.
            if "transport" not in kwargs:
                kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    return _MockAsyncClient
