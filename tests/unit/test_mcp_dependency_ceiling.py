"""mcp's pyproject.toml constraint must keep mcp 2.0+ out of a fresh
resolution, not just satisfy the committed uv.lock.

This is the fast, static half of the regression: a floor with no ceiling
("mcp>=1.28.1") is exactly the shape that let mcp 2.0 in undetected (mcp
2.0 removed `mcp.client.streamable_http.streamablehttp_client`, which
`gateway/services/mcp_client.py` imports at module scope, breaking every
gateway command on import - see #689). This check catches that shape
immediately, with a message pointing at #689, rather than only surfacing as
an obscure failure elsewhere.

It does not on its own prove a present ceiling is placed correctly (e.g.
catches "<3.0.0" is not that): that behavioral proof is
tests/integration/test_mcp_dependency_ceiling.py, which actually resolves
mcp fresh, ignoring uv.lock, and checks the real import.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _mcp_requirement() -> Requirement:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    for raw in pyproject["project"]["dependencies"]:
        requirement = Requirement(raw)
        if requirement.name == "mcp":
            return requirement
    pytest.fail("mcp is no longer declared as a direct dependency in pyproject.toml")


def _has_upper_bound(requirement: Requirement) -> bool:
    return any(spec.operator in ("<", "<=", "==", "~=") for spec in requirement.specifier)


def test_mcp_constraint_has_an_upper_bound() -> None:
    requirement = _mcp_requirement()
    assert _has_upper_bound(requirement), (
        f"mcp is declared as '{requirement}', a floor with no ceiling. A fresh, "
        "unlocked resolution can pick up a future breaking major release (mcp 2.0 "
        "already did this once, see #689) with nothing here to stop it."
    )
