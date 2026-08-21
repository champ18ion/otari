"""Fast, static half of the mcp dependency-ceiling regression (see #689).

Catches a floor with no ceiling at all ("mcp>=1.28.1"), immediately and by
name, rather than as an obscure failure elsewhere. Does not on its own prove
a present ceiling is placed correctly (e.g. "<3.0.0" passes this) - that's
tests/integration/test_mcp_dependency_ceiling.py, which has the full story
and does the real fresh-resolution proof.
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
