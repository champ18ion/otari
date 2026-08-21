"""mcp's pyproject.toml constraint must keep mcp 2.0+ out of a fresh
resolution, not just satisfy the committed uv.lock.

uv.lock pins mcp 1.28.1, so `uv sync` here never re-resolves and never
notices a missing ceiling: CI stays green even though a fresh, unlocked
resolution elsewhere (a fresh clone before the first `uv lock`, a consuming
superset build resolving otari's tree alongside its own) can pick up mcp
2.0, which removed `mcp.client.streamable_http.streamablehttp_client`.
`gateway/services/mcp_client.py` imports that at module scope, so every
gateway command dies with ImportError the moment that happens. See #689.

This resolves mcp into a throwaway venv using only the constraint declared
in pyproject.toml - uv.lock is never consulted - and proves the actual
import that broke still works against whatever version that constraint
allows today. If the ceiling is ever loosened or dropped, this installs the
newest matching mcp and fails the same way production did.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
import venv
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
    """Fast, static half of the regression: a floor with no ceiling ("mcp>=1.28.1")
    is exactly the shape that let mcp 2.0 in undetected. This fails immediately with
    a message pointing at #689, rather than only surfacing as an obscure failure in
    the fresh-install test below.
    """
    requirement = _mcp_requirement()
    assert _has_upper_bound(requirement), (
        f"mcp is declared as '{requirement}', a floor with no ceiling. A fresh, "
        "unlocked resolution can pick up a future breaking major release (mcp 2.0 "
        "already did this once, see #689) with nothing here to stop it."
    )


@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_mcp_constraint_resolves_to_an_importable_version(tmp_path: Path) -> None:
    """Slow, behavioral half of the regression: actually resolve mcp fresh, ignoring
    uv.lock entirely, and prove the import that broke in #689 still works. A ceiling
    that's present but placed too high (e.g. "<3.0.0") would pass the static check
    above yet still break here.

    Reruns cover a transient PyPI/index failure during the real install below, not
    the assertions themselves.
    """
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")

    requirement = _mcp_requirement()

    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True)
    venv_python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    install = subprocess.run(
        ["uv", "pip", "install", "--python", str(venv_python), str(requirement)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert install.returncode == 0, f"fresh install of '{requirement}' failed:\n{install.stderr}"

    imported = subprocess.run(
        [str(venv_python), "-c", "from mcp.client.streamable_http import streamablehttp_client"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert imported.returncode == 0, (
        f"a fresh install of '{requirement}' (ignoring uv.lock) cannot import "
        f"streamablehttp_client, the same break as #689:\n{imported.stderr}"
    )
