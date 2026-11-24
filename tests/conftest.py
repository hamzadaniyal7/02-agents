from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from actcheck.config import Config
from actcheck.tools import Registry, Sandbox, build_registry

WORKSPACE = Path(__file__).resolve().parents[1] / "workspace"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A throwaway copy of the sample workspace.

    Copied rather than shared because half these tests write files, and a test
    suite whose results depend on execution order is worse than no suite.
    """
    target = tmp_path / "workspace"
    shutil.copytree(WORKSPACE, target)
    return target


@pytest.fixture
def sandbox(workspace: Path) -> Sandbox:
    return Sandbox(workspace)


@pytest.fixture
def cfg(workspace: Path, tmp_path: Path) -> Config:
    """A config that runs entirely offline.

    `checker = "programmatic"` keeps the default tests free of the scripted
    model's canned verdicts, so a rule under test is the only thing that can
    produce the verdict being asserted.
    """
    return Config(
        workspace=str(workspace),
        provider="scripted",
        checker="programmatic",
        plan=False,
        trace_dir=str(tmp_path / "traces"),
        max_steps=12,
    )


@pytest.fixture
def registry(cfg: Config, sandbox: Sandbox) -> Registry:
    return build_registry(cfg, sandbox)
