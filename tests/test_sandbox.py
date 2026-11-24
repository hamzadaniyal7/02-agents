"""The workspace boundary is the one thing here that has to be airtight."""

from __future__ import annotations

from pathlib import Path

import pytest

from actcheck.errors import SandboxError
from actcheck.tools.sandbox import Sandbox


def test_resolves_relative_paths_against_the_root(sandbox: Sandbox) -> None:
    assert sandbox.resolve("README.md") == sandbox.root / "README.md"


def test_allows_the_root_itself(sandbox: Sandbox) -> None:
    assert sandbox.resolve(".") == sandbox.root


def test_rejects_parent_traversal(sandbox: Sandbox) -> None:
    with pytest.raises(SandboxError, match="escapes the workspace"):
        sandbox.resolve("../secrets.txt")


def test_rejects_deep_traversal_that_lands_outside(sandbox: Sandbox) -> None:
    with pytest.raises(SandboxError):
        sandbox.resolve("docs/../../../../etc/passwd")


def test_rejects_absolute_paths_outside(sandbox: Sandbox) -> None:
    with pytest.raises(SandboxError):
        sandbox.resolve("/etc/passwd")


def test_allows_absolute_paths_inside(sandbox: Sandbox) -> None:
    inside = str(sandbox.root / "README.md")
    assert sandbox.resolve(inside) == sandbox.root / "README.md"


def test_traversal_that_returns_inside_is_allowed(sandbox: Sandbox) -> None:
    """`a/../b` is a normal path, not an attack. Rejecting it would be wrong."""
    assert sandbox.resolve("docs/../README.md") == sandbox.root / "README.md"


def test_sibling_directory_with_a_shared_prefix_is_outside(tmp_path: Path) -> None:
    """The case a `startswith` check gets wrong: /work vs /work-other."""
    root = tmp_path / "work"
    root.mkdir()
    (tmp_path / "work-other").mkdir()
    sandbox = Sandbox(root)
    with pytest.raises(SandboxError):
        sandbox.resolve(str(tmp_path / "work-other" / "file.txt"))


def test_symlink_pointing_outside_is_rejected(sandbox: Sandbox, tmp_path: Path) -> None:
    """An agent that can write a file can write a symlink."""
    secret = tmp_path / "outside.txt"
    secret.write_text("classified")
    (sandbox.root / "escape").symlink_to(secret)
    with pytest.raises(SandboxError):
        sandbox.resolve("escape")


def test_symlinked_directory_pointing_outside_is_rejected(sandbox: Sandbox, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "file.txt").write_text("outside")
    (sandbox.root / "linkdir").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(SandboxError):
        sandbox.resolve("linkdir/file.txt")


def test_symlink_inside_the_workspace_is_fine(sandbox: Sandbox) -> None:
    (sandbox.root / "alias.md").symlink_to(sandbox.root / "README.md")
    assert sandbox.resolve("alias.md") == sandbox.root / "README.md"


def test_paths_that_do_not_exist_yet_are_allowed(sandbox: Sandbox) -> None:
    """Writing a new file means resolving a path with nothing at the end of it."""
    assert sandbox.resolve("new/nested/file.md").parent.name == "nested"


def test_must_exist_reports_a_missing_file(sandbox: Sandbox) -> None:
    with pytest.raises(SandboxError, match="no such file"):
        sandbox.resolve("nope.md", must_exist=True)


def test_relative_renders_workspace_relative_paths(sandbox: Sandbox) -> None:
    assert sandbox.relative(sandbox.root / "docs" / "runbook.md") == "docs/runbook.md"


def test_root_is_created_when_missing(tmp_path: Path) -> None:
    sandbox = Sandbox(tmp_path / "brand-new")
    assert sandbox.root.is_dir()


def test_root_that_is_a_file_is_rejected(tmp_path: Path) -> None:
    file = tmp_path / "a-file"
    file.write_text("x")
    with pytest.raises(SandboxError, match="not a directory"):
        Sandbox(file)


def test_the_filesystem_root_is_refused() -> None:
    """Confinement that works perfectly while confining an agent to the entire
    machine is not confinement."""
    with pytest.raises(SandboxError, match="refusing to use"):
        Sandbox("/")


def test_a_path_that_normalises_to_the_root_is_refused() -> None:
    with pytest.raises(SandboxError, match="refusing to use"):
        Sandbox("/nonexistent/../..")


def test_the_home_directory_is_refused() -> None:
    with pytest.raises(SandboxError, match="refusing to use"):
        Sandbox(Path.home())
