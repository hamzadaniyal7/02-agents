"""The workspace boundary.

Every path an agent supplies passes through `Sandbox.resolve`. The rule is
simple to state -- nothing outside the workspace root -- and surprisingly easy
to get wrong, so it is worth being explicit about what this does and does not
rely on.

**Not** string prefix matching. `"/work/../etc/passwd".startswith("/work")` is
True, and `/work-other` also starts with `/work`. Both are real escapes that
prefix checks wave through.

What it does instead: resolve the candidate path to a real, absolute, fully
symlink-expanded path, then ask the filesystem whether that path is the root or
is contained by it, component by component. A symlink inside the workspace that
points at `/etc` resolves to `/etc` and is rejected -- which is the whole point,
because an agent that can write a file can write a symlink.

For paths that do not exist yet (a file the agent is about to create) the
nearest existing ancestor is resolved and checked, since you cannot resolve
symlinks on something that is not there.
"""

from __future__ import annotations

import os
from pathlib import Path

from actcheck.errors import SandboxError


class Sandbox:
    """Confines filesystem access to a single directory tree."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise SandboxError(f"workspace root is not a directory: {self.root}")
        # A typo'd --workspace (or a path that normalises upward, like
        # "/nonexistent/../..") can land on "/" or a home directory. The
        # confinement below would then be doing its job perfectly while
        # confining the agent to the entire machine, so these are refused
        # outright: a workspace that broad is never what anyone meant.
        if self.root == Path(self.root.anchor) or self.root == Path.home():
            raise SandboxError(
                f"refusing to use {self.root} as a workspace -- it is the filesystem root "
                "or your home directory. Point --workspace at a specific project directory."
            )

    def resolve(self, candidate: str | Path, *, must_exist: bool = False) -> Path:
        """Return the absolute path for `candidate`, or raise `SandboxError`.

        Relative paths are taken relative to the workspace root, which is what
        makes `read_file("notes.md")` mean what the agent expects.
        """
        raw = Path(os.path.expanduser(str(candidate)))
        target = raw if raw.is_absolute() else self.root / raw

        resolved = self._resolve_existing_ancestor(target)
        if not self._contains(resolved):
            raise SandboxError(
                f"path escapes the workspace: {candidate!r} resolves to {resolved}, "
                f"which is outside {self.root}. Use a path inside the workspace."
            )
        if must_exist and not resolved.exists():
            raise SandboxError(f"no such file or directory: {self.relative(resolved)}")
        return resolved

    def _resolve_existing_ancestor(self, target: Path) -> Path:
        """Fully resolve `target`, tolerating a path that does not exist yet.

        `Path.resolve(strict=False)` on a non-existent path still expands any
        symlinks in the part that *does* exist, which is the behaviour we want.
        The explicit walk is here for the case where a parent directory is
        itself a dangling symlink -- resolve() would hand back the unresolved
        tail and we would check the wrong thing.
        """
        resolved = target.resolve()
        probe = resolved
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        # Re-attach the non-existent tail to the resolved real ancestor.
        return (probe.resolve() / resolved.relative_to(probe)) if probe != resolved else resolved

    def _contains(self, path: Path) -> bool:
        if path == self.root:
            return True
        return self.root in path.parents

    def relative(self, path: str | Path) -> str:
        """Workspace-relative display string. Agents should never see absolute
        paths -- they leak the host layout and invite absolute-path guesses.

        The unresolved path is tried first on purpose. Resolving for *display*
        renames entries: a symlink `notes -> archive/2024.md` would be listed
        as `archive/2024.md`, and a dangling link would be listed under the
        name of the file it fails to point at. Resolution is what the security
        check in `resolve` needs; it is the wrong thing for a label.
        """
        candidate = Path(path)
        for form in (candidate, candidate.resolve()):
            try:
                return str(form.relative_to(self.root)) or "."
            except ValueError:
                continue
        return str(candidate)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Sandbox({self.root})"
