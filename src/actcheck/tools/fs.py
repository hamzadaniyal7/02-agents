"""Filesystem tools.

Every path goes through the sandbox. Every result is written to be read by a
model: line numbers on file reads (so the agent can quote a location back),
sizes on directory listings (so it can tell a stub from the real thing), and
explicit "file is empty" rather than an empty string, because an empty
observation reads as a tool failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from actcheck.errors import ToolError
from actcheck.tools.base import Tool
from actcheck.tools.sandbox import Sandbox

# Files larger than this are refused outright rather than truncated: reading a
# 50MB binary into the context is never what the agent meant to do.
MAX_READ_BYTES = 2_000_000


class ReadFile(Tool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file from the workspace. Returns the contents with 1-indexed "
        "line numbers. Use start_line/end_line to read part of a large file."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace root."},
            "start_line": {"type": "integer", "description": "First line to return (1-indexed)."},
            "end_line": {"type": "integer", "description": "Last line to return, inclusive."},
        },
        "required": ["path"],
    }

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def run(
        self, path: str, start_line: int | None = None, end_line: int | None = None, **_: Any
    ) -> str:
        target = self.sandbox.resolve(path, must_exist=True)
        if target.is_dir():
            raise ToolError(f"{path} is a directory -- use list_dir to see what is in it.")
        if target.stat().st_size > MAX_READ_BYTES:
            raise ToolError(
                f"{path} is {target.stat().st_size} bytes, too large to read in one call. "
                "Use start_line and end_line, or grep for what you need."
            )
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"{path} is not UTF-8 text (it looks like a binary file).") from None

        lines = text.splitlines()
        if not lines:
            return f"{self.sandbox.relative(target)} is empty (0 lines)."

        lo = max(1, start_line or 1)
        hi = min(len(lines), end_line or len(lines))
        if lo > len(lines):
            raise ToolError(
                f"start_line {lo} is past the end of {path}, which has {len(lines)} lines."
            )
        if hi < lo:
            raise ToolError(f"end_line ({hi}) is before start_line ({lo}).")

        width = len(str(hi))
        body = "\n".join(f"{i:>{width}}  {lines[i - 1]}" for i in range(lo, hi + 1))
        header = f"{self.sandbox.relative(target)} (lines {lo}-{hi} of {len(lines)}):"
        return f"{header}\n{body}"


class WriteFile(Tool):
    name = "write_file"
    description = (
        "Write text to a file in the workspace, creating parent directories as needed. "
        "Overwrites the file if it already exists."
    )
    mutates = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace root."},
            "content": {"type": "string", "description": "Full contents to write."},
        },
        "required": ["path", "content"],
    }

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def run(self, path: str, content: str, **_: Any) -> str:
        target = self.sandbox.resolve(path)
        if target.is_dir():
            raise ToolError(f"{path} is a directory, not a file.")
        existed = target.exists()
        before = target.stat().st_size if existed else 0
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        verb = "Overwrote" if existed else "Created"
        # Reporting the before/after size is what lets a programmatic checker
        # tell "wrote the same bytes again" from "actually changed something".
        return (
            f"{verb} {self.sandbox.relative(target)}: "
            f"{len(content)} bytes, {len(content.splitlines())} lines"
            + (f" (was {before} bytes)" if existed else "")
        )


class ListDir(Tool):
    name = "list_dir"
    description = (
        "List the contents of a workspace directory. Directories are marked with a "
        "trailing slash; files show their size in bytes."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path; defaults to the root."},
            "recursive": {"type": "boolean", "description": "Walk subdirectories too."},
        },
    }

    def __init__(
        self, sandbox: Sandbox, *, max_entries: int = 200, scan_limit: int = 5_000
    ) -> None:
        self.sandbox = sandbox
        self.max_entries = max_entries
        # Capping what is *displayed* is not enough: a recursive listing that
        # first materialises and sorts every path under the root will sit there
        # for minutes on a large tree before any cap applies. The walk itself
        # has to stop.
        self.scan_limit = scan_limit

    def run(self, path: str = ".", recursive: bool = False, **_: Any) -> str:
        target = self.sandbox.resolve(path, must_exist=True)
        if not target.is_dir():
            raise ToolError(f"{path} is a file -- use read_file to read it.")

        entries: list[Path] = []
        overflowed = False
        walker = target.rglob("*") if recursive else target.iterdir()
        for scanned, candidate in enumerate(walker, start=1):
            if scanned > self.scan_limit:
                overflowed = True
                break
            if not _hidden(candidate, target):
                entries.append(candidate)

        if not entries:
            return f"{self.sandbox.relative(target)}/ is empty."

        entries.sort(key=lambda p: (not p.is_dir(), str(p).lower()))
        shown = entries[: self.max_entries]
        lines = [
            f"  {self.sandbox.relative(p)}/"
            if p.is_dir()
            else f"  {self.sandbox.relative(p)}  ({_size_of(p)} bytes)"
            for p in shown
        ]
        header = f"{self.sandbox.relative(target)}/ -- {len(entries)} entries"
        header += " (listing stopped early):" if overflowed else ":"
        out = [header, *lines]
        if len(entries) > len(shown):
            out.append(f"  ... and {len(entries) - len(shown)} more (narrow the path)")
        if overflowed:
            out.append(
                f"  (stopped after scanning {self.scan_limit} paths -- "
                "list a subdirectory instead of the whole tree)"
            )
        return "\n".join(out)


def _size_of(path: Path) -> int:
    """Size in bytes, or 0 for anything unreadable -- a broken symlink in a
    listing should not fail the whole listing."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _hidden(path: Path, root: Path) -> bool:
    """Skip dotfiles and caches. The agent is working on a project, not auditing
    one, and `.git/` alone would flood a recursive listing past any cap."""
    skip = {"__pycache__", ".git", ".venv", ".pytest_cache", ".ruff_cache", "node_modules"}
    rel = path.relative_to(root) if path != root else Path(".")
    return any(part in skip or part.startswith(".") for part in rel.parts)
