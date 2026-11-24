"""Search over the workspace.

A grep that reports *where* a match is, not just that one exists. An agent that
gets back `path:line: text` can go straight to `read_file` with a line range;
one that gets back a list of filenames has to read whole files to find out it
picked the wrong one, and that is three wasted steps and a lot of context.
"""

from __future__ import annotations

import re
from typing import Any

from actcheck.errors import ToolError
from actcheck.tools.base import Tool
from actcheck.tools.fs import _hidden
from actcheck.tools.sandbox import Sandbox

TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".cfg",
    ".ini",
    ".sh",
    ".html",
    ".css",
    ".sql",
    ".rst",
    ".csv",
    "",
}


class Grep(Tool):
    name = "grep"
    description = (
        "Search workspace files for a regular expression. Returns matching lines as "
        "path:line: text. Use this before read_file to find where something is defined."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regular expression."},
            "path": {"type": "string", "description": "File or directory to search; default root."},
            "glob": {"type": "string", "description": "Only search files matching this glob."},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive match."},
            "max_results": {
                "type": "integer",
                "description": "Cap on matching lines (default 50).",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    def run(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        ignore_case: bool = False,
        max_results: int = 50,
        **_: Any,
    ) -> str:
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ToolError(f"invalid regular expression {pattern!r}: {exc}") from None

        root = self.sandbox.resolve(path, must_exist=True)
        files = (
            [root]
            if root.is_file()
            else [
                p
                for p in sorted(root.rglob(glob or "*"))
                if p.is_file() and not _hidden(p, root) and p.suffix.lower() in TEXT_SUFFIXES
            ]
        )

        hits: list[str] = []
        scanned = 0
        for file in files:
            if len(hits) >= max_results:
                break
            try:
                text = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # binary or unreadable: not an error, just not searchable
            scanned += 1
            rel = self.sandbox.relative(file)
            for n, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{rel}:{n}: {line.strip()[:200]}")
                    if len(hits) >= max_results:
                        break

        if not hits:
            # An empty result is a real answer, and saying so plainly stops the
            # model reading it as a broken tool and retrying the same search.
            return (
                f"No matches for {pattern!r} in {self.sandbox.relative(root)} "
                f"({scanned} files searched)."
            )
        header = f"{len(hits)} match(es) for {pattern!r} in {scanned} file(s):"
        capped = (
            "\n  (result cap reached -- narrow the pattern for more)"
            if len(hits) >= max_results
            else ""
        )
        return header + "\n" + "\n".join(hits) + capped
