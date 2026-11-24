"""Filesystem, search and shell tools."""

from __future__ import annotations

import pytest

from actcheck.errors import ToolError
from actcheck.tools.fs import ListDir, ReadFile, WriteFile
from actcheck.tools.search import Grep
from actcheck.tools.shell import RunCommand


# ------------------------------------------------------------------ read_file
def test_read_file_returns_numbered_lines(sandbox) -> None:
    out = ReadFile(sandbox).run(path="README.md")
    assert "# Harbour" in out
    assert out.splitlines()[0].startswith("README.md (lines 1-")


def test_read_file_honours_a_line_range(sandbox) -> None:
    out = ReadFile(sandbox).run(path="README.md", start_line=1, end_line=2)
    assert len(out.splitlines()) == 3  # header plus two lines


def test_read_file_rejects_a_directory(sandbox) -> None:
    with pytest.raises(ToolError, match="use list_dir"):
        ReadFile(sandbox).run(path="docs")


def test_read_file_reports_a_missing_file(sandbox) -> None:
    with pytest.raises(ToolError, match="no such file"):
        ReadFile(sandbox).run(path="absent.md")


def test_read_file_rejects_a_start_past_the_end(sandbox) -> None:
    with pytest.raises(ToolError, match="past the end"):
        ReadFile(sandbox).run(path="README.md", start_line=9999)


def test_read_file_says_so_when_a_file_is_empty(sandbox) -> None:
    (sandbox.root / "empty.md").write_text("")
    assert "is empty" in ReadFile(sandbox).run(path="empty.md")


def test_read_file_rejects_binary(sandbox) -> None:
    (sandbox.root / "blob.txt").write_bytes(b"\xff\xfe\x00\x01binary")
    with pytest.raises(ToolError, match="not UTF-8"):
        ReadFile(sandbox).run(path="blob.txt")


# ----------------------------------------------------------------- write_file
def test_write_file_creates_and_reports_size(sandbox) -> None:
    out = WriteFile(sandbox).run(path="notes/new.md", content="hello\nworld\n")
    assert "Created" in out and "2 lines" in out
    assert (sandbox.root / "notes" / "new.md").read_text() == "hello\nworld\n"


def test_write_file_reports_the_previous_size_on_overwrite(sandbox) -> None:
    """The before/after size is what lets a checker tell a real change from a
    rewrite of identical bytes."""
    WriteFile(sandbox).run(path="x.md", content="aaaa")
    out = WriteFile(sandbox).run(path="x.md", content="bb")
    assert "Overwrote" in out and "was 4 bytes" in out


def test_write_file_cannot_escape_the_workspace(sandbox) -> None:
    with pytest.raises(ToolError):
        WriteFile(sandbox).run(path="../escaped.md", content="nope")


# -------------------------------------------------------------------- list_dir
def test_list_dir_lists_entries_with_sizes(sandbox) -> None:
    out = ListDir(sandbox).run(path=".")
    assert "README.md" in out and "bytes" in out
    assert "docs/" in out


def test_list_dir_recurses(sandbox) -> None:
    assert "docs/runbook.md" in ListDir(sandbox).run(path=".", recursive=True)


def test_list_dir_skips_dot_directories(sandbox) -> None:
    (sandbox.root / ".git").mkdir()
    (sandbox.root / ".git" / "config").write_text("x")
    assert ".git" not in ListDir(sandbox).run(path=".", recursive=True)


def test_list_dir_rejects_a_file(sandbox) -> None:
    with pytest.raises(ToolError, match="use read_file"):
        ListDir(sandbox).run(path="README.md")


def test_list_dir_stops_walking_a_huge_tree(sandbox) -> None:
    """Capping the display is not enough -- a recursive listing that first
    materialises every path sits there for minutes before any cap applies."""
    deep = sandbox.root / "deep"
    deep.mkdir()
    for i in range(60):
        (deep / f"f{i}.txt").write_text("x")
    out = ListDir(sandbox, max_entries=5, scan_limit=20).run(path=".", recursive=True)
    assert "stopped early" in out and "stopped after scanning 20 paths" in out


def test_list_dir_survives_a_broken_symlink(sandbox) -> None:
    (sandbox.root / "dangling").symlink_to(sandbox.root / "gone.txt")
    assert "dangling" in ListDir(sandbox).run(path=".")


def test_list_dir_caps_long_listings(sandbox) -> None:
    for i in range(30):
        (sandbox.root / f"f{i}.txt").write_text("x")
    out = ListDir(sandbox, max_entries=10).run(path=".")
    assert "and" in out and "more" in out


# ------------------------------------------------------------------------ grep
def test_grep_reports_path_and_line(sandbox) -> None:
    out = Grep(sandbox).run(pattern="Helsinki")
    assert "docs/runbook.md:" in out


def test_grep_is_case_insensitive_on_request(sandbox) -> None:
    assert "runbook" in Grep(sandbox).run(pattern="helsinki", ignore_case=True)


def test_grep_says_plainly_when_there_are_no_matches(sandbox) -> None:
    """An empty result has to read as an answer, not as a broken tool."""
    assert "No matches for" in Grep(sandbox).run(pattern="zzzznotthere")


def test_grep_rejects_an_invalid_regex(sandbox) -> None:
    with pytest.raises(ToolError, match="invalid regular expression"):
        Grep(sandbox).run(pattern="(unclosed")


def test_grep_respects_max_results(sandbox) -> None:
    out = Grep(sandbox).run(pattern="a", ignore_case=True, max_results=3)
    assert len([ln for ln in out.splitlines() if ":" in ln and ln.startswith(" ") is False]) <= 5


def test_grep_can_target_one_file(sandbox) -> None:
    out = Grep(sandbox).run(pattern="status", path="src/app.py")
    assert "src/app.py" in out and "runbook" not in out


# ----------------------------------------------------------------- run_command
@pytest.fixture
def shell(sandbox):
    return RunCommand(sandbox, allow=("echo", "ls", "false"), timeout=5.0)


def test_run_command_returns_output_and_exit_code(shell) -> None:
    out = shell.run(command="echo hello")
    assert "hello" in out and "exit code: 0" in out


def test_run_command_reports_a_non_zero_exit_without_raising(shell) -> None:
    """A failing command is an observation the agent needs, not an exception."""
    assert "exit code: 1" in shell.run(command="false")


def test_run_command_enforces_the_allowlist(shell) -> None:
    with pytest.raises(ToolError, match="not on the allowlist"):
        shell.run(command="rm -rf /")


def test_run_command_rejects_pipes(shell) -> None:
    with pytest.raises(ToolError, match="does not use a shell"):
        shell.run(command="ls | grep x")


def test_run_command_rejects_chaining(shell) -> None:
    """The escape an allowlist on argv[0] alone would miss."""
    with pytest.raises(ToolError, match="does not use a shell"):
        shell.run(command="echo hi; rm -rf /")


def test_run_command_rejects_substitution(shell) -> None:
    with pytest.raises(ToolError, match="does not use a shell"):
        shell.run(command="echo $(whoami)")


def test_run_command_rejects_redirection(shell) -> None:
    with pytest.raises(ToolError, match="does not use a shell"):
        shell.run(command="echo hi > /etc/passwd")


def test_run_command_rejects_an_empty_command(shell) -> None:
    with pytest.raises(ToolError, match="empty"):
        shell.run(command="   ")


def test_run_command_reports_unparseable_quoting(shell) -> None:
    with pytest.raises(ToolError, match="could not parse"):
        shell.run(command="echo 'unterminated")


def test_run_command_times_out(sandbox) -> None:
    tool = RunCommand(sandbox, allow=("sleep",), timeout=0.2)
    with pytest.raises(ToolError, match="did not finish"):
        tool.run(command="sleep 5")


def test_run_command_runs_in_the_workspace(shell, sandbox) -> None:
    assert "README.md" in shell.run(command="ls")
