"""The eval harness, including the ways a scorer can flatter itself."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from actcheck.config import Config
from actcheck.evaluate import EvalReport, TaskResult, _score, load_tasks, run_eval
from actcheck.types import Observation, RunResult, RunStatus, Step, ToolCall


def result(answer: str = "", status: RunStatus = RunStatus.SUCCESS, steps: int = 0) -> RunResult:
    run = RunResult(task="t", status=status, answer=answer)
    run.steps = [
        Step(index=i, tool_call=ToolCall(name="read_file"), observation=Observation(content="x"))
        for i in range(1, steps + 1)
    ]
    return run


def test_a_matching_status_and_answer_passes(tmp_path: Path) -> None:
    task = {
        "id": "t",
        "task": "q",
        "expect": {"status": "success", "answer_contains": ["make build"]},
    }
    assert _score(task, result("Run make build to build it."), tmp_path).passed


def test_a_wrong_status_fails(tmp_path: Path) -> None:
    task = {"id": "t", "task": "q", "expect": {"status": "failed"}}
    scored = _score(task, result(), tmp_path)
    assert not scored.passed and "status" in scored.failures[0]


def test_a_missing_string_fails_and_names_it(tmp_path: Path) -> None:
    task = {"id": "t", "task": "q", "expect": {"answer_contains": ["Helsinki"]}}
    scored = _score(task, result("The office is in Toronto."), tmp_path)
    assert not scored.passed and "Helsinki" in scored.failures[0]


def test_matching_is_case_insensitive(tmp_path: Path) -> None:
    task = {"id": "t", "task": "q", "expect": {"answer_contains": ["helsinki"]}}
    assert _score(task, result("Helsinki"), tmp_path).passed


def test_a_substring_check_would_flatter_a_short_answer(tmp_path: Path) -> None:
    """The reason answer_regex exists: '3' is satisfied by the order id
    'A-1003', which is how an eval reports a pass the agent did not earn."""
    task = {"id": "t", "task": "q", "expect": {"answer_contains": ["3"]}}
    assert _score(task, result("The ids are A-1001 and A-1003."), tmp_path).passed

    strict = {"id": "t", "task": "q", "expect": {"answer_regex": [r"\b3\b"]}}
    assert not _score(strict, result("The ids are A-1001 and A-1003."), tmp_path).passed


def test_a_bounded_pattern_matches_a_real_answer(tmp_path: Path) -> None:
    task = {"id": "t", "task": "q", "expect": {"answer_regex": [r"\b3\b"]}}
    assert _score(task, result("There are 3 refunded orders."), tmp_path).passed


def test_file_effects_are_checked_on_disk(tmp_path: Path) -> None:
    (tmp_path / "summary.md").write_text("Harbour is built with make build")
    task = {"id": "t", "task": "q", "expect": {"files": {"summary.md": "make build"}}}
    assert _score(task, result(), tmp_path).passed


def test_a_missing_file_fails(tmp_path: Path) -> None:
    task = {"id": "t", "task": "q", "expect": {"files": {"summary.md": "anything"}}}
    scored = _score(task, result(), tmp_path)
    assert not scored.passed and "was not created" in scored.failures[0]


def test_a_file_with_the_wrong_contents_fails(tmp_path: Path) -> None:
    (tmp_path / "summary.md").write_text("something else entirely")
    task = {"id": "t", "task": "q", "expect": {"files": {"summary.md": "make build"}}}
    assert not _score(task, result(), tmp_path).passed


def test_an_over_budget_run_fails_even_if_the_answer_is_right(tmp_path: Path) -> None:
    """Two configurations with equal success and a 2x step gap are not equally
    good."""
    task = {"id": "t", "task": "q", "expect": {"answer_contains": ["ok"], "max_steps": 2}}
    scored = _score(task, result("ok", steps=9), tmp_path)
    assert not scored.passed and "9 steps" in scored.failures[0]


# ----------------------------------------------------------------- reporting
def test_metrics_cover_the_headline_numbers() -> None:
    report = EvalReport(
        results=[
            TaskResult("a", "q", True, "success", "success", answer_match=True, steps=3),
            TaskResult("b", "q", False, "stuck", "success", answer_match=False, steps=7),
        ]
    )
    metrics = report.metrics()
    assert metrics["tasks"] == 2
    assert metrics["success"] == 0.5
    assert metrics["steps_mean"] == 5.0
    assert metrics["stuck"] == 1


def test_abstention_is_scored_on_the_impossible_tasks_only() -> None:
    report = EvalReport(
        results=[
            TaskResult("ok", "q", True, "success", "success"),
            TaskResult("absent", "q", False, "success", "failed"),
        ]
    )
    assert report.metrics()["abstention_correct"] == 0.0


def test_the_report_lists_what_failed() -> None:
    report = EvalReport(
        results=[TaskResult("b", "q", False, "stuck", "success", failures=["went in circles"])]
    )
    assert "went in circles" in report.render()


def test_the_report_serialises() -> None:
    report = EvalReport(results=[TaskResult("a", "q", True, "success", "success")])
    assert json.loads(report.to_json())["metrics"]["tasks"] == 1


# -------------------------------------------------------------------- tasks
def test_the_shipped_task_set_parses() -> None:
    tasks = load_tasks()
    assert len(tasks) >= 8
    assert all("id" in t and "task" in t for t in tasks)


def test_task_ids_are_unique() -> None:
    ids = [t["id"] for t in load_tasks()]
    assert len(ids) == len(set(ids))


def test_a_malformed_task_file_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"id": "a", "task": "x"}\nnot json\n')
    with pytest.raises(ValueError, match="tasks.jsonl:2"):
        load_tasks(path)


def test_each_task_runs_in_its_own_copy_of_the_workspace(cfg: Config, tmp_path: Path) -> None:
    """A task that writes files must not change the result of the next one."""
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "id": "writer",
                "task": "Create a file called marker.md",
                "expect": {"status": "success"},
            }
        )
        + "\n"
    )
    run_eval(cfg.merged({"max_steps": 4}), tasks_path=tasks)
    assert not (Path(cfg.workspace) / "marker.md").exists()


def test_only_filters_tasks(cfg: Config) -> None:
    report = run_eval(cfg.merged({"max_steps": 4}), only="oncall")
    assert len(report.results) == 1
    assert report.results[0].task_id == "oncall-office"
