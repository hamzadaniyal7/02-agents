"""The trace is the product; it has to survive the runs that go wrong."""

from __future__ import annotations

from pathlib import Path

from actcheck.trace import NullTrace, Trace, latest_trace, read_trace, render_trace
from actcheck.types import (
    Judgement,
    Observation,
    Plan,
    PlanStep,
    RunResult,
    RunStatus,
    Step,
    ToolCall,
    Verdict,
)


def a_step(index: int = 1) -> Step:
    return Step(
        index=index,
        thought="looking",
        tool_call=ToolCall(name="read_file", args={"path": "README.md"}),
        observation=Observation(content="# Harbour"),
        verdict=Verdict(Judgement.PROGRESS, "returned content"),
    )


def test_events_are_written_as_they_happen(tmp_path: Path) -> None:
    """A run that crashes is the one you most want the trace for."""
    trace = Trace(tmp_path / "t.jsonl", "run_1", "task")
    trace.step(a_step())
    # Deliberately not closed: the file must already be readable.
    assert len(read_trace(tmp_path / "t.jsonl")) == 2
    trace.close()


def test_every_event_carries_the_run_id(tmp_path: Path) -> None:
    trace = Trace(tmp_path / "t.jsonl", "run_7", "task")
    trace.note("something happened")
    trace.close()
    assert all(e["run_id"] == "run_7" for e in read_trace(tmp_path / "t.jsonl"))


def test_the_plan_is_recorded(tmp_path: Path) -> None:
    trace = Trace(tmp_path / "t.jsonl", "r", "task")
    trace.plan(Plan(goal="g", steps=[PlanStep(id=1, description="do it")]))
    trace.close()
    event = read_trace(tmp_path / "t.jsonl")[1]
    assert event["event"] == "plan" and event["plan"]["steps"][0]["description"] == "do it"


def test_the_terminal_event_records_the_outcome(tmp_path: Path) -> None:
    trace = Trace(tmp_path / "t.jsonl", "r", "task")
    trace.finish(RunResult(task="t", status=RunStatus.STUCK, reason="went in circles"))
    trace.close()
    end = read_trace(tmp_path / "t.jsonl")[-1]
    assert end["status"] == "stuck" and end["reason"] == "went in circles"


def test_a_truncated_final_line_does_not_lose_the_rest(tmp_path: Path) -> None:
    """The normal state of a trace from a killed process."""
    path = tmp_path / "t.jsonl"
    trace = Trace(path, "r", "task")
    trace.step(a_step())
    trace.close()
    with path.open("a") as fh:
        fh.write('{"event": "step", "index": 2, "trunca')
    assert len(read_trace(path)) == 2


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text('{"event": "run_start", "ts": 1}\n\n\n')
    assert len(read_trace(path)) == 1


def test_the_context_manager_records_an_exception(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    try:
        with Trace(path, "r", "task"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert read_trace(path)[-1]["error"].startswith("ValueError")


def test_rendering_shows_the_shape_of_the_run(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    trace = Trace(path, "r", "find the build command")
    trace.plan(Plan(goal="g", steps=[PlanStep(id=1, description="read it")]))
    trace.step(a_step())
    trace.note("loop guard: repeat")
    trace.finish(RunResult(task="t", status=RunStatus.SUCCESS, reason="done"))
    trace.close()

    out = render_trace(read_trace(path))
    assert "RUN   find the build command" in out
    assert "PLAN" in out and "STEP" in out and "NOTE" in out
    assert "END   [success]" in out


def test_verbose_rendering_includes_tool_output(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    trace = Trace(path, "r", "task")
    trace.step(a_step())
    trace.close()
    assert "# Harbour" in render_trace(read_trace(path), verbose=True)


def test_rendering_an_empty_trace_does_not_blow_up() -> None:
    assert render_trace([]) == ""


def test_null_trace_writes_nothing(tmp_path: Path) -> None:
    trace = NullTrace()
    trace.step(a_step())
    trace.finish(RunResult(task="t", status=RunStatus.SUCCESS))
    assert not list(tmp_path.glob("*.jsonl"))


def test_latest_trace_finds_the_newest(tmp_path: Path) -> None:
    import os
    import time

    for name in ("a", "b"):
        Trace(tmp_path / f"{name}.jsonl", name, "task").close()
        time.sleep(0.01)
    os.utime(tmp_path / "b.jsonl", (time.time() + 10, time.time() + 10))
    assert latest_trace(tmp_path).name == "b.jsonl"


def test_latest_trace_handles_an_empty_directory(tmp_path: Path) -> None:
    assert latest_trace(tmp_path / "nothing-here") is None
