"""Catching the agent going in circles."""

from __future__ import annotations

from actcheck.loopguard import LoopGuard
from actcheck.types import Judgement, ToolCall, Verdict


def call(path: str = "a.md") -> ToolCall:
    return ToolCall(name="read_file", args={"path": path})


def progress() -> Verdict:
    return Verdict(Judgement.PROGRESS, "fine")


def stalled() -> Verdict:
    return Verdict(Judgement.NO_PROGRESS, "nothing")


def test_one_call_does_not_trip() -> None:
    assert not LoopGuard().observe_call(call()).tripped


def test_repetition_trips_past_the_threshold() -> None:
    guard = LoopGuard(repeat_threshold=2)
    assert not guard.observe_call(call()).tripped
    assert not guard.observe_call(call()).tripped
    assert guard.observe_call(call()).tripped


def test_argument_order_does_not_count_as_a_different_call() -> None:
    """Models reorder keys between turns; that is not a new action."""
    guard = LoopGuard(repeat_threshold=1)
    guard.observe_call(ToolCall(name="grep", args={"pattern": "x", "path": "."}))
    report = guard.observe_call(ToolCall(name="grep", args={"path": ".", "pattern": "x"}))
    assert report.tripped


def test_a_different_call_resets_the_repeat_counter() -> None:
    guard = LoopGuard(repeat_threshold=2)
    guard.observe_call(call("a.md"))
    guard.observe_call(call("a.md"))
    guard.observe_call(call("b.md"))
    assert not guard.observe_call(call("a.md")).tripped


def test_the_first_trip_nudges_rather_than_killing_the_run() -> None:
    guard = LoopGuard(repeat_threshold=1)
    guard.observe_call(call())
    first = guard.observe_call(call())
    assert first.tripped and not first.fatal
    assert "different tool" in first.message


def test_repeating_after_a_nudge_is_fatal() -> None:
    guard = LoopGuard(repeat_threshold=1)
    guard.observe_call(call())
    guard.observe_call(call())
    assert guard.observe_call(call()).fatal


def test_stalling_trips_after_enough_non_advancing_verdicts() -> None:
    guard = LoopGuard(no_progress_threshold=3)
    assert not guard.observe_verdict(stalled()).tripped
    assert not guard.observe_verdict(stalled()).tripped
    assert guard.observe_verdict(stalled()).tripped


def test_progress_resets_the_stall_counter() -> None:
    guard = LoopGuard(no_progress_threshold=2)
    guard.observe_verdict(stalled())
    guard.observe_verdict(progress())
    assert not guard.observe_verdict(stalled()).tripped


def test_step_done_counts_as_progress() -> None:
    guard = LoopGuard(no_progress_threshold=2)
    guard.observe_verdict(stalled())
    guard.observe_verdict(Verdict(Judgement.STEP_DONE, "done"))
    assert guard.stalled_steps == 0


def test_replanning_resets_the_stall_counter() -> None:
    """A new plan deserves a fresh run of attempts, or replanning can never
    pay off."""
    guard = LoopGuard(no_progress_threshold=2)
    guard.observe_verdict(stalled())
    guard.reset_progress()
    assert not guard.observe_verdict(stalled()).tripped
