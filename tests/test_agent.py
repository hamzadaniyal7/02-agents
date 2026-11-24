"""The loop, end to end, against a scripted model."""

from __future__ import annotations

from actcheck.agent import Agent
from actcheck.config import Config
from actcheck.llm.scripted import ScriptedModel
from actcheck.trace import read_trace
from actcheck.types import Judgement, RunStatus, Step, StepStatus


def agent(cfg: Config, script: list[dict], **overrides) -> Agent:
    return Agent(cfg.merged(overrides), model=ScriptedModel(script))


READ_THEN_FINISH = [
    {"tool": "read_file", "args": {"path": "README.md"}},
    {"tool": "finish", "args": {"answer": "The build command is make build."}},
]


# ------------------------------------------------------------------- success
def test_a_completed_run_reports_success(cfg: Config) -> None:
    result = agent(cfg, READ_THEN_FINISH).run("What builds this?")
    assert result.status is RunStatus.SUCCESS
    assert result.ok
    assert "make build" in result.answer
    assert result.n_steps == 2


def test_the_finish_tool_decides_the_answer(cfg: Config) -> None:
    result = agent(cfg, READ_THEN_FINISH).run("task")
    assert result.answer == "The build command is make build."


def test_giving_up_is_a_failure_with_a_reason_not_a_crash(cfg: Config) -> None:
    script = [{"tool": "give_up", "args": {"reason": "no rate limit is documented anywhere"}}]
    result = agent(cfg, script).run("What is the rate limit?")
    assert result.status is RunStatus.FAILED
    assert "rate limit" in result.answer
    assert result.reason == "agent gave up"


def test_the_on_step_hook_sees_every_step(cfg: Config) -> None:
    seen: list[Step] = []
    Agent(cfg, model=ScriptedModel(READ_THEN_FINISH), on_step=seen.append).run("task")
    assert [s.tool_call.name for s in seen] == ["read_file", "finish"]


# ------------------------------------------------------------------ budgets
def test_the_step_ceiling_ends_the_run_and_says_so(cfg: Config) -> None:
    script = [{"tool": "read_file", "args": {"path": f"f{i}.md"}} for i in range(10)]
    result = agent(cfg, script, max_steps=3).run("task")
    assert result.status is RunStatus.BUDGET_EXHAUSTED
    assert "step budget" in result.reason
    assert result.n_steps == 3


def test_a_run_cut_short_still_reports_what_it_found(cfg: Config) -> None:
    """Reporting nothing throws away real work; reporting it as finished is
    worse. It is returned, labelled partial."""
    script = [{"tool": "read_file", "args": {"path": "README.md"}}] * 5
    result = agent(cfg, script, max_steps=2).run("task")
    assert "partial" in result.answer
    assert "Harbour" in result.answer


def test_an_exhausted_budget_is_not_reported_as_success(cfg: Config) -> None:
    script = [{"tool": "list_dir", "args": {"path": "."}}] * 5
    assert not agent(cfg, script, max_steps=2).run("task").ok


# --------------------------------------------------------------- loop guard
def test_repeating_a_call_gets_a_nudge_in_the_transcript(cfg: Config) -> None:
    script = [{"tool": "read_file", "args": {"path": "README.md"}}] * 4
    result = agent(cfg, script, repeat_threshold=1, max_steps=6).run("task")
    guarded = [s for s in result.steps if s.observation and s.observation.meta.get("source")]
    assert guarded, "the guard should have replaced a repeated call with a nudge"
    assert "identical arguments" in guarded[0].observation.content


def test_a_guarded_call_is_not_executed_again(cfg: Config) -> None:
    """Running it would cost a step to learn what we already know."""
    script = [{"tool": "read_file", "args": {"path": "README.md"}}] * 4
    result = agent(cfg, script, repeat_threshold=1, max_steps=6).run("task")
    guarded = [s for s in result.steps if s.observation.meta.get("source") == "loop_guard"]
    assert all(not s.observation.ok for s in guarded)


def test_repeating_after_a_nudge_ends_the_run(cfg: Config) -> None:
    """The guard nudges once. An agent that makes the identical call again
    after being told it will not help is not responding to feedback, and
    nothing later in the run is going to change that.

    Regression: `GuardReport.fatal` was computed and recorded but never acted
    on, so this run took 13 steps and 1870 tokens to reach the same conclusion
    it reaches here in 4.
    """
    script = [{"tool": "read_file", "args": {"path": "nonexistent.md"}}] * 20
    result = agent(cfg, script, repeat_threshold=2, max_steps=24, plan=False).run("task")
    assert result.status is RunStatus.STUCK
    assert result.n_steps == 4
    assert "after being told it would not help" in result.reason


def test_the_guard_ends_a_doomed_run_well_inside_the_budget(cfg: Config) -> None:
    script = [{"tool": "read_file", "args": {"path": "nonexistent.md"}}] * 30
    result = agent(cfg, script, max_steps=24, plan=True).run("task")
    assert result.n_steps < 24
    assert result.status is RunStatus.STUCK


def test_a_run_that_goes_nowhere_ends_as_stuck(cfg: Config) -> None:
    script = [{"tool": "read_file", "args": {"path": "missing.md"}}] * 10
    result = agent(cfg, script, no_progress_threshold=2, max_steps=12, plan=False).run("task")
    assert result.status is RunStatus.STUCK
    assert result.n_steps < 12, "it should stop well before the step ceiling"


def test_stuck_is_distinguishable_from_budget_exhausted(cfg: Config) -> None:
    script = [{"tool": "read_file", "args": {"path": "missing.md"}}] * 10
    result = agent(cfg, script, no_progress_threshold=2, max_steps=12, plan=False).run("t")
    assert result.status is not RunStatus.BUDGET_EXHAUSTED


# ------------------------------------------------------------ silent turns
def test_a_silent_turn_is_nudged_once(cfg: Config) -> None:
    script = [{"text": "I think the answer is probably in the README."}, *READ_THEN_FINISH]
    result = agent(cfg, script).run("task")
    assert result.status is RunStatus.SUCCESS


def test_two_silent_turns_end_the_run(cfg: Config) -> None:
    """A model that will not use the tools should not burn the whole budget."""
    script = [{"text": "Just talking."}, {"text": "Still talking."}]
    result = agent(cfg, script).run("task")
    assert result.status is RunStatus.FAILED
    assert "without calling finish" in result.reason
    assert result.answer == "Still talking."


# ------------------------------------------------------------- tool errors
def test_a_tool_error_does_not_end_the_run(cfg: Config) -> None:
    script = [{"tool": "read_file", "args": {"path": "nope.md"}}, *READ_THEN_FINISH]
    result = agent(cfg, script).run("task")
    assert result.status is RunStatus.SUCCESS
    assert result.tool_errors == 1


def test_a_bad_tool_name_is_reported_to_the_model_not_raised(cfg: Config) -> None:
    script = [{"tool": "teleport", "args": {}}, *READ_THEN_FINISH]
    result = agent(cfg, script).run("task")
    assert result.status is RunStatus.SUCCESS
    assert "no such tool" in result.steps[0].observation.content


# -------------------------------------------------------------------- plan
def test_planning_produces_a_plan_on_the_result(cfg: Config) -> None:
    result = agent(cfg, READ_THEN_FINISH, plan=True).run("task")
    assert result.plan is not None and result.plan.steps


def test_finishing_marks_outstanding_plan_steps_as_skipped(cfg: Config) -> None:
    result = agent(cfg, READ_THEN_FINISH, plan=True).run("task")
    assert all(s.status is not StepStatus.PENDING for s in result.plan.steps)


def test_the_loop_runs_without_a_plan(cfg: Config) -> None:
    result = agent(cfg, READ_THEN_FINISH, plan=False).run("task")
    assert result.plan is None and result.ok


# ----------------------------------------------------------------- sandbox
def test_the_agent_cannot_escape_the_workspace(cfg: Config) -> None:
    script = [{"tool": "read_file", "args": {"path": "../../../../etc/passwd"}}, *READ_THEN_FINISH]
    result = agent(cfg, script).run("task")
    assert "escapes the workspace" in result.steps[0].observation.content


def test_a_read_only_agent_has_no_write_tool(cfg: Config) -> None:
    script = [{"tool": "write_file", "args": {"path": "x.md", "content": "y"}}]
    result = agent(cfg, script, enabled_tools=("read_file", "grep")).run("task")
    assert "no such tool" in result.steps[0].observation.content


# ------------------------------------------------------------------- trace
def test_a_run_writes_a_readable_trace(cfg: Config) -> None:
    result = agent(cfg, READ_THEN_FINISH).run("task")
    events = read_trace(result.trace_path)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert kinds.count("step") == 2


def test_the_terminal_event_records_the_status(cfg: Config) -> None:
    """The bug this guards: closing the trace file before the status is known."""
    result = agent(cfg, READ_THEN_FINISH).run("task")
    end = read_trace(result.trace_path)[-1]
    assert end["event"] == "run_end" and end["status"] == "success"


def test_a_budget_exhausted_run_still_writes_its_trace(cfg: Config) -> None:
    script = [{"tool": "list_dir", "args": {"path": "."}}] * 5
    result = agent(cfg, script, max_steps=2).run("task")
    assert read_trace(result.trace_path)[-1]["status"] == "budget_exhausted"


def test_tracing_can_be_turned_off(cfg: Config) -> None:
    result = agent(cfg, READ_THEN_FINISH, trace=False).run("task")
    assert result.trace_path == ""


# ------------------------------------------------------------------- usage
def test_every_model_call_is_counted(cfg: Config) -> None:
    """Agent turns, planning and checking all spend tokens."""
    result = agent(cfg, READ_THEN_FINISH, plan=True, checker="both").run("task")
    assert result.usage.calls > result.n_steps


def test_the_run_is_timed(cfg: Config) -> None:
    assert agent(cfg, READ_THEN_FINISH).run("task").duration_ms >= 0


def test_verdicts_are_attached_to_every_step(cfg: Config) -> None:
    result = agent(cfg, READ_THEN_FINISH).run("task")
    assert all(s.verdict is not None for s in result.steps)
    assert result.steps[-1].verdict.judgement is Judgement.GOAL_DONE
