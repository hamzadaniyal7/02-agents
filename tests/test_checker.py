"""The 'check' in plan-act-check."""

from __future__ import annotations

import pytest

from actcheck.checker import Checker
from actcheck.config import Config
from actcheck.llm.scripted import ScriptedModel
from actcheck.types import Judgement, Observation, PlanStep, ToolCall


@pytest.fixture
def checker(cfg: Config) -> Checker:
    return Checker(None, cfg.merged({"checker": "programmatic"}))


def rule(checker: Checker, name: str = "read_file", **obs):
    return checker.check(
        goal="g",
        plan_step=PlanStep(id=1, description="d", success_criterion="c"),
        call=ToolCall(name=name, args={"path": "a.md"}),
        observation=Observation(**obs),
    )


def test_finish_ends_the_goal(checker: Checker) -> None:
    assert rule(checker, "finish", content="done").judgement is Judgement.GOAL_DONE


def test_give_up_is_blocked_not_failure(checker: Checker) -> None:
    assert (
        rule(checker, "give_up", content="no rate limit documented").judgement is Judgement.BLOCKED
    )


def test_a_failed_tool_is_no_progress(checker: Checker) -> None:
    verdict = rule(checker, content="boom", ok=False, error="boom")
    assert verdict.judgement is Judgement.NO_PROGRESS
    assert "tool failed" in verdict.reason


def test_an_empty_result_is_no_progress(checker: Checker) -> None:
    assert rule(checker, content="   ").judgement is Judgement.NO_PROGRESS


def test_a_successful_search_that_found_nothing_is_no_progress(checker: Checker) -> None:
    """Succeeding at the tool level and failing at the task level are different
    things, and telling them apart is the most useful rule here."""
    verdict = rule(checker, "grep", content="No matches for 'zzz' in . (6 files searched).")
    assert verdict.judgement is Judgement.NO_PROGRESS
    assert "found nothing" in verdict.reason


def test_an_entry_count_is_not_mistaken_for_emptiness(checker: Checker) -> None:
    """'10 entries' contains the substring '0 entries'."""
    verdict = rule(checker, "list_dir", content="./ -- 10 entries:\n  README.md  (100 bytes)")
    assert verdict.judgement is not Judgement.NO_PROGRESS


def test_real_content_is_progress(checker: Checker) -> None:
    assert rule(checker, content="# Harbour\n\nthe service").judgement is Judgement.PROGRESS


def test_the_same_bytes_twice_is_no_progress(checker: Checker) -> None:
    """Catches what the loop guard misses: two different calls returning the
    same thing, e.g. reading one file by two paths."""
    assert rule(checker, content="identical").judgement is Judgement.PROGRESS
    assert rule(checker, content="identical").judgement is Judgement.NO_PROGRESS


def test_checking_can_be_turned_off(cfg: Config) -> None:
    off = Checker(None, cfg.merged({"checker": "off"}))
    verdict = rule(off, content="")
    assert verdict.judgement is Judgement.PROGRESS and "disabled" in verdict.reason


def test_programmatic_only_never_closes_a_step(checker: Checker) -> None:
    """Only a judgement about the success criterion can rule a step done, and
    that is exactly what this configuration opted out of."""
    assert rule(checker, content="plenty of content here").judgement is not Judgement.STEP_DONE


def test_the_model_checker_is_consulted_when_rules_are_inconclusive(cfg: Config) -> None:
    both = Checker(ScriptedModel(), cfg.merged({"checker": "both"}))
    verdict = rule(both, content="# Harbour\n\nreal content")
    assert verdict.by == "model"


def test_the_model_checker_is_not_consulted_for_certain_rulings(cfg: Config) -> None:
    """Cheap-and-certain first: paying for a model call to learn the tool
    errored roughly doubles the cost of a run for nothing."""
    both = Checker(ScriptedModel(), cfg.merged({"checker": "both"}))
    verdict = rule(both, content="boom", ok=False, error="boom")
    assert verdict.by == "programmatic"
    assert both.usage.calls == 0


def test_an_unparseable_verdict_does_not_stall_the_run(cfg: Config) -> None:
    class Garbage(ScriptedModel):
        def complete(self, *a, **kw):
            response = super().complete(*a, **kw)
            response.text = "not json at all"
            return response

    checker = Checker(Garbage(), cfg.merged({"checker": "model"}))
    verdict = rule(checker, content="content")
    assert verdict.judgement is Judgement.PROGRESS
    assert "no usable verdict" in verdict.reason


def test_checker_usage_is_tracked(cfg: Config) -> None:
    both = Checker(ScriptedModel(), cfg.merged({"checker": "both"}))
    rule(both, content="some content")
    assert both.usage.calls == 1 and both.last_usage.calls == 1
