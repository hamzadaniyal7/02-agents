"""Planning, and surviving a planner that returns nonsense."""

from __future__ import annotations

import json

import pytest

from actcheck.config import Config
from actcheck.llm.base import Model, ModelResponse
from actcheck.planner import Planner, _loads
from actcheck.tools import build_registry
from actcheck.types import StepStatus, Usage


class Canned(Model):
    name = "canned"

    def __init__(self, text: str) -> None:
        self.text = text

    def respond(self, *a, **kw) -> ModelResponse:  # pragma: no cover - unused here
        raise NotImplementedError

    def complete(self, *a, **kw) -> ModelResponse:
        return ModelResponse(text=self.text, usage=Usage(output_tokens=10, calls=1))


PLAN = json.dumps(
    {
        "goal": "Find the build command",
        "steps": [
            {"description": "Read the README", "success_criterion": "README has been read"},
            {"description": "Report it", "success_criterion": "finish has been called"},
        ],
    }
)


@pytest.fixture
def registry(cfg: Config):
    return build_registry(cfg)


def test_parses_a_plan(cfg: Config, registry) -> None:
    plan = Planner(Canned(PLAN), cfg).plan("task", registry)
    assert plan.goal == "Find the build command"
    assert [s.id for s in plan.steps] == [1, 2]
    assert plan.steps[0].success_criterion == "README has been read"


def test_tolerates_a_code_fence(cfg: Config, registry) -> None:
    plan = Planner(Canned(f"```json\n{PLAN}\n```"), cfg).plan("task", registry)
    assert len(plan.steps) == 2


def test_tolerates_prose_around_the_json(cfg: Config, registry) -> None:
    plan = Planner(Canned(f"Here you go:\n{PLAN}\nHope that helps."), cfg).plan("t", registry)
    assert len(plan.steps) == 2


def test_a_broken_plan_degrades_instead_of_failing_the_run(cfg: Config, registry) -> None:
    """The loop works without a plan, so a malformed one must not be fatal."""
    plan = Planner(Canned("I refuse to plan."), cfg).plan("do the thing", registry)
    assert len(plan.steps) == 1
    assert plan.steps[0].description == "do the thing"
    assert "no usable plan" in plan.steps[0].note


def test_an_empty_step_list_degrades_too(cfg: Config, registry) -> None:
    plan = Planner(Canned('{"goal": "g", "steps": []}'), cfg).plan("task", registry)
    assert len(plan.steps) == 1


def test_steps_are_capped(cfg: Config, registry) -> None:
    many = json.dumps(
        {
            "goal": "g",
            "steps": [{"description": f"s{i}", "success_criterion": "c"} for i in range(20)],
        }
    )
    plan = Planner(Canned(many), cfg.merged({"plan_max_steps": 3})).plan("t", registry)
    assert len(plan.steps) == 3


def test_replanning_increments_the_revision(cfg: Config, registry) -> None:
    planner = Planner(Canned(PLAN), cfg)
    first = planner.plan("task", registry)
    second = planner.replan("task", first, [], registry, "it went wrong")
    assert second.revisions == first.revisions + 1


def test_replanning_carries_completed_steps_over(cfg: Config, registry) -> None:
    """Re-doing finished work is the commonest way a replanning agent wastes
    the budget it has left."""
    planner = Planner(Canned(PLAN), cfg)
    first = planner.plan("task", registry)
    first.steps[0].status = StepStatus.DONE
    second = planner.replan("task", first, [], registry, "evidence")
    assert second.steps[0].status is StepStatus.DONE
    assert "carried over" in second.steps[0].note


def test_planner_usage_is_tracked(cfg: Config, registry) -> None:
    planner = Planner(Canned(PLAN), cfg)
    planner.plan("task", registry)
    assert planner.usage.calls == 1 and planner.last_usage.calls == 1


@pytest.mark.parametrize(
    "text,expected",
    [('{"a": 1}', {"a": 1}), ('```json\n{"a": 1}\n```', {"a": 1}), ("nope", None), ("", None)],
)
def test_loads_handles_the_usual_wrappers(text: str, expected) -> None:
    assert _loads(text) == expected
