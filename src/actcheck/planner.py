"""Planning, and the more interesting half: replanning.

A plan is useful for three concrete reasons, none of which is that the model
needs to be told how to think:

1. It gives the checker something to check *against*. Without a stated success
   criterion per step, "did that help?" has no answer and progress collapses
   into "did the tool return a 200".
2. It makes a stuck run diagnosable. "Step 2 of 4 failed three times" is a
   sentence you can act on; "the agent did eleven things" is not.
3. It is cheap. One call up front, against many steps of context.

Replanning is where the design earns its keep. An agent that cannot revise its
plan treats the first guess as binding and spends its whole budget failing the
same way; one that revises on every setback never finishes anything. So:
replanning is triggered only by the loop guard or a failed step, it is handed
the evidence of what actually went wrong, and it is capped by `max_replans`.
Past that cap the run ends as STUCK, because a third rewrite of the plan is
not a plan problem.
"""

from __future__ import annotations

import json
import re
from typing import Any

from actcheck.config import Config
from actcheck.llm.base import Model
from actcheck.prompts import (
    PLAN_SCHEMA,
    PLANNER_SYSTEM,
    PLANNER_USER,
    REPLAN_USER,
    tool_summary,
)
from actcheck.tools.base import Registry
from actcheck.types import Plan, PlanStep, Step, StepStatus, Usage

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class Planner:
    """Turns a task into a plan, and a failing plan into a better one."""

    def __init__(self, model: Model, cfg: Config) -> None:
        self.model = model
        self.cfg = cfg
        self.usage = Usage()
        self.last_usage = Usage()

    def plan(self, task: str, registry: Registry) -> Plan:
        response = self.model.complete(
            PLANNER_SYSTEM.format(max_steps=self.cfg.plan_max_steps),
            PLANNER_USER.format(task=task, tools=tool_summary(registry)),
            schema=PLAN_SCHEMA,
            purpose="plan",
        )
        self.usage.add(response.usage)
        self.last_usage = response.usage
        return self._parse(response.text, task)

    def replan(
        self, task: str, current: Plan, steps: list[Step], registry: Registry, evidence: str
    ) -> Plan:
        response = self.model.complete(
            PLANNER_SYSTEM.format(max_steps=self.cfg.plan_max_steps),
            REPLAN_USER.format(
                task=task,
                plan=current.render(),
                evidence=evidence,
                tools=tool_summary(registry),
            ),
            schema=PLAN_SCHEMA,
            purpose="plan",
        )
        self.usage.add(response.usage)
        self.last_usage = response.usage
        revised = self._parse(response.text, task)
        revised.revisions = current.revisions + 1
        # Steps already finished stay finished. Rewriting the plan does not
        # un-read a file, and re-doing completed work is the most common way a
        # replanning agent wastes the budget it has left.
        done = {s.description.strip().lower() for s in current.steps if s.status is StepStatus.DONE}
        for step in revised.steps:
            if step.description.strip().lower() in done:
                step.status = StepStatus.DONE
                step.note = "carried over from the previous plan"
        return revised

    # ---------------------------------------------------------------- parsing
    def _parse(self, text: str, task: str) -> Plan:
        """Build a Plan from the model's JSON, tolerating the usual wrappers.

        A plan that fails to parse must not fail the run: the loop works
        without one (that is what `plan = false` does), so a malformed plan
        degrades to a single catch-all step rather than an exception.
        """
        data = _loads(text)
        if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
            return Plan(
                goal=task,
                steps=[
                    PlanStep(
                        id=1,
                        description=task,
                        success_criterion="The task has been completed and reported with finish.",
                        note="planner returned no usable plan; proceeding without one",
                    )
                ],
            )

        steps: list[PlanStep] = []
        for i, raw in enumerate(data["steps"][: self.cfg.plan_max_steps], start=1):
            if not isinstance(raw, dict):
                continue
            description = str(raw.get("description", "")).strip()
            if not description:
                continue
            steps.append(
                PlanStep(
                    id=i,
                    description=description,
                    success_criterion=str(raw.get("success_criterion", "")).strip(),
                )
            )

        if not steps:
            steps = [PlanStep(id=1, description=task, success_criterion="Task completed.")]
        return Plan(goal=str(data.get("goal") or task).strip(), steps=steps)


def _loads(text: str) -> Any:
    """Parse JSON that may be wrapped in prose or a code fence."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = JSON_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None
