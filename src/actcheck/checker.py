"""The 'check' in plan-act-check.

Without this, a loop has exactly one source of truth about its own progress:
the model that is doing the work. That is the same model that just decided the
failing approach was a good idea, and asking it to grade itself is how an agent
reports success it did not achieve.

Two checkers, used together by default:

**Programmatic.** Rules over the observation. Free, instant, certain, and
blind to meaning -- it knows a tool errored, returned nothing, or returned the
same bytes as last time. Most real failures are exactly this boring, and
catching them without a model call is why the default configuration is
affordable.

**Model.** Asked only when the programmatic rules are inconclusive, which is
the case that matters: the tool worked, returned plausible content, and the
question is whether that content satisfies the step's success criterion. That
is a judgement about meaning, and nothing but a model is going to make it. It
runs on a cheaper model than the agent, because grading one observation
against one sentence is a far smaller job than deciding what to do next.

The order is deliberate: cheap-and-certain first, expensive-and-fallible only
when needed. Inverting it roughly doubles the cost of a run to learn things the
rules already knew.
"""

from __future__ import annotations

from actcheck.config import Config
from actcheck.llm.base import Model
from actcheck.planner import _loads
from actcheck.prompts import CHECK_SCHEMA, CHECKER_SYSTEM, CHECKER_USER
from actcheck.tools.finish import Finish, GiveUp
from actcheck.types import Judgement, Observation, PlanStep, ToolCall, Usage, Verdict

# Phrases tools use to report "it worked, there was just nothing there". These
# are successes at the tool level and non-progress at the task level, and
# telling them apart is the single most useful rule in this file.
# Substring matching here is a trap: "10 entries" contains "0 entries". Each
# marker below is a phrase a tool emits verbatim and only when it means it.
EMPTY_MARKERS = ("no matches for", "is empty", "(no output)")


class Checker:
    """Rules on whether a step moved the run forward."""

    def __init__(self, model: Model | None, cfg: Config) -> None:
        self.model = model
        self.cfg = cfg
        self.usage = Usage()
        # Usage of the most recent check, so the budget can account for it
        # without double-counting the running total.
        self.last_usage = Usage()
        self._seen: dict[str, str] = {}

    # ------------------------------------------------------------------ api
    def check(
        self,
        *,
        goal: str,
        plan_step: PlanStep | None,
        call: ToolCall,
        observation: Observation,
    ) -> Verdict:
        self.last_usage = Usage()
        if self.cfg.checker == "off":
            return Verdict(Judgement.PROGRESS, "checking disabled", by="programmatic")

        verdict = self._programmatic(call, observation)
        if verdict is not None:
            return verdict

        if self.cfg.checker in ("model", "both") and self.model is not None:
            return self._model_check(goal, plan_step, call, observation)

        # Programmatic-only and nothing matched: the tool returned real content
        # and no rule objected. Credit it as progress, but not as done -- only
        # a judgement about the success criterion can close a step, and that is
        # exactly what this configuration has opted out of.
        return Verdict(Judgement.PROGRESS, "tool returned content", by="programmatic")

    # -------------------------------------------------------- programmatic
    def _programmatic(self, call: ToolCall, observation: Observation) -> Verdict | None:
        """Certain rulings only. Returns None when it genuinely cannot tell."""
        if call.name == Finish.name:
            return Verdict(Judgement.GOAL_DONE, "agent reported the task complete")
        if call.name == GiveUp.name:
            return Verdict(Judgement.BLOCKED, f"agent gave up: {observation.content[:160]}")

        if not observation.ok:
            return Verdict(
                Judgement.NO_PROGRESS,
                f"tool failed: {(observation.error or '')[:160]}",
            )

        body = observation.content.strip()
        if not body:
            return Verdict(Judgement.NO_PROGRESS, "tool returned nothing")

        lowered = body.lower()
        if any(marker in lowered for marker in EMPTY_MARKERS):
            return Verdict(Judgement.NO_PROGRESS, "tool succeeded but found nothing")

        # Same call, same bytes as before: new step, no new information. This
        # catches the case the loop guard misses -- two *different* calls that
        # happen to return identical content, e.g. reading a file by two paths.
        previous = self._seen.get(call.name)
        if previous is not None and previous == body:
            return Verdict(
                Judgement.NO_PROGRESS,
                f"{call.name} returned exactly what it returned last time",
            )
        self._seen[call.name] = body

        return None

    # --------------------------------------------------------------- model
    def _model_check(
        self,
        goal: str,
        plan_step: PlanStep | None,
        call: ToolCall,
        observation: Observation,
    ) -> Verdict:
        assert self.model is not None
        import json

        response = self.model.complete(
            CHECKER_SYSTEM,
            CHECKER_USER.format(
                goal=goal,
                step=plan_step.description if plan_step else "(no explicit step)",
                criterion=plan_step.success_criterion if plan_step else "Task progresses.",
                tool=call.name,
                args=json.dumps(call.args, default=str)[:400],
                observation=observation.render(width=self.cfg.max_observation_chars),
            ),
            schema=CHECK_SCHEMA,
            purpose="check",
            max_tokens=512,
        )
        self.usage.add(response.usage)
        self.last_usage = response.usage

        data = _loads(response.text)
        if not isinstance(data, dict):
            # An unparseable verdict must not stall the run. Falling back to
            # PROGRESS is the conservative choice: it neither closes the step
            # (which needs evidence) nor counts against the stall threshold
            # (which would punish the agent for the checker's failure).
            return Verdict(Judgement.PROGRESS, "checker returned no usable verdict", by="model")

        try:
            judgement = Judgement(str(data.get("judgement", "progress")))
        except ValueError:
            judgement = Judgement.PROGRESS

        return Verdict(
            judgement=judgement,
            reason=str(data.get("reason", "")).strip()[:300] or "no reason given",
            by="model",
            confidence=float(data.get("confidence", 0.0) or 0.0),
        )
