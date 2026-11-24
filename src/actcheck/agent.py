"""The loop.

Everything else in this package exists to serve the twenty lines in the middle
of `Agent.run`. The shape is old and simple:

    plan ──▶ ┌─▶ act ──▶ check ──┐ ──▶ answer
             └── recover ◀───────┘

What separates a loop you can run unattended from a demo is not the acting --
every framework does the acting. It is the four things around it:

* **Checking.** Something other than the model rules on whether a step helped.
* **Guarding.** Repetition and stalls are detected and named, in the
  transcript, while there is still budget left to do something about them.
* **Budgeting.** Three ceilings, checked before each step, and a run that hits
  one says so instead of reporting whatever it had.
* **Recording.** Every step lands in the trace as it happens.

A note on what is deliberately absent: there is no retry-with-backoff around
tool failures. A tool that failed because the model passed a bad path does not
want retrying, it wants the model to read the error -- which is what the
observation is. Retrying is the mechanism that turns one bad decision into
five identical ones.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from actcheck.budget import Budget
from actcheck.checker import Checker
from actcheck.config import Config
from actcheck.context import Transcript, step_digest
from actcheck.errors import ActcheckError, ProviderError
from actcheck.llm import build_model
from actcheck.llm.base import Model
from actcheck.loopguard import LoopGuard
from actcheck.planner import Planner
from actcheck.prompts import agent_system
from actcheck.tools import Registry, Sandbox, build_registry
from actcheck.tools.finish import Finish, GiveUp
from actcheck.trace import NullTrace, Trace
from actcheck.types import (
    Judgement,
    Observation,
    Plan,
    PlanStep,
    RunResult,
    RunStatus,
    Step,
    StepStatus,
    ToolCall,
)

StepHook = Callable[[Step], None]

# Consecutive turns where the model produced prose but called no tool. One is a
# stumble worth a nudge; two means it is not going to use the tools and the run
# should end with what it said rather than burning the budget on silence.
MAX_SILENT_TURNS = 2


class Agent:
    """Runs one task to a terminal state."""

    def __init__(
        self,
        cfg: Config,
        *,
        model: Model | None = None,
        registry: Registry | None = None,
        sandbox: Sandbox | None = None,
        replay_path: str | None = None,
        on_step: StepHook | None = None,
    ) -> None:
        self.cfg = cfg
        self.sandbox = sandbox or Sandbox(cfg.workspace_path)
        self.registry = registry or build_registry(cfg, self.sandbox)
        self.model = model or build_model(cfg, replay_path=replay_path)
        self.on_step = on_step

        # Model-based checking is pointless under replay -- the recorded run is
        # being re-executed, not re-decided -- and it would be the only thing
        # making a "free" replay cost money.
        checker_cfg = cfg
        if cfg.provider == "replay" and cfg.checker in ("model", "both"):
            checker_cfg = cfg.merged({"checker": "programmatic"})

        self.planner = Planner(self.model, cfg)
        self.checker = Checker(self._checker_model(checker_cfg), checker_cfg)
        self.guard = LoopGuard(
            repeat_threshold=cfg.repeat_threshold,
            no_progress_threshold=cfg.no_progress_threshold,
        )

    def _checker_model(self, cfg: Config) -> Model | None:
        """A separate, cheaper model for checking.

        Reusing the agent's own model would be simpler and would also mean the
        checker's opinion is worth exactly as much as the opinion it is
        checking. With the offline and replay providers there is nothing to
        separate, so the same instance is reused.
        """
        if cfg.checker in ("off", "programmatic"):
            return None
        if cfg.provider != "anthropic":
            return self.model
        from actcheck.llm.anthropic_llm import AnthropicModel

        return AnthropicModel(
            cfg.checker_model, effort=cfg.checker_effort, thinking=False, cache=False
        )

    # ------------------------------------------------------------------- run
    def run(self, task: str) -> RunResult:
        started = time.perf_counter()
        result = RunResult(task=task, status=RunStatus.ERROR, model=self.cfg.model)
        budget = Budget(
            max_steps=self.cfg.max_steps,
            max_tokens=self.cfg.max_tokens,
            max_seconds=self.cfg.max_seconds,
        )
        trace = self._open_trace(result.run_id, task)
        result.trace_path = str(trace.path)

        # try/finally rather than `with trace:` -- the terminal event has to be
        # written *after* the status is decided, including on the error paths,
        # and a context manager closes the file before that is known.
        try:
            self._loop(task, result, budget, trace)
        except ProviderError as exc:
            result.status = RunStatus.ERROR
            result.reason = str(exc)
            trace.write("error", error=str(exc))
        except ActcheckError as exc:
            result.status = RunStatus.ERROR
            result.reason = f"{type(exc).__name__}: {exc}"
            trace.write("error", error=result.reason)
        except KeyboardInterrupt:
            # Interrupting a run is a normal thing to do. It gets a real status
            # and a written trace, not a stack trace and a half-written file.
            result.status = RunStatus.ABORTED
            result.reason = "interrupted by the operator"
        finally:
            # The budget has seen every model call -- agent turns, planning,
            # and checking -- so it is the one true total.
            result.usage = budget.usage
            result.duration_ms = int((time.perf_counter() - started) * 1000)
            if trace.run_id:
                trace.finish(result)
            else:  # NullTrace
                result.trace_path = ""
            trace.close()
        return result

    # ------------------------------------------------------------------ loop
    def _loop(self, task: str, result: RunResult, budget: Budget, trace: Trace) -> None:
        plan = self._initial_plan(task, trace)
        result.plan = plan
        if plan is not None:
            budget.record_usage(self.planner.last_usage)

        transcript = Transcript(
            task,
            char_budget=self.cfg.context_char_budget,
            keep_recent_steps=self.cfg.keep_recent_steps,
            compact=self.cfg.compact,
        )
        silent_turns = 0
        warned_about_budget = False
        index = 0

        while True:
            exhausted = budget.exhausted()
            if exhausted:
                result.status = RunStatus.BUDGET_EXHAUSTED
                result.reason = exhausted
                result.answer = _best_effort_answer(result.steps)
                trace.note("budget exhausted", **budget.snapshot())
                return

            if not warned_about_budget and budget.pressure() >= 0.75:
                warned_about_budget = True
                transcript.add_note(
                    f"You have about {budget.remaining_steps()} steps of budget left. "
                    "Finish with what you have, or call give_up if you cannot."
                )
                trace.note("budget warning issued", **budget.snapshot())

            active = plan.active if plan else None
            response = self.model.respond(
                agent_system(
                    self.registry,
                    plan_text=plan.render() if plan else "",
                    budget_text=f"You have at most {budget.remaining_steps()} more actions.",
                ),
                transcript.messages(),
                self.registry.specs(),
                max_tokens=self.cfg.max_output_tokens,
            )
            budget.record_usage(response.usage)
            transcript.add_assistant(response.blocks, response.text)

            if not response.wants_tools:
                silent_turns += 1
                if silent_turns >= MAX_SILENT_TURNS:
                    result.status = RunStatus.FAILED
                    result.reason = "model stopped without calling finish"
                    result.answer = response.text
                    trace.note("model ended its turn without a tool call", text=response.text[:400])
                    return
                transcript.add_note(
                    "You ended your turn without taking an action. Use one of the tools, "
                    "or call finish with your answer if you are done."
                )
                trace.note("nudged a silent turn")
                continue
            silent_turns = 0

            for call in response.tool_calls:
                index += 1
                step = self._act(
                    call, index, plan, active, response.text, budget, transcript, trace
                )
                result.steps.append(step)
                if self.on_step is not None:
                    self.on_step(step)

                terminal = self._settle(step, call, result, plan, active)
                if terminal:
                    return

                stop = self._recover(step, task, plan, result, transcript, trace, budget)
                if stop:
                    return

                # Only the first call in a turn is planned against; a model that
                # emits several actions at once has already decided to batch
                # them, and re-reading the plan between them would be theatre.
                active = plan.active if plan else None

                if budget.exhausted():
                    break

    # ------------------------------------------------------------------ act
    def _act(
        self,
        call: ToolCall,
        index: int,
        plan: Plan | None,
        active: PlanStep | None,
        thought: str,
        budget: Budget,
        transcript: Transcript,
        trace: Trace,
    ) -> Step:
        """One action: guard, execute, check, record."""
        step = Step(index=index, thought=thought, tool_call=call)
        step.plan_step_id = active.id if active else None
        started = time.perf_counter()

        report = self.guard.observe_call(call)
        if report.tripped:
            # The nudge goes into the transcript as the *result* of this call,
            # in place of running it again. Running it would cost a step to
            # learn what we already know.
            trace.note(f"loop guard: {report.kind}", fatal=report.fatal, tool=call.name)
            transcript.add_tool_result(call, report.message, is_error=True)
            step.observation = _guard_observation(report.message, fatal=report.fatal)
            step.verdict = report.verdict()
            step.duration_ms = int((time.perf_counter() - started) * 1000)
            budget.record_step()
            trace.step(step)
            return step

        observation = self.registry.execute(call)
        step.observation = observation

        step.verdict = self.checker.check(
            goal=plan.goal if plan else "",
            plan_step=active,
            call=call,
            observation=observation,
        )
        transcript.add_tool_result(call, observation.render(), is_error=not observation.ok)

        step.duration_ms = int((time.perf_counter() - started) * 1000)
        step.usage = self.checker.last_usage
        budget.record_step(self.checker.last_usage)
        trace.step(step)
        return step

    # --------------------------------------------------------------- settle
    def _settle(
        self,
        step: Step,
        call: ToolCall,
        result: RunResult,
        plan: Plan | None,
        active: PlanStep | None,
    ) -> bool:
        """Apply the verdict to the plan. Returns True if the run is over."""
        verdict = step.verdict
        observation = step.observation
        assert verdict is not None and observation is not None

        if call.name == Finish.name:
            result.status = RunStatus.SUCCESS
            result.reason = "agent called finish"
            result.answer = observation.content
            if plan:
                for pending in plan.steps:
                    if pending.status in (StepStatus.PENDING, StepStatus.ACTIVE):
                        pending.status = StepStatus.SKIPPED
            return True

        if call.name == GiveUp.name:
            result.status = RunStatus.FAILED
            result.reason = "agent gave up"
            result.answer = observation.content
            return True

        if active is not None:
            active.attempts += 1
            if verdict.judgement is Judgement.STEP_DONE:
                active.status = StepStatus.DONE
                active.note = verdict.reason[:160]
            elif verdict.advanced:
                active.status = StepStatus.ACTIVE
            elif active.attempts >= self.cfg.step_attempt_limit:
                active.status = StepStatus.FAILED
                active.note = f"failed after {active.attempts} attempts: {verdict.reason[:120]}"
        return False

    # -------------------------------------------------------------- recover
    def _recover(
        self,
        step: Step,
        task: str,
        plan: Plan | None,
        result: RunResult,
        transcript: Transcript,
        trace: Trace,
        budget: Budget,
    ) -> bool:
        """React to a stalled run. Returns True if the run should end."""
        assert step.verdict is not None

        # The repeat guard nudges on its first trip; a second identical call
        # after being told it would not help is the agent failing to respond to
        # feedback, and nothing later in this run is going to change that.
        if step.observation is not None and step.observation.meta.get("fatal"):
            result.status = RunStatus.STUCK
            result.reason = (
                f"repeated the same {step.tool_call.name} call after being told it would not help"
            )
            result.answer = _best_effort_answer(result.steps)
            trace.note("loop guard: giving up on a repeated call", tool=step.tool_call.name)
            return True

        report = self.guard.observe_verdict(step.verdict)
        if not report.tripped:
            return False

        trace.note(f"loop guard: {report.kind}", stalled=self.guard.stalled_steps)

        if plan is not None and plan.revisions < self.cfg.max_replans:
            revised = self.planner.replan(
                task, plan, result.steps[-6:], self.registry, step_digest(result.steps[-6:])
            )
            plan.goal, plan.steps, plan.revisions = revised.goal, revised.steps, revised.revisions
            result.plan = plan
            budget.record_usage(self.planner.last_usage)
            self.guard.reset_progress()
            transcript.add_note(
                "Progress has stalled, so the plan has been revised. The new plan is:\n"
                + plan.render()
            )
            trace.plan(plan, revision=plan.revisions)
            return False

        if not report.fatal:
            transcript.add_note(report.message)
            return False

        result.status = RunStatus.STUCK
        result.reason = report.message.split(".")[0]
        result.answer = _best_effort_answer(result.steps)
        return True

    # ----------------------------------------------------------- scaffolding
    def _initial_plan(self, task: str, trace: Trace) -> Plan | None:
        if not self.cfg.plan:
            return None
        plan = self.planner.plan(task, self.registry)
        trace.plan(plan)
        return plan

    def _open_trace(self, run_id: str, task: str) -> Trace:
        if not self.cfg.trace:
            return NullTrace()
        path = Path(self.cfg.trace_dir) / f"{run_id}.jsonl"
        return Trace(
            path,
            run_id,
            task,
            meta={
                "model": self.cfg.model,
                "provider": self.cfg.provider,
                "checker": self.cfg.checker,
                "workspace": str(self.sandbox.root),
                "tools": self.registry.names(),
            },
        )


# --------------------------------------------------------------------- utils
def _guard_observation(message: str, *, fatal: bool = False) -> Observation:
    return Observation(
        content=message,
        ok=False,
        error=message,
        meta={"source": "loop_guard", "fatal": fatal},
    )


def _best_effort_answer(steps: list[Step]) -> str:
    """What to report when the run ended without the agent finishing.

    The last observation that actually succeeded -- clearly labelled as partial.
    Reporting nothing throws away work that may well have answered the
    question; reporting it as if it were a finished answer is worse.
    """
    for step in reversed(steps):
        if step.observation and step.observation.ok and step.observation.content.strip():
            return (
                "(partial -- the run ended before the agent finished; "
                "this is the last useful result)\n\n" + step.observation.content.strip()
            )
    return ""
