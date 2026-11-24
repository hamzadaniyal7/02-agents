"""Core data structures.

Everything the loop produces is a plain dataclass that serialises to JSON,
because the trace file *is* the product here. If a value matters for
understanding why a run went the way it did, it lives on one of these objects
and lands in the trace -- not in a log line that scrolls away.

The shape to keep in mind:

    Run ──▶ Plan ──▶ Step* ──▶ ToolCall ──▶ Observation ──▶ Verdict

One Step is one trip around the loop: the model picked a tool, the tool ran,
and the checker ruled on what came back.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class StepStatus(StrEnum):
    """Where a plan step stands. Steps move forward only when the checker says so."""

    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


class Judgement(StrEnum):
    """The checker's ruling on one observation.

    The distinction that earns its keep is NO_PROGRESS vs REGRESS. A tool that
    returns an empty result is not the same failure as a tool that broke
    something, and the recovery for each is different: the first wants a
    different approach, the second wants an undo.
    """

    PROGRESS = "progress"
    NO_PROGRESS = "no_progress"
    REGRESS = "regress"
    STEP_DONE = "step_done"
    GOAL_DONE = "goal_done"
    BLOCKED = "blocked"


class RunStatus(StrEnum):
    """How a run ended. Every terminal state is explicit -- there is no
    'the loop just stopped' outcome, because that is the bug this project is
    about preventing."""

    SUCCESS = "success"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STUCK = "stuck"
    ABORTED = "aborted"
    ERROR = "error"


@dataclass(slots=True)
class Usage:
    """Token and cost accounting, summed across every model call in a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.calls += other.calls

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_response(cls, usage: dict[str, Any] | None) -> Usage:
        usage = usage or {}
        return cls(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            calls=1,
        )


@dataclass(slots=True)
class ToolCall:
    """A tool invocation the model asked for.

    `call_id` comes from the provider when there is one; the loop generates it
    otherwise. It has to round-trip exactly, because that is how a tool result
    is matched back to its request in the next request's messages.
    """

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")

    def signature(self) -> str:
        """A stable identity for 'the same call again'.

        Used by the loop guard. Args are sorted and JSON-encoded so that
        `{"path": "a", "n": 1}` and `{"n": 1, "path": "a"}` collapse to one
        signature -- the model reorders keys between turns and that is not a
        different action.
        """
        return self.name + ":" + json.dumps(self.args, sort_keys=True, default=str)


@dataclass(slots=True)
class Observation:
    """What came back from running a tool.

    `ok=False` is a *normal* outcome, not an exception. A tool that fails is
    information the model needs in order to choose differently, so failures are
    fed back into the transcript in the same shape as successes.
    """

    content: str
    ok: bool = True
    error: str | None = None
    truncated: bool = False
    original_chars: int = 0
    duration_ms: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, error: str, **kw: Any) -> Observation:
        return cls(content=error, ok=False, error=error, **kw)

    def render(self, width: int = 2000) -> str:
        body = self.content if len(self.content) <= width else self.content[:width] + " ...[cut]"
        return f"ERROR: {body}" if not self.ok else body


@dataclass(slots=True)
class Verdict:
    """The checker's ruling on one step."""

    judgement: Judgement
    reason: str
    by: str = "programmatic"  # programmatic | model | guard
    confidence: float = 1.0

    @property
    def advanced(self) -> bool:
        return self.judgement in (Judgement.PROGRESS, Judgement.STEP_DONE, Judgement.GOAL_DONE)


@dataclass(slots=True)
class PlanStep:
    """One item in the plan.

    `success_criterion` is the field that makes the plan more than a to-do
    list: it is what the checker tests the observation against. A step without
    a checkable criterion is a wish, and the planner prompt says so.
    """

    id: int
    description: str
    success_criterion: str = ""
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    note: str = ""


@dataclass(slots=True)
class Plan:
    """The agent's intended route to the goal, and how often it has changed."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    revisions: int = 0

    @property
    def active(self) -> PlanStep | None:
        for step in self.steps:
            if step.status in (StepStatus.PENDING, StepStatus.ACTIVE):
                return step
        return None

    @property
    def complete(self) -> bool:
        return all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps)

    def render(self) -> str:
        marks = {
            StepStatus.PENDING: " ",
            StepStatus.ACTIVE: ">",
            StepStatus.DONE: "x",
            StepStatus.SKIPPED: "-",
            StepStatus.FAILED: "!",
        }
        lines = []
        for s in self.steps:
            line = f"  [{marks[s.status]}] {s.id}. {s.description}"
            if s.success_criterion:
                line += f"\n        check: {s.success_criterion}"
            if s.note:
                line += f"\n        note:  {s.note}"
            lines.append(line)
        return "\n".join(lines)


@dataclass(slots=True)
class Step:
    """One iteration of the loop, start to finish.

    This is the unit the trace is made of and the unit the eval harness counts.
    """

    index: int
    thought: str = ""
    tool_call: ToolCall | None = None
    observation: Observation | None = None
    verdict: Verdict | None = None
    plan_step_id: int | None = None
    usage: Usage = field(default_factory=Usage)
    started_at: float = field(default_factory=time.time)
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought,
            "plan_step_id": self.plan_step_id,
            "tool_call": asdict(self.tool_call) if self.tool_call else None,
            "observation": asdict(self.observation) if self.observation else None,
            "verdict": asdict(self.verdict) if self.verdict else None,
            "usage": asdict(self.usage),
            "duration_ms": self.duration_ms,
        }


@dataclass(slots=True)
class RunResult:
    """The end of a run.

    `status` plus `reason` always answer "why did it stop?" in one line. That
    pairing is deliberate: a run that hit the step ceiling and a run that
    finished are both "over", and conflating them is how agent demos end up
    reporting success they did not earn.
    """

    task: str
    status: RunStatus
    reason: str = ""
    answer: str = ""
    plan: Plan | None = None
    steps: list[Step] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    model: str = ""
    duration_ms: int = 0
    trace_path: str = ""

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.SUCCESS

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def tool_errors(self) -> int:
        return sum(1 for s in self.steps if s.observation and not s.observation.ok)

    def render(self) -> str:
        out = [self.answer.strip() or "(no answer produced)", ""]
        out.append(
            f"[{self.status}] {self.reason}".rstrip()
            + f"  --  {self.n_steps} steps, "
            + f"{self.usage.total_tokens} tokens, {self.duration_ms}ms"
        )
        if self.tool_errors:
            out.append(f"  tool errors: {self.tool_errors}")
        if self.trace_path:
            out.append(f"  trace: {self.trace_path}")
        return "\n".join(out)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "task": self.task,
                "status": str(self.status),
                "reason": self.reason,
                "answer": self.answer,
                "model": self.model,
                "duration_ms": self.duration_ms,
                "usage": asdict(self.usage),
                "plan": (
                    {
                        "goal": self.plan.goal,
                        "revisions": self.plan.revisions,
                        "steps": [asdict(s) for s in self.plan.steps],
                    }
                    if self.plan
                    else None
                ),
                "steps": [s.to_dict() for s in self.steps],
            },
            indent=indent,
            default=str,
        )
