"""Budgets: the three ways a run is allowed to end early.

An agent loop without ceilings is an open-ended spend authorisation. Three
independent ones, because they fail for different reasons:

* **steps** -- catches a loop that is making calls but getting nowhere. Cheap
  tools make this one invisible to a token budget.
* **tokens** -- catches context that grew faster than progress. Long tool
  outputs blow this long before the step count looks unusual.
* **wall clock** -- catches a slow dependency. Neither other ceiling moves
  while a command sits waiting on a timeout.

The budget is checked *before* each step, never during. Interrupting a tool
halfway leaves the workspace in a state nobody planned for; letting the step
finish and stopping afterwards costs one extra step at most and ends with the
workspace in a state the trace explains.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from actcheck.types import Usage


@dataclass(slots=True)
class Budget:
    """Ceilings for one run, and what has been used against them."""

    max_steps: int = 24
    max_tokens: int = 200_000
    max_seconds: float = 300.0

    steps_used: int = 0
    usage: Usage = field(default_factory=Usage)
    started_at: float = field(default_factory=time.monotonic)

    # ------------------------------------------------------------- recording
    def record_step(self, usage: Usage | None = None) -> None:
        self.steps_used += 1
        if usage is not None:
            self.usage.add(usage)

    def record_usage(self, usage: Usage) -> None:
        """Account for a model call that is not a loop step -- the planner and
        the checker both spend tokens, and a budget that ignored them would be
        lying by roughly the cost of the checker."""
        self.usage.add(usage)

    # -------------------------------------------------------------- checking
    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def exhausted(self) -> str | None:
        """The name of the ceiling that has been reached, or None.

        Returns a string rather than raising because the caller wants to end
        the run tidily -- write the trace, report a status -- not unwind.
        """
        if self.steps_used >= self.max_steps:
            return f"step budget exhausted: {self.steps_used}/{self.max_steps} steps"
        if self.usage.total_tokens >= self.max_tokens:
            return f"token budget exhausted: {self.usage.total_tokens}/{self.max_tokens} tokens"
        if self.elapsed >= self.max_seconds:
            return f"time budget exhausted: {self.elapsed:.1f}s/{self.max_seconds:.0f}s"
        return None

    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.steps_used)

    def pressure(self) -> float:
        """How close the tightest ceiling is, 0.0 to 1.0.

        The agent is told this once it passes 0.75. An agent that knows it has
        three steps left writes its answer; one that does not keep exploring
        and gets cut off mid-thought with nothing to show.
        """
        fractions = [
            self.steps_used / self.max_steps if self.max_steps else 0.0,
            self.usage.total_tokens / self.max_tokens if self.max_tokens else 0.0,
            self.elapsed / self.max_seconds if self.max_seconds else 0.0,
        ]
        return min(1.0, max(fractions))

    def snapshot(self) -> dict[str, float | int]:
        return {
            "steps_used": self.steps_used,
            "max_steps": self.max_steps,
            "tokens_used": self.usage.total_tokens,
            "max_tokens": self.max_tokens,
            "elapsed_s": round(self.elapsed, 2),
            "max_seconds": self.max_seconds,
            "pressure": round(self.pressure(), 3),
        }
