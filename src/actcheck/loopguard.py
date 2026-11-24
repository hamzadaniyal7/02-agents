"""Catching the agent going in circles.

The characteristic failure of a tool-using loop is not a crash. It is step
after step of the same call, each one costing tokens, each one returning the
same unhelpful result, until a budget finally ends it. The model is not broken
when this happens -- it is doing the most plausible next thing given a
transcript in which that thing has not yet worked.

Two detectors, because there are two distinct shapes of stuck:

* **Repetition** -- the same tool with the same arguments, back to back. Easy
  to detect exactly (see `ToolCall.signature`), and unambiguous: a file does
  not change between two reads one second apart.
* **No progress** -- different calls, none of which the checker credits. This
  is the subtler one: the agent looks busy, every step is a new action, and
  nothing is moving. Only a checker that can rule on outcomes can see it, which
  is why the checker is not optional in the default configuration.

What the guard does on firing is as important as the detection. It does not
kill the run on the first repeat -- it **tells the agent what it just did**, in
the transcript, as a tool result. That one message converts a hidden pattern
into visible evidence, and it is usually enough. Only when the nudge fails does
the run end as STUCK, which is an honest outcome and a much better one than
burning the whole budget to arrive at the same place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from actcheck.types import Judgement, ToolCall, Verdict


@dataclass(slots=True)
class GuardReport:
    """What the guard thinks the loop should do next."""

    tripped: bool = False
    kind: str = ""  # repeat | no_progress
    message: str = ""
    fatal: bool = False  # True once nudging has already been tried

    def verdict(self) -> Verdict:
        return Verdict(judgement=Judgement.NO_PROGRESS, reason=self.message, by="guard")


@dataclass(slots=True)
class LoopGuard:
    """Tracks repetition and progress across a run."""

    repeat_threshold: int = 2
    no_progress_threshold: int = 4

    _last_signature: str = ""
    _repeat_count: int = 0
    _no_progress_run: int = 0
    _nudged: set[str] = field(default_factory=set)

    # -------------------------------------------------------------- updating
    def observe_call(self, call: ToolCall) -> GuardReport:
        """Called before a tool runs. Detects exact repetition."""
        signature = call.signature()
        if signature == self._last_signature:
            self._repeat_count += 1
        else:
            self._last_signature = signature
            self._repeat_count = 1

        if self._repeat_count <= self.repeat_threshold:
            return GuardReport()

        already = signature in self._nudged
        self._nudged.add(signature)
        return GuardReport(
            tripped=True,
            kind="repeat",
            fatal=already,
            message=(
                f"You have now called {call.name} with identical arguments "
                f"{self._repeat_count} times in a row and the result has not changed. "
                "It will not change on another attempt. Either use a different tool or "
                "different arguments, or call give_up explaining what is blocking you."
            ),
        )

    def observe_verdict(self, verdict: Verdict) -> GuardReport:
        """Called after the checker rules. Detects a stalled run."""
        if verdict.advanced:
            self._no_progress_run = 0
            return GuardReport()

        self._no_progress_run += 1
        if self._no_progress_run < self.no_progress_threshold:
            return GuardReport()

        return GuardReport(
            tripped=True,
            kind="no_progress",
            fatal=self._no_progress_run > self.no_progress_threshold,
            message=(
                f"{self._no_progress_run} consecutive steps have produced no progress "
                "towards the goal. Stop and reconsider: is the plan wrong, or is the task "
                "not achievable with these tools? Revise the approach or call give_up."
            ),
        )

    # --------------------------------------------------------------- reading
    @property
    def stalled_steps(self) -> int:
        return self._no_progress_run

    def reset_progress(self) -> None:
        """Called when the plan is revised -- a new plan deserves a fresh run
        of attempts, otherwise replanning can never pay off."""
        self._no_progress_run = 0
