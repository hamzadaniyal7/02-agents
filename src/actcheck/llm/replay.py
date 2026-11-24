"""Replay a recorded run.

Point this at a trace and the loop re-executes it: the same tool calls, in the
same order, against live tools. The model is not consulted, so a replay costs
nothing and is byte-stable.

Two things it is for:

* **Regression tests.** Record a run that worked once, replay it in CI, and a
  change that breaks a tool's output format or the loop's bookkeeping shows up
  immediately -- without a model in the loop to paper over the difference by
  adapting.
* **Debugging.** "It did something strange on step 9" is much easier to work on
  when you can get back to step 9 on demand.

What a replay deliberately does *not* do is fake the model's judgement. If the
tools now return something different from what was recorded, the replayed calls
are the old ones and the run will likely fail -- which is the signal, not a
flaw. A replay that quietly adapted would be a replay that tests nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from actcheck.errors import ProviderError
from actcheck.llm.base import Message, Model, ModelResponse
from actcheck.llm.scripted import _blocks
from actcheck.trace import read_trace
from actcheck.types import ToolCall, Usage


class ReplayModel(Model):
    """Plays back the tool calls recorded in a trace file."""

    name = "replay"

    def __init__(self, trace_path: str | Path) -> None:
        self.trace_path = Path(trace_path)
        if not self.trace_path.is_file():
            raise ProviderError(f"no trace to replay at {self.trace_path}")
        self.events = read_trace(self.trace_path)
        self.calls: list[ToolCall] = [
            ToolCall(
                name=event["tool_call"]["name"],
                args=dict(event["tool_call"].get("args", {})),
            )
            for event in self.events
            if event.get("event") == "step" and event.get("tool_call")
        ]
        if not self.calls:
            raise ProviderError(f"{self.trace_path} contains no tool calls to replay")
        self._cursor = 0

    def respond(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        if self._cursor >= len(self.calls):
            # The recording ended without a terminal tool -- it was cut short by
            # a budget or an error. Say so rather than inventing an ending.
            return ModelResponse(
                text=f"Replay of {self.trace_path.name} ended after {len(self.calls)} calls.",
                stop_reason="end_turn",
                model=self.name,
                usage=Usage(calls=1),
            )
        call = self.calls[self._cursor]
        self._cursor += 1
        text = f"[replay {self._cursor}/{len(self.calls)}]"
        return ModelResponse(
            text=text,
            tool_calls=[call],
            stop_reason="tool_use",
            model=self.name,
            usage=Usage(calls=1),
            blocks=_blocks(text, [call]),
        )

    def complete(
        self,
        system: str,
        user: str,
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        purpose: str = "",
    ) -> ModelResponse:
        """Replay the recorded plan; refuse to invent anything else.

        A replayed run reuses the plan from the recording so the step
        bookkeeping lines up. Model-based checking is switched off during
        replay by the agent, so `purpose == "check"` should not arrive here.
        """
        import json

        if purpose == "plan":
            for event in self.events:
                if event.get("event") == "plan":
                    return ModelResponse(
                        text=json.dumps(event["plan"]), model=self.name, usage=Usage(calls=1)
                    )
        return ModelResponse(text="{}", model=self.name, usage=Usage(calls=1))
