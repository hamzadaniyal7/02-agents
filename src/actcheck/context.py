"""The transcript, and keeping it from eating the run.

Context growth is what kills long agent runs. Every step appends an assistant
turn and a tool result, tool results are the largest thing in the conversation,
and the cost of each step is the cost of re-sending everything before it. A
twenty-step run with no management does not cost twenty units; it costs
something closer to two hundred.

Three mechanisms, applied in this order:

1. **Per-observation caps** (in `tools/base.py`) stop one result from being
   enormous in the first place. Cheapest fix, applied earliest.
2. **Compaction** replaces the middle of the transcript with a digest once it
   outgrows the budget, keeping the task at the top and the recent steps
   verbatim at the bottom.
3. **Hard trimming** is the backstop: if even the kept tail is over budget,
   observations in it are shortened from the middle outward.

**Why the digest is built by code and not by a model.** A summariser call costs
tokens and latency on exactly the step where the run is already under pressure,
and it can invent a detail that then looks, to every later step, like something
the agent observed. A deterministic digest -- which tools ran, with what
arguments, and what the checker ruled -- is lossy but cannot be wrong, and the
thing an agent actually needs from its distant past is "I already tried that",
which the digest preserves exactly. The code hook for a model summariser is
`summarise`, if you want to measure the trade rather than take my word for it.

**Why recent steps are never compacted.** The immediate past is what stops the
agent repeating the action that just failed. Compacting it saves tokens and
costs you the loop guard's whole reason for existing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from actcheck.llm.base import Message
from actcheck.types import Step, ToolCall


class Transcript:
    """The message list sent to the model, with budget-aware compaction."""

    def __init__(
        self,
        task: str,
        *,
        char_budget: int = 60_000,
        keep_recent_steps: int = 6,
        compact: bool = True,
        summarise: Callable[[list[Message]], str] | None = None,
    ) -> None:
        self.task = task
        self.char_budget = char_budget
        self.keep_recent_steps = keep_recent_steps
        self.compact_enabled = compact
        self.summarise = summarise or _digest
        self.compactions = 0

        # The task message is pinned: it is never compacted and never trimmed.
        # An agent that loses the task statement is not recoverable.
        self._pinned: Message = {"role": "user", "content": [{"type": "text", "text": task}]}
        self._turns: list[Message] = []
        self._digest: str = ""

    # ---------------------------------------------------------------- append
    def add_assistant(self, blocks: list[dict[str, Any]], fallback_text: str = "") -> None:
        """Record the model's turn verbatim.

        Verbatim matters: thinking blocks carry signatures the API validates on
        the next request, so a reconstructed turn is a rejected request.
        """
        content = blocks or [{"type": "text", "text": fallback_text or "(no content)"}]
        self._turns.append({"role": "assistant", "content": content})

    def add_tool_result(self, call: ToolCall, content: str, *, is_error: bool = False) -> None:
        self._turns.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call.call_id,
                        "content": content or "(empty result)",
                        "is_error": is_error,
                    }
                ],
            }
        )

    def add_note(self, text: str) -> None:
        """Inject a message from the harness -- a guard nudge, a budget warning.

        It is addressed to the model as a user turn because that is the only
        channel it reads, but it is marked so the model can tell it apart from
        the task and from tool output.
        """
        self._turns.append(
            {"role": "user", "content": [{"type": "text", "text": f"[actcheck] {text}"}]}
        )

    # ----------------------------------------------------------------- build
    def messages(self) -> list[Message]:
        """The message list for the next model call."""
        turns = self._turns
        if self.compact_enabled and self._size(turns) > self.char_budget:
            turns = self._compacted(turns)
        turns = self._trimmed(turns)

        out: list[Message] = [self._pinned]
        if self._digest:
            out.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"[actcheck] Earlier steps:\n{self._digest}"}
                    ],
                }
            )
        out.extend(turns)
        return _repair(out)

    def _compacted(self, turns: list[Message]) -> list[Message]:
        """Digest everything but the recent tail."""
        keep = self.keep_recent_steps * 2  # one assistant turn + one result per step
        if len(turns) <= keep:
            return turns
        head, tail = turns[:-keep], turns[-keep:]
        new = self.summarise(head)
        # Digests accumulate: the previous digest stays at the top so the run's
        # early history is not lost on the second compaction.
        self._digest = f"{self._digest}\n{new}".strip() if self._digest else new
        self._turns = tail
        self.compactions += 1
        return tail

    def _trimmed(self, turns: list[Message]) -> list[Message]:
        """Backstop: shorten tool results until the tail fits.

        Oldest-first, because the newest observation is the one being reasoned
        about right now.
        """
        if self._size(turns) <= self.char_budget:
            return turns
        out = [dict(t) for t in turns]
        for message in out:
            if self._size(out) <= self.char_budget:
                break
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    body = str(block.get("content", ""))
                    if len(body) > 600:
                        block["content"] = (
                            body[:400]
                            + "\n...[trimmed to fit the context budget]...\n"
                            + body[-200:]
                        )
        return out

    # ----------------------------------------------------------------- stats
    @property
    def size_chars(self) -> int:
        return self._size(self.messages())

    @staticmethod
    def _size(messages: list[Message]) -> int:
        return sum(len(json.dumps(m, default=str)) for m in messages)


def _digest(messages: list[Message]) -> str:
    """Deterministic summary of compacted-away turns.

    One line per tool call: what ran, with what arguments, and whether it
    worked. That is the minimum an agent needs to avoid retrying its own past.
    """
    lines: list[str] = []
    pending: dict[str, str] = {}

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                args = json.dumps(block.get("input", {}), default=str)
                pending[str(block.get("id"))] = f"{block.get('name')}{args[:120]}"
            elif block.get("type") == "tool_result":
                label = pending.pop(str(block.get("tool_use_id")), "(unknown call)")
                body = block.get("content")
                if isinstance(body, list):
                    body = " ".join(str(p.get("text", "")) for p in body if isinstance(p, dict))
                outcome = "failed" if block.get("is_error") else "ok"
                first = str(body or "").strip().splitlines()
                gist = first[0][:100] if first else ""
                lines.append(f"  - {label} -> {outcome}: {gist}")

    return "\n".join(lines) if lines else "  (nothing to summarise)"


def _repair(messages: list[Message]) -> list[Message]:
    """Drop tool_results whose tool_use is no longer in the transcript.

    Compaction can cut between an assistant turn and its result. The API
    rejects a `tool_result` with no matching `tool_use` in the same
    conversation, so an orphan has to go -- and it is replaced with plain text
    rather than deleted, because the agent still needs to know that call
    happened.
    """
    seen_ids: set[str] = set()
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    seen_ids.add(str(block.get("id")))

    out: list[Message] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        blocks: list[dict[str, Any]] = []
        for block in content:
            orphan = (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and str(block.get("tool_use_id")) not in seen_ids
            )
            if orphan:
                body = str(block.get("content", ""))[:200]
                blocks.append({"type": "text", "text": f"[actcheck] earlier result: {body}"})
                continue
            blocks.append(block)
        if blocks:
            out.append({**message, "content": blocks})
    return out


def step_digest(steps: list[Step]) -> str:
    """Render recent steps as evidence for the replanner."""
    lines = []
    for step in steps:
        if not step.tool_call:
            continue
        verdict = step.verdict.judgement if step.verdict else "?"
        reason = step.verdict.reason if step.verdict else ""
        args = json.dumps(step.tool_call.args, default=str)[:100]
        lines.append(f"  step {step.index}: {step.tool_call.name}{args} -> {verdict} ({reason})")
    return "\n".join(lines) or "  (no steps taken yet)"
