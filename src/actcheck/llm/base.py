"""The model contract.

Two methods, because the loop asks a model two genuinely different questions:

* ``respond`` -- "here is the transcript and the tools; what next?" Multi-turn,
  returns tool calls.
* ``complete`` -- "here is a question; answer in this JSON shape." Single-turn,
  used by the planner and the checker, which are not part of the agent's own
  conversation and must not pollute it.

Keeping them separate is what lets the checker run on a cheaper model than the
agent, and what keeps the checker's opinion out of the context the agent sees.

**On the message format:** the transcript is carried in Anthropic's
content-block shape (`{"role", "content": [{"type": "tool_use" ...}]}`) rather
than a bespoke neutral format. That is a deliberate trade: it is the most
expressive of the shapes we target, and inventing a lossy intermediate just to
translate it back again adds a layer that can only lose information. The cost
is that a provider for a different API has to translate -- `ScriptedModel`
below shows how little that involves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from actcheck.types import ToolCall, Usage

Message = dict[str, Any]


@dataclass(slots=True)
class ModelResponse:
    """One turn from the model."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    # The raw assistant content blocks, kept verbatim so the next request can
    # replay this turn exactly. Reconstructing them from `text` + `tool_calls`
    # would drop thinking blocks, and dropping a thinking block invalidates the
    # signature the API checks on the following turn.
    blocks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class Model(ABC):
    name: str

    @abstractmethod
    def respond(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        """Next turn of the agent conversation."""

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        purpose: str = "",
    ) -> ModelResponse:
        """Single-turn completion, optionally constrained to a JSON schema.

        `purpose` ("plan" | "check" | "") is a hint, ignored by real providers
        and used by the offline stand-in to decide what kind of answer to
        fabricate. It is not sent to any API.
        """
