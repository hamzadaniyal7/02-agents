"""The Anthropic provider, with tool use.

Choices worth knowing about:

* **adaptive thinking** (``thinking={"type": "adaptive"}``) -- the model decides
  when reasoning is warranted. ``budget_tokens`` is gone on this generation and
  returns a 400.
* **streaming** -- a long agent turn can outlive an HTTP timeout. Streaming and
  then taking ``get_final_message()`` sidesteps that without changing the shape
  of the result.
* **prompt caching on the system prompt and tool definitions.** In an agent
  loop those two blocks are byte-identical on every single call while the
  transcript grows underneath them, which is the textbook case for a cache
  breakpoint. On a twenty-step run it is the difference between paying for the
  tool schemas twenty times and paying once.
* **thinking blocks are passed back verbatim.** They carry a signature the API
  validates on the next turn; rebuilding the assistant turn from parsed text
  would strip it and the request would fail.
"""

from __future__ import annotations

from typing import Any

from actcheck.errors import ProviderError
from actcheck.llm.base import Message, Model, ModelResponse
from actcheck.types import ToolCall, Usage

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# `output_config.effort` is not accepted by every model (Haiku 4.5 and the 4.5
# generation reject it). Sending it anyway is a 400, so it is filtered here
# rather than leaving a trap for anyone who swaps the model in actcheck.toml.
NO_EFFORT_MODELS = ("haiku", "-4-5")

NO_CREDENTIALS = (
    "no Anthropic credentials found. Either:\n"
    "  export ANTHROPIC_API_KEY=sk-ant-...        (see .env.example)\n"
    "  ant auth login                             (stores a profile the SDK reads)\n"
    "  actcheck run ... --provider scripted       (run the loop with no model)"
)


class AnthropicModel(Model):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5",
        *,
        effort: str = "medium",
        thinking: bool = True,
        temperature: float | None = None,
        cache: bool = True,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ImportError("pip install anthropic") from exc

        self._anthropic = anthropic
        # An unset ANTHROPIC_API_KEY does not mean there are no credentials: the
        # SDK also resolves ANTHROPIC_AUTH_TOKEN and an `ant auth login` profile.
        # Only the constructor can tell, so let it decide and translate its
        # failure into something actionable -- this is the first error a new
        # user is likely to hit.
        try:
            self.client = anthropic.Anthropic()
        except Exception as exc:
            raise ProviderError(NO_CREDENTIALS) from exc
        if not any(
            getattr(self.client, attr, None)
            for attr in ("api_key", "auth_token", "credentials", "_credentials")
        ):
            raise ProviderError(NO_CREDENTIALS)

        self.model = model
        self.effort = effort
        self.thinking = thinking
        self.temperature = temperature
        self.cache = cache

    # ------------------------------------------------------------------ api
    def respond(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": self._system_blocks(system),
            "messages": messages,
            "tools": self._tool_blocks(tools),
        }
        self._apply_common(params)
        message = self._stream(**params)
        return _to_response(message)

    def complete(
        self,
        system: str,
        user: str,
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        purpose: str = "",
    ) -> ModelResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        self._apply_common(params)
        if schema is not None:
            params.setdefault("output_config", {})["format"] = {
                "type": "json_schema",
                "schema": schema,
            }
        message = self._stream(**params)
        return _to_response(message)

    # ------------------------------------------------------------ internals
    def _apply_common(self, params: dict[str, Any]) -> None:
        if self.effort and self._supports_effort():
            params.setdefault("output_config", {})["effort"] = self.effort
        if self.thinking:
            params["thinking"] = {"type": "adaptive"}
        elif self.temperature is not None:
            # Temperature and thinking are mutually exclusive on this
            # generation; thinking wins because it is the better default.
            params["temperature"] = self.temperature

    def _supports_effort(self) -> bool:
        return not any(marker in self.model for marker in NO_EFFORT_MODELS)

    def _system_blocks(self, system: str) -> list[dict[str, Any]]:
        block: dict[str, Any] = {"type": "text", "text": system}
        if self.cache:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def _tool_blocks(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Mark the end of the tool list as a cache breakpoint.

        The breakpoint goes on the *last* tool because a cache entry covers
        everything above it. One marker therefore caches the whole tool array
        plus the system prompt, and the growing transcript stays outside it.
        """
        if not self.cache or not tools:
            return tools
        out = [dict(t) for t in tools]
        out[-1]["cache_control"] = {"type": "ephemeral"}
        return out

    def _stream(self, **params: Any) -> Any:
        try:
            with self.client.messages.stream(**params) as stream:
                return stream.get_final_message()
        except self._anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error {exc.status_code}: {exc}") from exc


def _to_response(message: Any) -> ModelResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    blocks: list[dict[str, Any]] = []

    for block in message.content:
        kind = getattr(block, "type", None)
        if kind == "text":
            text_parts.append(block.text)
            blocks.append({"type": "text", "text": block.text})
        elif kind == "tool_use":
            tool_calls.append(ToolCall(name=block.name, args=dict(block.input), call_id=block.id))
            blocks.append(
                {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)}
            )
        elif kind in ("thinking", "redacted_thinking"):
            # Verbatim, signature included -- see the module docstring.
            blocks.append(block.model_dump(exclude_none=True))

    usage = Usage()
    if getattr(message, "usage", None) is not None:
        usage = Usage(
            input_tokens=getattr(message.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(message.usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(message.usage, "cache_read_input_tokens", 0) or 0,
            calls=1,
        )

    return ModelResponse(
        text="".join(text_parts),
        tool_calls=tool_calls,
        stop_reason=getattr(message, "stop_reason", None),
        model=getattr(message, "model", ""),
        usage=usage,
        blocks=blocks,
    )
