"""The tool contract.

A tool is three things: a name, a JSON schema the model is shown, and a `run`
that takes validated arguments and returns text. Everything else -- timing,
truncation, error capture -- is done once, here, so no individual tool has to
remember to do it.

Two deliberate constraints:

* **`run` returns a string.** Not an object, not a dataclass. Whatever a tool
  produces is going into a language model's context as text, so the tool author
  is the right person to decide how it reads. A tool that returns JSON because
  that was convenient pushes the formatting decision onto the model.
* **Failures are values, not exceptions, by the time the loop sees them.**
  Tools raise `ToolError`; the registry converts it into a failed
  `Observation`. The loop never has to distinguish "tool said no" from "tool
  worked", and the model gets told what went wrong in both cases.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from actcheck.errors import ToolError
from actcheck.types import Observation, ToolCall


class Tool(ABC):
    """One capability the agent can invoke."""

    name: str
    description: str
    # JSON Schema for the arguments object, in the shape the Anthropic tool-use
    # API expects (`input_schema`).
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    # Tools that change state are called out in the system prompt and are the
    # ones a read-only agent is denied. Honest self-reporting here is what makes
    # `enabled_tools` a real permission boundary rather than a suggestion.
    mutates: bool = False

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Do the work. Raise `ToolError` with a message aimed at the model."""

    # ------------------------------------------------------------------ glue
    def spec(self) -> dict[str, Any]:
        """The tool definition sent to the model."""
        return {"name": self.name, "description": self.description, "input_schema": self.schema}

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        """Check `args` against the schema.

        This is a deliberately small subset of JSON Schema -- required keys,
        declared types, enums, unknown keys. It is not a conformance
        implementation and does not try to be: its job is to turn the three
        mistakes models actually make (missing a required field, passing a
        number as a string, inventing a parameter) into a clear message the
        model can act on, without pulling in a validation dependency.
        """
        props: dict[str, Any] = self.schema.get("properties", {})
        required: list[str] = self.schema.get("required", [])

        missing = [k for k in required if k not in args or args[k] is None]
        if missing:
            raise ToolError(
                f"{self.name}: missing required argument(s): {', '.join(missing)}. "
                f"Expected: {', '.join(props) or '(none)'}"
            )
        unknown = [k for k in args if k not in props]
        if unknown:
            raise ToolError(
                f"{self.name}: unknown argument(s): {', '.join(unknown)}. "
                f"Valid arguments: {', '.join(props) or '(none)'}"
            )

        out: dict[str, Any] = {}
        for key, value in args.items():
            out[key] = _coerce_arg(self.name, key, value, props[key])
        return out


def _coerce_arg(tool: str, key: str, value: Any, spec: dict[str, Any]) -> Any:
    """Coerce one argument to its declared type, or explain why it cannot be."""
    declared = spec.get("type")
    enum = spec.get("enum")

    if enum is not None and value not in enum:
        raise ToolError(f"{tool}: {key} must be one of {enum}, got {value!r}")

    if declared == "integer" and not isinstance(value, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ToolError(f"{tool}: {key} must be an integer, got {value!r}") from None
    if declared == "number" and not isinstance(value, bool):
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ToolError(f"{tool}: {key} must be a number, got {value!r}") from None
    if declared == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ToolError(f"{tool}: {key} must be true or false, got {value!r}")
    if declared == "string" and not isinstance(value, str):
        # Models sometimes send a number where a string is declared. Accepting
        # the scalar is strictly more useful than refusing it.
        if isinstance(value, (int, float)):
            return str(value)
        raise ToolError(f"{tool}: {key} must be a string, got {type(value).__name__}")
    if declared == "array" and not isinstance(value, list):
        raise ToolError(f"{tool}: {key} must be an array, got {type(value).__name__}")
    return value


class Registry:
    """The tools one agent is allowed to use, and the single place they run.

    Owning execution here is what guarantees every observation is timed,
    size-capped, and error-wrapped the same way -- including the ones from
    tools written later by someone who did not read this file.
    """

    def __init__(self, tools: list[Tool], *, max_observation_chars: int = 6000) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool
        self.max_observation_chars = max_observation_chars

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def execute(self, call: ToolCall) -> Observation:
        """Run one tool call. Never raises for anything the model did wrong."""
        started = time.perf_counter()
        tool = self._tools.get(call.name)
        if tool is None:
            return Observation.failure(
                f"no such tool: {call.name!r}. Available tools: {', '.join(self.names())}",
                duration_ms=0,
            )
        try:
            args = tool.validate(dict(call.args))
            raw = tool.run(**args)
        except ToolError as exc:
            return Observation.failure(str(exc), duration_ms=_ms(started))
        except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the run
            # An unexpected exception is a bug in the tool, not a decision by
            # the model. It is still reported as an observation so the run can
            # continue and the trace records what happened, but the type name
            # is included because that is what a debugger needs.
            return Observation.failure(
                f"{call.name} raised {type(exc).__name__}: {exc}", duration_ms=_ms(started)
            )

        text = raw if isinstance(raw, str) else str(raw)
        return self._cap(text, _ms(started))

    def _cap(self, text: str, duration_ms: int) -> Observation:
        """Truncate an oversized result and say so, in the result itself.

        Silent truncation is worse than no truncation: the model reads a
        half-file as the whole file and concludes the rest does not exist. The
        marker tells it to narrow the query instead.
        """
        limit = self.max_observation_chars
        if len(text) <= limit:
            return Observation(content=text, duration_ms=duration_ms, original_chars=len(text))
        head = int(limit * 0.7)
        tail = limit - head
        body = (
            text[:head]
            + f"\n\n... [{len(text) - limit} characters omitted by actcheck; "
            + "narrow the request to see the rest] ...\n\n"
            + text[-tail:]
        )
        return Observation(
            content=body,
            duration_ms=duration_ms,
            truncated=True,
            original_chars=len(text),
        )


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
