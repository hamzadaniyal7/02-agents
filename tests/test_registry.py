"""The registry is the single place tools run, so it owns every guarantee."""

from __future__ import annotations

import pytest

from actcheck.config import Config
from actcheck.errors import ConfigError, ToolError
from actcheck.tools import build_registry
from actcheck.tools.base import Registry, Tool
from actcheck.types import ToolCall


class Echo(Tool):
    name = "echo"
    description = "Echo a message back."
    schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "times": {"type": "integer"},
            "shout": {"type": "boolean"},
            "mode": {"type": "string", "enum": ["a", "b"]},
        },
        "required": ["message"],
    }

    def run(self, message: str, times: int = 1, shout: bool = False, mode: str = "a") -> str:
        out = message * times
        return out.upper() if shout else out


class Exploder(Tool):
    name = "explode"
    description = "Always raises something unexpected."
    schema = {"type": "object", "properties": {}}

    def run(self) -> str:
        raise ZeroDivisionError("boom")


class Firehose(Tool):
    name = "firehose"
    description = "Returns a lot of text."
    schema = {"type": "object", "properties": {}}

    def run(self) -> str:
        return "x" * 50_000


@pytest.fixture
def reg() -> Registry:
    return Registry([Echo(), Exploder(), Firehose()], max_observation_chars=1000)


def test_executes_a_tool(reg: Registry) -> None:
    obs = reg.execute(ToolCall(name="echo", args={"message": "hi"}))
    assert obs.ok and obs.content == "hi"


def test_unknown_tool_lists_what_is_available(reg: Registry) -> None:
    obs = reg.execute(ToolCall(name="nope", args={}))
    assert not obs.ok and "echo" in obs.content


def test_missing_required_argument_is_explained(reg: Registry) -> None:
    obs = reg.execute(ToolCall(name="echo", args={}))
    assert not obs.ok and "missing required" in obs.content


def test_unknown_argument_is_explained(reg: Registry) -> None:
    obs = reg.execute(ToolCall(name="echo", args={"message": "x", "colour": "red"}))
    assert not obs.ok and "unknown argument" in obs.content


def test_numeric_strings_are_coerced(reg: Registry) -> None:
    """Models send "2" where an integer is declared. Accepting it is strictly
    more useful than refusing it."""
    obs = reg.execute(ToolCall(name="echo", args={"message": "ab", "times": "2"}))
    assert obs.ok and obs.content == "abab"


def test_uncoercible_values_are_rejected(reg: Registry) -> None:
    obs = reg.execute(ToolCall(name="echo", args={"message": "x", "times": "lots"}))
    assert not obs.ok and "must be an integer" in obs.content


def test_enum_values_are_enforced(reg: Registry) -> None:
    obs = reg.execute(ToolCall(name="echo", args={"message": "x", "mode": "z"}))
    assert not obs.ok and "must be one of" in obs.content


def test_boolean_strings_are_coerced(reg: Registry) -> None:
    obs = reg.execute(ToolCall(name="echo", args={"message": "hi", "shout": "true"}))
    assert obs.ok and obs.content == "HI"


def test_a_tool_bug_does_not_kill_the_run(reg: Registry) -> None:
    obs = reg.execute(ToolCall(name="explode", args={}))
    assert not obs.ok
    assert "ZeroDivisionError" in obs.content  # the type name is what a debugger needs


def test_oversized_output_is_truncated_and_says_so(reg: Registry) -> None:
    """Silent truncation makes a model read half a file as the whole file."""
    obs = reg.execute(ToolCall(name="firehose", args={}))
    assert obs.truncated
    assert obs.original_chars == 50_000
    assert "characters omitted" in obs.content
    assert len(obs.content) < 1500


def test_execution_is_timed(reg: Registry) -> None:
    assert reg.execute(ToolCall(name="echo", args={"message": "x"})).duration_ms >= 0


def test_duplicate_tool_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate tool name"):
        Registry([Echo(), Echo()])


# ------------------------------------------------------------ build_registry
def test_build_registry_gives_every_tool_by_default(cfg: Config) -> None:
    assert set(build_registry(cfg).names()) >= {"read_file", "write_file", "grep", "run_command"}


def test_enabled_tools_is_a_real_boundary(cfg: Config) -> None:
    """A read-only agent must not merely be told not to write."""
    narrowed = cfg.merged({"enabled_tools": ("read_file", "grep")})
    names = build_registry(narrowed).names()
    assert "write_file" not in names
    assert "run_command" not in names


def test_terminal_tools_survive_narrowing(cfg: Config) -> None:
    """An agent with no way to say 'done' can only end by exhausting a budget."""
    names = build_registry(cfg.merged({"enabled_tools": ("read_file",)})).names()
    assert "finish" in names and "give_up" in names


def test_unknown_tool_in_config_is_rejected(cfg: Config) -> None:
    with pytest.raises(ConfigError, match="unknown tool"):
        build_registry(cfg.merged({"enabled_tools": ("teleport",)}))


def test_mutating_tools_declare_themselves(cfg: Config) -> None:
    registry = build_registry(cfg)
    assert registry.get("write_file").mutates
    assert not registry.get("read_file").mutates


def test_validate_raises_tool_error_not_value_error() -> None:
    with pytest.raises(ToolError):
        Echo().validate({})
