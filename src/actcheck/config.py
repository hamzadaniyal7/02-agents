"""Configuration: one dataclass, three sources, documented precedence.

Precedence (last wins):
  1. the defaults below
  2. ``actcheck.toml`` in the current directory (or the path given to ``load``)
  3. ``ACTCHECK_*`` environment variables
  4. explicit keyword overrides (what the CLI flags become)

Every knob that changes how the loop behaves lives here, so an experiment --
"does model-based checking beat programmatic checking on my tasks?" -- is a
config diff plus an `actcheck eval`, not a code change.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from actcheck.errors import ConfigError

CONFIG_FILENAME = "actcheck.toml"

CHECKERS = {"programmatic", "model", "both", "off"}
PROVIDERS = {"anthropic", "scripted", "replay"}
EFFORTS = {"low", "medium", "high", "xhigh", "max"}


@dataclass(slots=True)
class Config:
    # --- sandbox -----------------------------------------------------------
    # Every filesystem and shell tool is confined to this directory. Resolved
    # to an absolute path at validation time; see tools/sandbox.py for how the
    # confinement is actually enforced (it is not string prefix matching).
    workspace: str = "workspace"

    # --- budgets -----------------------------------------------------------
    # Three independent ceilings. A run stops at whichever it hits first, and
    # the stop reason always names which one -- "it just stopped" is the
    # failure mode this whole file exists to prevent.
    max_steps: int = 24
    max_tokens: int = 200_000
    max_seconds: float = 300.0
    # Per-tool-call ceiling. A tool that returns 4MB of JSON does not get to
    # eat the context window; it gets truncated and told so.
    max_observation_chars: int = 6_000

    # --- model -------------------------------------------------------------
    provider: str = "anthropic"  # anthropic | scripted | replay
    model: str = "claude-opus-5"
    max_output_tokens: int = 4_096
    effort: str = "medium"  # low | medium | high | xhigh | max
    thinking: bool = True
    temperature: float | None = None

    # --- planning ----------------------------------------------------------
    plan: bool = True
    plan_max_steps: int = 8
    # How many times the agent may rewrite the plan before the run is called
    # stuck. Replanning is the right response to surprise; replanning on a loop
    # is just a more expensive way to spin.
    max_replans: int = 2

    # --- checking ----------------------------------------------------------
    # programmatic: cheap rules over the observation (errors, emptiness, repeats)
    # model:        ask a model whether the step's success criterion is met
    # both:         programmatic first, model only when it is inconclusive
    # off:          trust the agent's own say-so (the baseline worth beating)
    checker: str = "both"
    checker_model: str = "claude-sonnet-5"
    checker_effort: str = "low"

    # --- loop guard --------------------------------------------------------
    # Identical tool call (name + args) this many times in a row -> intervene.
    repeat_threshold: int = 2
    # Consecutive non-advancing verdicts before the run is declared stuck.
    no_progress_threshold: int = 4
    # Consecutive failures of one plan step before it is marked failed.
    step_attempt_limit: int = 3

    # --- context -----------------------------------------------------------
    # Characters of transcript to carry into each model call. When the
    # transcript outgrows this, older steps are compacted into a summary
    # instead of being silently dropped.
    context_char_budget: int = 60_000
    # Never compact away the most recent N steps -- the model needs the
    # immediate past verbatim to avoid repeating the action that just failed.
    keep_recent_steps: int = 6
    compact: bool = True

    # --- tools -------------------------------------------------------------
    # Empty tuple means "every registered tool". Narrow it to give an agent
    # read-only access, which is the sane default for anything unattended.
    enabled_tools: tuple[str, ...] = ()
    shell_timeout: float = 20.0
    # Shell is opt-in per command name. An agent that can run `rm` is an agent
    # that will eventually run `rm`.
    shell_allow: tuple[str, ...] = (
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "find",
        "echo",
        "python3",
        "pytest",
        "git",
        "sort",
        "uniq",
        "cut",
        "sed",
        "awk",
    )

    # --- tracing -----------------------------------------------------------
    trace_dir: str = ".actcheck/traces"
    trace: bool = True

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> Config:
        cfg = cls()
        toml_path = Path(path) if path else Path(CONFIG_FILENAME)
        if toml_path.is_file():
            with toml_path.open("rb") as fh:
                data = tomllib.load(fh)
            cfg = cfg.merged(data.get("actcheck", data))
        cfg = cfg.merged(cls._from_env())
        if overrides:
            cfg = cfg.merged({k: v for k, v in overrides.items() if v is not None})
        cfg.validate()
        return cfg

    @staticmethod
    def _from_env() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(Config):
            raw = os.environ.get(f"ACTCHECK_{f.name.upper()}")
            if raw is not None:
                out[f.name] = raw
        return out

    def merged(self, data: dict[str, Any]) -> Config:
        """Return a copy with `data` applied, coercing strings to field types."""
        known = {f.name for f in fields(Config)}
        current = {f.name: getattr(self, f.name) for f in fields(Config)}
        for key, value in data.items():
            if key not in known:
                raise ConfigError(f"unknown config key: {key!r}")
            current[key] = _coerce(value, current[key], key)
        return Config(**current)

    def validate(self) -> None:
        if self.provider not in PROVIDERS:
            raise ConfigError(f"provider must be one of {sorted(PROVIDERS)}, got {self.provider!r}")
        if self.checker not in CHECKERS:
            raise ConfigError(f"checker must be one of {sorted(CHECKERS)}, got {self.checker!r}")
        if self.effort not in EFFORTS:
            raise ConfigError(f"effort must be one of {sorted(EFFORTS)}, got {self.effort!r}")
        if self.checker_effort not in EFFORTS:
            raise ConfigError(f"checker_effort must be one of {sorted(EFFORTS)}")
        if self.max_steps < 1:
            raise ConfigError("max_steps must be >= 1")
        if self.max_seconds <= 0:
            raise ConfigError("max_seconds must be > 0")
        if self.keep_recent_steps < 1:
            raise ConfigError("keep_recent_steps must be >= 1")
        if self.repeat_threshold < 1:
            raise ConfigError("repeat_threshold must be >= 1")
        if self.max_observation_chars < 200:
            raise ConfigError("max_observation_chars must be >= 200 to stay useful")
        if self.context_char_budget < self.max_observation_chars:
            raise ConfigError(
                "context_char_budget must be >= max_observation_chars, otherwise a single "
                "tool result cannot fit in the transcript"
            )

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace).expanduser().resolve()

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(Config)}


def _coerce(value: Any, like: Any, key: str) -> Any:
    """Coerce a TOML/env value to the type of the existing default.

    `like` is the current value, which carries the type. The one special case
    is `temperature`, whose default is None -- there is no type to read off it,
    so it is handled by name.
    """
    if key == "temperature":
        if value is None or value == "":
            return None
        return float(value)
    if isinstance(like, bool):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(like, int) and not isinstance(like, bool):
        return int(value)
    if isinstance(like, float):
        return float(value)
    if isinstance(like, tuple):
        if isinstance(value, str):
            return tuple(p.strip() for p in value.split(",") if p.strip())
        return tuple(value)
    return value
