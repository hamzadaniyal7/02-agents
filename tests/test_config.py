"""Configuration: precedence, coercion, and failing early."""

from __future__ import annotations

from pathlib import Path

import pytest

from actcheck.config import Config
from actcheck.errors import ConfigError


def test_defaults_are_valid() -> None:
    Config().validate()


def test_toml_overrides_defaults(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "actcheck.toml"
    path.write_text('[actcheck]\nmax_steps = 5\nchecker = "off"\n')
    cfg = Config.load(path)
    assert cfg.max_steps == 5 and cfg.checker == "off"


def test_env_overrides_toml(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "actcheck.toml"
    path.write_text("[actcheck]\nmax_steps = 5\n")
    monkeypatch.setenv("ACTCHECK_MAX_STEPS", "9")
    assert Config.load(path).max_steps == 9


def test_explicit_overrides_win(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ACTCHECK_MAX_STEPS", "9")
    assert Config.load(tmp_path / "none.toml", max_steps=3).max_steps == 3


def test_none_overrides_are_ignored() -> None:
    """CLI flags arrive as None when unset; they must not clobber the config."""
    assert Config.load(None, max_steps=None).max_steps == Config().max_steps


def test_env_strings_are_coerced_to_field_types(monkeypatch) -> None:
    monkeypatch.setenv("ACTCHECK_PLAN", "false")
    monkeypatch.setenv("ACTCHECK_MAX_SECONDS", "12.5")
    cfg = Config.load(None)
    assert cfg.plan is False and cfg.max_seconds == 12.5


def test_tuples_accept_comma_separated_strings(monkeypatch) -> None:
    monkeypatch.setenv("ACTCHECK_ENABLED_TOOLS", "read_file, grep")
    assert Config.load(None).enabled_tools == ("read_file", "grep")


def test_temperature_round_trips_through_none() -> None:
    """Its default is None, so there is no type to read the coercion off."""
    assert Config().merged({"temperature": "0.3"}).temperature == 0.3
    assert Config().merged({"temperature": ""}).temperature is None


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown config key"):
        Config().merged({"nonsense": 1})


@pytest.mark.parametrize(
    "bad",
    [
        {"provider": "openai"},
        {"checker": "vibes"},
        {"effort": "extreme"},
        {"max_steps": 0},
        {"max_seconds": 0},
        {"repeat_threshold": 0},
        {"max_observation_chars": 10},
    ],
)
def test_invalid_values_fail_before_the_first_model_call(bad: dict) -> None:
    with pytest.raises(ConfigError):
        Config().merged(bad).validate()


def test_context_budget_must_fit_one_observation() -> None:
    """Otherwise a single tool result can never fit in the transcript."""
    with pytest.raises(ConfigError, match="context_char_budget"):
        Config().merged({"context_char_budget": 1000, "max_observation_chars": 6000}).validate()
