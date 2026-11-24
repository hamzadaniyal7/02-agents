"""Model selection.

One function, so every entry point -- CLI, library, eval harness -- resolves a
provider the same way and `provider = "scripted"` means the same thing
everywhere.
"""

from __future__ import annotations

from actcheck.config import Config
from actcheck.errors import ConfigError
from actcheck.llm.base import Message, Model, ModelResponse
from actcheck.llm.scripted import ScriptedModel

__all__ = ["Message", "Model", "ModelResponse", "ScriptedModel", "build_model"]


def build_model(cfg: Config, *, replay_path: str | None = None) -> Model:
    """Construct the model described by `cfg`."""
    if cfg.provider == "scripted":
        return ScriptedModel()
    if cfg.provider == "replay":
        if not replay_path:
            raise ConfigError("provider 'replay' needs a trace path (actcheck replay <trace>)")
        from actcheck.llm.replay import ReplayModel

        return ReplayModel(replay_path)
    if cfg.provider == "anthropic":
        # Imported lazily so that `--provider scripted` works even if the
        # anthropic package is not importable in this environment.
        from actcheck.llm.anthropic_llm import AnthropicModel

        return AnthropicModel(
            cfg.model,
            effort=cfg.effort,
            thinking=cfg.thinking,
            temperature=cfg.temperature,
        )
    raise ConfigError(f"unknown provider: {cfg.provider!r}")
