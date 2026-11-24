"""Tool assembly.

`build_registry` is the single place that decides which tools an agent gets.
Keeping that decision here -- rather than letting the agent construct its own
tools -- is what makes `enabled_tools` in the config a real boundary: a tool
that is not built cannot be called, whatever the model asks for.
"""

from __future__ import annotations

from actcheck.config import Config
from actcheck.errors import ConfigError
from actcheck.tools.base import Registry, Tool
from actcheck.tools.finish import TERMINAL_TOOLS, Finish, GiveUp
from actcheck.tools.fs import ListDir, ReadFile, WriteFile
from actcheck.tools.sandbox import Sandbox
from actcheck.tools.search import Grep
from actcheck.tools.shell import RunCommand

__all__ = [
    "Finish",
    "GiveUp",
    "Grep",
    "ListDir",
    "ReadFile",
    "Registry",
    "RunCommand",
    "Sandbox",
    "Tool",
    "WriteFile",
    "build_registry",
]


def build_registry(cfg: Config, sandbox: Sandbox | None = None) -> Registry:
    """Construct the tool set described by `cfg`."""
    sandbox = sandbox or Sandbox(cfg.workspace_path)
    everything: list[Tool] = [
        ReadFile(sandbox),
        WriteFile(sandbox),
        ListDir(sandbox),
        Grep(sandbox),
        RunCommand(sandbox, allow=cfg.shell_allow, timeout=cfg.shell_timeout),
        Finish(),
        GiveUp(),
    ]

    if not cfg.enabled_tools:
        selected = everything
    else:
        by_name = {t.name: t for t in everything}
        unknown = [n for n in cfg.enabled_tools if n not in by_name]
        if unknown:
            raise ConfigError(
                f"unknown tool(s) in enabled_tools: {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(by_name))}"
            )
        # The terminal tools are always present. An agent with no way to say
        # "done" can only ever end by exhausting a budget, which turns every
        # successful run into a reported failure.
        wanted = set(cfg.enabled_tools) | TERMINAL_TOOLS
        selected = [by_name[n] for n in by_name if n in wanted]

    return Registry(selected, max_observation_chars=cfg.max_observation_chars)
