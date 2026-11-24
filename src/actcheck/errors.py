"""Typed errors.

The loop distinguishes three kinds of bad news, because each one gets a
different response:

* ``ToolError``   -- the tool refused or failed. Recoverable: it is handed back
                     to the model as an observation so it can try something else.
* ``BudgetError`` -- a ceiling was hit. Not recoverable: the run ends, cleanly,
                     with the ceiling named.
* ``ConfigError`` -- the operator set something impossible. Fails before the
                     first model call, so nobody pays tokens to find out.
"""

from __future__ import annotations


class ActcheckError(Exception):
    """Base class, so callers can catch everything this package raises."""


class ConfigError(ActcheckError):
    """Invalid configuration, raised before a run starts."""


class ToolError(ActcheckError):
    """A tool could not do what was asked.

    Raised inside a tool and caught by the registry, which turns it into a
    failed ``Observation``. Tools should raise this with a message written for
    *the model*: say what was wrong and what would work instead.
    """


class SandboxError(ToolError):
    """A tool tried to touch something outside the workspace root."""


class BudgetError(ActcheckError):
    """A step, token, or wall-clock ceiling was reached."""

    def __init__(self, kind: str, limit: int | float, used: int | float) -> None:
        super().__init__(f"{kind} budget exhausted: used {used} of {limit}")
        self.kind = kind
        self.limit = limit
        self.used = used


class ProviderError(ActcheckError):
    """The model provider failed in a way retrying will not fix."""
