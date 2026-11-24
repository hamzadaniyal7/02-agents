"""Running commands, carefully.

Shell access is the tool that makes an agent genuinely useful and the one that
makes it genuinely dangerous, so this module is built around one rule: the
agent chooses arguments, never syntax.

* **No shell interpretation.** The command is parsed with `shlex.split` and
  handed to `subprocess.run` as an argv list, with `shell=False`. There is no
  shell process, so `;`, `&&`, backticks and `$(...)` are not operators -- they
  are just characters, and they get rejected before they reach anything that
  would honour them.
* **Allowlist on the program name**, not a denylist on dangerous strings.
  Denylists lose: `rm` blocked is `/bin/rm` allowed, is `python3 -c "import
  os; os.remove(...)"` allowed. The allowlist is in config, defaults to
  read-mostly tools, and every entry on it is a decision someone made.
* **Confined and bounded.** `cwd` is the workspace root, there is a hard
  timeout, and output is capped by the registry like any other observation.

What this is not: a security boundary against a determined adversary. An
allowlisted `python3` can do anything Python can do, and `git` can reach the
network. It is a boundary against an agent making an expensive mistake, which
is the actual failure mode. Running untrusted task descriptions means running
this in a container -- see docs/05-decisions.md.
"""

from __future__ import annotations

import shlex
import subprocess
from typing import Any

from actcheck.errors import ToolError
from actcheck.tools.base import Tool
from actcheck.tools.sandbox import Sandbox

# Characters that only mean something to a shell. We never invoke one, so their
# presence means the model expected shell semantics it is not going to get --
# better to say so than to run half of what it intended.
SHELL_METACHARS = ("|", ";", "&", ">", "<", "`", "$(", "\n")


class RunCommand(Tool):
    name = "run_command"
    description = (
        "Run a single allowlisted command in the workspace and return its output. "
        "No shell features: pipes, redirection, and command chaining are not available. "
        "Run one command per call."
    )
    mutates = True
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command line, e.g. 'pytest -q tests/test_api.py'.",
            },
        },
        "required": ["command"],
    }

    def __init__(self, sandbox: Sandbox, allow: tuple[str, ...], timeout: float = 20.0) -> None:
        self.sandbox = sandbox
        self.allow = set(allow)
        self.timeout = timeout
        self.description += f" Allowed commands: {', '.join(sorted(self.allow))}."

    def run(self, command: str, **_: Any) -> str:
        found = [c for c in SHELL_METACHARS if c in command]
        if found:
            raise ToolError(
                f"run_command does not use a shell, so {found[0]!r} will not do what you expect. "
                "Run one plain command per call; use grep or read_file instead of pipes."
            )
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise ToolError(f"could not parse command {command!r}: {exc}") from None
        if not argv:
            raise ToolError("command is empty.")

        program = argv[0]
        if program not in self.allow:
            raise ToolError(
                f"{program!r} is not on the allowlist. Allowed: {', '.join(sorted(self.allow))}. "
                "Ask the operator to widen shell_allow if you genuinely need it."
            )

        try:
            proc = subprocess.run(  # noqa: S603 - argv list, shell=False, allowlisted program
                argv,
                cwd=self.sandbox.root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError:
            raise ToolError(f"{program!r} is not installed on this machine.") from None
        except subprocess.TimeoutExpired:
            raise ToolError(
                f"{program!r} did not finish within {self.timeout:g}s and was killed. "
                "Try a narrower command."
            ) from None

        return _format(proc, command)


def _format(proc: subprocess.CompletedProcess[str], command: str) -> str:
    """Render a completed process for a model.

    Both streams are included and labelled. A non-zero exit is reported as
    text, not raised: a failing test run is exactly the observation the agent
    needs in order to fix the test, and swallowing it as an error would hide
    the output that explains why.
    """
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    parts = [f"$ {command}", f"exit code: {proc.returncode}"]
    if out:
        parts.append(f"stdout:\n{out}")
    if err:
        parts.append(f"stderr:\n{err}")
    if not out and not err:
        parts.append("(no output)")
    return "\n".join(parts)
