"""Ending the run, on purpose.

An agent needs a way to say "I am done" and a way to say "I cannot do this",
and both have to be *tools* rather than a string the loop greps for. Parsing
"DONE" out of prose is how you get runs that end because the model happened to
quote the word, and runs that never end because it phrased it differently.

The two are kept separate rather than one tool with a `success` flag, because
the model picking `give_up` is a much stronger signal than it setting a boolean
to false, and the eval harness scores them differently.
"""

from __future__ import annotations

from typing import Any

from actcheck.tools.base import Tool


class Finish(Tool):
    name = "finish"
    description = (
        "Call this when the task is complete. Provide the final answer for the user, "
        "including whatever was asked for. This ends the run."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The complete answer or summary of what was done.",
            },
        },
        "required": ["answer"],
    }

    def run(self, answer: str, **_: Any) -> str:
        return answer


class GiveUp(Tool):
    name = "give_up"
    description = (
        "Call this if the task cannot be completed with the available tools, or if it "
        "is impossible as stated. Say precisely what blocked you. Stopping with a clear "
        "reason is a correct outcome; guessing an answer is not."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "What was tried and what specifically blocked progress.",
            },
        },
        "required": ["reason"],
    }

    def run(self, reason: str, **_: Any) -> str:
        return reason


TERMINAL_TOOLS = {Finish.name, GiveUp.name}
