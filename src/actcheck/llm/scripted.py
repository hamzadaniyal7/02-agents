"""A deterministic stand-in for the real model.

This is what makes the test suite and `actcheck eval` run with no API key, no
network, and no cost -- the same role `FakeLLM` plays in the 01-rag project.

It works in two modes.

**Scripted.** Hand it a list of turns and it plays them back in order. Tests
that need a precise sequence -- "call the same tool twice and prove the loop
guard fires" -- use this. There is no cleverness to get in the way.

**Reflex.** With no script it follows a fixed policy: look at the task, look at
what has already been tried, pick the next obviously-relevant tool, and when
there is nothing left to try, answer with the line from its observations that
best matches the task. It derives all of that from the transcript rather than
from instance state, so it behaves identically on a replayed run.

What it emphatically does not do is *reason*. It cannot decide that a file it
read is the wrong file, it cannot combine two observations into a fact that is
in neither, and it cannot tell "this is absent" from "I have not looked yet".
Those are the parts that need a real model, which is exactly why `actcheck
eval` reports its scores separately and why the README shows the gap instead of
hiding it. A harness whose offline numbers look like success is a harness that
is measuring itself.
"""

from __future__ import annotations

import json
import re
from typing import Any

from actcheck.llm.base import Message, Model, ModelResponse
from actcheck.types import ToolCall, Usage

FILENAME_RE = re.compile(r"\b[\w./-]+\.(?:md|txt|py|json|toml|ya?ml|csv|log|cfg|ini|sh)\b")
QUOTED_RE = re.compile(r"[\"'`]([^\"'`]{3,60})[\"'`]")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{2,}")

# Overlap score above which the reflex policy stops looking and answers.
# Tuned on the sample workspace: a line sharing three task words out of seven
# scores about 1.13, and a merely topical line scores well under 1.0.
FINISH_CONFIDENCE = 1.0

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "what",
    "which",
    "how",
    "many",
    "does",
    "did",
    "are",
    "was",
    "were",
    "has",
    "have",
    "you",
    "your",
    "its",
    "it's",
    "into",
    "out",
    "all",
    "any",
    "can",
    "find",
    "report",
    "return",
    "tell",
    "give",
    "list",
    "show",
    "there",
    "then",
    "than",
    "use",
    "using",
    "file",
    "files",
    "workspace",
    "about",
    "inside",
    "line",
    "lines",
    "word",
    "words",
    "name",
    "names",
    "answer",
    "please",
    "run",
}


class ScriptedModel(Model):
    """Offline model: plays a script, or follows the reflex policy."""

    name = "scripted"

    def __init__(self, script: list[dict[str, Any]] | None = None) -> None:
        self.script = list(script) if script else None
        self._cursor = 0

    # ------------------------------------------------------------------ api
    def respond(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        available = {t["name"] for t in tools}
        if self.script is not None:
            return self._play(available)
        return self._reflex(messages, available)

    def complete(
        self,
        system: str,
        user: str,
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        purpose: str = "",
    ) -> ModelResponse:
        """Fabricate a plan or a check verdict, deterministically.

        Both are deliberately generic. The point is to exercise the plumbing --
        that a plan parses, that a verdict reaches the loop -- not to pretend
        an offline matcher can judge whether a success criterion was met.
        """
        if purpose == "plan":
            payload = {
                "goal": _task_from_prompt(user),
                "steps": [
                    {
                        "description": "Locate the files relevant to the task",
                        "success_criterion": "At least one candidate file has been identified",
                    },
                    {
                        "description": "Read the relevant content",
                        "success_criterion": "The content needed to answer has been read",
                    },
                    {
                        "description": "Report the answer",
                        "success_criterion": "A final answer has been given with finish",
                    },
                ],
            }
        elif purpose == "check":
            looks_bad = "ERROR:" in user or "No matches" in user
            payload = {
                "judgement": "no_progress" if looks_bad else "progress",
                "reason": (
                    "offline checker: the observation reports an error or an empty result"
                    if looks_bad
                    else "offline checker: the tool returned content"
                ),
                "confidence": 0.3,
            }
        else:
            payload = {"text": ""}

        return ModelResponse(
            text=json.dumps(payload),
            model=self.name,
            usage=Usage(input_tokens=len(user) // 4, output_tokens=40, calls=1),
            stop_reason="end_turn",
        )

    # -------------------------------------------------------------- scripted
    def _play(self, available: set[str]) -> ModelResponse:
        assert self.script is not None
        if self._cursor >= len(self.script):
            # Running off the end of a script is a test authoring mistake, but
            # hanging or raising mid-loop makes it hard to see. Finishing
            # cleanly keeps the failure legible in the trace.
            return _finish_response("script exhausted", available)
        turn = self.script[self._cursor]
        self._cursor += 1

        if "tool" in turn:
            call = ToolCall(name=turn["tool"], args=dict(turn.get("args", {})))
            return ModelResponse(
                text=turn.get("text", ""),
                tool_calls=[call],
                stop_reason="tool_use",
                model=self.name,
                usage=Usage(output_tokens=20, calls=1),
                blocks=_blocks(turn.get("text", ""), [call]),
            )
        return ModelResponse(
            text=turn.get("text", ""),
            stop_reason="end_turn",
            model=self.name,
            usage=Usage(output_tokens=20, calls=1),
            blocks=_blocks(turn.get("text", ""), []),
        )

    # ---------------------------------------------------------------- reflex
    def _reflex(self, messages: list[Message], available: set[str]) -> ModelResponse:
        task = _task_from_messages(messages)
        done = _calls_made(messages)
        observations = _observations(messages)

        # Stop once something read looks like a direct answer. Without this the
        # policy works through every candidate tool before answering, which
        # burns a dozen steps to reach a conclusion it had after two -- and
        # makes every step-count measured against it meaningless.
        answer, confidence = _best_line_scored(task, observations)
        if observations and confidence >= FINISH_CONFIDENCE:
            return _finish_response(answer, available, tokens_in=_rough_tokens(messages))

        for call in self._candidates(task, available, observations):
            if call.signature() not in done:
                return ModelResponse(
                    text=f"Trying {call.name}.",
                    tool_calls=[call],
                    stop_reason="tool_use",
                    model=self.name,
                    usage=Usage(input_tokens=_rough_tokens(messages), output_tokens=24, calls=1),
                    blocks=_blocks(f"Trying {call.name}.", [call]),
                )

        return _finish_response(
            answer or "No answer could be extracted offline.",
            available,
            tokens_in=_rough_tokens(messages),
        )

    def _candidates(
        self, task: str, available: set[str], observations: list[str]
    ) -> list[ToolCall]:
        """Actions worth trying, most specific first."""
        out: list[ToolCall] = []

        if "read_file" in available:
            # Filenames named in the task, then filenames seen in observations:
            # the second is how `list_dir` -> `read_file` chains without the
            # policy needing to understand what a directory listing is.
            for path in _filenames(task) + _filenames("\n".join(observations)):
                out.append(ToolCall(name="read_file", args={"path": path}))

        if "grep" in available:
            for term in _search_terms(task):
                out.append(ToolCall(name="grep", args={"pattern": term, "ignore_case": True}))

        if "list_dir" in available:
            out.append(ToolCall(name="list_dir", args={"path": ".", "recursive": True}))

        return out


# --------------------------------------------------------------------- utils
def _finish_response(answer: str, available: set[str], tokens_in: int = 0) -> ModelResponse:
    if "finish" not in available:
        return ModelResponse(
            text=answer,
            stop_reason="end_turn",
            model="scripted",
            usage=Usage(input_tokens=tokens_in, output_tokens=30, calls=1),
        )
    call = ToolCall(name="finish", args={"answer": answer})
    return ModelResponse(
        text="",
        tool_calls=[call],
        stop_reason="tool_use",
        model="scripted",
        usage=Usage(input_tokens=tokens_in, output_tokens=30, calls=1),
        blocks=_blocks("", [call]),
    )


def _blocks(text: str, calls: list[ToolCall]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    blocks.extend(
        {"type": "tool_use", "id": c.call_id, "name": c.name, "input": c.args} for c in calls
    )
    return blocks


def _task_from_messages(messages: list[Message]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        for block in content or []:
            if isinstance(block, dict) and block.get("type") == "text":
                return str(block.get("text", ""))
    return ""


def _task_from_prompt(user: str) -> str:
    match = re.search(r"Task:\s*(.+?)\s*(?:\n\n|$)", user, re.DOTALL)
    return match.group(1).strip() if match else user.strip()[:200]


def _calls_made(messages: list[Message]) -> set[str]:
    """Signatures of every tool call already in the transcript."""
    out: set[str] = set()
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                out.add(ToolCall(name=block["name"], args=dict(block.get("input", {}))).signature())
    return out


def _observations(messages: list[Message]) -> list[str]:
    out: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                body = block.get("content")
                if isinstance(body, list):
                    body = " ".join(str(p.get("text", "")) for p in body if isinstance(p, dict))
                if body and not block.get("is_error"):
                    out.append(str(body))
    return out


def _filenames(text: str) -> list[str]:
    seen: list[str] = []
    for match in FILENAME_RE.findall(text):
        if match not in seen:
            seen.append(match)
    return seen[:8]


def _search_terms(task: str) -> list[str]:
    """Quoted phrases first, then the most distinctive bare word."""
    terms = [q.strip() for q in QUOTED_RE.findall(task)]
    words = sorted(
        {w for w in WORD_RE.findall(task) if w.lower() not in STOPWORDS and len(w) > 3},
        key=lambda w: (-len(w), w.lower()),
    )
    terms.extend(words[:2])
    return terms[:3]


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text) if w.lower() not in STOPWORDS]


def _best_line(task: str, observations: list[str]) -> str:
    return _best_line_scored(task, observations)[0]


def _best_line_scored(task: str, observations: list[str]) -> tuple[str, float]:
    """The observed line that overlaps the task most, and how strongly.

    Overlap is normalised by line length so a long paragraph that mentions one
    task word does not beat a short line that mentions three. This is the same
    bargain the 01-rag fake makes: extraction instead of reasoning, good enough
    to prove the plumbing carries real content end to end.
    """
    wanted = set(_tokens(task))
    if not wanted:
        return "", 0.0
    best, best_score = "", 0.0
    for observation in observations:
        for raw in observation.splitlines():
            line = re.sub(r"^\s*\d+\s\s", "", raw).strip()
            if len(line) < 12:
                continue
            tokens = _tokens(line)
            if not tokens:
                continue
            score = len(wanted & set(tokens)) / (len(tokens) ** 0.5)
            if score > best_score:
                best, best_score = line, score
    return best, best_score


def _rough_tokens(messages: list[Message]) -> int:
    return sum(len(json.dumps(m, default=str)) for m in messages) // 4
