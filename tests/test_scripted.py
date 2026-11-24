"""The offline model: what it can do, and what it honestly cannot."""

from __future__ import annotations

from actcheck.agent import Agent
from actcheck.config import Config
from actcheck.llm.scripted import ScriptedModel, _best_line_scored, _search_terms
from actcheck.types import RunStatus


def test_a_script_is_played_in_order() -> None:
    model = ScriptedModel([{"tool": "a", "args": {}}, {"tool": "b", "args": {}}])
    tools = [{"name": "a"}, {"name": "b"}, {"name": "finish"}]
    assert model.respond("", [], tools).tool_calls[0].name == "a"
    assert model.respond("", [], tools).tool_calls[0].name == "b"


def test_running_off_the_end_of_a_script_finishes_cleanly() -> None:
    """A test-authoring mistake should stay legible, not hang the loop."""
    model = ScriptedModel([{"tool": "a", "args": {}}])
    tools = [{"name": "a"}, {"name": "finish"}]
    model.respond("", [], tools)
    assert model.respond("", [], tools).tool_calls[0].name == "finish"


def test_the_reflex_policy_solves_a_lookup(cfg: Config) -> None:
    result = Agent(cfg.merged({"max_steps": 10})).run("Which office does primary on-call sit in?")
    assert result.status is RunStatus.SUCCESS
    assert "Helsinki" in result.answer


def test_the_reflex_policy_stops_once_it_has_an_answer(cfg: Config) -> None:
    """Without this it works through every tool before answering, which makes
    any step count measured against it meaningless."""
    result = Agent(cfg.merged({"max_steps": 16})).run("Which office does primary on-call sit in?")
    assert result.n_steps <= 6


def test_the_reflex_policy_does_not_repeat_itself(cfg: Config) -> None:
    result = Agent(cfg.merged({"max_steps": 16})).run("What is in this project?")
    signatures = [s.tool_call.signature() for s in result.steps if s.tool_call]
    assert len(signatures) == len(set(signatures))


def test_it_cannot_count(cfg: Config) -> None:
    """The honest limit: a lexical matcher extracts, it does not compute. This
    is asserted rather than hidden because the eval numbers depend on it."""
    result = Agent(cfg.merged({"max_steps": 12})).run(
        "What is the combined total_cents of every refunded order in data/orders.csv?"
    )
    assert "50999" not in result.answer


def test_it_cannot_tell_absent_from_adjacent(cfg: Config) -> None:
    """Asked something the workspace does not answer, it extracts the nearest
    line rather than abstaining. Knowing when to say 'I don't know' needs a
    real model."""
    result = Agent(cfg.merged({"max_steps": 12})).run(
        "What is Harbour's documented API rate limit per customer?"
    )
    assert result.status is RunStatus.SUCCESS  # not FAILED: it did not give up


def test_scoring_prefers_a_short_relevant_line() -> None:
    line, score = _best_line_scored(
        "which office does on-call sit in",
        ["Primary on-call sits in the Helsinki office.\nUnrelated sentence about queues."],
    )
    assert "Helsinki" in line and score > 0


def test_quoted_phrases_become_search_terms() -> None:
    assert "refund policy" in _search_terms('find the "refund policy" section')


def test_stopwords_are_not_search_terms() -> None:
    assert not any(t in {"what", "the", "find"} for t in _search_terms("what is the thing"))


def test_plan_completion_returns_parseable_json() -> None:
    import json

    response = ScriptedModel().complete("sys", "Task: do it", purpose="plan")
    assert isinstance(json.loads(response.text)["steps"], list)


def test_check_completion_notices_an_error() -> None:
    import json

    response = ScriptedModel().complete("sys", "ERROR: it broke", purpose="check")
    assert json.loads(response.text)["judgement"] == "no_progress"
