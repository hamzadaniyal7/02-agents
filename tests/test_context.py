"""The transcript, and keeping it from eating the run."""

from __future__ import annotations

from actcheck.context import Transcript, step_digest
from actcheck.types import Judgement, Observation, Step, ToolCall, Verdict


def add_step(transcript: Transcript, n: int, payload: str = "result") -> ToolCall:
    call = ToolCall(name="read_file", args={"path": f"file{n}.md"}, call_id=f"call_{n}")
    transcript.add_assistant(
        [{"type": "tool_use", "id": call.call_id, "name": call.name, "input": call.args}]
    )
    transcript.add_tool_result(call, payload)
    return call


def test_the_task_is_the_first_message() -> None:
    transcript = Transcript("do the thing")
    assert transcript.messages()[0]["content"][0]["text"] == "do the thing"


def test_steps_appear_in_order() -> None:
    transcript = Transcript("task")
    add_step(transcript, 1)
    roles = [m["role"] for m in transcript.messages()]
    assert roles == ["user", "assistant", "user"]


def test_tool_results_carry_the_call_id() -> None:
    """It is how a result is matched to its request on the next turn."""
    transcript = Transcript("task")
    call = add_step(transcript, 1)
    result = transcript.messages()[-1]["content"][0]
    assert result["tool_use_id"] == call.call_id


def test_an_empty_result_is_labelled_not_blank() -> None:
    transcript = Transcript("task")
    call = ToolCall(name="grep", args={}, call_id="c1")
    transcript.add_assistant([{"type": "tool_use", "id": "c1", "name": "grep", "input": {}}])
    transcript.add_tool_result(call, "")
    assert transcript.messages()[-1]["content"][0]["content"] == "(empty result)"


def test_notes_are_marked_as_coming_from_the_harness() -> None:
    transcript = Transcript("task")
    transcript.add_note("slow down")
    assert transcript.messages()[-1]["content"][0]["text"].startswith("[actcheck]")


def test_assistant_blocks_are_kept_verbatim() -> None:
    """Thinking blocks carry a signature the API validates on the next turn."""
    transcript = Transcript("task")
    block = {"type": "thinking", "thinking": "...", "signature": "abc123"}
    transcript.add_assistant([block])
    assert transcript.messages()[-1]["content"][0]["signature"] == "abc123"


# ----------------------------------------------------------------- compaction
def test_small_transcripts_are_not_compacted() -> None:
    transcript = Transcript("task", char_budget=100_000)
    for i in range(4):
        add_step(transcript, i)
    transcript.messages()
    assert transcript.compactions == 0


def test_compaction_fires_over_the_budget() -> None:
    transcript = Transcript("task", char_budget=3_000, keep_recent_steps=2)
    for i in range(10):
        add_step(transcript, i, payload="y" * 400)
    transcript.messages()
    assert transcript.compactions >= 1


def test_compaction_keeps_the_task() -> None:
    """An agent that loses the task statement is not recoverable."""
    transcript = Transcript("find the deadline", char_budget=2_000, keep_recent_steps=2)
    for i in range(12):
        add_step(transcript, i, payload="y" * 400)
    assert "find the deadline" in str(transcript.messages()[0])


def test_compaction_keeps_the_recent_steps_verbatim() -> None:
    """The immediate past is what stops the agent repeating what just failed."""
    transcript = Transcript("task", char_budget=2_000, keep_recent_steps=2)
    for i in range(12):
        add_step(transcript, i, payload=f"payload-{i}-" + "y" * 400)
    rendered = str(transcript.messages())
    assert "payload-11-" in rendered


def test_the_digest_records_what_was_already_tried() -> None:
    transcript = Transcript("task", char_budget=2_000, keep_recent_steps=1)
    for i in range(10):
        add_step(transcript, i, payload="y" * 400)
    rendered = str(transcript.messages())
    assert "Earlier steps" in rendered and "read_file" in rendered


def test_digests_accumulate_across_compactions() -> None:
    transcript = Transcript("task", char_budget=1_500, keep_recent_steps=1)
    for i in range(8):
        add_step(transcript, i, payload="y" * 400)
        transcript.messages()
    for i in range(8, 16):
        add_step(transcript, i, payload="y" * 400)
        transcript.messages()
    assert transcript.compactions >= 2
    assert "file0.md" in str(transcript.messages())


def test_orphaned_tool_results_are_rewritten_not_dropped() -> None:
    """The API rejects a tool_result with no matching tool_use, but the agent
    still needs to know the call happened."""
    transcript = Transcript("task")
    orphan = ToolCall(name="grep", args={}, call_id="missing")
    transcript.add_tool_result(orphan, "an earlier finding")
    rendered = transcript.messages()[-1]["content"][0]
    assert rendered["type"] == "text"
    assert "an earlier finding" in rendered["text"]


def test_hard_trimming_is_the_backstop() -> None:
    """Even the kept tail has to fit."""
    transcript = Transcript("task", char_budget=2_000, keep_recent_steps=20, compact=False)
    for i in range(6):
        add_step(transcript, i, payload="y" * 5_000)
    assert "trimmed to fit" in str(transcript.messages())


def test_compaction_can_be_turned_off() -> None:
    transcript = Transcript("task", char_budget=500, keep_recent_steps=1, compact=False)
    for i in range(6):
        add_step(transcript, i, payload="y" * 400)
    assert transcript.compactions == 0


# --------------------------------------------------------------- step_digest
def test_step_digest_lists_tools_and_verdicts() -> None:
    step = Step(
        index=1,
        tool_call=ToolCall(name="grep", args={"pattern": "x"}),
        observation=Observation(content="nothing"),
        verdict=Verdict(Judgement.NO_PROGRESS, "found nothing"),
    )
    out = step_digest([step])
    assert "grep" in out and "no_progress" in out and "found nothing" in out


def test_step_digest_handles_an_empty_run() -> None:
    assert "no steps" in step_digest([])
