"""The command line, exercised the way a user would."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from actcheck.cli import main


@pytest.fixture
def workdir(workspace: Path, tmp_path: Path, monkeypatch) -> Path:
    """Run the CLI from a directory with its own config, as a user would."""
    monkeypatch.chdir(tmp_path)
    Path("actcheck.toml").write_text(
        f'[actcheck]\nworkspace = "{workspace}"\nprovider = "scripted"\n'
        f'checker = "programmatic"\nplan = false\nmax_steps = 8\n'
        f'trace_dir = "{tmp_path / "traces"}"\n'
    )
    return tmp_path


def test_no_command_prints_help(capsys) -> None:
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out


def test_tools_lists_the_tool_set(workdir, capsys) -> None:
    assert main(["tools"]) == 0
    out = capsys.readouterr().out
    assert "read_file" in out and "(mutates)" in out


def test_tools_json_is_the_model_facing_schema(workdir, capsys) -> None:
    assert main(["tools", "--json"]) == 0
    specs = json.loads(capsys.readouterr().out)
    assert {"name", "description", "input_schema"} <= set(specs[0])


def test_run_prints_the_answer_and_exits_zero(workdir, capsys) -> None:
    assert main(["run", "Which office does primary on-call sit in?"]) == 0
    assert "Helsinki" in capsys.readouterr().out


def test_run_json_is_machine_readable(workdir, capsys) -> None:
    assert main(["run", "Which office does primary on-call sit in?", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "success" and payload["steps"]


def test_run_quiet_prints_only_the_answer(workdir, capsys) -> None:
    main(["run", "Which office does primary on-call sit in?", "--quiet"])
    assert "steps," not in capsys.readouterr().out


def test_a_failed_run_exits_non_zero(workdir, capsys) -> None:
    """So actcheck is usable in a script without parsing its output."""
    assert main(["run", "anything", "--max-steps", "1"]) == 1


def test_flags_override_the_config_file(workdir, capsys) -> None:
    main(["run", "Which office does primary on-call sit in?", "--max-steps", "2", "--json"])
    assert len(json.loads(capsys.readouterr().out)["steps"]) <= 2


def test_plan_shows_the_plan_without_running(workdir, capsys) -> None:
    assert main(["plan", "find the build command"]) == 0
    out = capsys.readouterr().out
    assert "goal:" in out and "1." in out


def test_trace_renders_the_most_recent_run(workdir, capsys) -> None:
    main(["run", "Which office does primary on-call sit in?", "--quiet"])
    capsys.readouterr()
    assert main(["trace"]) == 0
    out = capsys.readouterr().out
    assert "RUN" in out and "END   [success]" in out


def test_trace_list_shows_recorded_runs(workdir, capsys) -> None:
    main(["run", "Which office does primary on-call sit in?", "--quiet"])
    capsys.readouterr()
    assert main(["trace", "--list"]) == 0
    assert "[success]" in capsys.readouterr().out


def test_trace_with_nothing_recorded_is_an_error(workdir, capsys) -> None:
    assert main(["trace"]) == 2


def test_replay_reruns_a_recorded_trace(workdir, capsys) -> None:
    main(["run", "Which office does primary on-call sit in?", "--quiet"])
    capsys.readouterr()
    assert main(["replay"]) == 0
    assert "Helsinki" in capsys.readouterr().out


def test_replay_costs_no_tokens(workdir, capsys) -> None:
    main(["run", "Which office does primary on-call sit in?", "--quiet"])
    capsys.readouterr()
    main(["replay", "--quiet"])
    assert "0 tokens" in capsys.readouterr().out


def test_eval_runs_the_task_set(workdir, capsys) -> None:
    tasks = workdir / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "id": "oncall",
                "task": "Which office does primary on-call sit in?",
                "expect": {"status": "success", "answer_contains": ["Helsinki"]},
            }
        )
        + "\n"
    )
    assert main(["eval", "--tasks", str(tasks), "--no-save"]) == 0
    assert "success" in capsys.readouterr().out


def test_eval_exits_non_zero_when_a_task_fails(workdir, capsys) -> None:
    tasks = workdir / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"id": "impossible", "task": "x", "expect": {"answer_contains": ["zzz"]}}) + "\n"
    )
    assert main(["eval", "--tasks", str(tasks), "--no-save"]) == 1


def test_init_writes_a_starter_config(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    assert "[actcheck]" in (tmp_path / "actcheck.toml").read_text()


def test_init_refuses_to_clobber(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    main(["init"])
    assert main(["init"]) == 2
    assert main(["init", "--force"]) == 0


def test_a_workspace_that_normalises_to_the_root_is_refused(workdir, capsys) -> None:
    """A typo'd --workspace must not silently hand the agent the whole machine."""
    assert main(["run", "x", "--workspace", "/nonexistent/../.."]) == 2
    assert "refusing to use" in capsys.readouterr().err
