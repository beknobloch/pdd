from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from pdd.user_stories_schema import UserStory
from pdd.user_stories_runner import run_user_story


def test_run_user_story_evaluates_assertions(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "prompts").mkdir()
    (repo_root / "prompts" / "a.prompt").write_text("PROMPT A", encoding="utf-8")

    story = UserStory(
        id="s1",
        title="Test",
        prompts=["prompts/a.prompt"],
        steps=[{"user": "Do thing"}],
        assertions={"must_include": ["OK"], "must_not_include": ["ERROR"]},
        model={"min_strength": 0.7, "temperature": 0.0},
    )

    model_json = {
        "prompts_used": ["prompts/a.prompt"],
        "prompt_execution_path": ["a.prompt"],
        "step_trace": ["used a.prompt"],
        "failure_prompt_candidates": [],
        "final_response": "OK done",
    }

    with patch("pdd.user_stories_runner.llm_invoke") as mock_llm:
        mock_llm.return_value = {"result": json.dumps(model_json), "cost": 0.01, "model_name": "mock"}
        result = run_user_story(repo_root, story, verbose=False)

    assert result.passed is True
    assert result.reason == ""
    assert result.cost == 0.01


def test_run_user_story_fails_on_missing_required_text(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "prompts").mkdir()
    (repo_root / "prompts" / "a.prompt").write_text("PROMPT A", encoding="utf-8")

    story = UserStory(
        id="s1",
        title="Test",
        prompts=["prompts/a.prompt"],
        steps=[{"user": "Do thing"}],
        assertions={"must_include": ["REQUIRED"]},
        model={"min_strength": 0.7, "temperature": 0.0},
    )

    model_json = {
        "prompts_used": ["prompts/a.prompt"],
        "prompt_execution_path": ["a.prompt"],
        "step_trace": ["used a.prompt"],
        "failure_prompt_candidates": ["prompts/a.prompt"],
        "final_response": "OK done",
    }

    with patch("pdd.user_stories_runner.llm_invoke") as mock_llm:
        mock_llm.return_value = {"result": json.dumps(model_json), "cost": 0.0, "model_name": "mock"}
        result = run_user_story(repo_root, story, verbose=False)

    assert result.passed is False
    assert "Missing required text" in result.reason

