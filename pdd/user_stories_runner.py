from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import yaml
from pydantic import ValidationError

from .llm_invoke import llm_invoke
from .user_stories_schema import (
    StoryRunResult,
    StoryRunTrace,
    UserStory,
)


def load_user_story(path: Path) -> UserStory:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid user story file (expected mapping): {path}")
    return UserStory.model_validate(data)


def load_user_stories(stories_dir: Path) -> List[Tuple[Path, UserStory]]:
    if not stories_dir.exists():
        return []
    stories: List[Tuple[Path, UserStory]] = []
    for p in sorted(stories_dir.glob("*.yml")) + sorted(stories_dir.glob("*.yaml")):
        stories.append((p, load_user_story(p)))
    return stories


def _read_prompt_files(repo_root: Path, prompt_paths: Iterable[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for rel in prompt_paths:
        p = Path(rel)
        if not p.is_absolute():
            p = repo_root / p
        content = p.read_text(encoding="utf-8")
        out.append((str(rel), content))
    return out


def _build_story_prompt(story: UserStory, prompt_files: List[Tuple[str, str]]) -> str:
    # Keep this prompt deterministic and easy to parse.
    # We ask the model to "play through" the story using the provided prompts, then output JSON.
    prompts_blob = "\n\n".join(
        f"<prompt path=\"{path}\">\n{content}\n</prompt>"
        for path, content in prompt_files
    )

    steps_blob = "\n".join(
        f"- {step.get('user') or step.get('system') or json.dumps(step)}"
        for step in (story.steps or [])
    )

    return f"""You are a deterministic prompt unit-test runner.

## Inputs
story_id: {story.id}
title: {story.title}
inputs_json: {json.dumps(story.inputs, ensure_ascii=False)}

## Prompts under test
{prompts_blob}

## User story steps
{steps_blob}

## Task
Simulate how these prompts would behave for the user story. Produce:
1) a short step-by-step trace (what prompt is used when, and why)
2) a final_response that represents what the system would output
3) if the story fails, list likely failure_prompt_candidates (prompt file paths)

## Output format (MUST be valid JSON)
{{
  "prompts_used": ["path1.prompt", "path2.prompt"],
  "prompt_execution_path": ["..."],
  "step_trace": ["..."],
  "failure_prompt_candidates": ["..."],
  "final_response": "..."
}}
"""


def _evaluate_assertions(story: UserStory, trace: StoryRunTrace) -> Tuple[bool, str]:
    text = trace.final_response or ""
    for s in story.assertions.must_include:
        if s not in text:
            return False, f"Missing required text: {s!r}"
    for s in story.assertions.must_not_include:
        if s in text:
            return False, f"Found forbidden text: {s!r}"
    return True, "ok"


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(?P<body>[\s\S]*?)\n```\s*$", re.IGNORECASE)


def _coerce_json_payload(raw: object) -> object:
    """Best-effort: turn model output into a JSON-decodable object.

    Models often wrap JSON in Markdown fences (```json ... ```). We strip those.
    If parsing still fails, we fall back to extracting the first {...} block.
    """
    if not isinstance(raw, str):
        return raw

    s = raw.strip()
    m = _FENCE_RE.match(s)
    if m:
        s = (m.group("body") or "").strip()

    # Fast path: looks like JSON already
    if s.startswith("{") or s.startswith("["):
        return s

    # Fallback: find first JSON object in the text
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1].strip()

    return s


def run_user_story(
    repo_root: Path,
    story: UserStory,
    *,
    verbose: bool = False,
) -> StoryRunResult:
    prompt_files = _read_prompt_files(repo_root, story.prompts)
    prompt = _build_story_prompt(story, prompt_files)

    response = llm_invoke(
        # Use messages to avoid `{}` prompt-template formatting issues.
        messages=[{"role": "user", "content": prompt}],
        strength=story.model.min_strength,
        temperature=story.model.temperature,
        verbose=verbose,
    )

    raw = response.get("result", "")
    trace: StoryRunTrace
    try:
        payload = _coerce_json_payload(raw)
        obj = json.loads(payload) if isinstance(payload, str) else payload
        trace = StoryRunTrace.model_validate(obj)
    except Exception as e:
        # If the model didn't return JSON, treat as failure but preserve raw text.
        trace = StoryRunTrace(final_response=str(raw))
        return StoryRunResult(
            story_id=story.id,
            passed=False,
            reason=f"Failed to parse model JSON output: {type(e).__name__}: {e}",
            trace=trace,
            cost=float(response.get("cost", 0.0) or 0.0),
            model_name=str(response.get("model_name", "") or ""),
        )

    passed, reason = _evaluate_assertions(story, trace)
    return StoryRunResult(
        story_id=story.id,
        passed=passed,
        reason=reason if not passed else "",
        trace=trace,
        cost=float(response.get("cost", 0.0) or 0.0),
        model_name=str(response.get("model_name", "") or ""),
    )


def run_user_stories(
    repo_root: Path,
    stories: List[UserStory],
    *,
    verbose: bool = False,
) -> List[StoryRunResult]:
    results: List[StoryRunResult] = []
    for story in stories:
        results.append(run_user_story(repo_root, story, verbose=verbose))
    return results

