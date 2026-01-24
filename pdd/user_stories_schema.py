from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class UserStoryAssertions(BaseModel):
    """Local assertions evaluated against the model's final_response."""

    must_include: List[str] = Field(default_factory=list)
    must_not_include: List[str] = Field(default_factory=list)


class UserStoryModelConfig(BaseModel):
    """Model configuration hints for a user-story run."""

    min_strength: float = 0.7
    temperature: float = 0.0


class UserStory(BaseModel):
    """A prompt unit-test definition ("user story")."""

    id: str
    title: str = ""

    # Explicit prompt files involved in this story (MVP approach).
    prompts: List[str] = Field(default_factory=list)

    # Free-form inputs that can be referenced by the model in narration.
    inputs: Dict[str, Any] = Field(default_factory=dict)

    # Steps are natural-language actions. MVP keeps them as a list of messages.
    steps: List[Dict[str, str]] = Field(default_factory=list)

    assertions: UserStoryAssertions = Field(default_factory=UserStoryAssertions)
    model: UserStoryModelConfig = Field(default_factory=UserStoryModelConfig)

    # Optional knobs
    tags: List[str] = Field(default_factory=list)


class StoryRunTrace(BaseModel):
    """Structured trace returned by the model (used for reporting)."""

    prompts_used: List[str] = Field(default_factory=list)
    step_trace: List[str] = Field(default_factory=list)
    prompt_execution_path: List[str] = Field(default_factory=list)
    failure_prompt_candidates: List[str] = Field(default_factory=list)
    final_response: str = ""


class StoryRunResult(BaseModel):
    story_id: str
    passed: bool
    reason: str = ""
    trace: StoryRunTrace = Field(default_factory=StoryRunTrace)

    cost: float = 0.0
    model_name: str = ""


def default_user_stories_dir(repo_root: Path) -> Path:
    return repo_root / "user_stories"

