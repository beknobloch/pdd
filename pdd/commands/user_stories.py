from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import click

from ..core.errors import handle_error
from ..user_stories_runner import load_user_stories, run_user_stories


@click.group(name="user-stories")
def user_stories_group() -> None:
    """Prompt unit testing via user stories."""


@user_stories_group.command(name="test")
@click.option(
    "--dir",
    "stories_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="user_stories",
    show_default=True,
    help="Directory containing user story YAML files.",
)
@click.option(
    "--story",
    "story_id",
    default=None,
    help="Run only a specific story id (matches `id` field).",
)
@click.pass_context
def test_user_stories(
    ctx: click.Context,
    stories_dir: str,
    story_id: Optional[str],
) -> Optional[Tuple[Any, float, str]]:
    """Run user stories and report pass/fail."""
    ctx.ensure_object(dict)
    quiet = ctx.obj.get("quiet", False)
    verbose = ctx.obj.get("verbose", False)

    try:
        repo_root = Path.cwd()
        story_pairs = load_user_stories(repo_root / stories_dir)
        if story_id is not None:
            story_pairs = [(p, s) for (p, s) in story_pairs if s.id == story_id]

        if not story_pairs:
            if not quiet:
                click.echo(f"No user stories found in: {stories_dir}")
            return {"success": True, "passed": 0, "failed": 0, "results": []}, 0.0, "none"

        results = run_user_stories(repo_root, [s for _, s in story_pairs], verbose=verbose)

        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        total_cost = sum(r.cost for r in results)
        model_name = next((r.model_name for r in results if r.model_name), "unknown")

        if not quiet:
            click.echo("")
            click.echo("User stories")
            for r in results:
                status = "PASS" if r.passed else "FAIL"
                click.echo(f"- {status} {r.story_id}" + (f" ({r.reason})" if r.reason else ""))
                if verbose and r.trace.step_trace:
                    for line in r.trace.step_trace:
                        click.echo(f"    {line}")
                if not r.passed and r.trace.failure_prompt_candidates:
                    click.echo("    failure_prompt_candidates:")
                    for p in r.trace.failure_prompt_candidates:
                        click.echo(f"      - {p}")
            click.echo("")
            click.echo(f"Summary: {passed} passed, {failed} failed. Cost: ${total_cost:.6f}")

        payload = {
            "success": failed == 0,
            "passed": passed,
            "failed": failed,
            "total_cost": total_cost,
            "results": [r.model_dump() for r in results],
        }
        return payload, total_cost, model_name
    except click.Abort:
        raise
    except Exception as e:
        handle_error(e, "user-stories test", quiet)
        return None

