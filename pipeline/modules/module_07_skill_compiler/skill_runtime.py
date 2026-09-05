"""Render the compiled skill's prompt program for a new original story."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .jsonio import write_json
from .paths import output_dir, runs_dir, validate_author_id


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _run_id(value: str) -> str:
    result = str(value or "").strip()
    if not _RUN_ID_RE.fullmatch(result):
        raise ValueError("run_id may contain only letters, numbers, underscores, and hyphens")
    return result


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return payload


def _package_file(folder: Path, relative: str, legacy: str = "") -> Path:
    """Prefer the standard Skill layout while keeping old draft packages readable."""

    target = folder / relative
    if target.exists():
        return target
    fallback = folder / legacy if legacy else target
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"compiled skill file is missing: {relative}")


def _skill_files(author_id: str) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    folder = output_dir(validate_author_id(author_id))
    manifest = _json_object(folder / "skill_manifest.json")
    if not manifest.get("generated_files"):
        raise ValueError("compiled skill manifest is incomplete; run compile-skill first")
    return (
        folder,
        _json_object(_package_file(folder, "references/author_style_profile.json", "author_style_profile.json")),
        _json_object(_package_file(folder, "references/narrative_templates.json", "narrative_templates.json")),
        _json_object(_package_file(folder, "references/continuity_requirements.json", "continuity_requirements.json")),
        _json_object(_package_file(folder, "references/style_execution_spec.json", "style_execution_spec.json")),
    )


def _prompt_file(folder: Path, name: str) -> Path:
    return _package_file(folder, f"references/prompts/{name}", f"prompt_templates/{name}")


def _render(template: Path, values: dict[str, Any]) -> str:
    body = template.read_text(encoding="utf-8")
    for key, value in values.items():
        body = body.replace("{" + key + "}", json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value)
    unresolved = sorted(set(re.findall(r"\{[A-Z_]+\}", body)))
    if unresolved:
        raise ValueError(f"unresolved prompt placeholders: {', '.join(unresolved)}")
    return body


def _execution_mode_instruction(
    story_state: dict[str, Any],
    chapter_brief: dict[str, Any] | None = None,
    chapter_plan: dict[str, Any] | None = None,
) -> str:
    """Separate source-safe original writing from input-bounded fidelity evaluation."""

    modes = {
        str(payload.get("reconstruction_mode", "")).strip().lower()
        for payload in (story_state, chapter_brief or {}, chapter_plan or {})
        if isinstance(payload, dict)
    }
    if "fidelity_test_only" in modes:
        return (
            "当前为结构保真评测模式，不是原创发布模式。可使用 <OriginalStoryState>、"
            "<ChapterBrief> 与 <ChapterPlan> 已明确给出的角色名、地点、术语、事件和状态，"
            "以检验结构还原；不得自行补入这些输入之外的来源事实，更不得输入、复制或改写原文句子。"
        )
    return (
        "当前为原创写作模式。只能使用本次原创输入中提供的人物、世界、术语和事实；"
        "不得使用训练来源或其他作品中的专名、设定、情节、句子。"
    )


def _write_prompt(author_id: str, run_id: str, name: str, body: str, metadata: dict[str, Any]) -> Path:
    destination = runs_dir(author_id, _run_id(run_id)) / "skill_prompts"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / name
    target.write_text(body, encoding="utf-8")
    write_json(destination / f"{target.stem}.context.json", metadata)
    return target


def prepare_plan_prompt(author_id: str, story_state_path: Path, chapter_brief_path: Path, run_id: str = "skill") -> Path:
    """Render the first model call: an original, state-aware chapter plan."""

    folder, profile, library, requirements, execution_spec = _skill_files(author_id)
    story_state = _json_object(story_state_path)
    chapter_brief = _json_object(chapter_brief_path)
    if not str(chapter_brief.get("chapter_goal", "")).strip():
        raise ValueError("chapter brief requires a non-empty chapter_goal")
    prompt = _render(_prompt_file(folder, "chapter_plan.md"), {
        "AUTHOR_STYLE_PROFILE_JSON": profile,
        "NARRATIVE_TEMPLATE_LIBRARY_JSON": library,
        "CONTINUITY_REQUIREMENTS_JSON": requirements,
        "STYLE_EXECUTION_SPEC_JSON": execution_spec,
        "EXECUTION_MODE_INSTRUCTIONS": _execution_mode_instruction(story_state, chapter_brief),
        "ORIGINAL_STORY_STATE_JSON": story_state,
        "CHAPTER_BRIEF_JSON": chapter_brief,
    })
    return _write_prompt(author_id, run_id, "chapter_plan.prompt.md", prompt, {
        "stage": "planning", "story_state_path": str(story_state_path), "chapter_brief_path": str(chapter_brief_path),
    })


def _selected_templates(library: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    requested = plan.get("selected_template_ids", [])
    if not isinstance(requested, list) or not requested:
        raise ValueError("chapter plan requires non-empty selected_template_ids")
    by_id = {str(item.get("template_id", "")): item for item in library.get("templates", []) if isinstance(item, dict)}
    unknown = [str(item) for item in requested if str(item) not in by_id]
    if unknown:
        raise ValueError(f"chapter plan selected unknown template IDs: {', '.join(unknown)}")
    return [by_id[str(item)] for item in requested]


def prepare_draft_prompt(author_id: str, story_state_path: Path, chapter_plan_path: Path, run_id: str = "skill") -> Path:
    """Render the second model call using only templates selected by the plan."""

    folder, profile, library, requirements, execution_spec = _skill_files(author_id)
    story_state = _json_object(story_state_path)
    plan = _json_object(chapter_plan_path)
    selected = _selected_templates(library, plan)
    prompt = _render(_prompt_file(folder, "chapter_draft.md"), {
        "AUTHOR_STYLE_PROFILE_JSON": profile,
        "SELECTED_TEMPLATES_JSON": selected,
        "CONTINUITY_REQUIREMENTS_JSON": requirements,
        "STYLE_EXECUTION_SPEC_JSON": execution_spec,
        "EXECUTION_MODE_INSTRUCTIONS": _execution_mode_instruction(story_state, chapter_plan=plan),
        "ORIGINAL_STORY_STATE_JSON": story_state,
        "CHAPTER_PLAN_JSON": plan,
    })
    return _write_prompt(author_id, run_id, "chapter_draft.prompt.md", prompt, {
        "stage": "drafting", "story_state_path": str(story_state_path), "chapter_plan_path": str(chapter_plan_path),
        "selected_template_ids": [item["template_id"] for item in selected],
    })


def prepare_validation_prompt(
    author_id: str,
    story_state_path: Path,
    chapter_plan_path: Path,
    draft_path: Path,
    run_id: str = "skill",
) -> Path:
    """Render the third model call that returns a strict validation report."""

    folder, profile, _, requirements, execution_spec = _skill_files(author_id)
    story_state = _json_object(story_state_path)
    plan = _json_object(chapter_plan_path)
    draft = draft_path.read_text(encoding="utf-8").strip()
    if not draft:
        raise ValueError("draft must not be empty")
    prompt = _render(_prompt_file(folder, "chapter_validate.md"), {
        "AUTHOR_STYLE_PROFILE_JSON": profile,
        "CONTINUITY_REQUIREMENTS_JSON": requirements,
        "STYLE_EXECUTION_SPEC_JSON": execution_spec,
        "EXECUTION_MODE_INSTRUCTIONS": _execution_mode_instruction(story_state, chapter_plan=plan),
        "ORIGINAL_STORY_STATE_JSON": story_state,
        "CHAPTER_PLAN_JSON": plan,
        "DRAFT_TEXT": draft,
    })
    return _write_prompt(author_id, run_id, "chapter_validate.prompt.md", prompt, {
        "stage": "validation", "story_state_path": str(story_state_path), "chapter_plan_path": str(chapter_plan_path),
        "draft_path": str(draft_path),
    })
