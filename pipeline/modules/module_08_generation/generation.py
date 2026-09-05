"""Execute the compiled skill's planning, drafting, validation and repair loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .jsonio import write_json
from .model import ModelSettings, complete_json, complete_text
from .paths import runs_dir
from .skill_runtime import _execution_mode_instruction, _json_object, _render, _run_id, _selected_templates, _skill_files
from .style_execution import assess_style_budget, resolve_style_budget, style_execution_blueprint, style_execution_directives, style_repair_prompt


DEFAULT_MAX_REPAIRS = 3


def _prompt_paths(author_id: str, run_id: str) -> Path:
    target = runs_dir(author_id, _run_id(run_id)) / "generated_chapter"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _validation_prompt(
    folder: Path,
    profile: dict[str, Any],
    requirements: dict[str, Any],
    execution_spec: dict[str, Any],
    story_state: dict[str, Any],
    plan: dict[str, Any],
    draft: str,
) -> str:
    return _render(folder / "prompt_templates" / "chapter_validate.md", {
        "AUTHOR_STYLE_PROFILE_JSON": profile,
        "CONTINUITY_REQUIREMENTS_JSON": requirements,
        "STYLE_EXECUTION_SPEC_JSON": execution_spec,
        "EXECUTION_MODE_INSTRUCTIONS": _execution_mode_instruction(story_state, chapter_plan=plan),
        "ORIGINAL_STORY_STATE_JSON": story_state,
        "CHAPTER_PLAN_JSON": plan,
        "DRAFT_TEXT": draft,
    })


def _draft_prompt(
    folder: Path,
    profile: dict[str, Any],
    selected: list[dict[str, Any]],
    requirements: dict[str, Any],
    execution_spec: dict[str, Any],
    story_state: dict[str, Any],
    plan: dict[str, Any],
) -> str:
    return _render(folder / "prompt_templates" / "chapter_draft.md", {
        "AUTHOR_STYLE_PROFILE_JSON": profile,
        "SELECTED_TEMPLATES_JSON": selected,
        "CONTINUITY_REQUIREMENTS_JSON": requirements,
        "STYLE_EXECUTION_SPEC_JSON": execution_spec,
        "EXECUTION_MODE_INSTRUCTIONS": _execution_mode_instruction(story_state, chapter_plan=plan),
        "ORIGINAL_STORY_STATE_JSON": story_state,
        "CHAPTER_PLAN_JSON": plan,
    })


def _complete_draft(
    system: str,
    prompt: str,
    settings: ModelSettings,
    *,
    minimum_chars: int = 120,
    max_tokens: int | None = None,
) -> str:
    """Retry empty/placeholder model text before style measurement begins."""

    last_text = ""
    last_error: RuntimeError | None = None
    for _ in range(3):
        try:
            draft = complete_text(system, prompt, settings, attempts=6, max_tokens=min(settings.max_tokens, max_tokens or 9_000)).strip()
        except RuntimeError as exc:
            last_error = exc
            prompt += "\n\n上次请求未得到可用正文。请重新输出完整、连续的中文小说正文，至少包含多个可读段落；不要输出解释。"
            continue
        # Character measurement and model tokenisation differ slightly; rejecting a near-complete
        # scene causes needless retries while the chapter-level budget still performs the strict check.
        if len(draft) >= max(120, int(minimum_chars * 0.95)):
            return draft
        last_text = draft
        prompt += f"\n\n上次输出为空或过短。必须输出完整、连续的中文小说正文，至少 {minimum_chars} 个汉字或字符；不要输出解释。"
    if last_error is not None and not last_text:
        raise RuntimeError("model did not return a usable draft after retries") from last_error
    raise ValueError(f"model returned an empty or too-short draft after retries (last length={len(last_text)})")


def _chapter_minimum_chars(plan: dict[str, Any]) -> int:
    budget = plan.get("style_budget", {}) if isinstance(plan.get("style_budget", {}), dict) else {}
    for item in budget.get("metrics", []) if isinstance(budget.get("metrics", []), list) else []:
        if isinstance(item, dict) and item.get("metric_id") == "chapter_char_count":
            try:
                return max(0, int(float(item.get("preferred_min", 0))))
            except (TypeError, ValueError):
                return 0
    return 0


def _segment_plan(
    plan: dict[str, Any],
    scene: dict[str, Any],
    scene_no: int,
    scene_total: int,
    minimum_chars: int,
    scene_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep a long chapter within the model response budget by drafting one planned scene at a time."""

    segment = dict(plan)
    segment["scenes"] = [scene]
    segment["chapter_state_changes"] = [
        item for item in plan.get("chapter_state_changes", [])
        if isinstance(item, dict) and item.get("cause_scene_no") == scene.get("scene_no")
    ]
    segment.pop("covered_required_event_ids", None)
    segment.pop("covered_required_state_change_ids", None)
    budget = json.loads(json.dumps(plan.get("style_budget", {}), ensure_ascii=False))
    for item in budget.get("metrics", []) if isinstance(budget.get("metrics", []), list) else []:
        if not isinstance(item, dict) or item.get("metric_id") != "chapter_char_count":
            continue
        for key in ("target", "preferred_min", "preferred_max"):
            try:
                item[key] = round(float(item[key]) / scene_total)
            except (KeyError, TypeError, ValueError):
                pass
        item["preferred_min"] = max(120, minimum_chars)
        item["preferred_max"] = max(item["preferred_min"], item.get("preferred_max", minimum_chars))
    segment["style_budget"] = budget
    segment["style_execution_blueprint"] = style_execution_blueprint(budget)
    segment["style_execution_directives"] = style_execution_directives(budget)
    segment["current_segment"] = {
        "scene_no": scene.get("scene_no", scene_no), "sequence": f"{scene_no}/{scene_total}",
        "minimum_chars": minimum_chars,
        "instruction": "只写这一场景；写到该场景退出状态即收束，不写后续场景，不输出标题。",
    }
    if scene_contract:
        segment["scene_contract"] = scene_contract
    return segment


def _segment_story_state(story_state: dict[str, Any]) -> dict[str, Any]:
    """Hide whole-chapter target facts so a segment cannot restart another scene."""

    result = {
        "reconstruction_mode": story_state.get("reconstruction_mode", ""),
        "chapter_id": story_state.get("chapter_id", ""),
        "known_prior_chapters": story_state.get("known_prior_chapters", []),
    }
    for source_key, target_key in (("known_entities", "known_entities"), ("entities", "entities")):
        if isinstance(story_state.get(source_key), list):
            result[target_key] = story_state[source_key]
            break
    return result


def _draft_chapter(
    folder: Path,
    profile: dict[str, Any],
    selected: list[dict[str, Any]],
    requirements: dict[str, Any],
    execution_spec: dict[str, Any],
    story_state: dict[str, Any],
    plan: dict[str, Any],
    settings: ModelSettings,
    repair_note: str = "",
) -> tuple[str, bool]:
    """Draft directly for short chapters and scene-by-scene for long fidelity targets."""

    minimum_total = _chapter_minimum_chars(plan)
    scenes = [item for item in plan.get("scenes", []) if isinstance(item, dict)]
    segmented = minimum_total >= 3_000 and len(scenes) > 1
    system = "你是中文原创小说写作者。只输出连续正文，不输出标题、解释、Markdown 或 JSON。"
    if not segmented:
        prompt = _draft_prompt(folder, profile, selected, requirements, execution_spec, story_state, plan)
        if repair_note:
            prompt += "\n\n本轮修订要求：\n" + repair_note
        return _complete_draft(system, prompt, settings, minimum_chars=max(120, minimum_total)), False
    per_scene_minimum = max(600, int(minimum_total * 0.88 / len(scenes)))
    segments = []
    for index, scene in enumerate(scenes, start=1):
        contracts = plan.get("scene_contracts", []) if isinstance(plan.get("scene_contracts", []), list) else []
        contract = contracts[index - 1] if index <= len(contracts) and isinstance(contracts[index - 1], dict) else None
        segment = _segment_plan(plan, scene, index, len(scenes), per_scene_minimum, contract)
        prompt = _draft_prompt(folder, profile, selected, requirements, execution_spec, _segment_story_state(story_state), segment)
        prompt += f"\n\n分段生成硬要求：本次只写第 {index}/{len(scenes)} 个场景，正文不得少于 {per_scene_minimum} 个字符；先写完本场景的进入状态、行动、阻力、转折和退出状态，再停止。若 `<ChapterPlan>.scene_contract` 存在，必须逐项兑现它的事件、入口/出口事实，且不得写入该契约之外的其他场景完整事件。不得复述、预写或回顾其他编号场景的任何完整事件；前文只可作为当前 `entry_state` 的既成条件。\n\n输出前自检：严格按照本段 `style_execution_blueprint` 组织篇幅；不要把单句或单轮对话单独成段，优先把动作—感知—判断—结果组成完整句和完整段。每一场景至少补入蓝图要求的感知句与自然转场句，并让对话承担冲突或信息交接。"
        if repair_note:
            prompt += "\n\n本轮修订要求：\n" + repair_note
        segment_maximum = int(segment.get("style_execution_blueprint", {}).get("chapter_char_count", {}).get("maximum", per_scene_minimum * 2))
        segments.append(_complete_draft(
            system, prompt, settings, minimum_chars=per_scene_minimum,
            max_tokens=max(1_200, min(settings.max_tokens, segment_maximum + 200)),
        ))
    return "\n\n".join(segments), True


def _fidelity_plan_error(plan: dict[str, Any], brief: dict[str, Any]) -> str:
    """Require every test-only structural contract to be explicitly planned."""

    if str(brief.get("reconstruction_mode", "")).strip().lower() != "fidelity_test_only":
        return ""
    for source_key, plan_key in (
        ("required_events", "covered_required_event_ids"),
        ("required_state_changes", "covered_required_state_change_ids"),
    ):
        required = {
            str(item.get("contract_id", "")).strip()
            for item in brief.get(source_key, []) if isinstance(item, dict) and str(item.get("contract_id", "")).strip()
        }
        provided = {str(item).strip() for item in plan.get(plan_key, []) if str(item).strip()} if isinstance(plan.get(plan_key, []), list) else set()
        missing = sorted(required - provided)
        if missing:
            return f"{plan_key} 缺少结构契约：{', '.join(missing)}"
    end_contract = plan.get("chapter_end_contract", [])
    if not isinstance(end_contract, list) or not any(str(item).strip() for item in end_contract):
        return "chapter_end_contract 必须明确本章收束状态，防止写入后续事件"
    return ""


def _prefer_style_repair(current: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Never trade a measured style regression for an unverified rewrite."""

    if bool(candidate.get("passed")) and not bool(current.get("passed")):
        return True
    current_high = int(current.get("high_priority_in_range", 0))
    candidate_high = int(candidate.get("high_priority_in_range", 0))
    current_deviation = float(current.get("mean_relative_deviation", 1.0))
    candidate_deviation = float(candidate.get("mean_relative_deviation", 1.0))
    return candidate_high >= current_high and candidate_deviation + 0.02 < current_deviation


def generate_chapter(
    author_id: str,
    story_state_path: Path,
    chapter_brief_path: Path,
    *,
    run_id: str = "generated",
    max_repairs: int = DEFAULT_MAX_REPAIRS,
) -> dict[str, str | int | bool]:
    """Run model-backed plan → draft → validate → repair for original inputs."""

    if max_repairs < 0:
        raise ValueError("max_repairs must be non-negative")
    folder, profile, library, requirements, execution_spec = _skill_files(author_id)
    story_state = _json_object(story_state_path)
    brief = _json_object(chapter_brief_path)
    if not str(brief.get("chapter_goal", "")).strip():
        raise ValueError("chapter brief requires a non-empty chapter_goal")
    settings = ModelSettings.from_environment()
    if not settings.api_key:
        raise RuntimeError("模型密钥不可用：请在 config.yaml 或环境变量中配置后重试")
    destination = _prompt_paths(author_id, run_id)

    plan_prompt = _render(folder / "prompt_templates" / "chapter_plan.md", {
        "AUTHOR_STYLE_PROFILE_JSON": profile,
        "NARRATIVE_TEMPLATE_LIBRARY_JSON": library,
        "CONTINUITY_REQUIREMENTS_JSON": requirements,
        "STYLE_EXECUTION_SPEC_JSON": execution_spec,
        "EXECUTION_MODE_INSTRUCTIONS": _execution_mode_instruction(story_state, brief),
        "ORIGINAL_STORY_STATE_JSON": story_state,
        "CHAPTER_BRIEF_JSON": brief,
    })
    plan_error = ""
    plan: dict[str, Any] | None = None
    selected: list[dict[str, Any]] | None = None
    for _ in range(3):
        prompt = plan_prompt if not plan_error else plan_prompt + f"\n\n上次计划不合格：{plan_error}。请修复后只输出完整 JSON。"
        try:
            candidate = complete_json(
                "你是中文小说章节规划器。严格遵守提示词中的 JSON 契约与原创边界。",
                prompt, settings, attempts=6, max_tokens=min(settings.max_tokens, 9_000),
            )
        except RuntimeError as exc:
            plan_error = f"规划模型调用失败：{exc}"
            continue
        try:
            chosen = _selected_templates(library, candidate)
        except ValueError as exc:
            plan_error = str(exc)
            continue
        plan_error = _fidelity_plan_error(candidate, brief)
        if plan_error:
            continue
        plan = candidate
        selected = chosen
        break
    if plan is None or selected is None:
        raise ValueError(f"chapter planner could not produce a valid plan: {plan_error}")
    plan["style_budget"] = resolve_style_budget(
        execution_spec,
        plan.get("style_budget") if isinstance(plan.get("style_budget"), dict) else None,
        brief.get("style_budget_override") if isinstance(brief.get("style_budget_override"), dict) else None,
    )
    plan["style_execution_directives"] = style_execution_directives(plan["style_budget"], limit=6)
    plan["style_execution_blueprint"] = style_execution_blueprint(plan["style_budget"])
    if "fidelity_test_only" in {
        str(payload.get("reconstruction_mode", "")).strip().lower()
        for payload in (story_state, brief)
    }:
        plan["reconstruction_mode"] = "fidelity_test_only"
        if isinstance(brief.get("required_scenes"), list):
            plan["scene_contracts"] = brief["required_scenes"]
    write_json(destination / "generated_plan.json", plan)
    write_json(destination / "style_budget.json", plan["style_budget"])
    write_json(destination / "generation_status.json", {
        "stage": "drafting", "status": "in_progress", "selected_template_ids": [item["template_id"] for item in selected],
    })

    draft, segmented_generation = _draft_chapter(
        folder, profile, selected, requirements, execution_spec, story_state, plan, settings,
    )

    validation: dict[str, Any] = {}
    style_assessment: dict[str, Any] = {}
    repairs_used = 0
    for repair_index in range(max_repairs + 1):
        style_assessment = assess_style_budget(draft, plan["style_budget"])
        plan["style_execution_directives"] = style_execution_directives(plan["style_budget"], style_assessment, limit=6)
        plan["style_execution_blueprint"] = style_execution_blueprint(plan["style_budget"])
        write_json(destination / "generated_plan.json", plan)
        (destination / "generated_draft.txt").write_text(draft, encoding="utf-8")
        write_json(destination / "style_assessment.json", style_assessment)
        validation_prompt = _validation_prompt(
            folder, profile, requirements, execution_spec, story_state, plan, draft,
        )
        validation = complete_json(
            "你是中文小说章节校验器。只输出符合契约的合法 JSON object。",
            validation_prompt, settings, attempts=6, max_tokens=min(settings.max_tokens, 5_000),
        )
        write_json(destination / "validation.json", validation)
        continuity_passed = bool(validation.get("passed"))
        if continuity_passed and style_assessment["passed"]:
            break
        if repair_index >= max_repairs:
            break
        repair_context = style_repair_prompt(draft, plan, plan["style_budget"], style_assessment)
        repair_context += "\n\n连续性校验 JSON：\n" + json.dumps(validation, ensure_ascii=False)
        try:
            repaired = _complete_draft(
                "你是中文小说修订器。保持原计划的事件与状态，只输出修订后的完整正文。",
                repair_context, settings, minimum_chars=max(120, _chapter_minimum_chars(plan)),
            )
        except (RuntimeError, ValueError):
            repaired = draft
        repaired_assessment = assess_style_budget(repaired, plan["style_budget"])
        if _prefer_style_repair(style_assessment, repaired_assessment):
            draft = repaired
        repairs_used += 1

    (destination / "generated_draft.txt").write_text(draft, encoding="utf-8")
    write_json(destination / "style_assessment.json", style_assessment)
    write_json(destination / "validation.json", validation)
    write_json(destination / "generation_manifest.json", {
        "author_id": author_id,
        "run_id": run_id,
        "story_state_path": str(story_state_path),
        "chapter_brief_path": str(chapter_brief_path),
        "selected_template_ids": [item["template_id"] for item in selected],
        "repairs_used": repairs_used,
        "segmented_generation": segmented_generation,
        "continuity_passed": bool(validation.get("passed")),
        "style_passed": bool(style_assessment.get("passed")),
    })
    write_json(destination / "generation_status.json", {
        "stage": "complete", "status": "complete", "repairs_used": repairs_used,
        "continuity_passed": bool(validation.get("passed")), "style_passed": bool(style_assessment.get("passed")),
    })
    return {
        "output_dir": str(destination),
        "repairs_used": repairs_used,
        "continuity_passed": bool(validation.get("passed")),
        "style_passed": bool(style_assessment.get("passed")),
    }
