from __future__ import annotations

import copy
from dataclasses import is_dataclass
import json
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import config
from pipeline.core.state_validator import (
    build_forbidden_stage_terms,
    stage_jump_threshold,
    validate_enemy_power,
    validate_identity_transition,
    validate_location_transition,
    validate_reassembled_events,
    validate_skeleton_sequence,
    validate_stage_transition,
    validate_volume_outline,
)
from pipeline.core.story_state import (
    ProgressionStage,
    StateIssue,
    build_progression_stages,
    normalize_stage_name,
    stage_value,
    extract_max_stage_from_text,
    infer_stage_from_node,
    get_obj_field,
)
from pipeline.core.utils import chat_completion_json, get_deepseek_client


_DIRECT_WIN_KEYWORDS: Sequence[str] = (
    "击杀",
    "反杀",
    "斩杀",
    "正面击败",
    "碾压",
    "单独击败",
)

_LIMITED_OUTCOME_REWRITE = "主角没有正面取胜，而是借地形、禁制或外力周旋脱身，为后续反制留下空间。"
_RESIDUAL_REWRITE = "主角避开正面击杀更高阶段的威胁，只在外围获得线索、拖延时间或借势脱身。"

_BRIDGE_TRAVEL_PHRASES: Sequence[str] = (
    "接到任务后",
    "在调查推进中",
    "借正式调令",
    "在护送或追查过程中",
    "通过临时入口或队伍调度",
)

_SUMMARY_FIELDS: Sequence[str] = (
    "adapted_summary",
    "original_summary",
    "summary",
    "raw_text",
)

_TEXT_REWRITE_FIELDS: Sequence[str] = (
    "adapted_summary",
    "original_summary",
    "summary",
    "raw_text",
    "ending_state",
    "conflict_hint",
    "function_hint",
    "logic_card",
    "event_plan",
    "state_updates",
)


def classify_issue_action(issue: StateIssue) -> str:
    issue_type = str(issue.issue_type or "").strip()
    if issue_type in {"unknown_stage", "stage_field_mismatch"}:
        return "sync_field"
    if issue_type in {"power_gap_win", "power_gap_pressure"}:
        return "downgrade_opponent"
    if issue_type in {"location_jump", "identity_jump"}:
        return "insert_bridge"
    if issue_type in {"stage_regression"}:
        return "replace_node"
    if issue_type in {"stage_jump", "volume_stage_overflow", "ending_stage_overflow", "forbidden_stage_term", "volume_raw_stage_overflow"}:
        return "rewrite_summary"
    return "manual_review"


def get_allowed_opponent_stage(
    protagonist_stage: str,
    stages: List[ProgressionStage],
    max_gap: int = 1,
) -> str:
    canonical = normalize_stage_name(protagonist_stage, stages)
    if not canonical:
        return ""
    ordered = _sorted_stages(stages)
    position = _stage_position(canonical, ordered)
    if position < 0:
        return ""
    target = min(len(ordered) - 1, position + max(0, int(max_gap)))
    return ordered[target].name


def rewrite_overpowered_win_text_by_rule(
    text: str,
    protagonist_stage: str,
    stages: List[ProgressionStage],
) -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return raw_text
    if not _has_any_keyword(raw_text, _DIRECT_WIN_KEYWORDS):
        return raw_text

    protagonist_canonical = normalize_stage_name(protagonist_stage, stages)
    external_max_stage = extract_max_stage_from_text(raw_text, stages)
    if not protagonist_canonical or not external_max_stage:
        return raw_text

    protagonist_position = _stage_position(protagonist_canonical, stages)
    external_position = _stage_position(external_max_stage, stages)
    if protagonist_position < 0 or external_position < 0:
        return raw_text
    if external_position - protagonist_position <= stage_jump_threshold(stages):
        return raw_text

    if "残魂" in raw_text or "残念" in raw_text:
        return f"主角没有正面击溃{external_max_stage}层级的残念威压，而是避开锁定，从外围获取线索并保留后续隐患。"
    if _has_any_keyword(raw_text, ("追兵", "追杀", "围杀", "高阶", "威胁")):
        return f"面对{external_max_stage}层级的追击或压迫，主角没有正面取胜，而是借地形、禁制或外力周旋脱身。"
    return _replace_direct_win_phrases(raw_text, external_max_stage) or _RESIDUAL_REWRITE


def downgrade_overpowered_enemy_text(
    text: str,
    protagonist_stage: str,
    stages: List[ProgressionStage],
) -> str:
    return rewrite_overpowered_win_text_by_rule(text, protagonist_stage, stages)


def build_rewrite_prompt(
    node_or_event: Any,
    issues: List[StateIssue],
    allowed_state: dict,
    stages: List[ProgressionStage],
) -> str:
    record = _to_record(node_or_event)
    stage_lines = [f"{index + 1}. {stage.name} (stage_index={stage.stage_index})" for index, stage in enumerate(_sorted_stages(stages))]
    issue_lines = [f"- [{issue.severity}] {issue.issue_type}: {issue.message}" for issue in issues]
    forbidden_terms = allowed_state.get("forbidden_terms", [])
    max_stage = allowed_state.get("max_stage", "")
    allowed_opponent_stage = allowed_state.get("allowed_opponent_stage", "")
    summary_text = _get_primary_summary(record)
    schema = {
        "rewritten_summary": "重写后的摘要",
        "state_updates": {
            "power_state": "修复后的主角阶段，不得超过 max_stage",
            "identity_state": "如有需要可补充",
            "location": "如有需要可补充",
        },
        "repair_notes": ["说明采用了什么桥接或降级策略"],
    }
    prompt = (
        "你是 PlotWeaver 的状态连续性修复器。\n"
        "请在保留原事件功能的前提下，修复成长阶段、身份、地点和越级取胜问题。\n"
        "必须遵守：\n"
        f"1. 主角成长阶段不得超过 {max_stage or '当前允许上限'}。\n"
        f"2. 禁止使用这些阶段词：{', '.join(forbidden_terms) or '无'}。\n"
        f"3. 如果需要强敌，最多只能写到 {allowed_opponent_stage or '允许的外部威胁上限'}，并且只能作为压迫、追杀、远景威胁或间接影响。\n"
        "4. 不允许主角正面击杀、碾压或单独击败远高于自身阶段的敌人。\n"
        "5. 保留原事件的叙事功能、冲突功能和钩子功能。\n\n"
        f"阶段体系：\n{chr(10).join(stage_lines)}\n\n"
        f"当前摘要：\n{summary_text}\n\n"
        f"其他上下文：\n{json.dumps(record, ensure_ascii=False, indent=2)}\n\n"
        f"需要修复的问题：\n{chr(10).join(issue_lines) or '- 无'}\n\n"
        f"允许状态：\n{json.dumps(allowed_state, ensure_ascii=False, indent=2)}\n\n"
        f"输出 JSON Schema：\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
    return prompt


def repair_event_with_ai(
    node_or_event: Any,
    issues: List[StateIssue],
    allowed_state: dict,
    stages: List[ProgressionStage],
) -> dict:
    record = _to_record(node_or_event)
    _ensure_repair_notes(record)
    prompt = build_rewrite_prompt(record, issues, allowed_state, stages)
    try:
        client = get_deepseek_client()
        raw = chat_completion_json(
            client,
            system="你是严格遵守成长阶段约束的剧情修复器，只能输出合法 JSON。",
            user=prompt,
            json_mode=True,
            model_name=config.DEEPSEEK_LAST_STEPS_MODEL,
            temperature=0.2,
        )
        data = json.loads(raw)
        rewritten_summary = str(data.get("rewritten_summary", "") or "").strip()
        if rewritten_summary:
            _set_primary_summary(record, rewritten_summary)
        ai_state_updates = data.get("state_updates", {})
        if isinstance(ai_state_updates, dict):
            merged_updates = _merge_dicts(_coerce_dict(get_obj_field(record, "state_updates", {})), ai_state_updates)
            record["state_updates"] = merged_updates
            if "power_state" in ai_state_updates:
                _set_stage_fields(record, str(ai_state_updates.get("power_state", "")).strip(), stages)
            if "identity_state" in ai_state_updates and ai_state_updates.get("identity_state"):
                record["identity_state"] = ai_state_updates["identity_state"]
            if "location" in ai_state_updates and ai_state_updates.get("location"):
                record["location"] = ai_state_updates["location"]
        repair_notes = _flatten_texts(data.get("repair_notes", []))
        if repair_notes:
            record["repair_notes"].extend(repair_notes)
        record["repair_notes"].append("ai_rewrite_applied")
        return record
    except Exception as exc:
        record["repair_notes"].append(f"repair_failed: {exc}")
        return record


def make_bridge_node(prev_node: Any, next_node: Any, allowed_state: dict, stages: List[ProgressionStage]) -> dict:
    prev_record = _to_record(prev_node)
    next_record = _to_record(next_node)
    prev_id = _node_id(prev_record)
    next_id = _node_id(next_record)
    prev_stage = infer_stage_from_node(prev_record, stages)
    next_stage = infer_stage_from_node(next_record, stages)
    bridge_after = next_stage
    if prev_stage and next_stage:
        prev_pos = _stage_position(prev_stage, stages)
        next_pos = _stage_position(next_stage, stages)
        threshold = max(1, stage_jump_threshold(stages) - 1)
        if prev_pos >= 0 and next_pos > prev_pos + threshold:
            bridge_after = _stage_at_position(stages, prev_pos + threshold)
    if not bridge_after:
        bridge_after = allowed_state.get("max_stage", "") or prev_stage

    prev_identity = _infer_identity(prev_record)
    next_identity = _infer_identity(next_record)
    prev_location = _infer_location(prev_record)
    next_location = _infer_location(next_record)
    travel_phrase = _BRIDGE_TRAVEL_PHRASES[(_stable_hash(prev_id + next_id) % len(_BRIDGE_TRAVEL_PHRASES))]
    summary_parts = [
        travel_phrase,
        f"主角从{prev_location or '原有据点'}转向{next_location or '新场域'}" if prev_location or next_location else "主角完成场景切换",
        f"身份上从{prev_identity or '既有立场'}过渡到{next_identity or '新的任务身份'}" if prev_identity or next_identity else "并为下一事件建立新的立场接口",
        "过程中没有发生远高阶段的正面取胜，而是通过任务、引导或队伍调度完成衔接。",
    ]
    summary = "，".join(part for part in summary_parts if part)
    bridge_id = f"bridge_{prev_id}_{next_id}"
    return {
        "node_id": bridge_id,
        "event_id": bridge_id,
        "arc_name": _first_non_empty(get_obj_field(prev_record, "arc_name"), get_obj_field(next_record, "arc_name")),
        "pacing_role": "bridge_transition",
        "summary": summary,
        "original_summary": summary,
        "adapted_summary": summary,
        "is_bridge": True,
        "stage_before": prev_stage,
        "stage_after": bridge_after,
        "power_state": bridge_after,
        "identity_state": next_identity or prev_identity,
        "location": next_location or prev_location,
        "repair_notes": [f"bridge_inserted_between:{prev_id}->{next_id}"],
    }


def repair_skeleton_sequence(
    nodes: list,
    world: dict | object | None = None,
    max_rounds: int = 2,
) -> tuple[list, list[StateIssue]]:
    stages = build_progression_stages(world)
    working = [_prepare_record(item) for item in nodes]
    remaining_issues: List[StateIssue] = []

    for _round in range(max(1, int(max_rounds))):
        issues = validate_skeleton_sequence(working, world)
        if not issues:
            return working, []
        grouped = _group_issues_by_node(issues)
        repaired: List[dict] = []
        prev_record: dict | None = None

        for record in working:
            node_id = _node_id(record)
            node_issues = grouped.get(node_id, [])
            if prev_record and any(issue.issue_type in {"location_jump", "identity_jump"} for issue in node_issues):
                bridge_allowed_state = _build_allowed_state(prev_record, record, stages)
                repaired.append(make_bridge_node(prev_record, record, bridge_allowed_state, stages))
            allowed_state = _build_allowed_state(prev_record, record, stages)
            repaired_record = _repair_record(record, node_issues, allowed_state, stages)
            repaired.append(repaired_record)
            prev_record = repaired_record

        working = repaired
        remaining_issues = validate_skeleton_sequence(working, world)
        if not remaining_issues:
            break

    return working, remaining_issues


def repair_reassembled_events(
    events: list,
    world: dict | object | None = None,
    allowed_state: dict | None = None,
    max_rounds: int = 2,
) -> tuple[list, list[StateIssue]]:
    stages = build_progression_stages(world)
    working = [_prepare_record(item) for item in events]
    remaining_issues: List[StateIssue] = []

    for _round in range(max(1, int(max_rounds))):
        issues = list(validate_reassembled_events(working, world))
        issues.extend(_collect_identity_location_issues(working))
        if not issues:
            return working, []

        grouped = _group_issues_by_node(issues)
        repaired: List[dict] = []
        prev_record: dict | None = None
        for record in working:
            node_id = _node_id(record)
            node_issues = grouped.get(node_id, [])
            merged_allowed_state = _merge_dicts(_build_allowed_state(prev_record, record, stages), allowed_state or {})
            if prev_record and any(issue.issue_type in {"location_jump", "identity_jump"} for issue in node_issues):
                repaired.append(make_bridge_node(prev_record, record, merged_allowed_state, stages))
            repaired_record = _repair_record(record, node_issues, merged_allowed_state, stages)
            repaired.append(repaired_record)
            prev_record = repaired_record

        working = repaired
        remaining_issues = list(validate_reassembled_events(working, world))
        remaining_issues.extend(_collect_identity_location_issues(working))
        if not remaining_issues:
            break

    return working, remaining_issues


def repair_volume_outline(
    volume: dict,
    world: dict | object | None = None,
    allowed_state: dict | None = None,
    max_rounds: int = 2,
) -> tuple[dict, list[StateIssue]]:
    stages = build_progression_stages(world)
    working = _prepare_record(volume)
    remaining_issues: List[StateIssue] = []

    for _round in range(max(1, int(max_rounds))):
        issues = validate_volume_outline(working, world)
        if not issues:
            working["state_validation_issues"] = []
            return working, []

        base_allowed_state = _infer_volume_allowed_state(working, stages, allowed_state or {})
        chapter_summaries = _normalize_chapter_summaries(get_obj_field(working, "chapter_summaries", []))
        for issue in issues:
            if issue.issue_type in {"power_gap_win", "power_gap_pressure", "volume_stage_overflow", "ending_stage_overflow", "forbidden_stage_term"}:
                chapter_index = _extract_chapter_index(issue.node_id)
                if chapter_index is not None and 0 <= chapter_index < len(chapter_summaries):
                    summary = chapter_summaries[chapter_index]
                    rewritten = rewrite_overpowered_win_text_by_rule(summary, base_allowed_state.get("max_stage", ""), stages)
                    if rewritten == summary:
                        pseudo = {
                            "summary": summary,
                            "event_id": issue.node_id,
                            "state_updates": {},
                        }
                        ai_result = repair_event_with_ai(pseudo, [issue], base_allowed_state, stages)
                        rewritten = _get_primary_summary(ai_result) or summary
                    chapter_summaries[chapter_index] = rewritten
                    _append_repair_note(working, f"chapter_{chapter_index + 1}_repaired_for:{issue.issue_type}")
                elif issue.issue_type in {"ending_stage_overflow", "forbidden_stage_term"} and get_obj_field(working, "ending_state"):
                    ending_state = str(get_obj_field(working, "ending_state", "") or "")
                    rewritten = rewrite_overpowered_win_text_by_rule(ending_state, base_allowed_state.get("max_stage", ""), stages)
                    if rewritten == ending_state:
                        pseudo = {"summary": ending_state, "event_id": issue.node_id, "state_updates": {}}
                        ai_result = repair_event_with_ai(pseudo, [issue], base_allowed_state, stages)
                        rewritten = _get_primary_summary(ai_result) or ending_state
                    working["ending_state"] = rewritten
                    _append_repair_note(working, f"ending_state_repaired_for:{issue.issue_type}")
            if issue.issue_type == "volume_raw_stage_overflow" and get_obj_field(working, "raw_text"):
                raw_text = str(get_obj_field(working, "raw_text", "") or "")
                rewritten = rewrite_overpowered_win_text_by_rule(raw_text, base_allowed_state.get("max_stage", ""), stages)
                if rewritten != raw_text:
                    working["raw_text"] = rewritten
                    _append_repair_note(working, "raw_text_repaired_by_rule")

        working["chapter_summaries"] = chapter_summaries
        remaining_issues = validate_volume_outline(working, world)
        if not remaining_issues:
            break

    working["state_validation_issues"] = [_issue_to_dict(issue) for issue in remaining_issues]
    return working, remaining_issues


def _repair_record(
    record: dict,
    issues: List[StateIssue],
    allowed_state: dict,
    stages: List[ProgressionStage],
) -> dict:
    repaired = _prepare_record(record)
    if not issues:
        return repaired

    current_stage = infer_stage_from_node(repaired, stages) or allowed_state.get("max_stage", "")
    for issue in issues:
        action = classify_issue_action(issue)
        if action == "replace_node" and issue.issue_type == "stage_regression":
            target_stage = allowed_state.get("prev_stage", "") or allowed_state.get("max_stage", "") or current_stage
            if target_stage:
                _set_stage_fields(repaired, target_stage, stages)
                _cap_record_stage_mentions(repaired, target_stage, stages)
                _append_repair_note(repaired, f"stage_regression_replaced_with:{target_stage}")
                current_stage = target_stage
        elif action == "rewrite_summary" and issue.issue_type == "stage_jump":
            stage_ceiling = allowed_state.get("stage_ceiling", "") or allowed_state.get("max_stage", "")
            if stage_ceiling:
                _set_stage_fields(repaired, stage_ceiling, stages)
                _cap_record_stage_mentions(repaired, stage_ceiling, stages)
                _append_repair_note(repaired, f"stage_jump_capped_to:{stage_ceiling}")
                current_stage = stage_ceiling
        elif action == "downgrade_opponent":
            summary = _get_primary_summary(repaired)
            rewritten = rewrite_overpowered_win_text_by_rule(summary, current_stage, stages)
            if rewritten != summary:
                _set_primary_summary(repaired, rewritten)
                _append_repair_note(repaired, f"rule_rewrite_for:{issue.issue_type}")
            else:
                repaired = repair_event_with_ai(repaired, [issue], allowed_state, stages)
        elif action == "sync_field":
            target_stage = current_stage or allowed_state.get("max_stage", "")
            if target_stage:
                _set_stage_fields(repaired, target_stage, stages)
                _append_repair_note(repaired, f"synced_stage_fields:{target_stage}")
        elif action == "rewrite_summary":
            repaired = repair_event_with_ai(repaired, [issue], allowed_state, stages)
        else:
            _append_repair_note(repaired, f"manual_review_recommended:{issue.issue_type}")
    return repaired


def _collect_identity_location_issues(records: List[dict]) -> List[StateIssue]:
    issues: List[StateIssue] = []
    prev_record: dict | None = None
    for record in records:
        if prev_record is None:
            prev_record = record
            continue
        summary = _join_texts([_get_primary_summary(record), get_obj_field(record, "event_plan"), get_obj_field(record, "logic_notes")])
        issues.extend(
            validate_identity_transition(
                _infer_identity(prev_record),
                _infer_identity(record),
                summary,
                node_id=_node_id(record),
            )
        )
        issues.extend(
            validate_location_transition(
                _infer_location(prev_record),
                _infer_location(record),
                summary,
                node_id=_node_id(record),
            )
        )
        prev_record = record
    return issues


def _build_allowed_state(prev_record: dict | None, current_record: dict, stages: List[ProgressionStage]) -> dict:
    prev_stage = infer_stage_from_node(prev_record, stages) if prev_record else ""
    current_stage = infer_stage_from_node(current_record, stages)
    ordered = _sorted_stages(stages)
    stage_ceiling = current_stage or prev_stage
    if prev_stage:
        prev_pos = _stage_position(prev_stage, ordered)
        if prev_pos >= 0:
            ceiling_pos = min(len(ordered) - 1, prev_pos + stage_jump_threshold(stages))
            stage_ceiling = ordered[ceiling_pos].name
    max_stage = stage_ceiling or current_stage or prev_stage or (ordered[0].name if ordered else "")
    return {
        "prev_stage": prev_stage,
        "current_stage": current_stage,
        "max_stage": max_stage,
        "stage_ceiling": stage_ceiling or max_stage,
        "allowed_opponent_stage": get_allowed_opponent_stage(max_stage, stages, max_gap=1),
        "forbidden_terms": build_forbidden_stage_terms(max_stage, stages, margin=1),
    }


def _infer_volume_allowed_state(volume: dict, stages: List[ProgressionStage], override: dict) -> dict:
    working = dict(override)
    max_stage = str(working.get("max_stage", "") or "")
    if not max_stage:
        for field_name in ("realm_range", "stage_range", "power_range"):
            value = get_obj_field(volume, field_name)
            extracted = _extract_max_stage_from_any(value, stages)
            if extracted:
                max_stage = extracted
                break
    if not max_stage:
        max_stage = _extract_max_stage_from_any(get_obj_field(volume, "ending_state"), stages)
    if not max_stage and stages:
        max_stage = _sorted_stages(stages)[0].name
    working["max_stage"] = max_stage
    working.setdefault("allowed_opponent_stage", get_allowed_opponent_stage(max_stage, stages, max_gap=1))
    working.setdefault("forbidden_terms", build_forbidden_stage_terms(max_stage, stages, margin=1))
    return working


def _prepare_record(item: Any) -> dict:
    record = _to_record(item)
    _ensure_repair_notes(record)
    return record


def _to_record(item: Any) -> dict:
    if item is None:
        return {}
    if isinstance(item, dict):
        return copy.deepcopy(item)
    if is_dataclass(item) or hasattr(item, "__dict__"):
        return copy.deepcopy(vars(item))
    return {"value": copy.deepcopy(item)}


def _group_issues_by_node(issues: List[StateIssue]) -> Dict[str, List[StateIssue]]:
    grouped: Dict[str, List[StateIssue]] = {}
    for issue in issues:
        grouped.setdefault(str(issue.node_id or ""), []).append(issue)
    return grouped


def _ensure_repair_notes(record: dict) -> None:
    notes = record.get("repair_notes")
    if not isinstance(notes, list):
        record["repair_notes"] = []


def _append_repair_note(record: dict, note: str) -> None:
    _ensure_repair_notes(record)
    clean = str(note or "").strip()
    if clean and clean not in record["repair_notes"]:
        record["repair_notes"].append(clean)


def _set_stage_fields(record: dict, stage_name: str, stages: List[ProgressionStage]) -> None:
    if not stage_name:
        return
    canonical = normalize_stage_name(stage_name, stages) or stage_name
    index = stage_value(canonical, stages)
    for field_name in ("stage", "power_stage", "power_state", "realm"):
        if field_name in record or field_name == "power_state":
            record[field_name] = canonical
    for field_name in ("realm_level", "level", "rank"):
        if field_name in record and index >= 0:
            record[field_name] = index
    logic_card = _coerce_dict(get_obj_field(record, "logic_card", {}))
    if logic_card:
        logic_card["power_state"] = canonical
        record["logic_card"] = logic_card
    state_updates = _coerce_dict(get_obj_field(record, "state_updates", {}))
    if state_updates:
        state_updates["power_state"] = canonical
        record["state_updates"] = state_updates


def _cap_record_stage_mentions(record: dict, max_stage: str, stages: List[ProgressionStage]) -> None:
    max_canonical = normalize_stage_name(max_stage, stages)
    if not max_canonical:
        return
    max_position = _stage_position(max_canonical, stages)
    for field_name in _TEXT_REWRITE_FIELDS:
        if field_name not in record:
            continue
        record[field_name] = _cap_value_stage_mentions(record[field_name], max_canonical, max_position, stages)


def _cap_value_stage_mentions(value: Any, max_stage: str, max_position: int, stages: List[ProgressionStage]) -> Any:
    if isinstance(value, str):
        text = value
        for stage in _sorted_stages(stages):
            if _stage_position(stage.name, stages) > max_position:
                text = _replace_stage_aliases(text, stage, max_stage)
        return text
    if isinstance(value, dict):
        return {key: _cap_value_stage_mentions(item, max_stage, max_position, stages) for key, item in value.items()}
    if isinstance(value, list):
        return [_cap_value_stage_mentions(item, max_stage, max_position, stages) for item in value]
    return value


def _replace_stage_aliases(text: str, source_stage: ProgressionStage, target_stage: str) -> str:
    result = str(text or "")
    aliases = sorted({source_stage.name, *source_stage.aliases}, key=len, reverse=True)
    for alias in aliases:
        clean_alias = str(alias or "").strip()
        if not clean_alias:
            continue
        result = result.replace(clean_alias, target_stage)
    return result


def _get_primary_summary(record: dict) -> str:
    for field_name in _SUMMARY_FIELDS:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _set_primary_summary(record: dict, summary: str) -> None:
    clean = str(summary or "").strip()
    if not clean:
        return
    for field_name in _SUMMARY_FIELDS:
        if field_name in record:
            record[field_name] = clean
            return
    record["summary"] = clean


def _merge_dicts(base: dict, updates: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _coerce_dict(value: Any) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _issue_to_dict(issue: StateIssue) -> dict:
    return {
        "severity": issue.severity,
        "issue_type": issue.issue_type,
        "message": issue.message,
        "node_id": issue.node_id,
        "suggested_action": issue.suggested_action,
    }


def _extract_chapter_index(node_id: str) -> int | None:
    match = re.search(r"_chapter_(\d+)$", str(node_id or ""))
    if not match:
        return None
    return int(match.group(1)) - 1


def _normalize_chapter_summaries(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item or "") for item in value]
    if isinstance(value, dict):
        return [str(item or "") for item in value.values()]
    return [str(value or "")]


def _extract_max_stage_from_any(value: Any, stages: List[ProgressionStage]) -> str:
    text = _join_texts([value])
    return extract_max_stage_from_text(text, stages)


def _replace_direct_win_phrases(text: str, external_max_stage: str) -> str:
    result = str(text or "")
    replaced = False
    for keyword in _DIRECT_WIN_KEYWORDS:
        if keyword in result:
            result = result.replace(keyword, "借外力周旋摆脱")
            replaced = True
    if not replaced:
        return ""
    if external_max_stage and external_max_stage in result:
        return f"面对{external_max_stage}层级的强敌，主角没有正面取胜，而是借外力周旋摆脱。"
    return f"{result}，但最终结果改为脱身或取得外围线索，而非正面越级取胜。"


def _infer_identity(record: dict) -> str:
    return _first_non_empty(
        record.get("identity_state"),
        get_obj_field(record, "logic_card.identity_state"),
        get_obj_field(record, "event_plan.identity_state"),
        record.get("protagonist_identity"),
    )


def _infer_location(record: dict) -> str:
    return _first_non_empty(
        record.get("location"),
        record.get("raw_location"),
        get_obj_field(record, "logic_card.location"),
        get_obj_field(record, "event_plan.location"),
        record.get("scene"),
    )


def _node_id(record: dict) -> str:
    return _first_non_empty(record.get("node_id"), record.get("event_id"), record.get("id"), "unknown_node")


def _join_texts(values: Iterable[Any]) -> str:
    parts: List[str] = []
    for value in values:
        parts.extend(_flatten_texts(value))
    return " ".join(_dedupe_texts(parts))


def _flatten_texts(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        clean = value.strip()
        return [clean] if clean else []
    if isinstance(value, dict):
        parts: List[str] = []
        for item in value.values():
            parts.extend(_flatten_texts(item))
        return parts
    if isinstance(value, (list, tuple, set)):
        parts: List[str] = []
        for item in value:
            parts.extend(_flatten_texts(item))
        return parts
    clean = str(value).strip()
    return [clean] if clean else []


def _dedupe_texts(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        clean = str(value or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _has_any_keyword(text: str, keywords: Sequence[str]) -> bool:
    haystack = str(text or "")
    return any(keyword in haystack for keyword in keywords)


def _sorted_stages(stages: List[ProgressionStage]) -> List[ProgressionStage]:
    return sorted(stages, key=lambda item: (item.stage_index, item.name))


def _stage_position(stage_name: str, stages: List[ProgressionStage]) -> int:
    canonical = normalize_stage_name(stage_name, stages)
    if not canonical:
        return -1
    for index, stage in enumerate(_sorted_stages(stages)):
        if stage.name == canonical:
            return index
    return -1


def _stage_at_position(stages: List[ProgressionStage], position: int) -> str:
    ordered = _sorted_stages(stages)
    if not ordered:
        return ""
    index = max(0, min(len(ordered) - 1, int(position)))
    return ordered[index].name


def _stable_hash(text: str) -> int:
    value = 0
    for ch in str(text or ""):
        value = (value * 131 + ord(ch)) % 1_000_000_007
    return value


__all__ = [
    "classify_issue_action",
    "get_allowed_opponent_stage",
    "rewrite_overpowered_win_text_by_rule",
    "downgrade_overpowered_enemy_text",
    "build_rewrite_prompt",
    "repair_event_with_ai",
    "make_bridge_node",
    "repair_skeleton_sequence",
    "repair_reassembled_events",
    "repair_volume_outline",
]
