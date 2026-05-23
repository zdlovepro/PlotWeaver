from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

import config
from pipeline.core.common_json import read_json_file, safe_json_load, write_json_file
from pipeline.core.state_repair import (
    downgrade_overpowered_enemy_text,
    get_allowed_opponent_stage,
    repair_event_with_ai,
    repair_volume_outline,
)
from pipeline.core.state_validator import build_forbidden_stage_terms, validate_volume_outline
from pipeline.core.story_state import ProgressionStage, StateIssue, build_progression_stages, extract_max_stage_from_text, infer_stage_from_node, normalize_stage_name, stage_value
from pipeline.core.story_models import CharacterSheet
from pipeline.core.utils import chat_completion_json, get_deepseek_client
from pipeline.core.world_building_core import FusedWorld
from pipeline.step11_reassembly import ReassembledEvent


_STEP12_FILENAME = "step12_volume_outlines.json"
_GENERIC_VOLUME_MOTIFS = ("设局", "识破", "逆袭", "试炼", "拍卖会", "秘境", "暗中相助", "宝珠")


def _quality_chat_completion_json(client, **kwargs) -> str:
    return chat_completion_json(client, model_name=config.DEEPSEEK_LAST_STEPS_MODEL, **kwargs)


@dataclass
class VolumeOutline:
    volume_number: int
    volume_title: str
    realm_range: str
    chapter_summaries: List[str] = field(default_factory=list)
    tension_curve: str = ""
    ending_state: str = ""
    raw_text: str = ""
    volume_state_before: Dict[str, Any] = field(default_factory=dict)
    volume_state_after: Dict[str, Any] = field(default_factory=dict)
    max_realm_used: str = ""
    forbidden_terms_used: List[str] = field(default_factory=list)
    validation_notes: List[str] = field(default_factory=list)
    state_validation_issues: List[Dict[str, Any]] = field(default_factory=list)
    repair_notes: List[str] = field(default_factory=list)
    state_validation_failed: bool = False
    fatal_issues: List[Dict[str, Any]] = field(default_factory=list)


def generate_volumes(reassembled_events: List[ReassembledEvent], fused_world: FusedWorld, character_sheet: CharacterSheet) -> List[VolumeOutline]:
    client = get_deepseek_client()
    volume_groups = _group_events_into_volumes(reassembled_events)
    total_volumes = len(volume_groups)
    print(f"[Step 12] Generating {total_volumes} volumes. Expanding events into chapters...")

    volumes: List[VolumeOutline] = []
    previous_tail_chapters: List[str] = []
    previous_ending_state = ""

    for volume_number, (arc_name, events) in enumerate(tqdm(volume_groups.items(), desc="[Step 12] Generating volumes", unit="vol"), start=1):
        tqdm.write(f"[Step 12] Generating volume {volume_number}: {arc_name} (events: {len(events)})...")
        stage_network = _get_stage_network_for_volume(character_sheet, volume_number, total_volumes)
        system_context = _build_system_context(fused_world, character_sheet, stage_network)
        allowed_state = _build_volume_allowed_state("", events, previous_ending_state, fused_world)
        outline = _generate_single_volume(
            client=client,
            volume_number=volume_number,
            arc_name=arc_name,
            events=events,
            system_context=system_context,
            previous_ending_state=previous_ending_state,
            previous_tail_chapters=previous_tail_chapters,
            allowed_state=allowed_state,
        )
        outline = _postprocess_generated_volume_outline(outline, events, fused_world, allowed_state)
        volumes.append(outline)
        previous_ending_state = outline.ending_state
        previous_tail_chapters = outline.chapter_summaries[-5:] if len(outline.chapter_summaries) >= 5 else outline.chapter_summaries

    tqdm.write(f"[Step 12] All {len(volumes)} volumes generated.")
    save_step12_output(volumes)
    return volumes


def save_step12_output(volumes: List[VolumeOutline]) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _STEP12_FILENAME
    write_json_file(path, [asdict(volume) for volume in volumes])
    print(f"[Step 12] Intermediate output saved -> {path.name}")
    return path


def load_step12_output(intermediate_dir: str | Path | None = None) -> List[VolumeOutline]:
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)
    path = inter_dir / _STEP12_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Step 12 file not found: {path}")
    data = read_json_file(path)
    return [VolumeOutline(**item) for item in data]


def _group_events_into_volumes(events: List[ReassembledEvent]) -> Dict[str, List[ReassembledEvent]]:
    groups: Dict[str, List[ReassembledEvent]] = {}
    for event in events:
        groups.setdefault(event.arc_name, []).append(event)
    ordered: Dict[str, List[ReassembledEvent]] = {}
    for arc_name in sorted(groups.keys(), key=_arc_sort_key):
        ordered[arc_name] = sorted(groups[arc_name], key=_event_sort_key)
    return ordered


def _get_stage_network_for_volume(char_sheet: CharacterSheet, volume_number: int, total_volumes: int) -> str:
    if not char_sheet.relationship_networks:
        return "暂无关系网络。"
    ratio = (volume_number - 1) / total_volumes if total_volumes > 1 else 0
    stage_index = int(ratio * len(char_sheet.relationship_networks))
    active = char_sheet.relationship_networks[min(stage_index, len(char_sheet.relationship_networks) - 1)]
    return f"【本卷阶段】{active.stage}\n活跃角色：{', '.join(active.active_characters)}\n关系局势：{active.relationship_status}"


def _build_system_context(fused_world: FusedWorld, character_sheet: CharacterSheet, current_stage_network: str) -> str:
    protagonist = character_sheet.protagonist
    realms_text = "\n".join(
        f"{realm.name}(第{realm.level}境): {realm.breakthrough_condition}"
        for realm in fused_world.cultivation_realms
    )
    supporting_text = "\n".join(
        f"- [{item.role}] {item.name} | 执念: {item.dao_heart} | 背景: {item.background}"
        for item in character_sheet.supporting[:20]
    )
    return (
        "===== 世界设定 =====\n"
        f"世界名: {fused_world.world_name}\n"
        f"全局主题: {fused_world.global_theme}\n"
        f"力量来源: {fused_world.power_source}\n"
        f"世界背景: {fused_world.world_background}\n\n"
        f"修炼体系:\n{realms_text}\n\n"
        "===== 角色图鉴 =====\n"
        f"主角: {protagonist.name}\n"
        f"  执念: {protagonist.dao_heart}\n"
        f"  战斗: {protagonist.combat_style}\n"
        f"  缺陷: {protagonist.personality_flaw}\n"
        f"配角:\n{supporting_text}\n\n"
        "===== 当前关系局势 =====\n"
        f"{current_stage_network}\n"
    )


def _sorted_progression_stages(stages: List[ProgressionStage]) -> List[ProgressionStage]:
    return sorted(stages, key=lambda item: (item.stage_index, item.name))


def _stage_position(stage_name: str, stages: List[ProgressionStage]) -> int:
    canonical = normalize_stage_name(stage_name, stages)
    if not canonical:
        return -1
    for index, stage in enumerate(_sorted_progression_stages(stages)):
        if stage.name == canonical:
            return index
    return -1


def _canonical_stage_name(value: str, stages: List[ProgressionStage]) -> str:
    canonical = normalize_stage_name(value, stages)
    if canonical and stage_value(canonical, stages) >= 0:
        return canonical
    return ""


def _stage_from_numeric_level(level_hint: Any, stages: List[ProgressionStage]) -> str:
    if isinstance(level_hint, bool):
        return ""
    if isinstance(level_hint, (int, float)):
        numeric = int(level_hint)
        for stage in stages:
            if stage.stage_index == numeric:
                return stage.name
    return ""


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        clean = str(value or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _extract_stage_names_from_text(text: str, stages: List[ProgressionStage]) -> List[str]:
    haystack = str(text or "")
    matches: List[str] = []
    for stage in stages:
        aliases = _dedupe_preserve_order([stage.name] + list(stage.aliases))
        if any(alias and alias in haystack for alias in aliases):
            matches.append(stage.name)
    return _dedupe_preserve_order(matches)


def _resolve_volume_stage_bounds(
    realm_range_text: str,
    events: List[ReassembledEvent],
    previous_ending_state: str,
    fused_world: FusedWorld,
) -> Tuple[str, str, List[ProgressionStage]]:
    stages = build_progression_stages(fused_world)
    stage_names: List[str] = []

    if realm_range_text:
        stage_names.extend(_extract_stage_names_from_text(realm_range_text, stages))

    previous_stage = extract_max_stage_from_text(previous_ending_state, stages)
    if previous_stage:
        stage_names.append(previous_stage)

    for event in events:
        inferred_stage = infer_stage_from_node(event, stages)
        if inferred_stage:
            stage_names.append(inferred_stage)
        detected_stage = extract_max_stage_from_text(event.adapted_summary, stages)
        if detected_stage:
            stage_names.append(detected_stage)
        if isinstance(event.state_updates, dict):
            power_state = str(event.state_updates.get("power_state", "") or "").strip()
            if power_state:
                stage_names.append(power_state)
        stage_from_level = _stage_from_numeric_level(getattr(event, "realm_level", None), stages)
        if stage_from_level:
            stage_names.append(stage_from_level)

    normalized = [_canonical_stage_name(name, stages) for name in stage_names]
    normalized = [name for name in normalized if name]
    ordered = sorted(_dedupe_preserve_order(normalized), key=lambda item: (stage_value(item, stages), item))
    if not ordered and stages:
        ordered = [stages[0].name]
    if not ordered:
        return "", "", stages
    return ordered[0], ordered[-1], stages


def _is_at_or_below_named_stage(stage_name: str, stage_label: str, stages: List[ProgressionStage]) -> bool:
    current_position = _stage_position(stage_name, stages)
    if current_position < 0:
        return False
    candidate_positions: List[int] = []
    for stage in stages:
        aliases = _dedupe_preserve_order([stage.name] + list(stage.aliases))
        if any(stage_label in alias for alias in aliases):
            candidate_positions.append(_stage_position(stage.name, stages))
    candidate_positions = [position for position in candidate_positions if position >= 0]
    if candidate_positions:
        return current_position <= max(candidate_positions)
    if stage_label == "凝气十层":
        return "凝气" in stage_name
    if stage_label == "筑基后期":
        return "筑基" in stage_name
    return stage_label in stage_name


def _build_forbidden_terms_for_volume_max(volume_max_realm: str, stages: List[ProgressionStage]) -> List[str]:
    if volume_max_realm and _is_at_or_below_named_stage(volume_max_realm, "凝气十层", stages):
        return ["金丹", "元婴", "结婴", "化神", "古神传承", "元婴残魂"]
    if volume_max_realm and _is_at_or_below_named_stage(volume_max_realm, "筑基后期", stages):
        return ["元婴", "结婴", "化神", "古神传承"]
    return build_forbidden_stage_terms(volume_max_realm, stages, margin=1)


def _build_volume_allowed_state(
    realm_range_text: str,
    events: List[ReassembledEvent],
    previous_ending_state: str,
    fused_world: FusedWorld,
) -> Dict[str, Any]:
    volume_min_realm, volume_max_realm, stages = _resolve_volume_stage_bounds(
        realm_range_text=realm_range_text,
        events=events,
        previous_ending_state=previous_ending_state,
        fused_world=fused_world,
    )
    if not volume_max_realm and stages:
        volume_max_realm = stages[0].name
    if not volume_min_realm:
        volume_min_realm = volume_max_realm
    stage_range = volume_min_realm if volume_min_realm == volume_max_realm else f"{volume_min_realm}至{volume_max_realm}"
    forbidden_terms = _build_forbidden_terms_for_volume_max(volume_max_realm, stages)
    protagonist_stage = volume_max_realm
    return {
        "volume_min_realm": volume_min_realm,
        "volume_max_realm": volume_max_realm,
        "forbidden_terms": forbidden_terms,
        "protagonist_stage": protagonist_stage,
        "allowed_opponent_stage": get_allowed_opponent_stage(protagonist_stage, stages, max_gap=1) if protagonist_stage else "",
        "stage_range": stage_range,
    }


def _format_allowed_state_brief(allowed_state: Dict[str, Any]) -> str:
    forbidden_terms = [str(item).strip() for item in allowed_state.get("forbidden_terms", []) if str(item).strip()]
    lines = [
        "【本卷 allowed_state】",
        f"- 本卷阶段范围：{allowed_state.get('stage_range') or '未识别'}",
        f"- 本卷主角最高境界不得超过：{allowed_state.get('volume_max_realm') or '未识别'}",
        f"- 本卷主角起始参考境界：{allowed_state.get('volume_min_realm') or allowed_state.get('protagonist_stage') or '未识别'}",
        "- 敌人可以高于主角，但不能被主角正面击杀超过一个大境界的敌人。",
        "- 如果需要高阶存在，只能作为远景威胁、传闻、残留痕迹、规则压迫或追杀压力，不得直接交战并取胜。",
    ]
    if forbidden_terms:
        lines.append("- 禁止出现高阶词：" + "、".join(forbidden_terms))
    return "\n".join(lines)


def _issue_to_dict(issue: Any) -> Dict[str, Any]:
    return {
        "severity": getattr(issue, "severity", ""),
        "issue_type": getattr(issue, "issue_type", ""),
        "message": getattr(issue, "message", ""),
        "node_id": getattr(issue, "node_id", ""),
        "suggested_action": getattr(issue, "suggested_action", ""),
    }


def _find_forbidden_terms_used(texts: Iterable[str], forbidden_terms: List[str]) -> List[str]:
    found: List[str] = []
    haystack = "\n".join(str(text or "") for text in texts)
    for term in forbidden_terms:
        clean = str(term or "").strip()
        if clean and clean in haystack and clean not in found:
            found.append(clean)
    return found


def _collect_volume_texts(outline: VolumeOutline, include_raw_text: bool = False) -> List[str]:
    texts = [outline.ending_state, outline.realm_range] + list(outline.chapter_summaries)
    if include_raw_text:
        texts.insert(0, outline.raw_text)
    return texts


def _refresh_outline_state_fields(
    outline: VolumeOutline,
    allowed_state: Dict[str, Any],
    fused_world: FusedWorld,
) -> None:
    stages = build_progression_stages(fused_world)
    all_text = "\n".join(_collect_volume_texts(outline, include_raw_text=False))
    max_realm_used = extract_max_stage_from_text(all_text, stages) or allowed_state.get("volume_max_realm", "")
    outline.realm_range = allowed_state.get("stage_range", outline.realm_range)
    outline.volume_state_before = {
        "volume_min_realm": allowed_state.get("volume_min_realm", ""),
        "volume_max_realm": allowed_state.get("volume_max_realm", ""),
        "forbidden_terms": list(allowed_state.get("forbidden_terms", [])),
    }
    outline.volume_state_after = {
        "ending_state": outline.ending_state,
        "max_realm_used": max_realm_used,
    }
    outline.max_realm_used = max_realm_used
    outline.forbidden_terms_used = _find_forbidden_terms_used(
        _collect_volume_texts(outline, include_raw_text=False),
        list(allowed_state.get("forbidden_terms", [])),
    )


def _apply_volume_record_to_outline(outline: VolumeOutline, record: Dict[str, Any]) -> VolumeOutline:
    if not isinstance(record, dict):
        return outline
    for field_name in VolumeOutline.__dataclass_fields__:
        if field_name in record:
            setattr(outline, field_name, record[field_name])
    return outline


def _validate_and_repair_generated_volume(
    outline: VolumeOutline,
    fused_world: FusedWorld,
    allowed_state: Dict[str, Any],
) -> VolumeOutline:
    _refresh_outline_state_fields(outline, allowed_state, fused_world)
    raw_trigger_terms = _find_forbidden_terms_used(
        _collect_volume_texts(outline, include_raw_text=True),
        list(allowed_state.get("forbidden_terms", [])),
    )
    issues = list(validate_volume_outline(asdict(outline), fused_world))
    if raw_trigger_terms:
        outline.validation_notes.append(
            "forbidden_terms_detected_before_save: " + "、".join(raw_trigger_terms)
        )
    if raw_trigger_terms or outline.forbidden_terms_used or any(getattr(issue, "severity", "") in {"fatal", "major"} for issue in issues):
        repaired_record, remaining_issues = repair_volume_outline(
            asdict(outline),
            world=fused_world,
            allowed_state=allowed_state,
            max_rounds=2,
        )
        outline = _apply_volume_record_to_outline(outline, repaired_record)
        _refresh_outline_state_fields(outline, allowed_state, fused_world)
        issues = list(remaining_issues)
        if (raw_trigger_terms or outline.forbidden_terms_used) and "forbidden_terms_repair_applied" not in outline.repair_notes:
            outline.repair_notes.append("forbidden_terms_repair_applied")
    outline.state_validation_issues = [_issue_to_dict(issue) for issue in issues]
    if outline.forbidden_terms_used:
        raise RuntimeError(
            f"Step12 aborted: volume {outline.volume_number} still contains forbidden terms after repair: "
            + " | ".join(outline.forbidden_terms_used)
        )
    if any(getattr(issue, "severity", "") == "fatal" for issue in issues):
        fatal_labels = [f"[{issue.issue_type}] {issue.message}" for issue in issues if getattr(issue, "severity", "") == "fatal"]
        raise RuntimeError(
            f"Step12 aborted: volume {outline.volume_number} still has fatal state issues after repair: "
            + " | ".join(fatal_labels)
        )
    return outline


def _merge_allowed_state(base_allowed_state: Dict[str, Any], override_allowed_state: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base_allowed_state)
    for key in (
        "volume_min_realm",
        "volume_max_realm",
        "forbidden_terms",
        "protagonist_stage",
        "allowed_opponent_stage",
        "stage_range",
    ):
        if override_allowed_state.get(key):
            merged[key] = override_allowed_state[key]
    return merged


def _validate_and_repair_generated_volume_with_allowed_state(
    outline: VolumeOutline,
    source_events: List[ReassembledEvent],
    fused_world: FusedWorld,
    allowed_state: Dict[str, Any],
) -> VolumeOutline:
    recalculated_allowed_state = _build_volume_allowed_state(outline.realm_range, source_events, outline.ending_state, fused_world)
    effective_allowed_state = _merge_allowed_state(allowed_state, recalculated_allowed_state)
    _refresh_outline_state_fields(outline, effective_allowed_state, fused_world)
    raw_trigger_terms = _find_forbidden_terms_used(
        _collect_volume_texts(outline, include_raw_text=True),
        list(effective_allowed_state.get("forbidden_terms", [])),
    )
    issues = list(validate_volume_outline(asdict(outline), fused_world))
    if raw_trigger_terms:
        outline.validation_notes.append(
            "forbidden_terms_detected_before_save: " + "、".join(raw_trigger_terms)
        )
    if raw_trigger_terms or outline.forbidden_terms_used or any(getattr(issue, "severity", "") in {"fatal", "major"} for issue in issues):
        repaired_record, remaining_issues = repair_volume_outline(
            asdict(outline),
            world=fused_world,
            allowed_state=effective_allowed_state,
            max_rounds=2,
        )
        outline = _apply_volume_record_to_outline(outline, repaired_record)
        _refresh_outline_state_fields(outline, effective_allowed_state, fused_world)
        issues = list(remaining_issues)
        if (raw_trigger_terms or outline.forbidden_terms_used) and "forbidden_terms_repair_applied" not in outline.repair_notes:
            outline.repair_notes.append("forbidden_terms_repair_applied")
    outline.state_validation_issues = [_issue_to_dict(issue) for issue in issues]
    if outline.forbidden_terms_used:
        raise RuntimeError(
            f"Step12 aborted: volume {outline.volume_number} still contains forbidden terms after repair: "
            + " | ".join(outline.forbidden_terms_used)
        )
    if any(getattr(issue, "severity", "") == "fatal" for issue in issues):
        fatal_labels = [f"[{issue.issue_type}] {issue.message}" for issue in issues if getattr(issue, "severity", "") == "fatal"]
        raise RuntimeError(
            f"Step12 aborted: volume {outline.volume_number} still has fatal state issues after repair: "
            + " | ".join(fatal_labels)
        )
    return outline


def _extract_chapter_issue_number(node_id: str) -> Optional[int]:
    match = re.search(r"_chapter_(\d+)$", str(node_id or ""))
    if not match:
        return None
    return int(match.group(1))


def _non_empty_summary_positions(chapter_summaries: List[str]) -> List[int]:
    positions: List[int] = []
    for index, summary in enumerate(chapter_summaries):
        if str(summary or "").strip():
            positions.append(index)
    return positions


def _failure_term_for_volume_max(volume_max_realm: str) -> str:
    text = str(volume_max_realm or "")
    if "凝气" in text:
        return "筑基失败"
    if "筑基" in text:
        return "结丹失败"
    return f"{text}突破失败" if text else "突破失败"


def _repair_volume_text_by_rule(
    text: str,
    volume_max_realm: str,
    forbidden_terms: List[str],
    stages: List[ProgressionStage],
) -> str:
    rewritten = downgrade_overpowered_enemy_text(str(text or ""), volume_max_realm, stages)
    replacements = {
        "反杀金丹追兵": "借禁制拖住筑基追兵并趁乱逃脱",
        "击杀金丹追兵": "借禁制拖住筑基追兵并趁乱逃脱",
        "斩杀金丹追兵": "借禁制拖住筑基追兵并趁乱逃脱",
    }
    for source, target in replacements.items():
        if source in rewritten:
            rewritten = rewritten.replace(source, target)
    if any(token in rewritten for token in ("击杀元婴残魂", "斩杀元婴残魂", "反杀元婴残魂", "正面击败元婴残魂")):
        rewritten = "避开元婴残魂残念锁定，获得外围线索"
    if "结婴失败" in rewritten and "结婴" in forbidden_terms:
        rewritten = rewritten.replace("结婴失败", _failure_term_for_volume_max(volume_max_realm))
    return rewritten


def _apply_rule_repairs_to_volume_outline(
    outline: VolumeOutline,
    fused_world: FusedWorld,
    allowed_state: Dict[str, Any],
) -> bool:
    stages = build_progression_stages(fused_world)
    volume_max_realm = str(allowed_state.get("volume_max_realm", "") or allowed_state.get("max_stage", "")).strip()
    forbidden_terms = [str(item).strip() for item in allowed_state.get("forbidden_terms", []) if str(item).strip()]
    changed = False

    repaired_chapters: List[str] = []
    for chapter in outline.chapter_summaries:
        repaired = _repair_volume_text_by_rule(chapter, volume_max_realm, forbidden_terms, stages)
        if repaired != chapter:
            changed = True
        repaired_chapters.append(repaired)
    outline.chapter_summaries = repaired_chapters

    repaired_ending_state = _repair_volume_text_by_rule(outline.ending_state, volume_max_realm, forbidden_terms, stages)
    if repaired_ending_state != outline.ending_state:
        outline.ending_state = repaired_ending_state
        changed = True

    repaired_raw_text = _repair_volume_text_by_rule(outline.raw_text, volume_max_realm, forbidden_terms, stages)
    if repaired_raw_text != outline.raw_text:
        outline.raw_text = repaired_raw_text
        changed = True

    if changed:
        outline.repair_notes.append("rule_repaired_volume_outline")
    return changed


def _repair_volume_outline_with_ai(
    outline: VolumeOutline,
    issues: List[Any],
    allowed_state: Dict[str, Any],
    fused_world: FusedWorld,
) -> VolumeOutline:
    stages = build_progression_stages(fused_world)
    chapter_positions = _non_empty_summary_positions(outline.chapter_summaries)
    grouped_by_chapter: Dict[int, List[Any]] = {}
    volume_level_issues: List[Any] = []
    for issue in issues:
        chapter_no = _extract_chapter_issue_number(getattr(issue, "node_id", ""))
        if chapter_no is None:
            volume_level_issues.append(issue)
            continue
        grouped_by_chapter.setdefault(chapter_no, []).append(issue)

    for chapter_no, chapter_issues in grouped_by_chapter.items():
        chapter_index = chapter_no - 1
        if chapter_index < 0 or chapter_index >= len(chapter_positions):
            continue
        actual_index = chapter_positions[chapter_index]
        pseudo_record = {
            "event_id": f"volume_{outline.volume_number}_chapter_{chapter_no}",
            "summary": outline.chapter_summaries[actual_index],
            "raw_text": outline.raw_text,
            "chapter_summaries": outline.chapter_summaries,
            "ending_state": outline.ending_state,
            "state_updates": {"power_state": allowed_state.get("volume_max_realm", "")},
        }
        repaired = repair_event_with_ai(pseudo_record, chapter_issues, {
            "max_stage": allowed_state.get("volume_max_realm", ""),
            "forbidden_terms": allowed_state.get("forbidden_terms", []),
            "allowed_opponent_stage": allowed_state.get("allowed_opponent_stage", ""),
        }, stages)
        rewritten_summary = str(repaired.get("summary", "") or repaired.get("rewritten_summary", "") or "").strip()
        if rewritten_summary:
            outline.chapter_summaries[actual_index] = rewritten_summary
        for note in repaired.get("repair_notes", []):
            clean = str(note).strip()
            if clean and clean not in outline.repair_notes:
                outline.repair_notes.append(clean)

    if volume_level_issues:
        pseudo_record = {
            "event_id": f"volume_{outline.volume_number}",
            "summary": outline.ending_state or "\n".join(outline.chapter_summaries),
            "raw_text": outline.raw_text,
            "chapter_summaries": outline.chapter_summaries,
            "ending_state": outline.ending_state,
            "state_updates": {"power_state": allowed_state.get("volume_max_realm", "")},
        }
        repaired = repair_event_with_ai(pseudo_record, volume_level_issues, {
            "max_stage": allowed_state.get("volume_max_realm", ""),
            "forbidden_terms": allowed_state.get("forbidden_terms", []),
            "allowed_opponent_stage": allowed_state.get("allowed_opponent_stage", ""),
        }, stages)
        rewritten_summary = str(repaired.get("summary", "") or repaired.get("rewritten_summary", "") or "").strip()
        if rewritten_summary:
            outline.ending_state = rewritten_summary
        for note in repaired.get("repair_notes", []):
            clean = str(note).strip()
            if clean and clean not in outline.repair_notes:
                outline.repair_notes.append(clean)
        outline.repair_notes.append("ai_repaired_volume_outline")

    return outline


def _postprocess_generated_volume_outline(
    outline: VolumeOutline,
    source_events: List[ReassembledEvent],
    fused_world: FusedWorld,
    allowed_state: Dict[str, Any],
) -> VolumeOutline:
    effective_allowed_state = _merge_allowed_state(
        allowed_state,
        _build_volume_allowed_state(outline.realm_range, source_events, outline.ending_state, fused_world),
    )
    _refresh_outline_state_fields(outline, effective_allowed_state, fused_world)
    raw_trigger_terms = _find_forbidden_terms_used(
        _collect_volume_texts(outline, include_raw_text=True),
        list(effective_allowed_state.get("forbidden_terms", [])),
    )

    issues = list(validate_volume_outline(asdict(outline), fused_world))
    if raw_trigger_terms:
        outline.validation_notes.append(
            "forbidden_terms_detected_before_save: " + "、".join(raw_trigger_terms)
        )
    outline.state_validation_issues = [_issue_to_dict(issue) for issue in issues]
    if not issues and not raw_trigger_terms:
        outline.state_validation_failed = False
        outline.fatal_issues = []
        return outline

    blocking_issues = [issue for issue in issues if getattr(issue, "severity", "") in {"major", "fatal"}]
    if blocking_issues or raw_trigger_terms:
        _apply_rule_repairs_to_volume_outline(outline, fused_world, effective_allowed_state)
        _refresh_outline_state_fields(outline, effective_allowed_state, fused_world)
        issues = list(validate_volume_outline(asdict(outline), fused_world))
        raw_trigger_terms = _find_forbidden_terms_used(
            _collect_volume_texts(outline, include_raw_text=True),
            list(effective_allowed_state.get("forbidden_terms", [])),
        )
        blocking_issues = [issue for issue in issues if getattr(issue, "severity", "") in {"major", "fatal"}]

    ai_issues = list(blocking_issues)
    if raw_trigger_terms:
        ai_issues.append(
            StateIssue(
                severity="fatal",
                issue_type="forbidden_terms_remaining",
                message="Volume still contains forbidden terms after rule repair: " + "、".join(raw_trigger_terms),
                node_id=f"volume_{outline.volume_number}",
                suggested_action="只修正违规章节中的高阶内容，不改写整卷主线。",
            )
        )

    if any(getattr(issue, "severity", "") == "fatal" for issue in blocking_issues) or raw_trigger_terms:
        outline = _repair_volume_outline_with_ai(outline, ai_issues, effective_allowed_state, fused_world)
        _refresh_outline_state_fields(outline, effective_allowed_state, fused_world)
        issues = list(validate_volume_outline(asdict(outline), fused_world))
        raw_trigger_terms = _find_forbidden_terms_used(
            _collect_volume_texts(outline, include_raw_text=True),
            list(effective_allowed_state.get("forbidden_terms", [])),
        )

    outline.state_validation_issues = [_issue_to_dict(issue) for issue in issues]
    fatal_issues = [issue for issue in issues if getattr(issue, "severity", "") == "fatal"]
    outline.fatal_issues = [_issue_to_dict(issue) for issue in fatal_issues]
    if raw_trigger_terms:
        outline.fatal_issues.append(
            {
                "severity": "fatal",
                "issue_type": "forbidden_terms_remaining",
                "message": "Volume still contains forbidden terms after repair: " + "、".join(raw_trigger_terms),
                "node_id": f"volume_{outline.volume_number}",
                "suggested_action": "继续降低违规章节中的高阶设定或改写为外围威胁。",
            }
        )
    outline.state_validation_failed = bool(fatal_issues or raw_trigger_terms)
    if outline.state_validation_failed:
        outline.validation_notes.append("volume_outline_still_has_fatal_state_issues")
    return outline


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _generate_single_volume(
    client,
    volume_number: int,
    arc_name: str,
    events: List[ReassembledEvent],
    system_context: str,
    previous_ending_state: str,
    previous_tail_chapters: List[str],
    allowed_state: Dict[str, Any],
) -> VolumeOutline:
    max_realm_level = max([event.realm_level for event in events] + [1])
    allowed_state_brief = _format_allowed_state_brief(allowed_state)
    events_lines: List[str] = []
    for idx, event in enumerate(events, start=1):
        plan = event.event_plan or {}
        target_chapter_count = int(plan.get("target_chapter_count") or max(4, len(plan.get("chapter_blueprint", [])) or 4))
        events_lines.append(
            f"核心事件{idx}. [{event.pacing_role}] {event.adapted_summary}\n"
            f"  - 建议章数: {target_chapter_count}\n"
            f"  - 触发: {plan.get('trigger', '')}\n"
            f"  - 目标: {plan.get('goal', '')}\n"
            f"  - 阻碍: {plan.get('obstacle', '')}\n"
            f"  - 转折: {plan.get('reversal', '')}\n"
            f"  - 代价: {plan.get('cost', '')}\n"
            f"  - 活跃角色: {', '.join(event.active_characters) or '待生成'}\n"
            f"  - 章节蓝图: {json.dumps(plan.get('chapter_blueprint', []), ensure_ascii=False)}\n"
        )

    transition_parts: List[str] = []
    if previous_tail_chapters:
        transition_parts.append("【上卷末尾剧情】\n" + "\n".join(previous_tail_chapters))
    if previous_ending_state:
        transition_parts.append(f"【上卷结尾状态】\n{previous_ending_state}")
    transition_text = "\n\n".join(transition_parts)
    response_schema = {
        "volume_title": f"第{volume_number}卷标题",
        "realm_range": allowed_state.get("stage_range", arc_name),
        "expanded_events": [
            {
                "event_name": "事件标题",
                "chapters": [
                    {"chapter_number": 1, "summary": "章节摘要"}
                ],
            }
        ],
        "tension_curve": "卷内张力走势",
        "ending_state": "卷末状态",
        "volume_state_before": {
            "volume_min_realm": allowed_state.get("volume_min_realm", ""),
            "volume_max_realm": allowed_state.get("volume_max_realm", ""),
        },
        "volume_state_after": {
            "ending_state": "卷末主角状态",
            "relationship_state": "卷末关系状态",
        },
        "max_realm_used": allowed_state.get("volume_max_realm", ""),
        "forbidden_terms_used": [],
        "validation_notes": ["如无违规可返回空列表"],
    }

    user_prompt = (
        f"请为第{volume_number}卷（{arc_name}）生成详细中文章节大纲。\n\n"
        f"{transition_text}\n\n"
        "【本卷必须推进的核心事件骨架】\n"
        f"{chr(10).join(events_lines)}\n\n"
        f"{allowed_state_brief}\n\n"
        "【硬性要求】\n"
        "1. 每个核心事件按建议章数展开，允许章节功能重叠。\n"
        f"2. 战力与冲突强度不得明显超出第{max_realm_level}阶段的上限。\n"
        f"2.1 本卷主角最高境界不得超过 {allowed_state.get('volume_max_realm') or '当前卷允许上限'}。\n"
        f"2.2 禁止出现这些词：{', '.join(allowed_state.get('forbidden_terms', [])) or '无'}。\n"
        "2.3 敌人可以高于主角，但不能被主角正面击杀超过一个大境界的敌人。\n"
        "2.4 如果需要高阶存在，只能作为远景威胁、传闻、残留痕迹、规则压迫或追杀压力，不得直接交战并取胜。\n"
        "3. 每章摘要必须是自然中文，不要输出英文说明，不要输出 Python 字典字面量。\n"
        "4. 不要写“按原文”“时间线矛盾”“注：”这类元注释，也不要用回忆插叙打乱当前卷时间线。\n"
        "5. 每章摘要控制在约 80 到 120 个中文字符，前后章必须有因果承接。\n"
        f"{_build_volume_diversity_brief(events)}\n"
        "6. 必须尊重事件计划与状态变化，不能丢失后果、人物关系变化和资源代价。\n"
        f"请只返回 JSON，schema 如下：{json.dumps(response_schema, ensure_ascii=False)}"
    )
    system_prompt = (
        "你是一名中文网文大纲策划师。"
        "请写出简洁但连贯的中文章节摘要，保持章节因果连续、人物状态连续、时间线连续。\n\n"
        f"{system_context}"
    )
    raw = _quality_chat_completion_json(client, system=system_prompt, user=user_prompt, json_mode=True)
    data = _safe_json_load(raw)
    expanded_events = _normalize_expanded_events_payload(data.get("expanded_events", []), events)
    expanded_events = _validate_or_repair_volume_payload(client, arc_name, events, expanded_events)
    if not _is_volume_payload_usable(expanded_events, events):
        expanded_events = _build_fallback_expanded_events(events)

    flattened_chapters: List[str] = []
    chapter_index = 1
    for block in expanded_events:
        event_name = str(block.get("event_name") or block.get("event_title") or "主线推进").strip()
        flattened_chapters.append(f"#### 剧情点：{event_name}")
        for chapter in block.get("chapters", []):
            clean_text = _normalize_chapter_summary(chapter)
            flattened_chapters.append(f"**第{chapter_index}章**：{clean_text}")
            chapter_index += 1
        flattened_chapters.append("")

    return VolumeOutline(
        volume_number=volume_number,
        volume_title=data.get("volume_title", f"第{volume_number}卷"),
        realm_range=data.get("realm_range", allowed_state.get("stage_range", arc_name)),
        chapter_summaries=flattened_chapters,
        tension_curve=_normalize_tension_curve(data.get("tension_curve")),
        ending_state=data.get("ending_state", _build_fallback_ending_state(events)),
        raw_text=raw,
        volume_state_before=data.get("volume_state_before", {}),
        volume_state_after=data.get("volume_state_after", {}),
        max_realm_used=str(data.get("max_realm_used", allowed_state.get("volume_max_realm", "")) or ""),
        forbidden_terms_used=[
            str(item).strip()
            for item in data.get("forbidden_terms_used", [])
            if str(item).strip()
        ] if isinstance(data.get("forbidden_terms_used", []), list) else [],
        validation_notes=[
            str(item).strip()
            for item in data.get("validation_notes", [])
            if str(item).strip()
        ] if isinstance(data.get("validation_notes", []), list) else [],
    )


def _build_volume_diversity_brief(events: List[ReassembledEvent]) -> str:
    motif_hits = []
    for motif in _GENERIC_VOLUME_MOTIFS:
        count = sum(1 for event in events if motif in event.adapted_summary)
        if count >= 2:
            motif_hits.append(f"{motif} x{count}")
    lines = ["【本卷反重复提醒】"]
    lines.append("每个事件都必须使用明显不同的戏剧引擎，不能默认回到同一套设局、识破、逆袭。")
    if motif_hits:
        lines.append("骨架里已经重复出现的母题有：" + "、".join(motif_hits) + "。后续不要再把它们当默认解法。")
    else:
        lines.append("当前骨架未检测到特别明显的重复母题，但仍需主动拉开事件质感。")
    return "\n".join(lines)


def _is_volume_payload_usable(expanded_events: Any, source_events: List[ReassembledEvent]) -> bool:
    if not isinstance(expanded_events, list) or not expanded_events:
        return False
    if len(expanded_events) < max(1, len(source_events) - 1):
        return False
    valid_blocks = 0
    for index, block in enumerate(expanded_events):
        chapters = block.get("chapters", []) if isinstance(block, dict) else []
        if isinstance(chapters, list) and any(_normalize_chapter_summary(chapter) for chapter in chapters):
            if index < len(source_events):
                target_count = int((source_events[index].event_plan or {}).get("target_chapter_count") or max(4, len((source_events[index].event_plan or {}).get("chapter_blueprint", [])) or 4))
                if abs(len(chapters) - target_count) > 3:
                    continue
            valid_blocks += 1
    return valid_blocks >= max(1, len(source_events) // 2)


def _build_fallback_expanded_events(events: List[ReassembledEvent]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for event in events:
        plan = event.event_plan or {}
        event_name = _build_fallback_event_name(event)
        blueprint = plan.get("chapter_blueprint", [])
        chapters: List[Dict[str, Any]] = []
        local_chapter_index = 1
        if isinstance(blueprint, list) and blueprint:
            for beat in blueprint:
                primary = beat.get("primary_function", "推进")
                secondary = beat.get("secondary_functions", [])
                secondary_text = f"，并兼顾{'/'.join(secondary)}" if isinstance(secondary, list) and secondary else ""
                purpose = beat.get("purpose", "") or event.adapted_summary
                span = beat.get("chapter_span", [])
                if isinstance(span, list) and len(span) == 2:
                    start, end = int(span[0]), int(span[1])
                else:
                    start, end = 1, 1
                for _chapter_no in range(start, end + 1):
                    chapters.append(
                        {
                            "chapter_number": local_chapter_index,
                            "summary": f"{primary}{secondary_text}：{purpose}",
                        }
                    )
                    local_chapter_index += 1
        if not chapters:
            chapters = [
                {"chapter_number": 1, "summary": f"铺垫目标与阻碍：{event.adapted_summary}"},
                {"chapter_number": 2, "summary": f"冲突升级并迫使角色调整策略：围绕{plan.get('obstacle', event.pacing_role)}展开新的选择与对抗。"},
                {"chapter_number": 3, "summary": f"阶段性收束但留下后续压力：代价落在{plan.get('cost', '资源、人情或伤势')}上，并把主线推向下一步。"},
            ]
        results.append({"event_name": event_name, "chapters": chapters})
    return results


def _build_fallback_event_name(event: ReassembledEvent) -> str:
    summary = (event.adapted_summary or "").strip()
    if not summary:
        return "主线推进"
    return summary.replace("，", " ").replace("。", " ").split(" ", 1)[0][:18] or "主线推进"


def _build_fallback_ending_state(events: List[ReassembledEvent]) -> str:
    if not events:
        return "本卷完成阶段性推进。"
    return f"本卷收束于：{events[-1].adapted_summary}"


def _strip_leading_chapter_label(text: str) -> str:
    if not text:
        return "主线推进。"
    return re.sub(r"^\s*第[0-9一二三四五六七八九十百千]+章[：:]\s*", "", text).strip()


def _arc_sort_key(arc_name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(arc_name or ""))
    return (int(match.group(1)) if match else 9999, str(arc_name or ""))


def _event_sort_key(event: ReassembledEvent) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(event.event_id or ""))
    return (int(match.group(1)) if match else 9999, str(event.event_id or ""))


def _normalize_expanded_events_payload(expanded_events: Any, source_events: List[ReassembledEvent]) -> List[Dict[str, Any]]:
    if not isinstance(expanded_events, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for index, block in enumerate(expanded_events):
        if not isinstance(block, dict):
            continue
        event_name = str(block.get("event_name") or block.get("event_title") or f"主线事件{index + 1}").strip()
        raw_chapters = block.get("chapters", [])
        if not isinstance(raw_chapters, list):
            continue
        chapters: List[Dict[str, Any]] = []
        for chapter_index, chapter in enumerate(raw_chapters, start=1):
            summary = _normalize_chapter_summary(chapter)
            if not summary:
                continue
            chapter_number = chapter_index
            if isinstance(chapter, dict):
                try:
                    chapter_number = int(chapter.get("chapter_number", chapter_index))
                except (TypeError, ValueError):
                    chapter_number = chapter_index
            chapters.append({"chapter_number": chapter_number, "summary": summary})
        if chapters:
            normalized.append({"event_name": event_name, "chapters": chapters})
    if not normalized and source_events:
        return _build_fallback_expanded_events(source_events)
    return normalized


def _normalize_chapter_summary(chapter: Any) -> str:
    if isinstance(chapter, str):
        text = chapter.strip()
        if text.startswith("{") and text.endswith("}"):
            parsed = _safe_json_load(text)
            if isinstance(parsed, dict):
                return _normalize_chapter_summary(parsed)
    if isinstance(chapter, dict):
        for key in ("summary", "content", "chapter_summary", "outline"):
            value = chapter.get(key)
            if value:
                return _strip_leading_chapter_label(str(value).strip())
        return ""
    return _strip_leading_chapter_label(str(chapter or "").strip())


def _normalize_tension_curve(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        return text or "本卷张力持续推进。"
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if cleaned:
            return "章节张力大致依次为：" + " → ".join(cleaned)
    return "本卷张力持续推进。"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _validate_or_repair_volume_payload(client, arc_name: str, source_events: List[ReassembledEvent], expanded_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not expanded_events:
        return expanded_events
    event_specs = []
    for index, event in enumerate(source_events, start=1):
        plan = event.event_plan or {}
        blueprint = plan.get("chapter_blueprint", [])
        required_coverage = _required_coverage_from_blueprint(blueprint)
        event_specs.append(
            {
                "event_index": index,
                "event_summary": event.adapted_summary,
                "target_chapter_count": int(plan.get("target_chapter_count") or max(4, len(blueprint) or 4)),
                "required_coverage": required_coverage,
                "chapter_blueprint": blueprint,
            }
        )

    prompt = (
        f"你是卷级章节排布审校器。请检查第{arc_name}卷的章节展开是否真正遵守事件模板，而不是只写出一组松散章节。\n"
        f"事件规格: {json.dumps(event_specs, ensure_ascii=False)}\n"
        f"当前展开结果: {json.dumps(expanded_events, ensure_ascii=False)}\n"
        "检查重点：\n"
        "1. 每个事件的章节数是否大致符合 target_chapter_count；\n"
        "2. 章节排布是否覆盖 required_coverage；\n"
        "3. 事件内部是否有起承转合，而不是只列几个并排短句；\n"
        "4. 相邻事件之间是否有因果承接。\n"
        "只返回 JSON，字段包含 is_valid, issues, repaired_expanded_events。若当前结果可用，repaired_expanded_events 原样返回即可。"
    )
    raw = _quality_chat_completion_json(client, system="只返回合法 JSON。", user=prompt, json_mode=True, temperature=0.2)
    data = _safe_json_load(raw)
    if not isinstance(data, dict):
        return expanded_events
    repaired = _normalize_expanded_events_payload(data.get("repaired_expanded_events", expanded_events), source_events)
    return repaired or expanded_events


def _required_coverage_from_blueprint(blueprint: Any) -> List[str]:
    if not isinstance(blueprint, list):
        return []
    tags: List[str] = []
    for beat in blueprint:
        if not isinstance(beat, dict):
            continue
        primary = str(beat.get("primary_function", "")).strip()
        if primary:
            tags.append(primary)
        for item in beat.get("secondary_functions", []):
            text = str(item).strip()
            if text:
                tags.append(text)
    deduped = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped


_safe_json_load = safe_json_load
