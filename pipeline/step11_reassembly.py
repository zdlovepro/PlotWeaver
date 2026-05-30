from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

import config
from pipeline.core.common_json import read_json_file, safe_json_load
from pipeline.core.story_models import Character, CharacterSheet, EventRolePlan, NarrativeSkeleton, SkeletonNode
from pipeline.core.state_repair import repair_reassembled_events, rewrite_overpowered_win_text_by_rule
from pipeline.core.state_validator import build_forbidden_stage_terms, validate_reassembled_events
from pipeline.core.story_state import ProgressionStage, StateIssue, build_progression_stages, extract_max_stage_from_text, infer_stage_from_node, normalize_stage_name, stage_value
from pipeline.core.utils import chat_completion_json, get_deepseek_client
from pipeline.core.world_building_core import FusedWorld, KnowledgeBase


_STEP11_FILENAME = "step11_reassembled_plot.json"
_GENERIC_REPEAT_MOTIFS = ("设局", "识破", "逆袭", "试炼", "拍卖会", "秘境", "暗中相助", "宝珠")
_CANDIDATE_FOCUSES = [
    ("strategy", "强调谋略、试探和破局"),
    ("relationship", "强调人物关系推进和立场变化"),
    ("cost", "强调代价、后果和债务"),
]


def _quality_chat_completion_json(client, **kwargs) -> str:
    return chat_completion_json(client, model_name=config.DEEPSEEK_LAST_STEPS_MODEL, **kwargs)


@dataclass
class ReassembledEvent:
    event_id: str
    arc_name: str
    realm_level: int
    pacing_role: str
    adapted_summary: str
    used_trope: str = ""
    template_name: str = ""
    source_atom_ids: List[str] = field(default_factory=list)
    source_induced_event_ids: List[str] = field(default_factory=list)
    source_legacy_event_ids: List[str] = field(default_factory=list)
    source_chunk_ids: List[str] = field(default_factory=list)
    source_refs: List[Dict[str, Any]] = field(default_factory=list)
    is_bridge: bool = False
    event_plan: Dict[str, Any] = field(default_factory=dict)
    active_characters: List[str] = field(default_factory=list)
    state_updates: Dict[str, Any] = field(default_factory=dict)
    logic_notes: List[str] = field(default_factory=list)
    state_validation_issues: List[Dict[str, Any]] = field(default_factory=list)
    repair_notes: List[str] = field(default_factory=list)


_LEDGER_KEYS = [
    "relationships",
    "resources",
    "secrets",
    "injuries",
    "hooks",
    "identities",
    "positions",
    "objectives",
    "power_state",
]


def reassemble_plot(
    skeleton: NarrativeSkeleton,
    kb: KnowledgeBase,
    fused_world: FusedWorld,
    only_event_ids: Optional[Set[str]] = None,
    previous_events: Optional[List[ReassembledEvent]] = None,
) -> List[ReassembledEvent]:
    client = get_deepseek_client()
    if not skeleton.character_sheet:
        raise ValueError("Step 10 character sheet is required before Step 11.")

    char_sheet = skeleton.character_sheet
    role_plan_map = {plan.event_id: plan for plan in char_sheet.event_role_plans}
    recent_context: List[str] = []
    recent_micro_names: List[str] = []
    ledger = _initialize_story_ledger()
    prev_event_map = {event.event_id: event for event in (previous_events or [])}

    reassembled: List[ReassembledEvent] = []
    total_nodes = len(skeleton.nodes)

    for idx, node in enumerate(tqdm(skeleton.nodes, desc="[Step 11] Reassembling plot", unit="node")):
        event_id = f"adapted_{node.node_id}"
        role_plan = role_plan_map.get(node.node_id)

        if only_event_ids and node.node_id not in only_event_ids:
            existing = prev_event_map.get(event_id)
            if existing:
                reassembled.append(existing)
                _update_rolling_context(recent_context, existing.adapted_summary)
                _update_recent_micro_names(recent_micro_names, existing.used_trope)
                _apply_state_updates(ledger, existing.state_updates)
                continue

        characters_desc, active_characters = _format_characters_for_stage(
            char_sheet,
            idx + 1,
            role_plan,
            current_arc_name=node.arc_name,
            recent_events=reassembled[-3:],
        )
        progress_ratio = idx / total_nodes if total_nodes > 1 else 0
        active_macro_stages = _get_active_macro_stages(fused_world, progress_ratio)
        suggested_micro = _pick_micro_interaction(fused_world.micro_interactions, recent_micro_names)
        anti_repeat_rules = _build_anti_repetition_rules(recent_context, recent_micro_names)
        similar_events = kb.query_events(query=node.original_summary or node.pacing_role, n_results=3)
        template_brief = _select_event_template_brief(fused_world, node, similar_events)
        state_brief = _format_state_ledger(ledger)
        recent_story_brief = _format_recent_story_context(recent_context)

        event_plan = _plan_event(
            client=client,
            node=node,
            role_plan=role_plan,
            characters_desc=characters_desc,
            active_macro_stages=active_macro_stages,
            template_brief=template_brief,
            retrieved_events=similar_events,
            state_brief=state_brief,
            recent_story_brief=recent_story_brief,
            anti_repeat_rules=anti_repeat_rules,
        )
        event_plan, plan_logic_notes = _validate_or_repair_event_plan(
            client=client,
            node=node,
            event_plan=event_plan,
            characters_desc=characters_desc,
            state_brief=state_brief,
            recent_story_brief=recent_story_brief,
            active_macro_stages=active_macro_stages,
            anti_repeat_rules=anti_repeat_rules,
        )
        candidates = _generate_candidates(
            event_plan,
            node,
            characters_desc,
            active_macro_stages,
            suggested_micro,
            anti_repeat_rules,
            state_brief,
            recent_story_brief,
        )
        winner_index, final_summary = _judge_candidates(
            client,
            node,
            event_plan,
            candidates,
            anti_repeat_rules,
            state_brief,
            recent_story_brief,
        )
        final_summary, summary_logic_notes = _validate_or_repair_selected_summary(
            client=client,
            node=node,
            event_plan=event_plan,
            summary=final_summary,
            state_brief=state_brief,
            recent_story_brief=recent_story_brief,
            anti_repeat_rules=anti_repeat_rules,
        )
        final_summary = _strip_meta_continuity_notes(final_summary)

        used_trope = (suggested_micro or {}).get("interaction_name", "")
        if candidates and 0 <= winner_index < len(candidates):
            used_trope = candidates[winner_index].get("used_trope", used_trope)

        state_updates = event_plan.get("state_updates", {}) if isinstance(event_plan, dict) else {}
        logic_notes = [note for note in (plan_logic_notes + summary_logic_notes) if note]
        source_atom_ids = _dedupe_texts(list(getattr(node, "source_atom_ids", []) or []))
        source_induced_event_ids = _dedupe_texts(list(getattr(node, "source_induced_event_ids", []) or []))
        source_legacy_event_ids = _dedupe_texts(list(getattr(node, "source_legacy_event_ids", []) or []))
        source_chunk_ids = _dedupe_texts(list(getattr(node, "source_chunk_ids", []) or []))
        source_refs = _dedupe_source_refs(list(getattr(node, "source_refs", []) or []))
        fallback_source_novel = _dedupe_texts(list(getattr(node, "source_novels", []) or []))
        fallback_source_novel = fallback_source_novel[0] if fallback_source_novel else ""
        if not source_refs:
            source_refs.extend({"ref_type": "atom", "ref_id": ref_id, "source_novel": fallback_source_novel} for ref_id in source_atom_ids)
            source_refs.extend({"ref_type": "induced_event", "ref_id": ref_id, "source_novel": fallback_source_novel} for ref_id in source_induced_event_ids)
            source_refs.extend({"ref_type": "legacy_event", "ref_id": ref_id, "source_novel": fallback_source_novel} for ref_id in source_legacy_event_ids)
            source_refs.extend({"ref_type": "chunk", "ref_id": ref_id, "source_novel": fallback_source_novel} for ref_id in source_chunk_ids)
            source_refs = _dedupe_source_refs(source_refs)

        event = ReassembledEvent(
            event_id=event_id,
            arc_name=node.arc_name,
            realm_level=node.realm_level,
            pacing_role=node.pacing_role,
            adapted_summary=final_summary,
            used_trope=used_trope,
            template_name=node.template_hint,
            source_atom_ids=source_atom_ids,
            source_induced_event_ids=source_induced_event_ids,
            source_legacy_event_ids=source_legacy_event_ids,
            source_chunk_ids=source_chunk_ids,
            source_refs=source_refs,
            event_plan=event_plan,
            active_characters=active_characters,
            state_updates=state_updates,
            logic_notes=logic_notes,
        )
        previous_event = reassembled[-1] if reassembled else None
        event = _validate_and_repair_reassembled_event(event, node, fused_world, previous_event)
        reassembled.append(event)
        _apply_state_updates(ledger, event.state_updates)
        _update_rolling_context(recent_context, event.adapted_summary)
        _update_recent_micro_names(recent_micro_names, event.used_trope)

    tqdm.write(f"[Step 11] Reassembled {len(reassembled)} events.")
    return reassembled


def save_step11_output(events: List[ReassembledEvent]) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _STEP11_FILENAME
    from pipeline.core.common_json import write_json_file

    write_json_file(out_path, [asdict(event) for event in events])
    print(f"[Step 11] Intermediate output saved -> {out_path}")
    return out_path


def load_step11_output(intermediate_dir: str | Path | None = None) -> List[ReassembledEvent]:
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)
    in_path = inter_dir / _STEP11_FILENAME
    if not in_path.exists():
        raise FileNotFoundError(f"Step 11 file not found: {in_path}")
    data = read_json_file(in_path)
    return [ReassembledEvent(**_normalize_reassembled_event_record(item)) for item in data]


def _classify_source_ref_id(ref_id: str) -> str:
    text = str(ref_id or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if "_induced_event" in lowered:
        return "induced_event"
    if re.search(r"_atom_?\d+$", lowered):
        return "atom"
    if "chunk" in lowered:
        return "chunk"
    if re.search(r"(?:^|_)event_?\d+$", lowered):
        return "legacy_event"
    return ""


def _dedupe_source_refs(source_refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for raw_ref in source_refs or []:
        if not isinstance(raw_ref, dict):
            continue
        ref_type = str(raw_ref.get("ref_type", "") or "").strip()
        ref_id = str(raw_ref.get("ref_id", "") or "").strip()
        source_novel = str(raw_ref.get("source_novel", "") or "").strip()
        if not ref_type or not ref_id:
            continue
        key = (ref_type, ref_id, source_novel)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"ref_type": ref_type, "ref_id": ref_id, "source_novel": source_novel})
    return deduped


def _normalize_reassembled_event_record(record: Dict[str, Any]) -> Dict[str, Any]:
    # source_atom_ids is kept for backward compatibility, but it should only contain real Step2 atom ids.
    # Legacy event/chunk references are migrated into source_refs and the dedicated source_* fields below.
    payload = dict(record or {})
    payload.setdefault("source_atom_ids", [])
    payload.setdefault("source_induced_event_ids", [])
    payload.setdefault("source_legacy_event_ids", [])
    payload.setdefault("source_chunk_ids", [])
    payload.setdefault("source_refs", [])
    payload.setdefault("repair_notes", [])
    payload.setdefault("logic_notes", [])

    atom_ids: List[str] = []
    induced_event_ids = [str(item).strip() for item in payload.get("source_induced_event_ids", []) if str(item).strip()]
    legacy_event_ids = [str(item).strip() for item in payload.get("source_legacy_event_ids", []) if str(item).strip()]
    chunk_ids = [str(item).strip() for item in payload.get("source_chunk_ids", []) if str(item).strip()]
    source_refs = _dedupe_source_refs(list(payload.get("source_refs", []) or []))
    for ref in source_refs:
        ref_type = str(ref.get("ref_type", "") or "").strip()
        ref_id = str(ref.get("ref_id", "") or "").strip()
        if not ref_id:
            continue
        if ref_type == "atom":
            atom_ids.append(ref_id)
        elif ref_type == "induced_event" and ref_id not in induced_event_ids:
            induced_event_ids.append(ref_id)
        elif ref_type == "legacy_event" and ref_id not in legacy_event_ids:
            legacy_event_ids.append(ref_id)
        elif ref_type == "chunk" and ref_id not in chunk_ids:
            chunk_ids.append(ref_id)

    migrated_mixed_refs = False
    for raw_id in payload.get("source_atom_ids", []) or []:
        ref_id = str(raw_id or "").strip()
        if not ref_id:
            continue
        ref_type = _classify_source_ref_id(ref_id)
        if ref_type in ("", "atom"):
            atom_ids.append(ref_id)
            continue
        migrated_mixed_refs = True
        if ref_type == "induced_event" and ref_id not in induced_event_ids:
            induced_event_ids.append(ref_id)
        elif ref_type == "legacy_event" and ref_id not in legacy_event_ids:
            legacy_event_ids.append(ref_id)
        elif ref_type == "chunk" and ref_id not in chunk_ids:
            chunk_ids.append(ref_id)
        source_refs.append({"ref_type": ref_type, "ref_id": ref_id, "source_novel": ""})

    for ref_id in induced_event_ids:
        source_refs.append({"ref_type": "induced_event", "ref_id": ref_id, "source_novel": ""})
    for ref_id in legacy_event_ids:
        source_refs.append({"ref_type": "legacy_event", "ref_id": ref_id, "source_novel": ""})
    for ref_id in chunk_ids:
        source_refs.append({"ref_type": "chunk", "ref_id": ref_id, "source_novel": ""})
    for ref_id in atom_ids:
        source_refs.append({"ref_type": "atom", "ref_id": ref_id, "source_novel": ""})

    payload["source_atom_ids"] = _dedupe_texts(atom_ids)
    payload["source_induced_event_ids"] = _dedupe_texts(induced_event_ids)
    payload["source_legacy_event_ids"] = _dedupe_texts(legacy_event_ids)
    payload["source_chunk_ids"] = _dedupe_texts(chunk_ids)
    payload["source_refs"] = _dedupe_source_refs(source_refs)
    if migrated_mixed_refs:
        note = "migrated_legacy_source_ids_out_of_source_atom_ids"
        if note not in payload["repair_notes"]:
            payload["repair_notes"].append(note)
    return payload


def parse_event_index(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    matches = re.findall(r"(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])


def _supporting_character_priority(
    character: Character,
    preferred_names: Set[str],
    required_roles: Set[str],
    recent_names: Set[str],
    current_arc_name: str = "",
) -> Tuple[int, int, int, int, str]:
    role_slots = {str(slot).strip() for slot in character.role_slots if str(slot).strip()}
    char_arc_name = str(getattr(character, "primary_arc", "") or getattr(character, "arc_name", "") or "").strip()
    tier_score = 0
    if character.tier == "core":
        tier_score = 3
    elif character.tier == "volume":
        tier_score = 2
    elif character.tier == "transient":
        tier_score = 1
    return (
        int(character.name in preferred_names),
        int(bool(required_roles and (required_roles & role_slots))),
        int(character.name in recent_names),
        tier_score + int(bool(current_arc_name and char_arc_name == current_arc_name)),
        character.name,
    )


def _format_characters_for_stage(
    char_sheet: CharacterSheet,
    current_idx: int,
    role_plan: Optional[EventRolePlan],
    current_arc_name: str = "",
    recent_events: Optional[List[ReassembledEvent]] = None,
) -> Tuple[str, List[str]]:
    protagonist = char_sheet.protagonist
    lines = [f"[主角] {protagonist.name} | 执念: {protagonist.dao_heart} | 缺陷: {protagonist.personality_flaw} | 战斗: {protagonist.combat_style}"]
    active_names = [protagonist.name]
    preferred_names = {
        str(name).strip()
        for name in (role_plan.candidate_names if role_plan and role_plan.candidate_names else [])
        if str(name).strip()
    }
    required_roles = {
        str(role).strip()
        for role in (role_plan.role_slots if role_plan and role_plan.role_slots else [])
        if str(role).strip()
    }
    recent_names: Set[str] = set()
    for event in recent_events or []:
        for name in getattr(event, "active_characters", []) or []:
            text = str(name).strip()
            if text:
                recent_names.add(text)

    active_supporting: List[Character] = []
    for character in char_sheet.supporting:
        if _is_char_active(character, current_idx, current_arc_name=current_arc_name):
            active_supporting.append(character)

    if len(active_supporting) > 11:
        active_supporting = sorted(
            active_supporting,
            key=lambda item: _supporting_character_priority(item, preferred_names, required_roles, recent_names, current_arc_name),
            reverse=True,
        )[:11]

    for character in active_supporting:
        lines.append(f"[{character.role}] {character.name} | 功能槽位: {character.role_slots} | 背景: {character.background}")
        active_names.append(character.name)
    if role_plan and role_plan.candidate_names:
        prioritized = [name for name in role_plan.candidate_names if name in active_names]
        if prioritized:
            lines.append(f"[优先事件角色] {prioritized}")
    return "【当前事件可调角色池】\n" + "\n".join(lines), active_names


def _is_char_active(character: Character, current_idx: int, current_arc_name: str = "") -> bool:
    entry = parse_event_index(character.entry_event)
    exit_ = parse_event_index(character.exit_event)
    if entry is not None and current_idx < entry:
        return False
    if exit_ is not None and current_idx > exit_:
        return False
    char_arc_name = str(getattr(character, "primary_arc", "") or getattr(character, "arc_name", "") or "").strip()
    if current_arc_name and char_arc_name and char_arc_name != current_arc_name:
        return False
    return True


def _get_active_macro_stages(fused_world: FusedWorld, progress_ratio: float) -> str:
    lines: List[str] = []
    if fused_world.macro_tropes:
        trope = fused_world.macro_tropes[0]
        stages = trope.get("stages", [])
        if stages:
            stage_idx = min(int(progress_ratio * len(stages)), len(stages) - 1)
            lines.append(f"主套路: {trope.get('name', '')} -> 当前阶段 {stages[stage_idx]}")
    if fused_world.plot_threads:
        thread = fused_world.plot_threads[0]
        stages = thread.get("stages", [])
        if stages:
            stage_idx = min(int(progress_ratio * len(stages)), len(stages) - 1)
            lines.append(f"长线关系: {thread.get('name', '')} -> 当前阶段 {stages[stage_idx]}")
    return "\n".join(lines) if lines else "暂无额外长线约束。"


def _initialize_story_ledger() -> Dict[str, List[str]]:
    return {key: [] for key in _LEDGER_KEYS}


def _format_state_ledger(ledger: Dict[str, List[str]]) -> str:
    parts = [f"{key}: {values[-4:]}" for key, values in ledger.items() if values]
    return "\n".join(parts) if parts else "暂无前序状态负债。"


def _apply_state_updates(ledger: Dict[str, List[str]], state_updates: Dict[str, Any]) -> None:
    if not isinstance(state_updates, dict):
        return
    for key, raw_values in state_updates.items():
        if key not in ledger:
            ledger[key] = []
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        ledger[key].extend(str(value).strip() for value in values if str(value).strip())
        ledger[key] = ledger[key][-8:]


def _issue_to_dict(issue: StateIssue) -> Dict[str, Any]:
    return {
        "severity": issue.severity,
        "issue_type": issue.issue_type,
        "message": issue.message,
        "node_id": issue.node_id,
        "suggested_action": issue.suggested_action,
    }


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


def _stage_from_level_hint(level_hint: Any, stages: List[ProgressionStage]) -> str:
    if isinstance(level_hint, bool):
        return ""
    if isinstance(level_hint, (int, float)):
        numeric = int(level_hint)
        for stage in stages:
            if stage.stage_index == numeric:
                return stage.name
    return ""


def _next_stage_name(stage_name: str, stages: List[ProgressionStage]) -> str:
    ordered = _sorted_progression_stages(stages)
    position = _stage_position(stage_name, stages)
    if position < 0 or not ordered:
        return stage_name
    return ordered[min(len(ordered) - 1, position + 1)].name


def _build_event_allowed_state(
    node: SkeletonNode,
    previous_event: Optional[ReassembledEvent],
    fused_world: FusedWorld,
) -> tuple[Dict[str, Any], List[ProgressionStage]]:
    stages = build_progression_stages(fused_world)
    protagonist_stage = ""
    if previous_event:
        protagonist_stage = (
            str(previous_event.state_updates.get("power_state", "")).strip()
            if isinstance(previous_event.state_updates, dict)
            else ""
        )
        if not protagonist_stage:
            protagonist_stage = infer_stage_from_node(previous_event, stages)
    if not protagonist_stage:
        protagonist_stage = str((node.logic_card or {}).get("power_state", "") or "").strip()
    if not protagonist_stage:
        protagonist_stage = infer_stage_from_node(node, stages)
    if not protagonist_stage:
        protagonist_stage = _stage_from_level_hint(node.realm_level, stages)

    max_stage = protagonist_stage or _stage_from_level_hint(node.realm_level, stages)
    if not max_stage and stages:
        max_stage = _sorted_progression_stages(stages)[0].name

    forbidden_terms = build_forbidden_stage_terms(max_stage, stages, margin=1)
    if "凝气" in max_stage:
        forbidden_terms.extend(["金丹", "元婴", "结婴", "化神", "古神传承"])
    elif "筑基" in max_stage:
        forbidden_terms.extend(["元婴", "结婴", "化神", "古神传承"])

    deduped_terms: List[str] = []
    for term in forbidden_terms:
        clean = str(term or "").strip()
        if clean and clean not in deduped_terms:
            deduped_terms.append(clean)

    return {
        "max_realm": node.realm_level,
        "max_stage": max_stage,
        "stage_ceiling": max_stage,
        "protagonist_realm": protagonist_stage or max_stage,
        "allowed_opponent_stage": _next_stage_name(protagonist_stage or max_stage, stages),
        "forbidden_terms": deduped_terms,
    }, stages


def _collect_event_validation_issues(
    event: ReassembledEvent,
    fused_world: FusedWorld,
    allowed_state: Dict[str, Any],
    stages: List[ProgressionStage],
) -> List[StateIssue]:
    issues = list(validate_reassembled_events([event], fused_world))
    text_parts = [
        event.adapted_summary,
        json.dumps(event.event_plan, ensure_ascii=False) if isinstance(event.event_plan, dict) else str(event.event_plan or ""),
        json.dumps(event.state_updates, ensure_ascii=False) if isinstance(event.state_updates, dict) else str(event.state_updates or ""),
    ]
    combined_text = " ".join(part for part in text_parts if part).strip()
    max_stage = str(allowed_state.get("max_stage", "") or "").strip()
    max_position = _stage_position(max_stage, stages)
    protagonist_stage = str(allowed_state.get("protagonist_realm", "") or max_stage).strip()

    if max_stage and combined_text:
        detected_max = extract_max_stage_from_text(combined_text, stages)
        detected_position = _stage_position(detected_max, stages)
        if detected_max and max_position >= 0 and detected_position > max_position + 1:
            severity = "fatal" if detected_position > max_position + 2 else "major"
            issues.append(
                StateIssue(
                    severity=severity,
                    issue_type="event_stage_overflow",
                    message=f"Event references stage {detected_max}, which exceeds current allowed stage {max_stage}.",
                    node_id=event.event_id,
                    suggested_action="降低敌方/外部威胁阶段，或改写为压迫、逃脱、借力与外围线索。",
                )
            )

    current_stage = infer_stage_from_node(event, stages)
    current_position = _stage_position(current_stage, stages)
    if current_stage and max_position >= 0 and current_position > max_position:
        issues.append(
            StateIssue(
                severity="major",
                issue_type="event_protagonist_stage_overflow",
                message=f"Event protagonist stage {current_stage} exceeds allowed stage ceiling {max_stage}.",
                node_id=event.event_id,
                suggested_action="将事件的主角成长状态压回当前阶段上限。",
            )
        )

    for forbidden_term in allowed_state.get("forbidden_terms", []):
        if forbidden_term and forbidden_term in combined_text:
            issues.append(
                StateIssue(
                    severity="major",
                    issue_type="forbidden_stage_term",
                    message=f"Event text contains forbidden term beyond current stage band: {forbidden_term}.",
                    node_id=event.event_id,
                    suggested_action="替换超阶术语，或改写为当前阶段可接触的外围压力与线索。",
                )
            )
            break

    if protagonist_stage and stage_value(protagonist_stage, stages) == -1 and max_stage:
        issues.append(
            StateIssue(
                severity="minor",
                issue_type="unknown_protagonist_stage",
                message=f"Unable to normalize protagonist stage from allowed state: {protagonist_stage}.",
                node_id=event.event_id,
                suggested_action="补充 node.logic_card.power_state 或上一个事件的 power_state。",
            )
        )

    return issues


def _apply_rule_based_event_repairs(
    event: ReassembledEvent,
    allowed_state: Dict[str, Any],
    stages: List[ProgressionStage],
) -> bool:
    changed = False
    protagonist_stage = str(allowed_state.get("protagonist_realm", "") or allowed_state.get("max_stage", "")).strip()
    rewritten_summary = rewrite_overpowered_win_text_by_rule(event.adapted_summary, protagonist_stage, stages)
    if rewritten_summary and rewritten_summary != event.adapted_summary:
        event.adapted_summary = rewritten_summary
        event.repair_notes.append("rule_rewrite_overpowered_summary")
        changed = True

    if isinstance(event.event_plan, dict):
        for key in ("summary_seed", "outcome", "reversal", "cost"):
            value = event.event_plan.get(key)
            if isinstance(value, str):
                rewritten_value = rewrite_overpowered_win_text_by_rule(value, protagonist_stage, stages)
                if rewritten_value and rewritten_value != value:
                    event.event_plan[key] = rewritten_value
                    changed = True
        state_updates = event.event_plan.get("state_updates")
        if isinstance(state_updates, dict) and protagonist_stage:
            power_state = str(state_updates.get("power_state", "") or "").strip()
            if power_state and _stage_position(power_state, stages) > _stage_position(str(allowed_state.get("max_stage", "") or ""), stages):
                state_updates["power_state"] = allowed_state.get("max_stage", protagonist_stage)
                event.event_plan["state_updates"] = state_updates
                event.repair_notes.append("rule_cap_event_plan_power_state")
                changed = True

    if isinstance(event.state_updates, dict) and protagonist_stage:
        power_state = str(event.state_updates.get("power_state", "") or "").strip()
        if power_state and _stage_position(power_state, stages) > _stage_position(str(allowed_state.get("max_stage", "") or ""), stages):
            event.state_updates["power_state"] = allowed_state.get("max_stage", protagonist_stage)
            event.repair_notes.append("rule_cap_state_updates_power_state")
            changed = True

    return changed


def _apply_repaired_record_to_event(event: ReassembledEvent, record: Dict[str, Any]) -> ReassembledEvent:
    if not isinstance(record, dict):
        return event
    for field_name in ReassembledEvent.__dataclass_fields__:
        if field_name in record:
            setattr(event, field_name, record[field_name])
    if isinstance(record.get("summary"), str) and not event.adapted_summary:
        event.adapted_summary = record["summary"]
    return event


def _validate_and_repair_reassembled_event(
    event: ReassembledEvent,
    node: SkeletonNode,
    fused_world: FusedWorld,
    previous_event: Optional[ReassembledEvent],
) -> ReassembledEvent:
    allowed_state, stages = _build_event_allowed_state(node, previous_event, fused_world)
    issues = _collect_event_validation_issues(event, fused_world, allowed_state, stages)
    event.state_validation_issues = [_issue_to_dict(issue) for issue in issues]

    if any(issue.severity in {"fatal", "major"} for issue in issues):
        _apply_rule_based_event_repairs(event, allowed_state, stages)
        issues = _collect_event_validation_issues(event, fused_world, allowed_state, stages)
        if any(issue.severity in {"fatal", "major"} for issue in issues):
            repaired_events, _remaining = repair_reassembled_events(
                [event],
                world=fused_world,
                allowed_state=allowed_state,
                max_rounds=2,
            )
            if repaired_events:
                repaired_record = repaired_events[-1]
                event = _apply_repaired_record_to_event(event, repaired_record if isinstance(repaired_record, dict) else {})
            issues = _collect_event_validation_issues(event, fused_world, allowed_state, stages)

    event.state_validation_issues = [_issue_to_dict(issue) for issue in issues]
    if any(issue.severity == "fatal" for issue in issues):
        fatal_labels = [f"[{issue.issue_type}] {issue.message}" for issue in issues if issue.severity == "fatal"]
        event.logic_notes.extend(label for label in fatal_labels if label not in event.logic_notes)
        if "fatal_state_validation_after_repair" not in event.repair_notes:
            event.repair_notes.append("fatal_state_validation_after_repair")
        raise RuntimeError(
            f"Step11 aborted: event {event.event_id} still has fatal state issues after repair: "
            + " | ".join(fatal_labels)
        )
    return event


def _format_recent_story_context(recent_context: List[str]) -> str:
    if not recent_context:
        return "暂无前序事件摘要。"
    lines = [f"前序事件{i + 1}: {summary}" for i, summary in enumerate(recent_context[-3:])]
    return "\n".join(lines)


def _update_rolling_context(context_list: List[str], new_summary: str, max_history: int = 3) -> None:
    summary = str(new_summary or "").strip()
    if not summary:
        return
    context_list.append(summary)
    if len(context_list) > max_history:
        context_list.pop(0)


def _pick_micro_interaction(interactions: List[Dict[str, Any]], recent_micro_names: List[str]) -> Optional[Dict[str, Any]]:
    if not interactions:
        return None
    recent_name = recent_micro_names[-1] if recent_micro_names else ""
    candidates = [item for item in interactions if item.get("interaction_name", "") != recent_name]
    return random.choice(candidates or interactions)


def _update_recent_micro_names(recent_micro_names: List[str], used_trope: str, max_history: int = 3) -> None:
    trope = str(used_trope or "").strip()
    if not trope:
        return
    recent_micro_names.append(trope)
    if len(recent_micro_names) > max_history:
        recent_micro_names.pop(0)


def _build_anti_repetition_rules(recent_context: List[str], recent_micro_names: List[str]) -> str:
    repeated = [motif for motif in _GENERIC_REPEAT_MOTIFS if sum(1 for summary in recent_context[-3:] if motif in summary) >= 2]
    lines = [
        "Anti-repetition rules:",
        "- The next event must use a different obstacle, reversal, and payoff from the last 3 events.",
        "- Do not reuse the exact combo of trap -> clue spotting -> secret help -> sudden reversal.",
        f"- Recently used micro templates: {', '.join(recent_micro_names[-2:]) or 'none'}.",
    ]
    if repeated:
        lines.append("- Recently overused motifs to avoid: " + ", ".join(repeated))
    return "\n".join(lines)


def _select_event_template_brief(fused_world: FusedWorld, node: SkeletonNode, similar_events: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    if node.template_hint:
        lines.append(f"骨架模板: {node.template_hint}")
    if node.conflict_hint or node.function_hint:
        lines.append(f"骨架冲突/功能: {node.conflict_hint} / {node.function_hint}")
    executable_template = _match_executable_template(fused_world, node)
    if executable_template:
        lines.append(f"可执行模板: {executable_template.get('template_name', '')} | 层级 {executable_template.get('level', 'event')}")
        lines.append(f"模板功能: {executable_template.get('abstract_function', '')}")
        lines.append(f"冲突发动机: {executable_template.get('conflict_engine', '')}")
        lines.append("角色槽位: " + json.dumps(executable_template.get("role_slots", {}), ensure_ascii=False))
        preconditions = executable_template.get("required_preconditions", []) or []
        if preconditions:
            lines.append("触发前提: " + "；".join(str(item).strip() for item in preconditions[:4] if str(item).strip()))
        beat_lines = []
        for beat in executable_template.get("beat_sequence", [])[:4]:
            beat_index = beat.get("beat_index", "")
            beat_func = beat.get("function", "")
            beat_purpose = beat.get("purpose", "")
            beat_lines.append(f"{beat_index}.{beat_func}:{beat_purpose}")
        if beat_lines:
            lines.append("事件流程: " + " | ".join(beat_lines))
        state_delta = executable_template.get("state_delta", {}) or {}
        if state_delta:
            lines.append("状态变化: " + json.dumps(state_delta, ensure_ascii=False))
        variation_axes = executable_template.get("variation_axes", {}) or {}
        if variation_axes:
            compact_axes = {
                key: [str(item).strip() for item in values[:4]]
                for key, values in variation_axes.items()
                if isinstance(values, list) and values
            }
            lines.append("变形轴: " + json.dumps(compact_axes, ensure_ascii=False))
        forbidden = [str(item).strip() for item in executable_template.get("forbidden_source_details", []) if str(item).strip()]
        if forbidden:
            lines.append("禁用来源细节: " + "；".join(forbidden[:8]))
        lines.append("使用约束: 只能使用模板功能和槽位，不要复用 forbidden_source_details 中的具体设定。")
    for template in fused_world.event_templates or []:
        if node.conflict_hint and template.get("conflict_type") != node.conflict_hint:
            continue
        if node.function_hint and template.get("narrative_function") not in ("", node.function_hint):
            continue
        lines.append(f"事件模板参考: {template.get('name')} | 槽位 {template.get('role_slots', [])} | 发动机 {template.get('conflict_engines', [])}")
        break
    if similar_events:
        lines.append("检索参考: " + " | ".join(
            f"{item.get('metadata', {}).get('conflict_type', '')}/{item.get('metadata', {}).get('narrative_function', '')}"
            for item in similar_events[:2]
        ))
    return "\n".join(lines) if lines else "暂无模板提示。"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _plan_event(client, node: SkeletonNode, role_plan: Optional[EventRolePlan], characters_desc: str, active_macro_stages: str, template_brief: str, retrieved_events: List[Dict[str, Any]], state_brief: str, recent_story_brief: str, anti_repeat_rules: str) -> Dict[str, Any]:
    target_chapter_count = max(4, node.chapter_count or len(node.chapter_blueprint) or 4)
    prompt = (
        f"你是仙侠大纲规划师。\n"
        f"骨架节点: {node.original_summary}\n"
        f"当前节奏角色: {node.pacing_role}\n"
        f"建议章节数: {target_chapter_count}\n"
        f"章节蓝图: {json.dumps(node.chapter_blueprint or [], ensure_ascii=False)}\n"
        f"优先功能槽位: {(role_plan.role_slots if role_plan else node.role_slots)}\n"
        f"角色池:\n{characters_desc}\n"
        f"长线约束:\n{active_macro_stages}\n"
        f"前序事件:\n{recent_story_brief}\n"
        f"状态账本:\n{state_brief}\n"
        f"模板提示:\n{template_brief}\n"
        "只能抽象复用模板功能和角色槽位，禁止照搬禁用来源细节中的人物名、地点名、功法名、法宝名和标志性桥段。\n"
        f"检索灵感:\n{json.dumps(retrieved_events[:2], ensure_ascii=False)}\n"
        f"{anti_repeat_rules}\n"
        "请只返回 JSON，包含 story_purpose, target_chapter_count, trigger, goal, obstacle, choice, reversal, outcome, cost, required_roles, preconditions, state_updates, chapter_blueprint, summary_seed。"
    )
    raw = _quality_chat_completion_json(client, system="只返回合法 JSON。", user=prompt, json_mode=True, temperature=0.4)
    data = _safe_json_load(raw)
    if isinstance(data, dict) and data.get("goal") and data.get("obstacle"):
        data.setdefault("target_chapter_count", target_chapter_count)
        data.setdefault("chapter_blueprint", node.chapter_blueprint or [])
        data.setdefault("summary_seed", node.original_summary)
        data.setdefault("preconditions", [])
        data.setdefault("state_updates", _empty_state_updates())
        return data
    return _fallback_event_plan(node, role_plan)


def _fallback_event_plan(node: SkeletonNode, role_plan: Optional[EventRolePlan]) -> Dict[str, Any]:
    return {
        "story_purpose": node.pacing_role,
        "target_chapter_count": max(4, node.chapter_count or len(node.chapter_blueprint) or 4),
        "trigger": node.original_summary[:40],
        "goal": "推动当前主线并争取局部优势",
        "obstacle": node.conflict_hint or "当前势力和资源压制",
        "choice": "主动应对而不是被动承受",
        "reversal": "局面出现意外变化",
        "outcome": node.original_summary[:80],
        "cost": "付出资源、伤势或人情债",
        "required_roles": (role_plan.role_slots if role_plan else node.role_slots) or ["阻碍者", "短期盟友"],
        "preconditions": [],
        "state_updates": _empty_state_updates(),
        "chapter_blueprint": node.chapter_blueprint or [],
        "summary_seed": node.original_summary,
    }


def _empty_state_updates() -> Dict[str, List[str]]:
    return {key: [] for key in _LEDGER_KEYS}


def _match_executable_template(fused_world: FusedWorld, node: SkeletonNode) -> Dict[str, Any] | None:
    templates = fused_world.executable_templates or []
    if not templates:
        return None

    best_template: Dict[str, Any] | None = None
    best_score = -999.0
    template_hint = str(node.template_hint or "").strip()
    conflict_hint = str(node.conflict_hint or "").strip()
    function_hint = str(node.function_hint or "").strip()
    logic_template_name = str((node.logic_card or {}).get("template_name", "") or "").strip()

    for template in templates:
        score = 0.0
        haystack = " ".join(
            [
                str(template.get("template_name", "") or ""),
                str(template.get("abstract_function", "") or ""),
                str(template.get("conflict_engine", "") or ""),
                str(template.get("source_pattern_summary", "") or ""),
            ]
        )
        if function_hint and function_hint in haystack:
            score += 4
        if conflict_hint and conflict_hint in haystack:
            score += 4
        if template_hint and template_hint in haystack:
            score += 1
        if logic_template_name and logic_template_name in haystack:
            score += 1
        if node.pacing_role and node.pacing_role in haystack:
            score += 1
        if score > best_score:
            best_score = score
            best_template = template

    return best_template if best_score > 0 else None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _validate_or_repair_event_plan(
    client,
    node: SkeletonNode,
    event_plan: Dict[str, Any],
    characters_desc: str,
    state_brief: str,
    recent_story_brief: str,
    active_macro_stages: str,
    anti_repeat_rules: str,
) -> Tuple[Dict[str, Any], List[str]]:
    prompt = (
        f"你是剧情连续性审校器。\n"
        f"骨架节点: {node.original_summary}\n"
        f"前序事件:\n{recent_story_brief}\n"
        f"状态账本:\n{state_brief}\n"
        f"长线阶段:\n{active_macro_stages}\n"
        f"可用角色:\n{characters_desc}\n"
        f"{anti_repeat_rules}\n"
        f"待审事件计划: {json.dumps(event_plan, ensure_ascii=False)}\n"
        "请检查这份事件计划是否存在时间倒退、境界跳跃、身份错位、关系断裂、资源凭空出现、伏笔无承接等问题。"
        "只返回 JSON，字段包含 is_consistent, issues, corrected_plan。corrected_plan 必须保持原 schema。"
    )
    raw = _quality_chat_completion_json(client, system="只返回合法 JSON。", user=prompt, json_mode=True, temperature=0.2)
    data = _safe_json_load(raw)
    if not isinstance(data, dict):
        return event_plan, []
    issues = [str(item).strip() for item in data.get("issues", []) if str(item).strip()]
    corrected = data.get("corrected_plan")
    if isinstance(corrected, dict) and corrected.get("goal") and corrected.get("obstacle"):
        corrected.setdefault("target_chapter_count", event_plan.get("target_chapter_count", max(4, node.chapter_count or 4)))
        corrected.setdefault("chapter_blueprint", event_plan.get("chapter_blueprint", node.chapter_blueprint or []))
        corrected.setdefault("summary_seed", event_plan.get("summary_seed", node.original_summary))
        corrected.setdefault("preconditions", event_plan.get("preconditions", []))
        corrected.setdefault("state_updates", event_plan.get("state_updates", _empty_state_updates()))
        return corrected, issues
    return event_plan, issues


def _generate_candidates(event_plan: Dict[str, Any], node: SkeletonNode, characters_desc: str, active_macro_stages: str, suggested_micro: Optional[Dict[str, Any]], anti_repeat_rules: str, state_brief: str, recent_story_brief: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(_CANDIDATE_FOCUSES)) as executor:
        futures = [
            executor.submit(
                _generate_single_candidate,
                focus_name,
                focus_desc,
                event_plan,
                node,
                characters_desc,
                active_macro_stages,
                suggested_micro,
                anti_repeat_rules,
                state_brief,
                recent_story_brief,
            )
            for focus_name, focus_desc in _CANDIDATE_FOCUSES
        ]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:
                continue
    if not results:
        results.append({
            "focus": "fallback",
            "adapted_summary": event_plan.get("summary_seed", node.original_summary),
            "distinctive_engine": "fallback",
            "used_trope": (suggested_micro or {}).get("interaction_name", ""),
        })
    return results


def _generate_single_candidate(focus_name: str, focus_desc: str, event_plan: Dict[str, Any], node: SkeletonNode, characters_desc: str, active_macro_stages: str, suggested_micro: Optional[Dict[str, Any]], anti_repeat_rules: str, state_brief: str, recent_story_brief: str) -> Dict[str, Any]:
    client = get_deepseek_client()
    prompt = (
        f"当前事件骨架: {node.original_summary}\n"
        f"事件计划: {json.dumps(event_plan, ensure_ascii=False)}\n"
        f"前序事件:\n{recent_story_brief}\n"
        f"角色池:\n{characters_desc}\n"
        f"长线阶段:\n{active_macro_stages}\n"
        f"状态账本:\n{state_brief}\n"
        f"微模板: {json.dumps(suggested_micro or {}, ensure_ascii=False)}\n"
        f"候选重点: {focus_desc}\n"
        f"{anti_repeat_rules}\n"
        "请写一段 120 到 180 字的中文剧情摘要，必须符合当前状态账本与前序事件，不能出现时间倒退、无因跳跃。只返回 JSON，字段包含 adapted_summary 和 distinctive_engine。"
    )
    raw = _quality_chat_completion_json(client, system="只返回合法 JSON。", user=prompt, json_mode=True, temperature=0.75)
    data = _safe_json_load(raw)
    summary = data.get("adapted_summary") if isinstance(data, dict) else ""
    if not summary:
        summary = event_plan.get("summary_seed", node.original_summary)
    return {
        "focus": focus_name,
        "adapted_summary": summary,
        "distinctive_engine": data.get("distinctive_engine", focus_desc) if isinstance(data, dict) else focus_desc,
        "used_trope": (suggested_micro or {}).get("interaction_name", ""),
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _judge_candidates(client, node: SkeletonNode, event_plan: Dict[str, Any], candidates: List[Dict[str, Any]], anti_repeat_rules: str, state_brief: str, recent_story_brief: str) -> Tuple[int, str]:
    if len(candidates) == 1:
        return 0, candidates[0].get("adapted_summary", node.original_summary)
    prompt = (
        f"骨架节点: {node.original_summary}\n"
        f"事件计划: {json.dumps(event_plan, ensure_ascii=False)}\n"
        f"前序事件:\n{recent_story_brief}\n"
        f"状态账本:\n{state_brief}\n"
        f"{anti_repeat_rules}\n"
        f"候选列表: {json.dumps(candidates, ensure_ascii=False)}\n"
        "请选择最符合逻辑、最不重复、与前序状态最连贯的一版。只返回 JSON，字段包含 winner_index、adapted_summary、winner_reason。winner_index 从 1 开始。"
    )
    raw = _quality_chat_completion_json(client, system="只返回合法 JSON。", user=prompt, json_mode=True, temperature=0.2)
    data = _safe_json_load(raw)
    if isinstance(data, dict):
        winner_index = max(0, min(int(data.get("winner_index", 1)) - 1, len(candidates) - 1))
        summary = data.get("adapted_summary", candidates[winner_index].get("adapted_summary", node.original_summary))
        return winner_index, summary
    return 0, candidates[0].get("adapted_summary", node.original_summary)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _validate_or_repair_selected_summary(
    client,
    node: SkeletonNode,
    event_plan: Dict[str, Any],
    summary: str,
    state_brief: str,
    recent_story_brief: str,
    anti_repeat_rules: str,
) -> Tuple[str, List[str]]:
    prompt = (
        f"你是剧情连续性终审器。\n"
        f"骨架节点: {node.original_summary}\n"
        f"事件计划: {json.dumps(event_plan, ensure_ascii=False)}\n"
        f"前序事件:\n{recent_story_brief}\n"
        f"状态账本:\n{state_brief}\n"
        f"{anti_repeat_rules}\n"
        f"待审摘要: {summary}\n"
        "请检查这段摘要是否与前文逻辑连续。重点检查：时间线是否倒退，境界与身份是否跳跃，人物关系是否突然变化，资源/伤势/秘密是否凭空出现或消失。"
        "只返回 JSON，字段包含 is_consistent, issues, repaired_summary。若原摘要可用，repaired_summary 返回原文即可。"
    )
    raw = _quality_chat_completion_json(client, system="只返回合法 JSON。", user=prompt, json_mode=True, temperature=0.2)
    data = _safe_json_load(raw)
    if not isinstance(data, dict):
        return summary, []
    issues = [str(item).strip() for item in data.get("issues", []) if str(item).strip()]
    repaired = str(data.get("repaired_summary", "") or "").strip()
    if repaired:
        return repaired, issues
    return summary, issues


def _strip_meta_continuity_notes(summary: str) -> str:
    text = str(summary or "").strip()
    if not text:
        return ""
    patterns = [
        r"（[^）]*(?:按原文|时间线|错位|保留原文|注：)[^）]*）",
        r"\([^)]*(?:按原文|时间线|错位|retain|note)[^)]*\)",
        r"回忆插叙[:：]?",
        r"按原文[^，。；]*",
        r"时间线[^，。；]*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", "", text)
    return text.strip("，。；; ")


_safe_json_load = safe_json_load
