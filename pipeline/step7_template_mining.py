from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import config
from pipeline.core.common_json import write_json_file
from pipeline.core.world_building_core import (
    FusedWorld,
    _ROLE_SLOT_LIBRARY,
    _coerce_text,
    _dedupe_text_values,
    _flatten_text_values,
    _match_role_slot_template,
    _normalize_text,
    _top_values,
    load_world_snapshot,
    save_world_snapshot,
)
from pipeline.step2_extraction import PlotAtom
from pipeline.step3_event_induction import InducedEvent


_STEP7_FILENAME = "step7_template_mining.json"
_STEP7_CARDS_FILENAME = "step7_template_cards.json"


def derive_templates(
    all_atoms: Dict[str, List[PlotAtom]],
    fused_world: FusedWorld,
    induced_events_by_novel: Dict[str, List[InducedEvent]] | None = None,
) -> FusedWorld:
    all_atoms_flat = [atom for atoms in all_atoms.values() for atom in atoms]
    all_induced_events_flat = [event for events in (induced_events_by_novel or {}).values() for event in events]
    print("[Step 7] Deriving template cards...")
    fused_world.role_slot_templates = [dict(item) for item in _ROLE_SLOT_LIBRARY]
    fused_world.event_templates = _derive_event_templates(all_induced_events_flat, all_atoms_flat)
    fused_world.volume_templates = _derive_volume_templates(all_induced_events_flat, all_atoms_flat)
    fused_world.event_flow_templates = _derive_event_flow_templates(all_induced_events_flat, all_atoms_flat)
    return fused_world


def save_step7_output(fused_world: FusedWorld):
    path = save_world_snapshot(_STEP7_FILENAME, fused_world)
    cards_path = _save_template_cards_output(fused_world)
    print(f"[Step 7] Intermediate output saved -> {path.name}; template cards -> {cards_path.name}")
    return path


def load_step7_output() -> FusedWorld:
    return load_world_snapshot(_STEP7_FILENAME)


def _save_template_cards_output(fused_world: FusedWorld) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _STEP7_CARDS_FILENAME
    payload = {
        "role_slot_templates": fused_world.role_slot_templates,
        "event_templates": fused_world.event_templates,
        "volume_templates": fused_world.volume_templates,
        "event_flow_templates": fused_world.event_flow_templates,
    }
    return write_json_file(path, payload)


def _derive_event_templates(induced_events: List[InducedEvent], atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    if induced_events:
        return _derive_event_templates_from_induced_events(induced_events)
    return _derive_event_templates_from_atoms(atoms)


def _derive_event_templates_from_atoms(atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[PlotAtom]] = defaultdict(list)
    for atom in atoms:
        key = _normalize_text(atom.conflict_type or "default") + "|" + _normalize_text(atom.narrative_function or "progress")
        grouped[key].append(atom)

    templates: List[Dict[str, Any]] = []
    ranked = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)[:18]
    for index, (_key, bucket) in enumerate(ranked, start=1):
        first = bucket[0]
        combined_text = " ".join(_flatten_text_values([first.conflict_type, first.narrative_function, first.summary, first.core_action]))
        matched = _match_role_slot_template(combined_text)
        templates.append(
            {
                "template_id": f"event_template_{index}",
                "name": f"{_coerce_text(first.conflict_type) or '推进'}-{_coerce_text(first.narrative_function) or '事件'}",
                "conflict_type": _coerce_text(first.conflict_type),
                "narrative_function": _coerce_text(first.narrative_function),
                "role_slots": matched.get("role_slots", []),
                "conflict_engines": matched.get("conflict_engines", []),
                "sample_summaries": [atom.summary for atom in bucket[:3] if atom.summary],
            }
        )
    return templates


def _derive_event_templates_from_induced_events(events: List[InducedEvent]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[InducedEvent]] = defaultdict(list)
    for event in events:
        key = _normalize_text(event.conflict_hint or "default") + "|" + _normalize_text(event.function_hint or "progress")
        grouped[key].append(event)

    templates: List[Dict[str, Any]] = []
    ranked = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)[:18]
    for index, (_key, bucket) in enumerate(ranked, start=1):
        first = bucket[0]
        combined_text = " ".join(
            _flatten_text_values([first.conflict_hint, first.function_hint, first.summary, first.raw_summary, first.identity_state])
        )
        matched = _match_role_slot_template(combined_text)
        chapter_counts = [max(1, int(event.chapter_count or 1)) for event in bucket]
        templates.append(
            {
                "template_id": f"event_template_{index}",
                "name": f"{_coerce_text(first.conflict_hint) or '鎺ㄨ繘'}-{_coerce_text(first.function_hint) or '浜嬩欢'}",
                "conflict_type": _coerce_text(first.conflict_hint),
                "narrative_function": _coerce_text(first.function_hint),
                "role_slots": matched.get("role_slots", []),
                "conflict_engines": matched.get("conflict_engines", []),
                "chapter_count_range": [min(chapter_counts), max(chapter_counts)],
                "sample_summaries": [event.summary for event in bucket[:3] if event.summary],
            }
        )
    return templates


def _derive_volume_templates(induced_events: List[InducedEvent], atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    if induced_events:
        return _derive_volume_templates_from_induced_events(induced_events)
    return _derive_volume_templates_from_atoms(atoms)


def _derive_volume_templates_from_atoms(atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[PlotAtom]] = defaultdict(list)
    for atom in atoms:
        grouped[atom.arc_name or "Unnamed Arc"].append(atom)

    templates: List[Dict[str, Any]] = []
    for arc_name, bucket in grouped.items():
        dominant_conflicts = _top_values([atom.conflict_type for atom in bucket], limit=3)
        dominant_functions = _top_values([atom.narrative_function for atom in bucket], limit=3)
        combined_text = " ".join(filter(None, dominant_conflicts + dominant_functions + [atom.summary for atom in bucket[:3]]))
        matched = _match_role_slot_template(combined_text)
        templates.append(
            {
                "arc_name": arc_name,
                "event_count": len(bucket),
                "dominant_conflicts": dominant_conflicts,
                "dominant_functions": dominant_functions,
                "suggested_role_slots": matched.get("role_slots", []),
                "conflict_engines": matched.get("conflict_engines", []),
                "recommended_new_characters": max(3, min(8, len(bucket) // 2)),
                "sample_events": [atom.summary for atom in bucket[:4] if atom.summary],
            }
        )
    return templates


def _derive_volume_templates_from_induced_events(events: List[InducedEvent]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[InducedEvent]] = defaultdict(list)
    for event in events:
        grouped[event.arc_name or "Unnamed Arc"].append(event)

    templates: List[Dict[str, Any]] = []
    for arc_name, bucket in grouped.items():
        dominant_conflicts = _top_values([event.conflict_hint for event in bucket], limit=3)
        dominant_functions = _top_values([event.function_hint for event in bucket], limit=3)
        combined_text = " ".join(filter(None, dominant_conflicts + dominant_functions + [event.summary for event in bucket[:3]]))
        matched = _match_role_slot_template(combined_text)
        templates.append(
            {
                "arc_name": arc_name,
                "event_count": len(bucket),
                "dominant_conflicts": dominant_conflicts,
                "dominant_functions": dominant_functions,
                "suggested_role_slots": matched.get("role_slots", []),
                "conflict_engines": matched.get("conflict_engines", []),
                "recommended_new_characters": max(3, min(8, len(bucket) // 2)),
                "sample_events": [event.summary for event in bucket[:4] if event.summary],
                "start_event_index": 0,
                "end_event_index": max(0, len(bucket) - 1),
            }
        )
    return templates


def _derive_event_flow_templates(induced_events: List[InducedEvent], atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    if induced_events:
        return _derive_event_flow_templates_from_induced_events(induced_events)
    return _derive_event_flow_templates_from_atoms(atoms)


def _derive_event_flow_templates_from_atoms(atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[PlotAtom]] = defaultdict(list)
    for atom in atoms:
        grouped[atom.arc_name or "Unnamed Arc"].append(atom)

    windows: List[Dict[str, Any]] = []
    for bucket in grouped.values():
        ordered = sorted(bucket, key=lambda atom: (getattr(atom, "chapter_start", 0), getattr(atom, "chapter_end", 0), atom.atom_id))
        windows.extend(_build_atomic_windows(ordered))

    signature_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for window in windows:
        signature_groups[window["group_key"]].append(window)

    templates: List[Dict[str, Any]] = []
    ranked_groups = sorted(signature_groups.items(), key=lambda item: len(item[1]), reverse=True)[:24]
    for index, (_signature, items) in enumerate(ranked_groups, start=1):
        chapter_counts = [item["chapter_count"] for item in items]
        conflicts = _top_values([item["dominant_conflict"] for item in items], limit=3)
        aggregated_blueprint = _aggregate_beat_blueprint(items)
        templates.append(
            {
                "template_id": f"event_flow_{index}",
                "name": f"{(conflicts[0] if conflicts else 'generic')}_{min(chapter_counts)}-{max(chapter_counts)}_chapters",
                "chapter_count_range": [min(chapter_counts), max(chapter_counts)],
                "dominant_conflicts": conflicts,
                "required_coverage": _merge_required_coverage(items),
                "beat_blueprint": aggregated_blueprint,
                "source_window_count": len(items),
                "beat_count": len(aggregated_blueprint),
            }
        )

    templates.sort(key=lambda item: item.get("source_window_count", 0), reverse=True)
    return templates


def _derive_event_flow_templates_from_induced_events(events: List[InducedEvent]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[InducedEvent]] = defaultdict(list)
    for event in events:
        grouped[event.arc_name or "Unnamed Arc"].append(event)

    templates: List[Dict[str, Any]] = []
    template_index = 1
    for bucket in grouped.values():
        ordered = sorted(bucket, key=lambda event: (event.chapter_start, event.chapter_end, event.event_id))
        for event in ordered:
            beat_blueprint = _build_event_blueprint_from_induced_event(event)
            templates.append(
                {
                    "template_id": f"event_flow_{template_index}",
                    "name": f"{(_coerce_text(event.conflict_hint) or 'generic')}_{max(1, int(event.chapter_count or 1))}_chapters",
                    "chapter_count_range": [max(1, int(event.chapter_count or 1)), max(1, int(event.chapter_count or 1))],
                    "dominant_conflicts": _top_values([event.conflict_hint], limit=1),
                    "required_coverage": _derive_required_coverage(beat_blueprint),
                    "beat_blueprint": beat_blueprint,
                    "source_window_count": 1,
                    "beat_count": len(beat_blueprint),
                }
            )
            template_index += 1

    templates.sort(key=lambda item: (item.get("beat_count", 0), item.get("chapter_count_range", [0, 0])[0]), reverse=True)
    return templates[:24]


def _build_event_blueprint_from_induced_event(event: InducedEvent) -> List[Dict[str, Any]]:
    chapter_total = max(1, int(event.chapter_count or 1))
    if chapter_total == 1:
        primary_functions = ["涓荤嚎鎺ㄨ繘"]
    elif chapter_total == 2:
        primary_functions = ["閾哄灚鍏ュ満", "浣欐尝鏀舵潫"]
    elif chapter_total == 3:
        primary_functions = ["閾哄灚鍏ュ満", "鍐茬獊鍗囩骇", "浣欐尝鏀舵潫"]
    else:
        middle_count = max(1, chapter_total - 3)
        primary_functions = ["閾哄灚鍏ュ満"] + ["鍐茬獊鍗囩骇"] * middle_count + ["灞€鍔胯浆鎶?", "浣欐尝鏀舵潫"]
        primary_functions = primary_functions[:chapter_total]

    blueprint: List[Dict[str, Any]] = []
    for index, primary in enumerate(primary_functions, start=1):
        secondary: List[str] = []
        if index == 1 and event.identity_state:
            secondary.append("浜虹墿浜浉")
        if index == len(primary_functions) and event.hook_open:
            secondary.append("涓嬩竴浜嬩欢閽╁瓙")
        if index >= max(2, len(primary_functions) - 1) and event.resource_delta:
            secondary.append("浠ｄ环钀藉湴")
        if event.relationship_delta and 1 < index < len(primary_functions):
            secondary.append("鍏崇郴鎷夋壇")
        purpose_parts = [
            _coerce_text(event.conflict_hint) or "褰撳墠鍐茬獊",
            _coerce_text(event.function_hint) or "涓荤嚎鎺ㄨ繘",
        ]
        blueprint.append(
            {
                "beat_index": index,
                "chapter_span": [index, index],
                "chapter_share": round(1 / max(chapter_total, 1), 3),
                "primary_function": primary,
                "secondary_functions": _dedupe_text_values(secondary),
                "purpose": f"鍥寸粫{purpose_parts[0]}鎵挎媴{primary}锛屽苟鎺ㄥ姩{purpose_parts[1]}",
            }
        )
    return blueprint


def _build_atomic_windows(atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    start = 0
    while start < len(atoms):
        current: List[PlotAtom] = []
        index = start
        while index < len(atoms):
            atom = atoms[index]
            current.append(atom)
            chapter_total = _window_chapter_total(current)
            should_close = chapter_total >= 4 and (
                chapter_total >= 10 or _atom_has_event_boundary(atom) or _next_atom_resets_pressure(atoms, index)
            )
            if should_close or chapter_total >= 12:
                break
            index += 1
        if current:
            windows.append(_window_to_flow_record(current))
        start += max(1, len(current) // 2 or 1)
    return windows


def _window_to_flow_record(window: List[PlotAtom]) -> Dict[str, Any]:
    beat_blueprint: List[Dict[str, Any]] = []
    relative_spans = _relative_chapter_spans_from_atoms(window)
    dominant_conflict = _top_values([atom.conflict_type for atom in window], limit=1)
    chapter_total = _window_chapter_total(window)
    for beat_index, atom in enumerate(window, start=1):
        primary, secondary = _infer_chapter_functions(window, beat_index - 1)
        span = relative_spans[beat_index - 1]
        span_len = max(1, int(span[1]) - int(span[0]) + 1)
        beat_blueprint.append(
            {
                "beat_index": beat_index,
                "chapter_span": span,
                "chapter_share": round(span_len / max(chapter_total, 1), 3),
                "primary_function": primary,
                "secondary_functions": secondary,
                "purpose": _build_blueprint_purpose(atom, primary, secondary),
            }
        )
    beat_count = len(beat_blueprint)
    return {
        "signature": "|".join(item["primary_function"] for item in beat_blueprint),
        "group_key": f"{dominant_conflict[0] if dominant_conflict else 'generic'}|{beat_count}|{'|'.join(item['primary_function'] for item in beat_blueprint)}",
        "chapter_count": chapter_total,
        "dominant_conflict": dominant_conflict[0] if dominant_conflict else "generic progression",
        "required_coverage": _derive_required_coverage(beat_blueprint),
        "beat_blueprint": beat_blueprint,
    }


def _window_chapter_total(window: List[PlotAtom]) -> int:
    covered: set[int] = set()
    for atom in window:
        start = int(getattr(atom, "chapter_start", 0) or 0)
        end = int(getattr(atom, "chapter_end", 0) or start or 0)
        if start and end and end >= start:
            covered.update(range(start, end + 1))
        else:
            covered.add(len(covered) + 1)
    return max(1, len(covered))


def _atom_chapter_span(atom: PlotAtom) -> int:
    start = int(getattr(atom, "chapter_start", 0) or 0)
    end = int(getattr(atom, "chapter_end", 0) or start or 0)
    if start and end and end >= start:
        return max(1, end - start + 1)
    return max(1, int(getattr(atom, "chapter_count", 1) or 1))


def _relative_chapter_spans_from_atoms(window: List[PlotAtom]) -> List[List[int]]:
    covered = sorted(
        {
            chapter_no
            for atom in window
            for chapter_no in range(
                int(getattr(atom, "chapter_start", 0) or 0),
                int(getattr(atom, "chapter_end", getattr(atom, "chapter_start", 0)) or getattr(atom, "chapter_start", 0) or 0) + 1,
            )
            if chapter_no
        }
    )
    if not covered:
        cursor = 1
        spans: List[List[int]] = []
        for atom in window:
            span = _atom_chapter_span(atom)
            spans.append([cursor, cursor + span - 1])
            cursor += span
        return spans

    index_map = {chapter_no: index + 1 for index, chapter_no in enumerate(covered)}
    spans: List[List[int]] = []
    for atom in window:
        start = int(getattr(atom, "chapter_start", 0) or 0)
        end = int(getattr(atom, "chapter_end", 0) or start or 0)
        if start and end and start in index_map and end in index_map:
            spans.append([index_map[start], index_map[end]])
        else:
            fallback = spans[-1][1] if spans else 0
            spans.append([fallback + 1, fallback + _atom_chapter_span(atom)])
    return spans


def _infer_chapter_functions(window: List[PlotAtom], index: int) -> tuple[str, List[str]]:
    atom = window[index]
    total = len(window)
    position = index / max(total - 1, 1)
    text = " ".join(_flatten_text_values([atom.narrative_function, atom.conflict_type, atom.summary, atom.causality_consequence, atom.emotion]))
    tags: List[str] = []
    if index == 0:
        tags.extend(["铺垫入场", "人物亮相"])
    if position < 0.35:
        tags.append("试探推进")
    if atom.tension_level >= 7 or any(keyword in text for keyword in ["冲突", "争夺", "围杀", "试炼", "追杀"]):
        tags.append("冲突升级")
    if any(keyword in text for keyword in ["反转", "逆袭", "翻盘", "揭露", "破局", "真相"]):
        tags.append("局势转折")
    if any(keyword in text for keyword in ["代价", "重伤", "暴露", "损失", "后果", "牺牲"]):
        tags.append("代价落地")
    if index == total - 1:
        tags.append("余波收束")
    if index == total - 1 or any(keyword in text for keyword in ["伏笔", "线索", "后续", "隐患", "更大", "下一"]):
        tags.append("下一事件钩子")
    ordered = _dedupe_and_rank_tags(tags)
    primary = ordered[0] if ordered else "主线推进"
    return primary, ordered[1:]


def _build_blueprint_purpose(atom: PlotAtom, primary: str, secondary: List[str]) -> str:
    conflict = _coerce_text(atom.conflict_type) or "当前冲突"
    function = _coerce_text(atom.narrative_function) or "主线推进"
    suffix = f"，兼顾{'/'.join(secondary)}" if secondary else ""
    return f"围绕{conflict}完成{primary}，并推动{function}{suffix}"


def _derive_required_coverage(beat_blueprint: List[Dict[str, Any]]) -> List[str]:
    labels = []
    all_tags = [beat.get("primary_function", "") for beat in beat_blueprint]
    all_tags.extend(tag for beat in beat_blueprint for tag in beat.get("secondary_functions", []))
    if any("转折" in tag for tag in all_tags):
        labels.append("局势转折")
    if any("代价" in tag for tag in all_tags):
        labels.append("代价落地")
    if any("钩子" in tag for tag in all_tags):
        labels.append("下一事件钩子")
    if any("冲突" in tag for tag in all_tags):
        labels.append("冲突升级")
    if any("铺垫" in tag or "试探" in tag for tag in all_tags):
        labels.append("铺垫推进")
    return _dedupe_text_values(labels)


def _merge_required_coverage(items: List[Dict[str, Any]]) -> List[str]:
    merged: List[str] = []
    for item in items:
        merged.extend(item.get("required_coverage", []))
    return _dedupe_text_values(merged)


def _aggregate_beat_blueprint(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return []
    beat_count = max(len(item.get("beat_blueprint", [])) for item in items)
    aggregated: List[Dict[str, Any]] = []
    for beat_index in range(beat_count):
        beat_variants = [item["beat_blueprint"][beat_index] for item in items if beat_index < len(item.get("beat_blueprint", []))]
        if not beat_variants:
            continue
        primary = _majority_primary_function(beat_variants)
        secondary = _top_secondary_functions(beat_variants)
        avg_share = sum(float(beat.get("chapter_share", 0.0) or 0.0) for beat in beat_variants) / max(len(beat_variants), 1)
        purpose = _aggregate_purpose_texts(beat_variants, primary, secondary)
        aggregated.append(
            {
                "beat_index": beat_index + 1,
                "chapter_span": [beat_index + 1, beat_index + 1],
                "chapter_share": round(avg_share, 3),
                "primary_function": primary,
                "secondary_functions": secondary,
                "purpose": purpose,
            }
        )
    return aggregated


def _majority_primary_function(beat_variants: List[Dict[str, Any]]) -> str:
    values = [str(beat.get("primary_function", "")).strip() for beat in beat_variants if str(beat.get("primary_function", "")).strip()]
    if not values:
        return "主线推进"
    ranked = _top_values(values, limit=1)
    return ranked[0] if ranked else values[0]


def _top_secondary_functions(beat_variants: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    values: List[str] = []
    for beat in beat_variants:
        values.extend(str(item).strip() for item in beat.get("secondary_functions", []) if str(item).strip())
    return _top_values(values, limit=limit)


def _aggregate_purpose_texts(beat_variants: List[Dict[str, Any]], primary: str, secondary: List[str]) -> str:
    purposes = _top_values([beat.get("purpose", "") for beat in beat_variants], limit=2)
    if purposes:
        base = purposes[0]
        if len(base) > 60:
            base = base[:60].rstrip("，。；; ")
        return base
    suffix = f"，兼顾{'/'.join(secondary)}" if secondary else ""
    return f"承担{primary}{suffix}"


def _atom_has_event_boundary(atom: PlotAtom) -> bool:
    text = " ".join(_flatten_text_values([atom.narrative_function, atom.summary, atom.causality_consequence, atom.emotion]))
    if any(keyword in text for keyword in ["转折", "翻盘", "余波", "收束", "伏笔", "隐患", "后果", "钩子"]):
        return True
    return int(getattr(atom, "tension_level", 5) or 5) >= 8


def _next_atom_resets_pressure(atoms: List[PlotAtom], index: int) -> bool:
    if index + 1 >= len(atoms):
        return True
    next_atom = atoms[index + 1]
    text = " ".join(_flatten_text_values([next_atom.narrative_function, next_atom.summary]))
    if any(keyword in text for keyword in ["开篇", "铺垫", "入场", "试探"]):
        return True
    return int(getattr(next_atom, "tension_level", 5) or 5) <= 4


def _dedupe_and_rank_tags(tags: List[str]) -> List[str]:
    priority = {
        "铺垫入场": 0,
        "试探推进": 1,
        "冲突升级": 2,
        "局势转折": 3,
        "代价落地": 4,
        "余波收束": 5,
        "下一事件钩子": 6,
        "人物亮相": 7,
        "主线推进": 8,
    }
    unique = _dedupe_text_values(tags)
    return sorted(unique, key=lambda item: priority.get(item, 99))
