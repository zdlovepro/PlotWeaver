from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import config
from pipeline.core.artifacts import (
    ExecutableTemplate,
    pipeline_state_snapshot_from_fused_world,
    save_json_artifact,
    template_mining_from_fused_world,
)
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
_STEP7_TEMPLATES_FILENAME = "step7_templates.json"
_STEP7_PIPELINE_STATE_FILENAME = "pipeline_state_after_step7.json"

_GENERIC_SOURCE_DETAILS = {
    "主角",
    "反派",
    "长老",
    "弟子",
    "宗门",
    "家族",
    "城池",
    "秘境",
    "遗迹",
    "传承",
    "功法",
    "法宝",
    "丹药",
    "资源",
    "试炼",
    "考核",
}

_ROLE_SLOT_SEMANTIC_MAP: Dict[str, Dict[str, str]] = {
    "trial_competition": {
        "protagonist": "被压制者/破局者",
        "oppressor": "竞争对手或规则利用者",
        "authority": "裁决者/守关者",
        "witness": "旁观者或舆论放大者",
        "ally": "有限支援者",
    },
    "market_transaction": {
        "protagonist": "争取资源者/议价者",
        "oppressor": "抬价者或截胡者",
        "broker": "掌柜主事或中介",
        "valuator": "识货者/真伪判断者",
        "witness": "维持秩序者或场外观察者",
    },
    "secret_realm": {
        "protagonist": "进入者/破局者",
        "guardian": "守护者或门槛设定者",
        "rival": "争夺者",
        "ally": "临时盟友",
        "guide": "引路者或信息持有者",
    },
    "hunt_escape": {
        "protagonist": "逃亡者/破局者",
        "pursuer": "追杀者",
        "hunter_support": "线人或封锁协助者",
        "ally": "掩护者",
        "shelter": "收容者或退路提供者",
    },
    "inheritance_mentor": {
        "protagonist": "求法者/继承候选",
        "mentor": "试探前辈或传承持有者",
        "rival": "同门竞争者",
        "guardian": "护道者",
        "authority": "规则维护者",
    },
    "revenge_rivalry": {
        "protagonist": "清算者/承压者",
        "oppressor": "宿敌或宿敌代理人",
        "witness": "见证者",
        "mediator": "调停失败者或失控维稳者",
        "ally": "有限支援者",
    },
    "emotion_misalignment": {
        "protagonist": "关系修复者/被误解者",
        "counterpart": "误解对象或立场相左者",
        "agitator": "挑拨者",
        "mediator": "撮合者",
        "ally": "共患难者",
    },
    "default_pressure": {
        "protagonist": "承压者/破局者",
        "oppressor": "主要阻碍者",
        "ally": "短期盟友",
        "witness": "旁观者或放大器",
        "source": "消息提供者",
    },
}

_LEGACY_BEAT_NAME_MAP = {
    "閾哄灚鍏ュ満": "铺垫入场",
    "璇曟帰鎺ㄨ繘": "试探推进",
    "鍐茬獊鍗囩骇": "冲突升级",
    "灞€鍔胯浆鎶?": "局势转折",
    "浠ｄ环钀藉湴": "代价落地",
    "浣欐尝鏀舵潫": "余波收束",
    "涓嬩竴浜嬩欢閽╁瓙": "下一事件钩子",
    "浜虹墿浜浉": "人物亮相",
    "涓荤嚎鎺ㄨ繘": "主线推进",
}


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
    fused_world.executable_templates = _derive_executable_templates(
        induced_events=all_induced_events_flat,
        atoms=all_atoms_flat,
        event_flow_templates=fused_world.event_flow_templates,
    )
    return fused_world


def save_step7_output(fused_world: FusedWorld):
    legacy_path = save_world_snapshot(_STEP7_FILENAME, fused_world)
    templates_path = _save_template_artifact(_STEP7_TEMPLATES_FILENAME, fused_world)
    cards_path = _save_template_artifact(_STEP7_CARDS_FILENAME, fused_world)
    pipeline_state_path = save_json_artifact(
        Path(config.INTERMEDIATE_DIR) / _STEP7_PIPELINE_STATE_FILENAME,
        pipeline_state_snapshot_from_fused_world(
            fused_world,
            metadata={
                "step": 7,
                "artifact_type": "pipeline_state_snapshot",
                "primary_step_artifact": _STEP7_TEMPLATES_FILENAME,
            },
        ),
    )
    print(
        f"[Step 7] Intermediate output saved -> {legacy_path.name}; "
        f"templates -> {templates_path.name}; pipeline state -> {pipeline_state_path.name}; "
        f"template cards -> {cards_path.name}"
    )
    return legacy_path


def load_step7_output() -> FusedWorld:
    return load_world_snapshot(_STEP7_FILENAME)


def _save_template_artifact(filename: str, fused_world: FusedWorld) -> Path:
    return save_json_artifact(Path(config.INTERMEDIATE_DIR) / filename, template_mining_from_fused_world(fused_world))


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
                "name": f"{_coerce_text(first.conflict_type) or '主线推进'}-{_coerce_text(first.narrative_function) or '事件'}",
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
                "name": f"{_coerce_text(first.conflict_hint) or '主线推进'}-{_coerce_text(first.function_hint) or '事件'}",
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
                "name": f"{(conflicts[0] if conflicts else '通用冲突')}_{min(chapter_counts)}-{max(chapter_counts)}章流程",
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
                    "name": f"{(_coerce_text(event.conflict_hint) or '通用冲突')}_{max(1, int(event.chapter_count or 1))}章流程",
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


def _derive_executable_templates(
    induced_events: List[InducedEvent],
    atoms: List[PlotAtom],
    event_flow_templates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sources = _collect_template_sources(induced_events, atoms)
    if not sources:
        return []

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for source in sources:
        key = _normalize_text(source.get("conflict_hint", "") or "default") + "|" + _normalize_text(source.get("function_hint", "") or "progress")
        grouped[key].append(source)

    executable_templates: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    ranked_groups = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)[:18]
    for index, (_group_key, bucket) in enumerate(ranked_groups, start=1):
        payload = _build_executable_template_from_event_group(index, bucket, event_flow_templates)
        template_name = str(payload.get("template_name", "")).strip()
        if not template_name or template_name in seen_names:
            continue
        seen_names.add(template_name)
        executable_templates.append(payload)
    return executable_templates


def _build_executable_template_from_event_group(
    index: int,
    event_group: List[Dict[str, Any]],
    event_flow_templates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    abstract_function = _infer_abstract_function(event_group)
    conflict_engine = _infer_conflict_engine(event_group)
    beat_sequence = _infer_beat_sequence(event_group, event_flow_templates, abstract_function, conflict_engine)
    state_delta = _infer_state_delta(event_group, abstract_function, conflict_engine, beat_sequence)
    payload = ExecutableTemplate(
        template_id=f"executable_template_{index}",
        template_name=_build_template_name(abstract_function, conflict_engine),
        level=_infer_template_level(event_group, abstract_function),
        abstract_function=abstract_function,
        conflict_engine=conflict_engine,
        required_preconditions=_infer_required_preconditions(event_group, abstract_function, conflict_engine),
        role_slots=_infer_role_slots(event_group, abstract_function, conflict_engine),
        beat_sequence=beat_sequence,
        state_delta=state_delta,
        variation_axes=_infer_variation_axes(event_group, abstract_function, conflict_engine),
        forbidden_source_details=_extract_forbidden_source_details(event_group),
        source_refs=_dedupe_text_values([source.get("source_ref", "") for source in event_group])[:8],
        source_pattern_summary=_build_source_pattern_summary(event_group, abstract_function, conflict_engine, beat_sequence, state_delta),
    )
    return payload.__dict__


def _collect_template_sources(induced_events: List[InducedEvent], atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    atoms_by_id = {atom.atom_id: atom for atom in atoms}
    if induced_events:
        return [_template_source_from_event(event, atoms_by_id) for event in induced_events]
    return [_template_source_from_atom(atom) for atom in atoms]


def _template_source_from_event(event: InducedEvent, atoms_by_id: Dict[str, PlotAtom]) -> Dict[str, Any]:
    source_atoms = [atoms_by_id[atom_id] for atom_id in event.source_atom_ids if atom_id in atoms_by_id]
    raw_locations = _dedupe_text_values([getattr(atom, "raw_location", "") or atom.location for atom in source_atoms])
    cultivation_elements = _dedupe_text_values([item for atom in source_atoms for item in getattr(atom, "cultivation_elements", [])])
    raw_characters = _dedupe_text_values(
        [item for atom in source_atoms for item in (getattr(atom, "raw_characters", []) or getattr(atom, "characters", []))]
    ) or event.raw_characters[:]
    summaries = [event.raw_summary or event.summary] + [atom.raw_summary or atom.summary for atom in source_atoms]
    return {
        "source_id": event.event_id,
        "source_ref": f"{event.novel_source}:{event.event_id}",
        "novel_source": event.novel_source,
        "arc_name": event.arc_name,
        "summary": event.summary,
        "raw_summary": event.raw_summary or event.summary,
        "conflict_hint": event.conflict_hint,
        "function_hint": event.function_hint,
        "chapter_count": max(1, int(event.chapter_count or 1)),
        "character_keys": event.character_keys[:],
        "raw_characters": raw_characters,
        "raw_locations": raw_locations,
        "cultivation_elements": cultivation_elements,
        "state_inputs": event.state_inputs[:],
        "state_outputs": event.state_outputs[:],
        "relationship_delta": event.relationship_delta[:],
        "resource_delta": event.resource_delta[:],
        "hook_open": event.hook_open[:],
        "identity_state": event.identity_state,
        "sample_summaries": _dedupe_text_values([event.summary] + [atom.summary for atom in source_atoms if atom.summary])[:3],
        "source_summaries": _dedupe_text_values(summaries)[:6],
    }


def _template_source_from_atom(atom: PlotAtom) -> Dict[str, Any]:
    return {
        "source_id": atom.atom_id,
        "source_ref": f"{atom.novel_source}:{atom.atom_id}",
        "novel_source": atom.novel_source,
        "arc_name": atom.arc_name,
        "summary": atom.summary,
        "raw_summary": atom.raw_summary or atom.summary,
        "conflict_hint": atom.conflict_type,
        "function_hint": atom.narrative_function,
        "chapter_count": max(1, int(getattr(atom, "chapter_count", 1) or 1)),
        "character_keys": atom.character_keys[:],
        "raw_characters": atom.raw_characters[:] or atom.characters[:],
        "raw_locations": _dedupe_text_values([atom.raw_location or atom.location]),
        "cultivation_elements": atom.cultivation_elements[:],
        "state_inputs": _dedupe_text_values([atom.causality_precondition, atom.motivation])[:4],
        "state_outputs": _dedupe_text_values([atom.causality_consequence, atom.core_action, atom.summary])[:4],
        "relationship_delta": [f"相关角色关系转向{atom.emotion}"] if atom.emotion and len(atom.character_keys) >= 2 else [],
        "resource_delta": [item for item in atom.cultivation_elements if any(keyword in item for keyword in ("法宝", "丹", "药", "灵石", "功法", "传承", "令牌"))][:4],
        "hook_open": [atom.causality_consequence] if atom.causality_consequence else [],
        "identity_state": "",
        "sample_summaries": [atom.summary] if atom.summary else [],
        "source_summaries": _dedupe_text_values([atom.raw_summary or atom.summary])[:3],
    }


def _infer_abstract_function(event_group: List[Dict[str, Any]]) -> str:
    text = _group_haystack(event_group)
    if _contains_any(text, ["逆袭", "打脸", "翻盘", "力压", "证明", "扬名", "洗刷", "反击"]):
        if _contains_any(text, ["审判", "裁决", "考核", "大比", "听证", "公开", "众目", "擂台", "验货", "审查"]):
            return "公开反转"
        return "低位逆袭"
    if _contains_any(text, ["追杀", "逃亡", "围杀", "围堵", "追捕", "脱身", "通缉"]):
        return "逃亡破局"
    if _contains_any(text, ["秘境", "遗迹", "机缘", "宝物", "拍卖", "争夺"]):
        return "机缘争夺"
    if _contains_any(text, ["传承", "拜师", "授法", "道统"]):
        if _contains_any(text, ["筛选", "资格", "试炼", "考核"]):
            return "传承筛选"
        return "传承确认"
    if _contains_any(text, ["复仇", "宿敌", "旧怨", "清算", "报复"]):
        return "旧怨升级"
    if _contains_any(text, ["炼丹", "炼器", "异象", "失控", "反噬", "爆炉"]):
        return "技术失控"
    if _contains_any(text, ["误会", "立场", "互救", "婚约", "心结", "背叛", "同盟"]):
        return "关系转折"
    if _contains_any(text, ["审问", "裁决", "执法", "规矩", "审查", "听证"]):
        return "规则压迫"
    if _contains_any(text, ["争夺", "资源", "资格", "名额", "拍卖", "竞价"]):
        return "资源争夺"
    return "主线推进"


def _infer_conflict_engine(event_group: List[Dict[str, Any]]) -> str:
    text = _group_haystack(event_group)
    if _contains_any(text, ["审问", "裁决", "执法", "规矩", "听证", "资格审查"]):
        return "规则压迫"
    if _contains_any(text, ["考核", "大比", "比斗", "选拔", "资格", "试炼"]):
        return "规则竞争"
    if _contains_any(text, ["机缘", "宝物", "拍卖", "竞价", "资源", "令牌"]):
        return "资源竞争"
    if _contains_any(text, ["追杀", "围杀", "追捕", "通缉", "围堵"]):
        return "围追堵截"
    if _contains_any(text, ["复仇", "宿敌", "旧怨", "家族", "清算"]):
        return "旧怨升级"
    if _contains_any(text, ["秘密", "真相", "线索", "隐瞒", "卧底", "潜伏"]):
        return "信息差"
    if _contains_any(text, ["身份", "暴露", "揭穿", "冒名"]):
        return "身份暴露"
    if _contains_any(text, ["炼丹", "炼器", "异象", "失控", "反噬"]):
        return "技术风险"
    matched = _match_role_slot_template(text)
    engines = matched.get("conflict_engines", [])
    return str(engines[0]).strip() if engines else "局势施压"


def _infer_required_preconditions(event_group: List[Dict[str, Any]], abstract_function: str, conflict_engine: str) -> List[str]:
    preconditions: List[str] = []
    text = _group_haystack(event_group)

    if abstract_function in {"公开反转", "低位逆袭"}:
        preconditions.extend(["主角处于低位、被质疑或被压制", "对手拥有规则、身份或资源优势"])
    if abstract_function == "逃亡破局":
        preconditions.extend(["主角已被锁定、通缉或围堵", "存在必须立刻脱身的高压场景"])
    if abstract_function in {"机缘争夺", "资源争夺"}:
        preconditions.extend(["场景中存在稀缺资源、资格或入场门槛", "至少两方目标一致且资源不可共享"])
    if abstract_function in {"传承筛选", "传承确认"}:
        preconditions.extend(["前文已埋下传承、师承或道统线索", "主角具备被考验或被筛选的资格"])
    if abstract_function == "关系转折":
        preconditions.extend(["关键关系已形成信任裂缝、误会或利益错位", "双方仍有继续接触或合作的必要"])
    if abstract_function == "技术失控":
        preconditions.extend(["前文已埋下技术、法宝或资源使用风险", "主角必须在压力下继续操作或承担后果"])

    if conflict_engine in {"规则压迫", "规则竞争"} and not any("规则" in item for item in preconditions):
        preconditions.append("存在公开裁决、考核、审查或规则执行场景")
    if conflict_engine in {"信息差", "身份暴露"}:
        preconditions.append("前文已埋下隐藏能力、秘密或身份信息差")
    if conflict_engine == "围追堵截":
        preconditions.append("追击方已形成封锁、悬赏或人数优势")

    preconditions.extend(_collect_logic_fragments(event_group, "state_inputs", limit=2))
    if _contains_any(text, ["隐藏", "底牌", "秘密", "线索"]) and not any("信息差" in item or "秘密" in item for item in preconditions):
        preconditions.append("主角已有隐藏能力、秘密或可反用的线索")
    return _dedupe_text_values(preconditions)[:6]


def _infer_role_slots(event_group: List[Dict[str, Any]], abstract_function: str, conflict_engine: str) -> Dict[str, str]:
    matched = _match_role_slot_template(_group_haystack(event_group))
    slots = dict(_ROLE_SLOT_SEMANTIC_MAP.get(matched.get("name", "default_pressure"), _ROLE_SLOT_SEMANTIC_MAP["default_pressure"]))

    if abstract_function in {"公开反转", "低位逆袭"}:
        slots.update(
            {
                "protagonist": "被压制者/破局者",
                "oppressor": "规则利用者或公开打压者",
                "witness": "舆论放大者或围观者",
            }
        )
    if conflict_engine in {"规则压迫", "规则竞争"}:
        slots.setdefault("authority", "裁决者或规则执行者")
    if conflict_engine == "信息差":
        slots.setdefault("source", "掌握情报或秘密的人")
    if conflict_engine == "围追堵截":
        slots.setdefault("pursuer", "追击者")
    if abstract_function == "关系转折":
        slots.setdefault("counterpart", "关系对侧或立场摇摆者")
    return slots


def _infer_beat_sequence(
    event_group: List[Dict[str, Any]],
    event_flow_templates: List[Dict[str, Any]],
    abstract_function: str,
    conflict_engine: str,
) -> List[Dict[str, Any]]:
    matched_flow_template = _find_matching_flow_template(event_group, event_flow_templates)
    beat_sequence = _normalize_beat_sequence_from_flow_template(matched_flow_template, abstract_function, conflict_engine)
    if beat_sequence:
        return beat_sequence
    return _build_default_beat_sequence(abstract_function, conflict_engine)


def _infer_state_delta(
    event_group: List[Dict[str, Any]],
    abstract_function: str,
    conflict_engine: str,
    beat_sequence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}

    if abstract_function in {"公开反转", "低位逆袭"}:
        delta.update({"reputation": "+", "enemy_attention": "+", "secret_exposure": "optional"})
    elif abstract_function == "逃亡破局":
        delta.update({"pressure": "-", "enemy_attention": "+", "injury": "optional", "open_hook": "+"})
    elif abstract_function in {"机缘争夺", "资源争夺"}:
        delta.update({"resources": "+/-", "enemy_attention": "+", "open_hook": "+"})
    elif abstract_function in {"传承筛选", "传承确认"}:
        delta.update({"inheritance_progress": "+", "relationship_shift": "optional", "open_hook": "+"})
    elif abstract_function == "关系转折":
        delta.update({"relationship_shift": "+/-", "trust": "+/-", "secret_exposure": "optional"})
    elif abstract_function == "技术失控":
        delta.update({"resources": "-", "injury": "optional", "enemy_attention": "optional"})
    else:
        delta.update({"pressure": "+/-", "open_hook": "+"})

    if conflict_engine in {"规则压迫", "规则竞争"}:
        delta.setdefault("status", "+/-")
    if conflict_engine in {"信息差", "身份暴露"}:
        delta.setdefault("secret_exposure", "optional")
    if _collect_logic_fragments(event_group, "relationship_delta", limit=1):
        delta.setdefault("relationship_shift", "optional")
    if _collect_logic_fragments(event_group, "resource_delta", limit=1):
        delta.setdefault("resources", delta.get("resources", "+/-"))
    if _collect_logic_fragments(event_group, "hook_open", limit=1):
        delta["open_hook"] = "+"
    if any("代价" in str(beat.get("function", "")) for beat in beat_sequence):
        delta.setdefault("cost_paid", "+")
    return delta


def _infer_variation_axes(
    event_group: List[Dict[str, Any]],
    abstract_function: str,
    conflict_engine: str,
) -> Dict[str, List[str]]:
    if abstract_function in {"公开反转", "低位逆袭"} or conflict_engine in {"规则压迫", "规则竞争"}:
        return {
            "scene": ["审判庭", "任务复盘会", "商会验货", "遗迹资格审查", "城防听证"],
            "pressure_method": ["规则指控", "资源封锁", "证据栽赃", "舆论围攻", "身份质疑"],
            "reversal_method": ["证据反推", "规则反用", "心理博弈", "第三方矛盾", "牺牲代价"],
            "cost": ["暴露底牌", "受伤", "背负债务", "失去信任", "被高层关注"],
        }
    if abstract_function == "逃亡破局" or conflict_engine == "围追堵截":
        return {
            "scene": ["城门封锁", "荒野追缉", "秘境出口", "边境关卡", "护送途中"],
            "pressure_method": ["围堵搜捕", "悬赏通缉", "内鬼出卖", "地形封锁", "资源耗尽"],
            "reversal_method": ["声东击西", "借势反咬", "临时同盟", "环境利用", "舍弃诱饵"],
            "cost": ["留下伤势", "暴露行踪", "失去物资", "欠下人情", "引来更强追兵"],
        }
    if abstract_function in {"机缘争夺", "资源争夺"} or conflict_engine == "资源竞争":
        return {
            "scene": ["拍卖场", "资源库", "试炼秘境", "商路据点", "传承台"],
            "pressure_method": ["抬价截胡", "资格卡位", "先手夺取", "信息封锁", "联盟排挤"],
            "reversal_method": ["隐藏筹码", "规则反用", "情报换手", "临时合作", "代价交换"],
            "cost": ["资源透支", "暴露底牌", "招惹宿敌", "欠下债务", "触发更大争端"],
        }
    if abstract_function in {"传承筛选", "传承确认"}:
        return {
            "scene": ["传承殿", "祖师遗迹", "收徒仪式", "封闭试炼场", "长老议事堂"],
            "pressure_method": ["资格怀疑", "理念冲突", "同门竞争", "考验加码", "守护者试探"],
            "reversal_method": ["道心证明", "补足缺口", "揭示契合", "越级完成试炼", "承担代价"],
            "cost": ["失去退路", "暴露天赋", "背负期待", "结下旧怨", "卷入道统争端"],
        }
    if abstract_function == "关系转折":
        return {
            "scene": ["联手任务", "密谈场合", "危机救援", "误会对峙", "利益谈判"],
            "pressure_method": ["立场错位", "信任裂痕", "第三方挑拨", "共同负债", "旧事翻出"],
            "reversal_method": ["坦白真相", "共同承担", "交换秘密", "反向救援", "以退为进"],
            "cost": ["信任受损", "关系失衡", "欠下人情", "暴露秘密", "牵连同伴"],
        }
    if abstract_function == "技术失控" or conflict_engine == "技术风险":
        return {
            "scene": ["炼丹房", "炼器台", "闭关地", "试验场", "临时阵台"],
            "pressure_method": ["材料不足", "步骤失控", "外力干扰", "规则反噬", "时间压迫"],
            "reversal_method": ["临场改法", "借力导流", "冒险提纯", "牺牲部件", "求援协作"],
            "cost": ["受伤", "资源报废", "暴露技术", "引发异象", "被高层盯上"],
        }
    del event_group
    return {
        "scene": ["公开场合", "封闭试炼场", "交易节点", "路途中转点", "高压会面场景"],
        "pressure_method": ["规则压制", "资源稀缺", "信息封锁", "旧怨施压", "身份质疑"],
        "reversal_method": ["信息反推", "借势破局", "第三方介入", "临时代价交换", "局势错位利用"],
        "cost": ["暴露底牌", "受伤", "失去资源", "欠下人情", "引来后续风险"],
    }


def _extract_forbidden_source_details(event_group: List[Dict[str, Any]]) -> List[str]:
    details: List[str] = []
    for source in event_group:
        for raw_character in source.get("raw_characters", []) or []:
            cleaned = _clean_named_fragment(raw_character)
            if cleaned and cleaned not in _GENERIC_SOURCE_DETAILS:
                details.append(cleaned)
        for raw_location in source.get("raw_locations", []) or []:
            cleaned = _clean_named_fragment(raw_location)
            if cleaned and cleaned not in _GENERIC_SOURCE_DETAILS:
                details.append(cleaned)
        for item in source.get("cultivation_elements", []) or []:
            cleaned = _clean_named_fragment(item)
            if cleaned and _looks_like_specific_setting_term(cleaned):
                details.append(cleaned)
        for summary in source.get("source_summaries", []) or []:
            details.extend(_extract_named_summary_fragments(summary))
    return _dedupe_text_values(details)[:16]


def _build_event_blueprint_from_induced_event(event: InducedEvent) -> List[Dict[str, Any]]:
    chapter_total = max(1, int(event.chapter_count or 1))
    if chapter_total == 1:
        primary_functions = ["主线推进"]
    elif chapter_total == 2:
        primary_functions = ["铺垫入场", "余波收束"]
    elif chapter_total == 3:
        primary_functions = ["铺垫入场", "冲突升级", "余波收束"]
    else:
        middle_count = max(1, chapter_total - 3)
        primary_functions = ["铺垫入场"] + ["冲突升级"] * middle_count + ["局势转折", "余波收束"]
        primary_functions = primary_functions[:chapter_total]

    blueprint: List[Dict[str, Any]] = []
    for index, primary in enumerate(primary_functions, start=1):
        secondary: List[str] = []
        if index == 1 and event.identity_state:
            secondary.append("人物亮相")
        if index == len(primary_functions) and event.hook_open:
            secondary.append("下一事件钩子")
        if index >= max(2, len(primary_functions) - 1) and event.resource_delta:
            secondary.append("代价落地")
        if event.relationship_delta and 1 < index < len(primary_functions):
            secondary.append("关系拉扯")
        purpose_parts = [
            _coerce_text(event.conflict_hint) or "当前冲突",
            _coerce_text(event.function_hint) or "主线推进",
        ]
        blueprint.append(
            {
                "beat_index": index,
                "chapter_span": [index, index],
                "chapter_share": round(1 / max(chapter_total, 1), 3),
                "primary_function": primary,
                "secondary_functions": _dedupe_text_values(secondary),
                "purpose": f"围绕{purpose_parts[0]}承担{primary}，并推动{purpose_parts[1]}",
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
        "关系拉扯": 9,
    }
    unique = _dedupe_text_values(tags)
    return sorted(unique, key=lambda item: priority.get(item, 99))


def _group_haystack(event_group: Sequence[Dict[str, Any]]) -> str:
    values: List[Any] = []
    for source in event_group:
        values.extend(
            [
                source.get("summary", ""),
                source.get("raw_summary", ""),
                source.get("conflict_hint", ""),
                source.get("function_hint", ""),
                source.get("identity_state", ""),
                source.get("sample_summaries", []),
                source.get("cultivation_elements", []),
                source.get("raw_locations", []),
            ]
        )
    return " ".join(_flatten_text_values(values))


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    normalized = _normalize_text(text)
    return any(keyword and keyword in normalized for keyword in keywords)


def _infer_template_level(event_group: List[Dict[str, Any]], abstract_function: str) -> str:
    chapter_counts = [max(1, int(source.get("chapter_count", 1) or 1)) for source in event_group]
    avg_chapters = sum(chapter_counts) / max(len(chapter_counts), 1)
    if avg_chapters >= 8:
        return "volume"
    if avg_chapters <= 2 and abstract_function in {"关系转折", "公开反转", "低位逆袭"}:
        return "micro"
    return "event"


def _build_template_name(abstract_function: str, conflict_engine: str) -> str:
    if abstract_function == "公开反转":
        return "公开规则场景中的低位反转模板"
    if abstract_function == "低位逆袭":
        return "承压积累后的低位逆袭模板"
    if abstract_function == "逃亡破局":
        return "围追堵截下的逃亡破局模板"
    if abstract_function == "机缘争夺":
        return "高压资源场景中的机缘争夺模板"
    if abstract_function == "资源争夺":
        return "稀缺资源竞争中的破局模板"
    if abstract_function == "传承筛选":
        return "传承资格筛选中的证明模板"
    if abstract_function == "传承确认":
        return "师承与道统确认模板"
    if abstract_function == "旧怨升级":
        return "旧怨清算中的压迫升级模板"
    if abstract_function == "关系转折":
        return "立场错位下的关系转折模板"
    if abstract_function == "技术失控":
        return "高压操作中的技术失控模板"
    if abstract_function == "规则压迫":
        return "规则压迫下的破局模板"
    return f"{conflict_engine}下的{abstract_function}模板"


def _find_matching_flow_template(
    event_group: List[Dict[str, Any]],
    event_flow_templates: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    if not event_flow_templates:
        return None
    conflict = _top_values([source.get("conflict_hint", "") for source in event_group], limit=1)
    target_conflict = conflict[0] if conflict else ""
    avg_chapters = sum(max(1, int(source.get("chapter_count", 1) or 1)) for source in event_group) / max(len(event_group), 1)

    best_template: Dict[str, Any] | None = None
    best_score = -999.0
    for template in event_flow_templates:
        score = 0.0
        chapter_range = template.get("chapter_count_range", [1, 12])
        if isinstance(chapter_range, list) and len(chapter_range) == 2:
            low, high = int(chapter_range[0]), int(chapter_range[1])
            if low <= avg_chapters <= high:
                score += 3
            else:
                score -= min(abs(avg_chapters - low), abs(avg_chapters - high)) * 0.2
        if target_conflict and target_conflict in template.get("dominant_conflicts", []):
            score += 5
        if template.get("beat_count", 0):
            score += min(int(template.get("beat_count", 0)), 6) * 0.1
        if score > best_score:
            best_score = score
            best_template = template
    return best_template


def _normalize_beat_sequence_from_flow_template(
    flow_template: Dict[str, Any] | None,
    abstract_function: str,
    conflict_engine: str,
) -> List[Dict[str, Any]]:
    if not flow_template:
        return []
    beat_blueprint = flow_template.get("beat_blueprint", []) or []
    normalized: List[Dict[str, Any]] = []
    total = len(beat_blueprint)
    for beat_index, beat in enumerate(beat_blueprint, start=1):
        function_name = _normalize_beat_function(beat.get("primary_function") or beat.get("function") or "主线推进")
        purpose = str(beat.get("purpose", "") or "").strip() or f"承担{function_name}"
        normalized.append(
            {
                "beat_index": beat_index,
                "function": function_name,
                "purpose": purpose,
                "state_delta": _infer_beat_state_delta(function_name, abstract_function, conflict_engine, beat_index, total),
            }
        )
    return normalized


def _normalize_beat_function(raw_value: Any) -> str:
    text = str(raw_value or "").strip()
    if not text:
        return "主线推进"
    return _LEGACY_BEAT_NAME_MAP.get(text, text)


def _infer_beat_state_delta(
    function_name: str,
    abstract_function: str,
    conflict_engine: str,
    beat_index: int,
    total: int,
) -> Dict[str, Any]:
    del abstract_function, conflict_engine
    if function_name == "公开压迫":
        return {"reputation": "-", "pressure": "+", "enemy_confidence": "+"}
    if function_name == "试探与误判":
        return {"enemy_confidence": "+", "information_tension": "+"}
    if function_name == "反向破局":
        return {"reputation": "+", "secret_exposure": "optional", "enemy_confidence": "-"}
    if function_name == "代价与钩子":
        return {"enemy_attention": "+", "future_hook": "+", "cost_paid": "optional"}
    if function_name == "危机逼近":
        return {"pressure": "+", "escape_window": "-"}
    if function_name == "围堵加压":
        return {"pressure": "+", "resources": "-", "enemy_attention": "+"}
    if function_name == "险中脱身":
        return {"pressure": "-", "position": "unstable", "future_hook": "+"}
    if function_name == "代价与追缉升级":
        return {"injury": "optional", "resources": "-", "enemy_attention": "+"}
    if function_name == "资格显现":
        return {"goal_clarity": "+", "competition": "+"}
    if function_name == "竞争加码":
        return {"pressure": "+", "resources": "-/optional", "enemy_attention": "+"}
    if function_name == "关键破局":
        return {"resources": "+/-", "reputation": "+", "position": "+"}
    if function_name == "收获与余波":
        return {"resources": "+", "future_hook": "+", "enemy_attention": "+"}
    if function_name == "资格门槛":
        return {"qualification_pressure": "+", "goal_clarity": "+"}
    if function_name == "试炼与怀疑":
        return {"pressure": "+", "trust": "-/optional"}
    if function_name == "证明与确认":
        return {"reputation": "+", "inheritance_progress": "+", "secret_exposure": "optional"}
    if function_name == "收束与后效":
        return {"future_hook": "+", "enemy_attention": "optional", "cost_paid": "optional"}
    if function_name == "立场错位":
        return {"relationship_shift": "-", "pressure": "+"}
    if function_name == "情绪加压":
        return {"trust": "-", "pressure": "+", "misunderstanding": "+"}
    if function_name == "关键选择":
        return {"relationship_shift": "+/-", "secret_exposure": "optional"}
    if function_name == "关系重估":
        return {"trust": "+/-", "future_hook": "+", "relationship_shift": "+/-"}
    if function_name == "操作启动":
        return {"technical_risk": "+", "resources": "-/optional"}
    if function_name == "风险扩大":
        return {"technical_risk": "+", "pressure": "+", "resources": "-"}
    if function_name == "硬性补救":
        return {"technical_risk": "-", "cost_paid": "+", "secret_exposure": "optional"}
    if function_name == "后果落地":
        return {"injury": "optional", "resources": "-", "enemy_attention": "optional"}
    if function_name in {"铺垫入场", "人物亮相"}:
        return {"pressure": "+", "position": "unstable"}
    if function_name == "试探推进":
        return {"enemy_confidence": "+", "information_tension": "+"}
    if function_name == "冲突升级":
        return {"pressure": "+", "resources": "-/optional"}
    if function_name == "局势转折":
        return {"reputation": "+", "secret_exposure": "optional", "enemy_confidence": "-"}
    if function_name == "代价落地":
        return {"cost_paid": "+", "injury": "optional"}
    if function_name == "余波收束":
        return {"position": "+/-", "future_hook": "+" if beat_index == total else "optional"}
    if function_name == "下一事件钩子":
        return {"open_hook": "+"}
    return {"state_shift": "optional"}


def _build_default_beat_sequence(abstract_function: str, conflict_engine: str) -> List[Dict[str, Any]]:
    if abstract_function in {"公开反转", "低位逆袭"}:
        blueprints = [
            ("公开压迫", "让主角在公开场景中处于不利位置"),
            ("试探与误判", "对手确认主角暂时无力反击"),
            ("反向破局", "主角利用隐藏能力、信息差或规则漏洞完成反转"),
            ("代价与钩子", "胜利后引出更高层关注或更大风险"),
        ]
    elif abstract_function == "逃亡破局":
        blueprints = [
            ("危机逼近", "明确追击压力和立即脱身需求"),
            ("围堵加压", "封锁升级，主角退路被进一步压缩"),
            ("险中脱身", "利用环境、同盟或信息差打开缺口"),
            ("代价与追缉升级", "脱身成功但留下新的伤势、债务或追踪线索"),
        ]
    elif abstract_function in {"机缘争夺", "资源争夺"}:
        blueprints = [
            ("资格显现", "稀缺资源或机缘正式出现"),
            ("竞争加码", "多方对资源进行争夺或卡位"),
            ("关键破局", "主角通过筹码、规则或合作赢得主动"),
            ("收获与余波", "收益落地，同时引出新的敌意或更大诱饵"),
        ]
    elif abstract_function in {"传承筛选", "传承确认"}:
        blueprints = [
            ("资格门槛", "交代传承、师承或资格限制"),
            ("试炼与怀疑", "主角遭遇筛选、竞争或理念质疑"),
            ("证明与确认", "主角用行动或代价证明自身契合"),
            ("收束与后效", "得到承认，但也承担道统带来的连锁后果"),
        ]
    elif abstract_function == "关系转折":
        blueprints = [
            ("立场错位", "关系双方在目标或立场上发生冲突"),
            ("情绪加压", "误会、债务或第三方影响让关系继续恶化"),
            ("关键选择", "有人坦白、牺牲或共同承担风险"),
            ("关系重估", "关系发生偏转，并为后续合作或裂痕埋钩子"),
        ]
    elif abstract_function == "技术失控":
        blueprints = [
            ("操作启动", "主角在高压下启动关键技术或资源流程"),
            ("风险扩大", "外部干扰、材料不足或规则反噬让失控显性化"),
            ("硬性补救", "主角以临场改法、代价交换或协作强行稳住局面"),
            ("后果落地", "问题暂时解决，但伤势、消耗或异象被外界记录"),
        ]
    else:
        blueprints = [
            ("铺垫入场", "建立当前场景、角色位置和核心压力"),
            ("冲突升级", "让对立目标与阻碍条件变得不可回避"),
            ("局势转折", "通过信息差、代价或外力改变局面"),
            ("余波收束", "收束当前事件并引向下一节点"),
        ]

    beats: List[Dict[str, Any]] = []
    total = len(blueprints)
    for beat_index, (function_name, purpose) in enumerate(blueprints, start=1):
        normalized_function = _normalize_beat_function(function_name)
        beats.append(
            {
                "beat_index": beat_index,
                "function": normalized_function,
                "purpose": purpose,
                "state_delta": _infer_beat_state_delta(normalized_function, abstract_function, conflict_engine, beat_index, total),
            }
        )
    return beats


def _build_source_pattern_summary(
    event_group: List[Dict[str, Any]],
    abstract_function: str,
    conflict_engine: str,
    beat_sequence: List[Dict[str, Any]],
    state_delta: Dict[str, Any],
) -> str:
    beat_names = [str(item.get("function", "")).strip() for item in beat_sequence[:3] if str(item.get("function", "")).strip()]
    state_keys = [key for key in list(state_delta.keys())[:3] if key]
    avg_chapters = round(sum(max(1, int(source.get("chapter_count", 1) or 1)) for source in event_group) / max(len(event_group), 1), 1)
    beat_text = " -> ".join(beat_names) if beat_names else "铺垫 -> 升压 -> 破局"
    state_text = "、".join(state_keys) if state_keys else "局势"
    return f"该模板常见于{conflict_engine}驱动的场景，平均覆盖约{avg_chapters}章，通常沿着{beat_text}推进，并最终改写{state_text}。"


def _collect_logic_fragments(event_group: List[Dict[str, Any]], field_name: str, limit: int) -> List[str]:
    values: List[str] = []
    for source in event_group:
        raw_value = source.get(field_name, [])
        for item in _flatten_text_values([raw_value]):
            for piece in _split_logic_fragments(item):
                if piece and piece not in values:
                    values.append(piece)
                if len(values) >= limit:
                    return values
    return values


def _split_logic_fragments(text: Any) -> List[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    fragments = []
    for piece in cleaned.replace("；", "，").replace("。", "，").split("，"):
        item = piece.strip("、/ \t\r\n")
        if len(item) >= 2:
            fragments.append(item[:32])
    return fragments


def _clean_named_fragment(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for splitter in ("(", "（", ")", "）", ":", "：", "/", "｜", "|"):
        text = text.split(splitter, 1)[0].strip()
    return text[:16]


def _looks_like_specific_setting_term(text: str) -> bool:
    if not text or text in _GENERIC_SOURCE_DETAILS:
        return False
    if len(text) < 2 or len(text) > 16:
        return False
    keywords = ("诀", "经", "印", "鼎", "珠", "塔", "剑", "刀", "火", "焰", "雷", "血", "体", "骨", "符", "令", "盘", "丹", "脉")
    return any(keyword in text for keyword in keywords) or text.endswith(("功", "法", "阵", "炎"))


def _extract_named_summary_fragments(text: Any) -> List[str]:
    summary = str(text or "").strip()
    if not summary:
        return []
    patterns = [
        "大比",
        "拍卖会",
        "秘境",
        "遗迹",
        "审判庭",
        "执法堂",
        "听证",
        "试炼",
        "验货",
        "资格审查",
        "传承台",
        "洞府",
    ]
    fragments: List[str] = []
    for piece in _split_logic_fragments(summary):
        if any(keyword in piece for keyword in patterns):
            cleaned = _clean_named_fragment(piece)
            if cleaned and cleaned not in _GENERIC_SOURCE_DETAILS:
                fragments.append(cleaned)
    return fragments[:4]
