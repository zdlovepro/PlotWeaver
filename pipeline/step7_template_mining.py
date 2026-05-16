from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import config
from pipeline.core.artifacts import (
    ExecutableTemplate,
    pipeline_state_snapshot_from_fused_world,
    save_json_artifact,
    template_mining_from_fused_world,
)
from pipeline.core.common_text import coerce_text, dedupe_text_values, flatten_text_values, normalize_text
from pipeline.core.world_building_core import FusedWorld, load_world_snapshot, save_world_snapshot
from pipeline.step2_extraction import PlotAtom
from pipeline.step3_event_induction import InducedEvent


_STEP7_FILENAME = "step7_template_mining.json"
_STEP7_CARDS_FILENAME = "step7_template_cards.json"
_STEP7_TEMPLATES_FILENAME = "step7_templates.json"
_STEP7_PIPELINE_STATE_FILENAME = "pipeline_state_after_step7.json"

_STEP7_MAX_EXECUTABLE_TEMPLATES = int(getattr(config, "STEP7_MAX_EXECUTABLE_TEMPLATES", 120) or 120)
_STEP7_MAX_EVENT_TEMPLATES = int(getattr(config, "STEP7_MAX_EVENT_TEMPLATES", 100) or 100)
_STEP7_MAX_EVENT_FLOW_TEMPLATES = int(getattr(config, "STEP7_MAX_EVENT_FLOW_TEMPLATES", 100) or 100)

_GENERIC_FORBIDDEN_DETAILS = {
    "主角",
    "反派",
    "宗门",
    "家族",
    "城市",
    "秘境",
    "传承",
    "功法",
    "法宝",
    "丹药",
}

_ROLE_SLOT_TEMPLATE_LIBRARY: List[Dict[str, Any]] = [
    {
        "name": "public_judgment",
        "keywords": ["审判", "裁决", "执法", "规矩", "资格"],
        "role_slots": ["protagonist", "oppressor", "authority", "witness", "ally"],
    },
    {
        "name": "resource_bidding",
        "keywords": ["拍卖", "交易", "竞价", "验货", "争夺"],
        "role_slots": ["protagonist", "oppressor", "broker", "valuator", "witness"],
    },
    {
        "name": "inheritance_trial",
        "keywords": ["试炼", "传承", "拜师", "授法", "考核"],
        "role_slots": ["protagonist", "authority", "rival", "guardian", "ally"],
    },
    {
        "name": "hunt_escape",
        "keywords": ["追杀", "围杀", "逃亡", "围堵", "截杀"],
        "role_slots": ["protagonist", "oppressor", "hunter_support", "ally", "shelter"],
    },
    {
        "name": "relationship_turn",
        "keywords": ["误会", "和解", "背叛", "结盟", "互救"],
        "role_slots": ["protagonist", "counterpart", "agitator", "mediator", "ally"],
    },
    {
        "name": "default_pressure",
        "keywords": [],
        "role_slots": ["protagonist", "oppressor", "ally", "witness", "source"],
    },
]

_ROLE_SLOT_MEANINGS: Dict[str, Dict[str, str]] = {
    "public_judgment": {
        "protagonist": "被压制者/破局者",
        "oppressor": "规则利用者",
        "authority": "裁决者",
        "witness": "舆论放大者",
        "ally": "有限支援者",
    },
    "resource_bidding": {
        "protagonist": "资源争夺者",
        "oppressor": "截胡者/抬价者",
        "broker": "秩序维护者",
        "valuator": "真伪判定者",
        "witness": "场外放大者",
    },
    "inheritance_trial": {
        "protagonist": "候选者/求法者",
        "authority": "筛选者",
        "rival": "同台竞争者",
        "guardian": "门槛守卫者",
        "ally": "有限协助者",
    },
    "hunt_escape": {
        "protagonist": "被围猎者/突围者",
        "oppressor": "追杀主导者",
        "hunter_support": "封锁协助者",
        "ally": "掩护者",
        "shelter": "退路提供者",
    },
    "relationship_turn": {
        "protagonist": "主动修复者/被误解者",
        "counterpart": "关键关系对象",
        "agitator": "挑拨者",
        "mediator": "调停者",
        "ally": "共患难支撑者",
    },
    "default_pressure": {
        "protagonist": "破局者",
        "oppressor": "主要阻碍者",
        "ally": "临时盟友",
        "witness": "旁观放大者",
        "source": "信息提供者",
    },
}

_ABSTRACT_FUNCTION_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("公开反转", ("打脸", "翻盘", "公开", "逆袭", "证明自己")),
    ("低位逆袭", ("低位", "受压", "被轻视", "破局", "反制")),
    ("资源争夺", ("资源", "拍卖", "宝物", "机缘", "争夺")),
    ("逃亡破局", ("追杀", "逃亡", "围杀", "堵截", "脱身")),
    ("关系转折", ("误会", "和解", "背叛", "结盟", "互救")),
    ("传承筛选", ("传承", "拜师", "授法", "考核", "试炼")),
    ("代价落地", ("代价", "后果", "债务", "伤势", "余波")),
    ("旧怨升级", ("复仇", "宿敌", "旧怨", "清算", "报复")),
    ("技术失控", ("炼丹", "炼器", "异象", "失控", "副作用")),
]

_CONFLICT_ENGINE_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("规则压迫", ("裁决", "执法", "规矩", "资格", "审查")),
    ("资源竞争", ("资源", "拍卖", "机缘", "争夺", "竞价")),
    ("围追堵截", ("追杀", "围杀", "围堵", "截杀", "追捕")),
    ("旧怨升级", ("复仇", "宿敌", "旧怨", "清算", "报复")),
    ("信息差", ("秘密", "真相", "线索", "身份", "误导")),
    ("技术风险", ("炼丹", "炼器", "异象", "失控", "反噬")),
    ("试炼筛选", ("试炼", "传承", "考核", "拜师", "选拔")),
    ("权力审判", ("权力", "审判", "裁决", "高层", "堂会")),
]

_VARIATION_AXES_MAP: Dict[str, Dict[str, List[str]]] = {
    "公开反转": {
        "scene": ["审判庭", "任务复盘会", "商会验货", "资格审查", "公开擂台"],
        "pressure_method": ["规则指控", "资源卡位", "证据栽赃", "舆论围攻", "身份质疑"],
        "reversal_method": ["证据反推", "规则反用", "心理博弈", "第三方矛盾", "代价换胜"],
        "cost": ["暴露底牌", "被高层关注", "失去退路", "背负人情", "树立新敌"],
    },
    "资源争夺": {
        "scene": ["拍卖会", "遗迹入口", "商会库房", "任务分配会"],
        "pressure_method": ["抬价", "封锁", "资格剥夺", "临时背刺"],
        "reversal_method": ["识货", "暗线交易", "第三方插手", "借势反咬"],
        "cost": ["欠债", "受伤", "秘密暴露", "失去盟友信任"],
    },
    "逃亡破局": {
        "scene": ["城防封锁", "山脉追击", "秘境出口", "黑市暗巷"],
        "pressure_method": ["围追堵截", "悬赏追捕", "地形封锁", "身份曝光"],
        "reversal_method": ["地形利用", "牺牲诱饵", "反向设局", "第三方冲突"],
        "cost": ["体力透支", "受伤", "底牌暴露", "同伴失散"],
    },
}

_BEAT_PURPOSES = {
    "铺垫入场": "建立场景、立场或资源格局",
    "冲突升级": "把核心压力推到前台",
    "局势转折": "触发误判、反咬或立场改变",
    "反向破局": "让主角或关键角色完成突破",
    "余波收束": "结算短期后果并稳定局势",
    "下一事件钩子": "为后续更大冲突留下牵引",
    "代价落地": "让胜利或失败带来明确成本",
    "主线推进": "把阶段性目标推向下一层",
}

_RARE_ABSTRACT_FUNCTIONS = {"技术失控", "代价落地", "旧怨升级"}


def derive_templates(
    all_atoms: Dict[str, List[PlotAtom]],
    fused_world: FusedWorld,
    induced_events_by_novel: Dict[str, List[InducedEvent]] | None = None,
) -> FusedWorld:
    atoms = [atom for bucket in all_atoms.values() for atom in bucket]
    induced_events = [event for bucket in (induced_events_by_novel or {}).values() for event in bucket]

    template_candidates = []
    template_candidates.extend(_mine_template_candidates_from_atoms(atoms))
    template_candidates.extend(_mine_template_candidates_from_induced_events(induced_events))
    template_candidates.extend(_mine_template_candidates_from_step6_patterns(fused_world))

    template_clusters = _cluster_template_candidates(template_candidates)
    selected_clusters = _consolidate_template_clusters(template_clusters)

    fused_world.role_slot_templates = [dict(item) for item in _ROLE_SLOT_TEMPLATE_LIBRARY]
    fused_world.template_candidates = template_candidates
    fused_world.template_clusters = template_clusters
    fused_world.event_templates = _build_event_templates_from_clusters(selected_clusters)
    fused_world.volume_templates = _build_volume_templates(induced_events, atoms)
    fused_world.event_flow_templates = _build_event_flow_templates_from_clusters(selected_clusters)
    fused_world.executable_templates = _build_executable_templates_from_clusters(selected_clusters)
    fused_world.template_mining_coverage_report = _build_step7_coverage_report(
        atoms=atoms,
        induced_events=induced_events,
        template_candidates=template_candidates,
        template_clusters=template_clusters,
        event_templates=fused_world.event_templates,
        event_flow_templates=fused_world.event_flow_templates,
        executable_templates=fused_world.executable_templates,
    )

    print(f"[Step 7] Atoms: {len(atoms)}")
    print(f"[Step 7] Induced events: {len(induced_events)}")
    print(f"[Step 7] Template candidates: {len(template_candidates)}")
    print(f"[Step 7] Template clusters: {len(template_clusters)}")
    print(f"[Step 7] Executable templates: {len(fused_world.executable_templates)}")
    print(f"[Step 7] Coverage ratio: {round(fused_world.template_mining_coverage_report.get('covered_event_ratio', 0.0) * 100, 2)}%")
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


def _mine_template_candidates_from_atoms(atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for atom in atoms:
        text = _source_text_from_atom(atom)
        if not text:
            continue
        abstract_function = _infer_abstract_function(text, atom.narrative_function)
        conflict_engine = _infer_conflict_engine(text, atom.conflict_type)
        role_template = _match_role_template(text)
        candidate = {
            "candidate_id": f"template_candidate_{len(candidates) + 1:04d}",
            "source_refs": [atom.atom_id],
            "novel_sources": [atom.novel_source],
            "arc_names": [atom.arc_name],
            "source_type": "atom",
            "level_hint": "micro",
            "abstract_function": abstract_function,
            "conflict_engine": conflict_engine,
            "role_slot_signature": role_template.get("role_slots", ["protagonist", "oppressor", "ally"]),
            "beat_signature": _infer_candidate_beats_from_atom(atom, abstract_function),
            "state_delta_signature": _infer_state_delta_signature(getattr(atom, "state_delta", {})),
            "required_preconditions": _infer_required_preconditions(abstract_function, conflict_engine),
            "forbidden_source_details": _extract_forbidden_source_details(
                raw_summary=atom.raw_summary or atom.summary,
                raw_characters=getattr(atom, "raw_characters", []),
                raw_location=getattr(atom, "raw_location", "") or getattr(atom, "location", ""),
                cultivation_elements=getattr(atom, "cultivation_elements", []),
            ),
            "source_pattern_summary": _build_source_pattern_summary(
                abstract_function,
                conflict_engine,
                _infer_candidate_beats_from_atom(atom, abstract_function),
                _infer_state_delta_signature(getattr(atom, "state_delta", {})),
            ),
            "support_hint": 1,
            "chapter_count_hint": max(1, int(getattr(atom, "chapter_count", 1) or 1)),
        }
        candidates.append(candidate)
    return candidates


def _mine_template_candidates_from_induced_events(events: List[InducedEvent]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for event in events:
        text = _source_text_from_event(event)
        if not text:
            continue
        abstract_function = _infer_abstract_function(text, event.function_hint)
        conflict_engine = _infer_conflict_engine(text, event.conflict_hint)
        role_template = _match_role_template(text)
        state_delta_signature = _dedupe_state_tags(
            list(getattr(event, "relationship_delta", []))
            + list(getattr(event, "resource_delta", []))
            + list(getattr(event, "state_outputs", []))
        )
        candidate = {
            "candidate_id": f"template_candidate_{len(candidates) + 1:04d}",
            "source_refs": [event.event_id],
            "novel_sources": [event.novel_source],
            "arc_names": [event.arc_name],
            "source_type": "induced_event",
            "level_hint": "event",
            "abstract_function": abstract_function,
            "conflict_engine": conflict_engine,
            "role_slot_signature": role_template.get("role_slots", ["protagonist", "oppressor", "authority"]),
            "beat_signature": _infer_candidate_beats_from_event(event, abstract_function),
            "state_delta_signature": state_delta_signature,
            "required_preconditions": _infer_required_preconditions(abstract_function, conflict_engine),
            "forbidden_source_details": _extract_forbidden_source_details(
                raw_summary=event.raw_summary or event.summary,
                raw_characters=getattr(event, "raw_characters", []),
                raw_location="",
                cultivation_elements=list(getattr(event, "state_outputs", [])),
            ),
            "source_pattern_summary": _build_source_pattern_summary(
                abstract_function,
                conflict_engine,
                _infer_candidate_beats_from_event(event, abstract_function),
                state_delta_signature,
            ),
            "support_hint": max(1, int(getattr(event, "chapter_count", 1) or 1)),
            "chapter_count_hint": max(1, int(getattr(event, "chapter_count", 1) or 1)),
        }
        candidates.append(candidate)
    return candidates


def _mine_template_candidates_from_step6_patterns(fused_world: FusedWorld) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for item in fused_world.micro_interactions or []:
        interaction_type = coerce_text(item.get("interaction_type", "")) or "关系互动"
        conflict_engine = coerce_text(item.get("conflict_engine", "")) or _infer_conflict_engine(interaction_type, interaction_type)
        phases = flatten_text_values([item.get("phase_sequence", []), item.get("the_dance_of_interaction", {})])[:5]
        candidates.append(
            {
                "candidate_id": f"template_candidate_step6_micro_{len(candidates) + 1:04d}",
                "source_refs": list(item.get("source_refs", [])),
                "novel_sources": list(item.get("novel_sources", [])),
                "arc_names": [],
                "source_type": "step6_pattern",
                "level_hint": "event",
                "abstract_function": _infer_abstract_function(interaction_type, interaction_type),
                "conflict_engine": conflict_engine,
                "role_slot_signature": ["protagonist", "counterpart", "ally"],
                "beat_signature": phases or ["铺垫入场", "冲突升级", "局势转折", "余波收束"],
                "state_delta_signature": _dedupe_state_tags([item.get("relationship_delta", ""), item.get("emotional_turn", "")]),
                "required_preconditions": _infer_required_preconditions(interaction_type, conflict_engine),
                "forbidden_source_details": _extract_forbidden_source_details(
                    raw_summary=item.get("summary", ""),
                    raw_characters=[item.get("role_A", ""), item.get("role_B", "")],
                    raw_location=item.get("applicable_scene", ""),
                    cultivation_elements=[],
                ),
                "source_pattern_summary": _clip_text(item.get("summary", interaction_type), 120),
                "support_hint": max(1, int(item.get("support_count", 1) or 1)),
                "chapter_count_hint": 2,
            }
        )

    for item in fused_world.macro_tropes or []:
        stages = list(item.get("stage_sequence", item.get("stages", [])))
        abstract = _infer_abstract_function(" ".join(flatten_text_values([stages, item.get("description", "")])), item.get("name", ""))
        conflict = coerce_text(item.get("dominant_conflicts", [])) or _infer_conflict_engine(item.get("description", ""), "")
        candidates.append(
            {
                "candidate_id": f"template_candidate_step6_macro_{len(candidates) + 1:04d}",
                "source_refs": list(item.get("source_refs", [])),
                "novel_sources": list(item.get("novel_sources", [])),
                "arc_names": [],
                "source_type": "step6_pattern",
                "level_hint": "volume",
                "abstract_function": abstract,
                "conflict_engine": conflict,
                "role_slot_signature": ["protagonist", "oppressor", "ally", "witness"],
                "beat_signature": stages or ["铺垫入场", "冲突升级", "局势转折", "主线推进"],
                "state_delta_signature": _dedupe_state_tags(flatten_text_values([item.get("state_progression", {})])),
                "required_preconditions": _infer_required_preconditions(abstract, conflict),
                "forbidden_source_details": _extract_forbidden_source_details(
                    raw_summary=item.get("description", ""),
                    raw_characters=[],
                    raw_location="",
                    cultivation_elements=[],
                ),
                "source_pattern_summary": _clip_text(item.get("description", abstract), 120),
                "support_hint": max(1, int(item.get("support_count", 1) or 1)),
                "chapter_count_hint": max(3, len(stages) or 4),
            }
        )

    for item in fused_world.plot_threads or []:
        stages = list(item.get("stage_sequence", item.get("stages", [])))
        abstract = _infer_abstract_function(" ".join(flatten_text_values([item.get("thread_type", ""), stages])), item.get("thread_type", ""))
        conflict = _infer_conflict_engine(" ".join(flatten_text_values([item.get("thread_type", ""), item.get("summary", "")])), "")
        candidates.append(
            {
                "candidate_id": f"template_candidate_step6_thread_{len(candidates) + 1:04d}",
                "source_refs": list(item.get("source_refs", [])),
                "novel_sources": list(item.get("novel_sources", [])),
                "arc_names": [],
                "source_type": "step6_pattern",
                "level_hint": "event",
                "abstract_function": abstract,
                "conflict_engine": conflict,
                "role_slot_signature": ["protagonist", "counterpart", "ally"],
                "beat_signature": stages or ["铺垫入场", "冲突升级", "关系转折", "余波收束"],
                "state_delta_signature": _dedupe_state_tags(item.get("relationship_delta_sequence", [])),
                "required_preconditions": _infer_required_preconditions(abstract, conflict),
                "forbidden_source_details": _extract_forbidden_source_details(
                    raw_summary=item.get("summary", ""),
                    raw_characters=item.get("display_characters", []),
                    raw_location="",
                    cultivation_elements=[],
                ),
                "source_pattern_summary": _clip_text(item.get("summary", item.get("thread_type", "")), 120),
                "support_hint": max(1, int(item.get("support_count", 1) or 1)),
                "chapter_count_hint": max(2, len(stages) or 3),
            }
        )
    return candidates


def _cluster_template_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for candidate in candidates:
        best_index = -1
        best_score = 0.0
        for index, cluster in enumerate(clusters):
            score = _template_similarity(candidate, cluster["representative_candidate"])
            if score > best_score:
                best_score = score
                best_index = index
        if best_index >= 0 and best_score >= 0.65:
            clusters[best_index]["members"].append(candidate)
        else:
            clusters.append(
                {
                    "cluster_id": f"template_cluster_{len(clusters) + 1:04d}",
                    "cluster_key": _template_cluster_key(candidate),
                    "members": [candidate],
                }
            )
    normalized: List[Dict[str, Any]] = []
    for cluster in clusters:
        members = cluster.pop("members")
        representative = _pick_representative_candidate(members)
        merged_beats = _merge_tag_lists(item.get("beat_signature", []) for item in members)
        merged_state = _merge_tag_lists(item.get("state_delta_signature", []) for item in members)
        merged_roles = _merge_tag_lists(item.get("role_slot_signature", []) for item in members)
        forbidden = _merge_tag_lists(item.get("forbidden_source_details", []) for item in members)[:50]
        normalized.append(
            {
                "cluster_id": cluster["cluster_id"],
                "cluster_key": cluster["cluster_key"],
                "candidate_ids": [item.get("candidate_id", "") for item in members],
                "source_refs": _merge_tag_lists(item.get("source_refs", []) for item in members),
                "support_count": len(members),
                "novel_sources": _merge_tag_lists(item.get("novel_sources", []) for item in members),
                "abstract_function": representative.get("abstract_function", ""),
                "conflict_engine": representative.get("conflict_engine", ""),
                "merged_beat_signature": merged_beats,
                "merged_state_delta_signature": merged_state,
                "merged_role_slot_signature": merged_roles,
                "forbidden_source_details": forbidden,
                "representative_candidate": representative,
                "is_rare_pattern": False,
                "level_hints": _merge_tag_lists(item.get("level_hint", "") for item in members),
                "chapter_count_range": _chapter_count_range(item.get("chapter_count_hint", 1) for item in members),
                "source_pattern_summary": _build_source_pattern_summary(
                    representative.get("abstract_function", ""),
                    representative.get("conflict_engine", ""),
                    merged_beats,
                    merged_state,
                ),
            }
        )
    return normalized


def _consolidate_template_clusters(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not clusters:
        return []
    ranked = sorted(clusters, key=_template_cluster_rank_key, reverse=True)
    selected: List[Dict[str, Any]] = []
    covered_functions: set[str] = set()

    for cluster in ranked:
        if cluster.get("support_count", 0) >= 2:
            cluster["is_rare_pattern"] = False
            selected.append(cluster)
            covered_functions.add(cluster.get("abstract_function", ""))

    for cluster in ranked:
        if cluster in selected:
            continue
        abstract_function = cluster.get("abstract_function", "")
        keep_rare = (
            cluster.get("support_count", 0) == 1
            and (
                abstract_function in _RARE_ABSTRACT_FUNCTIONS
                or len(cluster.get("forbidden_source_details", [])) >= 4
                or len(cluster.get("merged_beat_signature", [])) >= 4
            )
        )
        cluster["is_rare_pattern"] = keep_rare
        if keep_rare and len(selected) < _STEP7_MAX_EXECUTABLE_TEMPLATES:
            selected.append(cluster)
            covered_functions.add(abstract_function)

    for cluster in ranked:
        if cluster in selected:
            continue
        abstract_function = cluster.get("abstract_function", "")
        if abstract_function and abstract_function not in covered_functions:
            cluster["is_rare_pattern"] = cluster.get("support_count", 0) == 1
            selected.append(cluster)
            covered_functions.add(abstract_function)
        if len(selected) >= _STEP7_MAX_EXECUTABLE_TEMPLATES:
            break

    return selected[:_STEP7_MAX_EXECUTABLE_TEMPLATES]


def _build_executable_templates_from_clusters(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    templates: List[Dict[str, Any]] = []
    for index, cluster in enumerate(clusters[:_STEP7_MAX_EXECUTABLE_TEMPLATES], start=1):
        abstract_function = cluster.get("abstract_function", "")
        conflict_engine = cluster.get("conflict_engine", "")
        role_template = _match_role_template(" ".join(flatten_text_values([abstract_function, conflict_engine])))
        role_slots = _role_slots_from_signature(
            cluster.get("merged_role_slot_signature", []),
            role_template.get("name", "default_pressure"),
        )
        beat_sequence = _beat_sequence_from_signature(
            cluster.get("merged_beat_signature", []),
            abstract_function,
            conflict_engine,
        )
        state_delta = _state_delta_map_from_signature(cluster.get("merged_state_delta_signature", []))
        template = ExecutableTemplate(
            template_id=f"executable_template_{index:04d}",
            template_name=_build_template_name(abstract_function, conflict_engine),
            level=_infer_template_level(cluster.get("level_hints", [])),
            abstract_function=abstract_function,
            conflict_engine=conflict_engine,
            required_preconditions=_infer_required_preconditions(abstract_function, conflict_engine),
            role_slots=role_slots,
            beat_sequence=beat_sequence,
            state_delta=state_delta,
            variation_axes=_infer_variation_axes(abstract_function, conflict_engine),
            forbidden_source_details=cluster.get("forbidden_source_details", [])[:50],
            source_refs=cluster.get("source_refs", []),
            source_pattern_summary=_build_source_pattern_summary(
                abstract_function,
                conflict_engine,
                cluster.get("merged_beat_signature", []),
                cluster.get("merged_state_delta_signature", []),
            ),
        )
        templates.append(template.__dict__)
    return templates


def _build_event_templates_from_clusters(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    templates: List[Dict[str, Any]] = []
    for index, cluster in enumerate(clusters[:_STEP7_MAX_EVENT_TEMPLATES], start=1):
        rep = cluster.get("representative_candidate", {})
        templates.append(
            {
                "template_id": f"event_template_{index:04d}",
                "name": _build_template_name(cluster.get("abstract_function", ""), cluster.get("conflict_engine", "")),
                "conflict_type": cluster.get("conflict_engine", ""),
                "narrative_function": cluster.get("abstract_function", ""),
                "role_slots": list(cluster.get("merged_role_slot_signature", [])),
                "conflict_engines": [cluster.get("conflict_engine", "")] if cluster.get("conflict_engine", "") else [],
                "chapter_count_range": list(cluster.get("chapter_count_range", [1, 1])),
                "sample_summaries": [rep.get("source_pattern_summary", rep.get("source_pattern_summary", ""))][:1],
                "support_count": cluster.get("support_count", 0),
                "source_refs": cluster.get("source_refs", []),
            }
        )
    return templates


def _build_volume_templates(induced_events: List[InducedEvent], atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    source_events = induced_events or _pseudo_events_from_atoms(atoms)
    grouped: Dict[str, List[Any]] = defaultdict(list)
    for event in source_events:
        grouped[getattr(event, "arc_name", "") or "Unnamed Arc"].append(event)
    templates: List[Dict[str, Any]] = []
    for arc_name, bucket in grouped.items():
        dominant_conflicts = _top_values(
            (getattr(item, "conflict_hint", getattr(item, "conflict_type", "")) for item in bucket),
            limit=3,
        )
        dominant_functions = _top_values(
            (getattr(item, "function_hint", getattr(item, "narrative_function", "")) for item in bucket),
            limit=3,
        )
        templates.append(
            {
                "arc_name": arc_name,
                "event_count": len(bucket),
                "dominant_conflicts": dominant_conflicts,
                "dominant_functions": dominant_functions,
                "suggested_role_slots": _match_role_template(" ".join(dominant_conflicts + dominant_functions)).get("role_slots", []),
                "conflict_engines": dominant_conflicts[:3],
                "recommended_new_characters": max(3, min(8, len(bucket) // 2 or 3)),
                "sample_events": [_clip_text(getattr(item, "summary", ""), 60) for item in bucket[:2] if getattr(item, "summary", "")],
                "start_event_index": 0,
                "end_event_index": max(0, len(bucket) - 1),
            }
        )
    return templates


def _build_event_flow_templates_from_clusters(clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    templates: List[Dict[str, Any]] = []
    for index, cluster in enumerate(clusters[:_STEP7_MAX_EVENT_FLOW_TEMPLATES], start=1):
        beats = cluster.get("merged_beat_signature", []) or ["铺垫入场", "冲突升级", "局势转折", "余波收束"]
        beat_blueprint = _beat_blueprint_from_signature(beats)
        templates.append(
            {
                "template_id": f"event_flow_{index:04d}",
                "name": f"{cluster.get('abstract_function', '事件模板')}_{cluster.get('conflict_engine', '通用驱动')}",
                "chapter_count_range": list(cluster.get("chapter_count_range", [1, max(1, len(beats))])),
                "dominant_conflicts": [cluster.get("conflict_engine", "")] if cluster.get("conflict_engine", "") else [],
                "required_coverage": [beat.get("primary_function", "") for beat in beat_blueprint if beat.get("primary_function", "")],
                "beat_blueprint": beat_blueprint,
                "source_window_count": cluster.get("support_count", 0),
                "beat_count": len(beat_blueprint),
                "support_count": cluster.get("support_count", 0),
                "source_refs": cluster.get("source_refs", []),
            }
        )
    return templates


def _build_step7_coverage_report(
    atoms: List[PlotAtom],
    induced_events: List[InducedEvent],
    template_candidates: List[Dict[str, Any]],
    template_clusters: List[Dict[str, Any]],
    event_templates: List[Dict[str, Any]],
    event_flow_templates: List[Dict[str, Any]],
    executable_templates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    total_refs = {event.event_id for event in induced_events} if induced_events else {atom.atom_id for atom in atoms}
    covered_refs = {
        str(ref)
        for item in template_candidates
        for ref in flatten_text_values([item.get("source_refs", [])])
    }
    abstract_distribution = Counter(
        cluster.get("abstract_function", "")
        for cluster in template_clusters
        if cluster.get("abstract_function", "")
    )
    conflict_distribution = Counter(
        cluster.get("conflict_engine", "")
        for cluster in template_clusters
        if cluster.get("conflict_engine", "")
    )
    notes: List[str] = []
    if len(template_candidates) > len(executable_templates):
        notes.append("Candidates outnumber final executable templates, indicating programmatic consolidation is active.")
    if any(cluster.get("is_rare_pattern", False) for cluster in template_clusters):
        notes.append("Rare template clusters were retained when they added distinct structure or conflict engines.")
    return {
        "atom_count": len(atoms),
        "induced_event_count": len(induced_events),
        "template_candidate_count": len(template_candidates),
        "template_cluster_count": len(template_clusters),
        "final_event_template_count": len(event_templates),
        "final_flow_template_count": len(event_flow_templates),
        "final_executable_template_count": len(executable_templates),
        "covered_event_ratio": round((len(covered_refs & total_refs) / len(total_refs)) if total_refs else 0.0, 4),
        "abstract_function_distribution": dict(abstract_distribution),
        "conflict_engine_distribution": dict(conflict_distribution),
        "notes": notes,
    }


def _source_text_from_atom(atom: PlotAtom) -> str:
    return " ".join(
        flatten_text_values(
            [
                atom.summary,
                atom.raw_summary,
                atom.core_action,
                atom.raw_core_action,
                atom.conflict_type,
                atom.narrative_function,
                atom.location,
                atom.raw_location,
                atom.cultivation_elements,
                atom.state_delta,
            ]
        )
    )


def _source_text_from_event(event: InducedEvent) -> str:
    return " ".join(
        flatten_text_values(
            [
                event.summary,
                event.raw_summary,
                event.conflict_hint,
                event.function_hint,
                event.relationship_delta,
                event.resource_delta,
                event.state_outputs,
                event.hook_open,
                event.hook_close,
            ]
        )
    )


def _infer_abstract_function(text: str, fallback: Any) -> str:
    haystack = " ".join(flatten_text_values([text, fallback]))
    for label, keywords in _ABSTRACT_FUNCTION_RULES:
        if any(keyword in haystack for keyword in keywords):
            return label
    return coerce_text(fallback) or "主线推进"


def _infer_conflict_engine(text: str, fallback: Any) -> str:
    haystack = " ".join(flatten_text_values([text, fallback]))
    for label, keywords in _CONFLICT_ENGINE_RULES:
        if any(keyword in haystack for keyword in keywords):
            return label
    return coerce_text(fallback) or "局势施压"


def _infer_required_preconditions(abstract_function: str, conflict_engine: str) -> List[str]:
    if abstract_function == "公开反转":
        return ["主角处于低位或被质疑", "存在公开裁决或旁观场景", "对手暂时掌握规则优势"]
    if abstract_function == "资源争夺":
        return ["存在稀缺资源或资格入口", "竞争双方利益明确", "前文埋下资源价值或需求"]
    if abstract_function == "逃亡破局":
        return ["主角已暴露或被锁定", "追击方具有压倒性追踪优势", "场景允许路径选择或牺牲换位"]
    if abstract_function == "关系转折":
        return ["双方已有立场冲突或情感债务", "存在误判、误会或外部挑拨", "事件需要明确的信任变化"]
    if abstract_function == "传承筛选":
        return ["存在门槛、考核或择徒机制", "候选人之间存在比较", "前文已埋下资格或传承价值"]
    return [f"存在{conflict_engine}压力", f"事件围绕{abstract_function}展开"]


def _infer_candidate_beats_from_atom(atom: PlotAtom, abstract_function: str) -> List[str]:
    text = _source_text_from_atom(atom)
    beats: List[str] = ["铺垫入场"]
    if any(keyword in text for keyword in ("压制", "争夺", "追杀", "冲突", "误会", "试炼")):
        beats.append("冲突升级")
    if any(keyword in text for keyword in ("翻盘", "反制", "揭露", "破局", "和解", "结盟")):
        beats.append("局势转折")
    if any(keyword in text for keyword in ("代价", "后果", "受伤", "暴露", "关注")):
        beats.append("代价落地")
    if any(keyword in text for keyword in ("伏笔", "线索", "后续", "下一步", "更大")):
        beats.append("下一事件钩子")
    if abstract_function in {"公开反转", "低位逆袭"} and "局势转折" not in beats:
        beats.append("局势转折")
    if "冲突升级" not in beats:
        beats.insert(1, "冲突升级")
    if beats[-1] not in {"代价落地", "下一事件钩子"}:
        beats.append("余波收束")
    return _dedupe_state_tags(beats)


def _infer_candidate_beats_from_event(event: InducedEvent, abstract_function: str) -> List[str]:
    beats = ["铺垫入场"]
    if getattr(event, "hook_open", []):
        beats.append("冲突升级")
    if getattr(event, "state_outputs", []) or getattr(event, "relationship_delta", []):
        beats.append("局势转折")
    if getattr(event, "hook_close", []):
        beats.append("余波收束")
    if abstract_function in {"公开反转", "关系转折", "代价落地"}:
        beats.append("代价落地")
    if len(beats) < 4:
        beats.append("下一事件钩子")
    return _dedupe_state_tags(beats)


def _infer_state_delta_signature(state_delta: Any) -> List[str]:
    if not isinstance(state_delta, dict):
        return []
    tags: List[str] = []
    for key, value in state_delta.items():
        clean_key = str(key or "").strip()
        clean_value = str(value or "").strip()
        if clean_key and clean_value:
            tags.append(f"{clean_key}:{clean_value}")
    return _dedupe_state_tags(tags)


def _dedupe_state_tags(values: Iterable[Any]) -> List[str]:
    return dedupe_text_values([str(value or "").strip() for value in values if str(value or "").strip()])


def _extract_forbidden_source_details(
    raw_summary: Any,
    raw_characters: Any,
    raw_location: Any,
    cultivation_elements: Any,
) -> List[str]:
    details: List[str] = []
    details.extend(_extract_specific_terms(raw_summary))
    details.extend(flatten_text_values([raw_characters, raw_location, cultivation_elements]))
    filtered = []
    for detail in details:
        clean = str(detail or "").strip()
        if not clean or clean in _GENERIC_FORBIDDEN_DETAILS or len(clean) <= 1:
            continue
        filtered.append(clean)
    return dedupe_text_values(filtered)[:50]


def _extract_specific_terms(text: Any) -> List[str]:
    raw = str(text or "")
    fragments = []
    current = []
    for ch in raw:
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff":
            current.append(ch)
            continue
        if current:
            fragments.append("".join(current))
            current = []
    if current:
        fragments.append("".join(current))
    return [fragment for fragment in fragments if len(fragment) >= 2]


def _build_source_pattern_summary(
    abstract_function: str,
    conflict_engine: str,
    beat_signature: Sequence[Any],
    state_delta_signature: Sequence[Any],
) -> str:
    beats = " -> ".join(flatten_text_values(list(beat_signature))[:4]) or "阶段推进"
    states = "、".join(flatten_text_values(list(state_delta_signature))[:4]) or "状态变更"
    return f"该模板围绕{abstract_function}展开，由{conflict_engine}驱动，通常经历{beats}，最终引出{states}。"


def _match_role_template(text: str) -> Dict[str, Any]:
    haystack = str(text or "")
    for item in _ROLE_SLOT_TEMPLATE_LIBRARY:
        keywords = item.get("keywords", [])
        if keywords and any(keyword in haystack for keyword in keywords):
            return item
    return _ROLE_SLOT_TEMPLATE_LIBRARY[-1]


def _template_cluster_key(candidate: Dict[str, Any]) -> str:
    return "|".join(
        [
            normalize_text(candidate.get("abstract_function", "")),
            normalize_text(candidate.get("conflict_engine", "")),
            "-".join(_normalize_sequence(candidate.get("beat_signature", []))),
        ]
    )


def _template_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    abstract_score = 1.0 if normalize_text(left.get("abstract_function")) == normalize_text(right.get("abstract_function")) else 0.0
    conflict_score = 1.0 if normalize_text(left.get("conflict_engine")) == normalize_text(right.get("conflict_engine")) else _token_overlap(left.get("conflict_engine", ""), right.get("conflict_engine", ""))
    beat_score = _sequence_overlap(left.get("beat_signature", []), right.get("beat_signature", []))
    state_score = _sequence_overlap(left.get("state_delta_signature", []), right.get("state_delta_signature", []))
    role_score = _sequence_overlap(left.get("role_slot_signature", []), right.get("role_slot_signature", []))
    return round(abstract_score * 0.35 + conflict_score * 0.25 + beat_score * 0.2 + state_score * 0.1 + role_score * 0.1, 4)


def _template_cluster_rank_key(cluster: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
    return (
        int(cluster.get("support_count", 0)),
        len(cluster.get("novel_sources", [])),
        len(cluster.get("merged_beat_signature", [])),
        len(cluster.get("merged_state_delta_signature", [])),
        1 if cluster.get("forbidden_source_details", []) else 0,
        len(cluster.get("source_refs", [])),
    )


def _pick_representative_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(
        candidates,
        key=lambda item: (
            int(item.get("support_hint", 1) or 1),
            len(item.get("beat_signature", [])),
            len(item.get("state_delta_signature", [])),
            len(item.get("source_refs", [])),
        ),
    )


def _role_slots_from_signature(signature: Sequence[Any], role_template_name: str) -> Dict[str, str]:
    template_map = _ROLE_SLOT_MEANINGS.get(role_template_name, _ROLE_SLOT_MEANINGS["default_pressure"])
    ordered_slots = flatten_text_values(list(signature)) or list(template_map.keys())
    role_slots: Dict[str, str] = {}
    for slot in ordered_slots:
        clean_slot = str(slot or "").strip()
        if not clean_slot:
            continue
        role_slots[clean_slot] = template_map.get(clean_slot, "功能槽位")
    return role_slots


def _beat_sequence_from_signature(
    beat_signature: Sequence[Any],
    abstract_function: str,
    conflict_engine: str,
) -> List[Dict[str, Any]]:
    beats = flatten_text_values(list(beat_signature))
    if not beats:
        beats = _default_beats_for_function(abstract_function, conflict_engine)
    sequence: List[Dict[str, Any]] = []
    for index, beat in enumerate(beats, start=1):
        sequence.append(
            {
                "beat_index": index,
                "function": beat,
                "purpose": _BEAT_PURPOSES.get(beat, f"承接{abstract_function}推进"),
                "state_delta": _default_state_delta_for_beat(beat),
            }
        )
    return sequence


def _state_delta_map_from_signature(signature: Sequence[Any]) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    for item in flatten_text_values(list(signature)):
        if ":" in item:
            key, value = item.split(":", 1)
            delta[key.strip()] = value.strip()
    if not delta:
        delta = {
            "reputation": "optional",
            "resources": "optional",
            "relationship_shift": "optional",
            "open_hook": "+",
        }
    return delta


def _infer_variation_axes(abstract_function: str, conflict_engine: str) -> Dict[str, List[str]]:
    base = _VARIATION_AXES_MAP.get(abstract_function)
    if base:
        return dict(base)
    return {
        "scene": ["宗门议事", "边境据点", "商会交易场", "遗迹入口"],
        "pressure_method": [conflict_engine or "局势压迫", "资格卡位", "情报误导", "资源封锁"],
        "reversal_method": ["规则反用", "借势破局", "第三方矛盾", "代价换胜"],
        "cost": ["暴露底牌", "欠下人情", "受伤", "被注意到"],
    }


def _build_template_name(abstract_function: str, conflict_engine: str) -> str:
    return f"{abstract_function}模板({conflict_engine or '通用驱动'})"


def _infer_template_level(level_hints: Sequence[Any]) -> str:
    hints = set(flatten_text_values(list(level_hints)))
    if "volume" in hints:
        return "volume"
    if "micro" in hints and len(hints) == 1:
        return "micro"
    return "event"


def _default_beats_for_function(abstract_function: str, conflict_engine: str) -> List[str]:
    if abstract_function in {"公开反转", "低位逆袭"}:
        return ["铺垫入场", "冲突升级", "局势转折", "代价落地", "下一事件钩子"]
    if abstract_function == "逃亡破局":
        return ["铺垫入场", "冲突升级", "反向破局", "余波收束", "下一事件钩子"]
    if abstract_function == "关系转折":
        return ["铺垫入场", "冲突升级", "局势转折", "余波收束"]
    if conflict_engine in {"规则压迫", "权力审判"}:
        return ["铺垫入场", "冲突升级", "局势转折", "代价落地"]
    return ["铺垫入场", "冲突升级", "主线推进", "余波收束"]


def _default_state_delta_for_beat(beat: str) -> Dict[str, str]:
    if beat == "冲突升级":
        return {"pressure": "+", "enemy_attention": "+"}
    if beat == "局势转折":
        return {"reputation": "+", "secret_exposure": "optional"}
    if beat == "反向破局":
        return {"resources": "+/-", "relationship_shift": "optional"}
    if beat == "代价落地":
        return {"injury": "optional", "resources": "-", "open_hook": "+"}
    if beat == "下一事件钩子":
        return {"open_hook": "+"}
    return {"state_progress": "+"}


def _beat_blueprint_from_signature(signature: Sequence[Any]) -> List[Dict[str, Any]]:
    beats = flatten_text_values(list(signature)) or ["铺垫入场", "冲突升级", "局势转折", "余波收束"]
    chapter_share = round(1 / max(len(beats), 1), 3)
    blueprint: List[Dict[str, Any]] = []
    for index, beat in enumerate(beats, start=1):
        blueprint.append(
            {
                "beat_index": index,
                "chapter_span": [index, index],
                "chapter_share": chapter_share,
                "primary_function": beat,
                "secondary_functions": ["代价落地"] if beat == "局势转折" else [],
                "purpose": _BEAT_PURPOSES.get(beat, "推进模板结构"),
            }
        )
    return blueprint


def _pseudo_events_from_atoms(atoms: List[PlotAtom]) -> List[Any]:
    grouped: Dict[str, List[PlotAtom]] = defaultdict(list)
    for atom in atoms:
        grouped[atom.arc_name or "Unnamed Arc"].append(atom)
    pseudo_events = []
    for arc_name, bucket in grouped.items():
        pseudo_events.append(
            type(
                "PseudoEvent",
                (),
                {
                    "arc_name": arc_name,
                    "summary": " / ".join(_clip_text(atom.summary or atom.core_action, 30) for atom in bucket[:3]),
                    "conflict_hint": coerce_text(_top_values(atom.conflict_type for atom in bucket, limit=2)),
                    "function_hint": coerce_text(_top_values(atom.narrative_function for atom in bucket, limit=2)),
                },
            )()
        )
    return pseudo_events


def _merge_tag_lists(values: Iterable[Any]) -> List[str]:
    merged: List[str] = []
    for value in values:
        merged.extend(flatten_text_values([value]))
    return dedupe_text_values(merged)


def _chapter_count_range(values: Iterable[Any]) -> List[int]:
    numbers = [max(1, int(value or 1)) for value in values]
    if not numbers:
        return [1, 1]
    return [min(numbers), max(numbers)]


def _normalize_sequence(values: Sequence[Any]) -> List[str]:
    return [normalize_text(item) for item in flatten_text_values(list(values))]


def _sequence_overlap(left: Sequence[Any], right: Sequence[Any]) -> float:
    left_set = set(_normalize_sequence(left))
    right_set = set(_normalize_sequence(right))
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _token_overlap(left: Any, right: Any) -> float:
    left_tokens = set(_normalize_sequence([left]))
    right_tokens = set(_normalize_sequence([right]))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _top_values(values: Iterable[Any], limit: int = 3) -> List[str]:
    counter = Counter(value for value in flatten_text_values(list(values)) if value)
    return [item for item, _count in counter.most_common(limit)]


def _clip_text(text: Any, limit: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."
