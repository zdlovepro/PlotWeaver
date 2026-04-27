from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
from tenacity import retry, stop_after_attempt, wait_exponential
from pipeline.core.common_json import read_json_file, safe_json_load, write_json_file
from pipeline.core.common_text import dedupe_texts, flatten_text_values
from pipeline.core.story_models import Character, CharacterSheet, EventRolePlan, NarrativeSkeleton, SkeletonNode, StageNetwork
from pipeline.core.utils import chat_completion_json, get_deepseek_client
from pipeline.core.world_building_core import FusedWorld
from pipeline.step1_chunking import VolumeArc
from pipeline.step2_extraction import PlotAtom
from pipeline.step3_event_induction import InducedEvent


_DEFAULT_ROLE_SLOTS = ["阻碍者", "消息提供者", "短期盟友", "旁观者"]
_SKELETON_BEAM_WIDTH = 6
_SKELETON_SLOT_CANDIDATE_LIMIT = 6


@dataclass
class _SkeletonSearchPath:
    nodes: List[SkeletonNode] = field(default_factory=list)
    used_refs: set[str] = field(default_factory=set)
    usage_counts: Dict[str, int] = field(default_factory=dict)
    recent_picks: List[Dict[str, Any]] = field(default_factory=list)
    ledger_outputs: set[str] = field(default_factory=set)
    open_hooks: set[str] = field(default_factory=set)
    character_keys: set[str] = field(default_factory=set)
    score: float = 0.0


def _quality_chat_completion_json(client, **kwargs) -> str:
    return chat_completion_json(client, model_name=config.DEEPSEEK_LAST_STEPS_MODEL, **kwargs)

_CORE_CAST_SCHEMA = {
    "protagonist": {
        "name": "原创姓名",
        "dao_heart": "核心执念",
        "combat_style": "战斗风格",
        "personality_flaw": "性格缺陷",
        "background": "出身背景",
    },
    "core_cast": [
        {
            "name": "角色名",
            "role": "师尊/宿敌/盟友/红颜",
            "bond_depth": "核心",
            "entry_event": "事件1",
            "exit_event": "大结局",
            "return_event": "",
            "dao_heart": "执念",
            "combat_style": "战斗风格",
            "personality_flaw": "缺陷",
            "background": "背景",
            "role_slots": ["护道者", "规则维护者"],
        }
    ],
}

_VOLUME_CAST_SCHEMA = {
    "volume_cast": [
        {
            "name": "角色名",
            "role": "卷级人物定位",
            "bond_depth": "卷级",
            "entry_event": "事件3",
            "exit_event": "事件7",
            "return_event": "",
            "dao_heart": "执念",
            "combat_style": "战斗方式",
            "personality_flaw": "缺陷",
            "background": "背景",
            "role_slots": ["掌柜主事", "维持秩序者"],
        }
    ]
}

_TRANSIENT_CAST_SCHEMA = {
    "transient_cast": [
        {
            "name": "角色名",
            "role": "事件功能位",
            "bond_depth": "过客",
            "entry_event": "事件8",
            "exit_event": "事件9",
            "return_event": "",
            "dao_heart": "个人欲望",
            "combat_style": "手段",
            "personality_flaw": "弱点",
            "background": "一句背景",
            "role_slots": ["追杀者", "线人"],
        }
    ]
}

_RELATIONSHIP_SCHEMA = {
    "relationship_networks": [
        {
            "stage": "卷一前中段",
            "active_characters": ["角色A", "角色B"],
            "relationship_status": "局势、利益和关系变化",
        }
    ]
}


def save_node_map(filename: str, per_novel_nodes: Dict[str, List[SkeletonNode]]) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    payload = {novel: [asdict(node) for node in nodes] for novel, nodes in per_novel_nodes.items()}
    return write_json_file(path, payload)


def load_node_map(filename: str) -> Dict[str, List[SkeletonNode]]:
    path = Path(config.INTERMEDIATE_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(f"Skeleton node map not found: {path}")
    raw = read_json_file(path)
    return {novel: [SkeletonNode(**item) for item in nodes] for novel, nodes in raw.items()}


def save_skeleton_snapshot(filename: str, skeleton: NarrativeSkeleton) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    payload = {"base_novel": skeleton.base_novel, "nodes": [asdict(node) for node in skeleton.nodes], "character_sheet": None}
    if skeleton.character_sheet:
        payload["character_sheet"] = {
            "protagonist": asdict(skeleton.character_sheet.protagonist),
            "supporting": [asdict(item) for item in skeleton.character_sheet.supporting],
            "relationship_networks": [asdict(item) for item in skeleton.character_sheet.relationship_networks],
            "event_role_plans": [asdict(item) for item in skeleton.character_sheet.event_role_plans],
        }
    return write_json_file(path, payload)


def load_skeleton_snapshot(filename: str) -> NarrativeSkeleton:
    path = Path(config.INTERMEDIATE_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(f"Skeleton snapshot not found: {path}")
    raw = read_json_file(path)
    character_sheet = None
    raw_sheet = raw.get("character_sheet")
    if isinstance(raw_sheet, dict):
        character_sheet = CharacterSheet(
            protagonist=Character(**raw_sheet.get("protagonist", {})),
            supporting=[Character(**item) for item in raw_sheet.get("supporting", [])],
            relationship_networks=[StageNetwork(**item) for item in raw_sheet.get("relationship_networks", [])],
            event_role_plans=[EventRolePlan(**item) for item in raw_sheet.get("event_role_plans", [])],
        )
    return NarrativeSkeleton(
        base_novel=raw.get("base_novel", ""),
        nodes=[SkeletonNode(**item) for item in raw.get("nodes", [])],
        character_sheet=character_sheet,
    )


_safe_json_load = safe_json_load
_flatten_text_values = flatten_text_values
_dedupe_texts = dedupe_texts


def _atom_text(value: Any) -> str:
    return " / ".join(_flatten_text_values([value]))


def _realm_level_from_arc(arc_name: str) -> int:
    match = re.search(r"(\d+)", arc_name)
    return int(match.group(1)) if match else 0


def _event_label(index: int) -> str:
    return f"事件{index + 1}"


def _fallback_chapter_blueprint(chapter_count: int) -> List[Dict[str, Any]]:
    total = max(4, min(10, int(chapter_count or 4)))
    blueprint: List[Dict[str, Any]] = []
    for idx in range(total):
        if idx == 0:
            primary = "铺垫入场"
            secondary = ["人物亮相"]
        elif idx < total // 2:
            primary = "冲突升级"
            secondary = []
        elif idx == total // 2:
            primary = "局势转折"
            secondary = ["信息翻面"]
        elif idx >= total - 2:
            primary = "余波收束"
            secondary = ["代价落地", "下一事件钩子"]
        else:
            primary = "主线推进"
            secondary = []
        blueprint.append(
            {
                "beat_index": idx + 1,
                "chapter_span": [idx + 1, idx + 1],
                "primary_function": primary,
                "secondary_functions": secondary,
                "purpose": f"第{idx + 1}章承担{primary}",
            }
        )
    return blueprint


def _blueprint_chapter_count(chapter_blueprint: List[Dict[str, Any]]) -> int:
    last = 0
    for beat in chapter_blueprint:
        span = beat.get("chapter_span", [])
        if isinstance(span, list) and len(span) == 2:
            last = max(last, int(span[1]))
    return last


def _fit_blueprint_to_chapter_count(chapter_blueprint: List[Dict[str, Any]], target_chapter_count: int) -> List[Dict[str, Any]]:
    if not chapter_blueprint:
        return _fallback_chapter_blueprint(target_chapter_count)

    total_chapters = max(1, int(target_chapter_count or 1))
    if total_chapters < len(chapter_blueprint):
        sampled_indices = [round(i * (len(chapter_blueprint) - 1) / max(total_chapters - 1, 1)) for i in range(total_chapters)]
        reduced: List[Dict[str, Any]] = []
        for new_index, source_index in enumerate(sampled_indices, start=1):
            beat_copy = dict(chapter_blueprint[source_index])
            beat_copy["beat_index"] = new_index
            beat_copy["chapter_share"] = round(1 / total_chapters, 3)
            beat_copy["chapter_span"] = [new_index, new_index]
            reduced.append(beat_copy)
        return reduced

    shares: List[float] = []
    for beat in chapter_blueprint:
        share = float(beat.get("chapter_share", 0.0) or 0.0)
        if share <= 0:
            span = beat.get("chapter_span", [])
            if isinstance(span, list) and len(span) == 2:
                share = max(1, int(span[1]) - int(span[0]) + 1)
            else:
                share = 1.0
        shares.append(share)

    total_share = sum(shares) or float(len(shares))
    normalized = [share / total_share for share in shares]
    raw_lengths = [max(1.0, ratio * total_chapters) for ratio in normalized]
    lengths = [max(1, int(round(value))) for value in raw_lengths]

    delta = total_chapters - sum(lengths)
    if delta != 0:
        order = sorted(range(len(lengths)), key=lambda idx: raw_lengths[idx] - lengths[idx], reverse=(delta > 0))
        pointer = 0
        while delta != 0 and order:
            target_idx = order[pointer % len(order)]
            if delta > 0:
                lengths[target_idx] += 1
                delta -= 1
            else:
                if lengths[target_idx] > 1:
                    lengths[target_idx] -= 1
                    delta += 1
            pointer += 1

    fitted: List[Dict[str, Any]] = []
    cursor = 1
    for beat_index, beat in enumerate(chapter_blueprint, start=1):
        beat_length = lengths[beat_index - 1] if beat_index - 1 < len(lengths) else 1
        beat_copy = dict(beat)
        beat_copy["beat_index"] = beat_index
        beat_copy["chapter_share"] = round(normalized[beat_index - 1], 3) if beat_index - 1 < len(normalized) else round(1 / max(len(chapter_blueprint), 1), 3)
        beat_copy["chapter_span"] = [cursor, cursor + beat_length - 1]
        fitted.append(beat_copy)
        cursor += beat_length
    if fitted:
        fitted[-1]["chapter_span"][1] = total_chapters
    return fitted


def _select_base_novel(client, novel_arcs: Dict[str, List[VolumeArc]], fused_world: FusedWorld) -> str:
    del client, fused_world
    return max(novel_arcs, key=lambda novel: len(novel_arcs[novel])) if novel_arcs else ""


def _match_template_for_induced_event(event: InducedEvent, fused_world: FusedWorld) -> Dict[str, Any]:
    templates = fused_world.event_flow_templates or []
    if not templates:
        return {
            "name": f"generic_{max(4, event.chapter_count)}_chapter_event",
            "beat_blueprint": _fallback_chapter_blueprint(event.chapter_count),
            "required_coverage": ["铺垫推进", "冲突升级", "局势转折", "代价落地", "下一事件钩子"],
        }
    best = templates[0]
    best_score = -999.0
    for template in templates:
        score = 0.0
        chapter_range = template.get("chapter_count_range", [4, 8])
        if isinstance(chapter_range, list) and len(chapter_range) == 2:
            low, high = int(chapter_range[0]), int(chapter_range[1])
            if low <= event.chapter_count <= high:
                score += 3
        if event.conflict_hint and event.conflict_hint in template.get("dominant_conflicts", []):
            score += 4
        if score > best_score:
            best_score = score
            best = template
    return best


def _role_slots_for_induced_event(event: InducedEvent, fused_world: FusedWorld, matched_template: Dict[str, Any]) -> List[str]:
    haystack = " ".join(
        _flatten_text_values([event.summary, event.conflict_hint, event.function_hint, matched_template.get("name", "")])
    ).lower()
    for item in fused_world.role_slot_templates or []:
        keywords = [str(keyword).lower() for keyword in item.get("keywords", [])]
        if keywords and any(keyword and keyword in haystack for keyword in keywords):
            slots = item.get("role_slots", [])
            if slots:
                return slots[:6]
    for template in fused_world.event_templates or []:
        if event.conflict_hint and template.get("conflict_type") == event.conflict_hint:
            slots = template.get("role_slots", [])
            if slots:
                return slots[:6]
    return _DEFAULT_ROLE_SLOTS[:]


def _pacing_role_for_induced_event(event: InducedEvent, index: int, total: int) -> str:
    if event.function_hint:
        return event.function_hint
    ratio = index / max(total - 1, 1) if total > 1 else 0
    if ratio < 0.12:
        return "开篇落点"
    if ratio > 0.88:
        return "卷末收束"
    if event.chapter_count >= 6:
        return "阶段主事件"
    return "主线推进"


def _nodes_from_induced_events(induced_events: List[InducedEvent], fused_world: FusedWorld) -> List[SkeletonNode]:
    grouped_by_arc: Dict[str, List[InducedEvent]] = {}
    arc_order: List[str] = []
    for event in induced_events:
        arc_name = event.arc_name or "未命名卷"
        if arc_name not in grouped_by_arc:
            grouped_by_arc[arc_name] = []
            arc_order.append(arc_name)
        grouped_by_arc[arc_name].append(event)

    ordered: List[InducedEvent] = []
    for arc_name in arc_order:
        ordered.extend(
            sorted(
                grouped_by_arc[arc_name],
                key=lambda item: (item.chapter_start, item.chapter_end, item.event_id),
            )
        )
    total = len(ordered)
    nodes: List[SkeletonNode] = []
    for idx, event in enumerate(ordered):
        template = _match_template_for_induced_event(event, fused_world)
        template_blueprint = template.get("beat_blueprint") or _fallback_chapter_blueprint(event.chapter_count)
        blueprint = _fit_blueprint_to_chapter_count(template_blueprint, event.chapter_count)
        nodes.append(
            SkeletonNode(
                node_id=event.event_id,
                arc_name=event.arc_name,
                realm_level=_realm_level_from_arc(event.arc_name),
                pacing_role=_pacing_role_for_induced_event(event, idx, total),
                original_summary=event.summary,
                conflict_hint=event.conflict_hint,
                function_hint=event.function_hint,
                role_slots=_role_slots_for_induced_event(event, fused_world, template),
                template_hint=template.get("name", ""),
                source_event_ids=event.source_atom_ids[:],
                chapter_start=event.chapter_start,
                chapter_end=event.chapter_end,
                chapter_count=event.chapter_count,
                chapter_blueprint=blueprint,
                character_keys=event.character_keys[:],
                source_novels=[event.novel_source],
                logic_card={
                    "required_coverage": template.get("required_coverage", []),
                    "template_name": template.get("name", ""),
                    "preconditions": event.state_inputs[:],
                    "state_outputs": event.state_outputs[:],
                    "power_state": event.power_state or f"第{max(1, _realm_level_from_arc(event.arc_name))}阶段附近",
                    "identity_state": event.identity_state,
                    "relationship_delta": "；".join(event.relationship_delta[:3]),
                    "timeline_stage": _pacing_role_for_induced_event(event, idx, total),
                    "hook_open": event.hook_open[:],
                    "hook_close": event.hook_close[:],
                    "resource_delta": event.resource_delta[:],
                },
            )
        )
    return nodes


def _extract_skeleton_nodes(arcs: List[VolumeArc], atoms: List[PlotAtom], fused_world: FusedWorld) -> List[SkeletonNode]:
    from pipeline.step8_skeleton_extraction import _extract_skeleton_nodes as extract_impl

    return extract_impl(arcs, atoms, fused_world)


def _blend_source_skeleton_nodes(base_novel: str, per_novel_nodes: Dict[str, List[SkeletonNode]]) -> List[SkeletonNode]:
    if not per_novel_nodes:
        return []
    base_nodes = per_novel_nodes.get(base_novel) or next(iter(per_novel_nodes.values()))
    candidate_pool = _flatten_node_candidates(per_novel_nodes)
    if not candidate_pool:
        return base_nodes

    slots = [
        _build_pacing_slot(
            reference_novel=base_novel,
            base_node=base_node,
            slot_index=index,
            total_slots=len(base_nodes),
            source_novel_count=len(per_novel_nodes),
        )
        for index, base_node in enumerate(base_nodes)
    ]
    source_limits = _build_source_target_limits(base_novel, per_novel_nodes, len(base_nodes))
    initial_path = _SkeletonSearchPath(usage_counts={novel: 0 for novel in per_novel_nodes})
    beam: List[_SkeletonSearchPath] = [initial_path]

    for index, slot in enumerate(slots):
        expanded_paths: List[_SkeletonSearchPath] = []
        for path in beam:
            ranked_candidates = _top_path_candidates(
                slot=slot,
                candidate_pool=candidate_pool,
                path=path,
                source_limits=source_limits,
                limit=_SKELETON_SLOT_CANDIDATE_LIMIT,
            )
            if not ranked_candidates:
                fallback_node = _build_slot_fallback_node(slot, base_nodes[index], index)
                expanded_paths.append(_extend_search_path(path, fallback_node, candidate_score=-0.4))
                continue

            for candidate_score, candidate in ranked_candidates:
                expanded_paths.append(_extend_search_path(path, _build_slot_node(slot, candidate, index), candidate_score))

        if not expanded_paths:
            return [_build_slot_fallback_node(slot, base_nodes[idx], idx) for idx, slot in enumerate(slots)]
        beam = _prune_search_paths(expanded_paths, width=_SKELETON_BEAM_WIDTH)

    if not beam:
        return base_nodes
    return beam[0].nodes


def _flatten_node_candidates(per_novel_nodes: Dict[str, List[SkeletonNode]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for novel_name, nodes in per_novel_nodes.items():
        total = len(nodes)
        for source_index, node in enumerate(nodes):
            progress_ratio = source_index / max(total - 1, 1) if total > 1 else 0.0
            candidates.append(
                {
                    "novel": novel_name,
                    "index": source_index,
                    "total": total,
                    "progress_ratio": progress_ratio,
                    "node": node,
                }
            )
    return candidates


def _build_source_target_limits(base_novel: str, per_novel_nodes: Dict[str, List[SkeletonNode]], slot_count: int) -> Dict[str, int]:
    if not per_novel_nodes:
        return {}
    novel_count = max(len(per_novel_nodes), 1)
    even_share = max(1, round(slot_count / novel_count))
    limits: Dict[str, int] = {}
    for novel_name, nodes in per_novel_nodes.items():
        available = max(len(nodes), 1)
        limit = min(available, even_share)
        if novel_name == base_novel and novel_count > 1:
            limit = min(available, max(1, round(slot_count / (novel_count + 1))))
        limits[novel_name] = max(1, limit)
    return limits


def _top_path_candidates(
    slot: Dict[str, Any],
    candidate_pool: List[Dict[str, Any]],
    path: _SkeletonSearchPath,
    source_limits: Dict[str, int],
    limit: int,
) -> List[tuple[float, Dict[str, Any]]]:
    ranked: List[tuple[float, Dict[str, Any]]] = []
    for candidate in candidate_pool:
        candidate_ref = _candidate_ref(candidate)
        if candidate_ref in path.used_refs:
            continue
        base_score = _score_candidate_for_slot(slot, candidate, path.recent_picks, path.usage_counts, source_limits)
        if base_score <= -1e8:
            continue
        total_score = base_score + _state_ledger_score(path, candidate["node"])
        ranked.append((total_score, candidate))
    ranked.sort(
        key=lambda item: (
            -item[0],
            path.usage_counts.get(item[1]["novel"], 0),
            item[1]["novel"],
            item[1]["node"].node_id,
        )
    )
    return ranked[:limit]


def _extend_search_path(path: _SkeletonSearchPath, node: SkeletonNode, candidate_score: float) -> _SkeletonSearchPath:
    logic_card = _normalize_logic_card(node.logic_card, node.pacing_role, node.chapter_count)
    node.logic_card = logic_card
    heuristic_issues = _heuristic_transition_flags(path.nodes[-2:], node, None)
    transition_penalty = len(heuristic_issues) * 1.2

    new_nodes = path.nodes + [node]
    new_used_refs = set(path.used_refs)
    if node.selection_ref:
        new_used_refs.add(node.selection_ref)

    new_usage_counts = dict(path.usage_counts)
    if node.selection_source_novel:
        new_usage_counts[node.selection_source_novel] = new_usage_counts.get(node.selection_source_novel, 0) + 1

    new_recent_picks = list(path.recent_picks)
    if node.selection_source_novel:
        new_recent_picks.append(
            {
                "novel": node.selection_source_novel,
                "index": node.selection_source_index,
                "progress_ratio": 0.0,
                "node": node,
            }
        )
        new_recent_picks[:] = new_recent_picks[-3:]

    new_ledger_outputs = set(path.ledger_outputs)
    new_ledger_outputs.update(_extract_logic_terms(logic_card.get("state_outputs", [])))
    new_ledger_outputs.update(_extract_logic_terms([logic_card.get("identity_state", ""), logic_card.get("relationship_delta", ""), logic_card.get("power_state", "")]))

    new_open_hooks = set(path.open_hooks)
    new_open_hooks.difference_update(_extract_logic_terms(logic_card.get("hook_close", [])))
    new_open_hooks.update(_extract_logic_terms(logic_card.get("hook_open", [])))

    new_character_keys = set(path.character_keys)
    new_character_keys.update(node.character_keys)

    diversity_bonus = len({item.selection_source_novel for item in new_nodes if item.selection_source_novel}) * 0.15
    new_score = path.score + candidate_score + diversity_bonus - transition_penalty
    return _SkeletonSearchPath(
        nodes=new_nodes,
        used_refs=new_used_refs,
        usage_counts=new_usage_counts,
        recent_picks=new_recent_picks,
        ledger_outputs=new_ledger_outputs,
        open_hooks=new_open_hooks,
        character_keys=new_character_keys,
        score=new_score,
    )


def _prune_search_paths(paths: List[_SkeletonSearchPath], width: int) -> List[_SkeletonSearchPath]:
    deduped: Dict[str, _SkeletonSearchPath] = {}
    for path in paths:
        key = "|".join(node.selection_ref or node.node_id for node in path.nodes)
        current = deduped.get(key)
        if current is None or _search_path_sort_key(path) < _search_path_sort_key(current):
            deduped[key] = path
    ranked = sorted(deduped.values(), key=_search_path_sort_key)
    return ranked[:width]


def _search_path_sort_key(path: _SkeletonSearchPath) -> Tuple[float, float, int, int]:
    distinct_sources = len({node.selection_source_novel for node in path.nodes if node.selection_source_novel})
    unresolved_hooks = len(path.open_hooks)
    return (-path.score, -distinct_sources, unresolved_hooks, len(path.nodes))


def _state_ledger_score(path: _SkeletonSearchPath, node: SkeletonNode) -> float:
    logic_card = _normalize_logic_card(node.logic_card, node.pacing_role, node.chapter_count)
    score = 0.0

    preconditions = _extract_logic_terms(logic_card.get("preconditions", []))
    if preconditions:
        overlap = len(preconditions & path.ledger_outputs)
        if overlap:
            score += overlap * 1.4
        else:
            score -= min(len(preconditions), 3) * 1.2

    hook_close = _extract_logic_terms(logic_card.get("hook_close", []))
    if hook_close:
        matched = len(hook_close & path.open_hooks)
        if matched:
            score += matched * 1.1
        elif path.open_hooks:
            score -= 0.6

    future_open_hooks = (path.open_hooks - hook_close) | _extract_logic_terms(logic_card.get("hook_open", []))
    if len(future_open_hooks) > 6:
        score -= (len(future_open_hooks) - 6) * 0.5

    if path.character_keys and node.character_keys:
        overlap = len(set(node.character_keys) & path.character_keys)
        if overlap:
            score += min(overlap, 3) * 0.5
        elif len(path.nodes) >= 1 and node.pacing_role not in {"开篇落点", "铺垫与试探"}:
            score -= 0.4

    return score


def _build_pacing_slot(
    reference_novel: str,
    base_node: SkeletonNode,
    slot_index: int,
    total_slots: int,
    source_novel_count: int,
) -> Dict[str, Any]:
    chapter_blueprint = [dict(beat) for beat in (base_node.chapter_blueprint or _fallback_chapter_blueprint(base_node.chapter_count or 4))]
    chapter_count = _blueprint_chapter_count(chapter_blueprint) or max(1, base_node.chapter_count or 1)
    progress_ratio = slot_index / max(total_slots - 1, 1) if total_slots > 1 else 0.0
    return {
        "reference_novel": reference_novel,
        "slot_index": slot_index,
        "total_slots": total_slots,
        "progress_ratio": progress_ratio,
        "source_novel_count": source_novel_count,
        "arc_name": base_node.arc_name,
        "realm_level": base_node.realm_level,
        "pacing_role": base_node.pacing_role,
        "function_hint": base_node.function_hint,
        "conflict_hint": base_node.conflict_hint,
        "role_slots": base_node.role_slots[:],
        "template_hint": base_node.template_hint,
        "chapter_count": chapter_count,
        "chapter_blueprint": chapter_blueprint,
        "character_keys": base_node.character_keys[:],
        "logic_card": dict(base_node.logic_card or {}),
        "excluded_node_id": base_node.node_id,
        "excluded_event_ids": set(base_node.source_event_ids),
    }


def _select_candidate_for_slot(
    slot: Dict[str, Any],
    candidate_pool: List[Dict[str, Any]],
    used_refs: set[str],
    recent_picks: List[Dict[str, Any]],
    usage_counts: Dict[str, int],
    source_limits: Dict[str, int],
) -> Optional[Dict[str, Any]]:
    ranked: List[tuple[float, Dict[str, Any]]] = []
    for candidate in candidate_pool:
        if _candidate_ref(candidate) in used_refs:
            continue
        score = _score_candidate_for_slot(slot, candidate, recent_picks, usage_counts, source_limits)
        if score <= -1e8:
            continue
        ranked.append((score, candidate))
    if not ranked:
        return None
    ranked.sort(
        key=lambda item: (
            -item[0],
            usage_counts.get(item[1]["novel"], 0),
            0 if item[1]["novel"] != slot["reference_novel"] else 1,
            abs(item[1]["index"] - slot["slot_index"]),
            item[1]["novel"],
            item[1]["node"].node_id,
        )
    )
    return ranked[0][1]


def _score_candidate_for_slot(
    slot: Dict[str, Any],
    candidate: Dict[str, Any],
    recent_picks: List[Dict[str, Any]],
    usage_counts: Dict[str, int],
    source_limits: Dict[str, int],
) -> float:
    node: SkeletonNode = candidate["node"]
    novel_name = str(candidate["novel"])
    source_index = int(candidate["index"])

    if novel_name == slot["reference_novel"] and node.node_id == slot["excluded_node_id"]:
        return -1e9
    if slot["excluded_event_ids"] and any(event_id in slot["excluded_event_ids"] for event_id in node.source_event_ids):
        return -1e9

    score = 0.0
    if slot["pacing_role"] and node.pacing_role == slot["pacing_role"]:
        score += 3.0
    if slot["function_hint"] and node.function_hint == slot["function_hint"]:
        score += 2.4
    if slot["template_hint"] and node.template_hint == slot["template_hint"]:
        score += 1.5
    if slot["conflict_hint"] and node.conflict_hint == slot["conflict_hint"]:
        score += 0.8
    if node.arc_name == slot["arc_name"]:
        score += 0.4

    score -= abs((node.chapter_count or 1) - slot["chapter_count"]) * 0.7
    role_overlap = len(set(node.role_slots) & set(slot["role_slots"]))
    score += role_overlap * 0.45
    progress_gap = abs(float(candidate.get("progress_ratio", 0.0)) - float(slot["progress_ratio"]))
    score -= progress_gap * 6.0
    if progress_gap <= 0.12:
        score += 1.2
    elif progress_gap >= 0.45:
        score -= 3.0
    score += _transition_reasonableness_score(candidate, recent_picks)

    current_usage = usage_counts.get(novel_name, 0)
    score -= current_usage * 0.9
    soft_limit = source_limits.get(novel_name, max(slot["total_slots"], 1))
    if current_usage >= soft_limit:
        score -= (current_usage - soft_limit + 1) * 2.5

    if recent_picks:
        prev_pick = recent_picks[-1]
        if prev_pick["novel"] == novel_name:
            score -= 2.8
            distance = abs(int(prev_pick["index"]) - source_index)
            if distance <= 1:
                score -= 4.2
            elif distance <= 3:
                score -= 2.0
    if len(recent_picks) >= 2 and all(item["novel"] == novel_name for item in recent_picks[-2:]):
        score -= 5.5

    if novel_name == slot["reference_novel"] and slot["source_novel_count"] > 1:
        score -= 2.2
        distance_from_slot = abs(source_index - slot["slot_index"])
        score += min(distance_from_slot, 10) * 0.25
        if distance_from_slot <= 1:
            score -= 3.5
    elif slot["source_novel_count"] > 1:
        score += 0.8

    return score


def _fallback_candidate_for_slot(
    slot: Dict[str, Any],
    candidate_pool: List[Dict[str, Any]],
    used_refs: set[str],
) -> Optional[Dict[str, Any]]:
    ranked: List[tuple[int, int, Dict[str, Any]]] = []
    for candidate in candidate_pool:
        if _candidate_ref(candidate) in used_refs:
            continue
        node: SkeletonNode = candidate["node"]
        if candidate["novel"] == slot["reference_novel"] and node.node_id == slot["excluded_node_id"]:
            continue
        same_ref = 1 if candidate["novel"] == slot["reference_novel"] else 0
        distance = abs(int(candidate["index"]) - slot["slot_index"])
        ranked.append((same_ref, -distance, candidate))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1], item[2]["novel"], item[2]["node"].node_id))
    return ranked[0][2]


def _transition_reasonableness_score(candidate: Dict[str, Any], recent_picks: List[Dict[str, Any]]) -> float:
    if not recent_picks:
        return 0.0

    node: SkeletonNode = candidate["node"]
    prev_node: SkeletonNode = recent_picks[-1]["node"]
    score = 0.0

    if prev_node.pacing_role and node.pacing_role and prev_node.pacing_role == node.pacing_role:
        score -= 0.7
    if prev_node.function_hint and node.function_hint and prev_node.function_hint == node.function_hint:
        score -= 0.6
    if prev_node.conflict_hint and node.conflict_hint and prev_node.conflict_hint == node.conflict_hint:
        score -= 0.35

    chapter_delta = abs(int(prev_node.chapter_count or 1) - int(node.chapter_count or 1))
    score -= chapter_delta * 0.08

    if recent_picks[-1]["novel"] == candidate["novel"]:
        prev_index = int(recent_picks[-1]["index"])
        current_index = int(candidate["index"])
        if current_index < prev_index:
            score -= min(prev_index - current_index, 6) * 1.2
        else:
            score += min(current_index - prev_index, 4) * 0.1

    return score


def _candidate_ref(candidate: Dict[str, Any]) -> str:
    return f"{candidate['novel']}::{candidate['node'].node_id}"


def _coerce_logic_list(value: Any) -> List[str]:
    return _dedupe_texts(_flatten_text_values([value]))


def _normalize_logic_card(logic_card: Dict[str, Any], pacing_role: str, chapter_count: int) -> Dict[str, Any]:
    card = dict(logic_card or {})
    card["preconditions"] = _coerce_logic_list(card.get("preconditions", []))
    card["state_outputs"] = _coerce_logic_list(card.get("state_outputs", []))
    card["hook_open"] = _coerce_logic_list(card.get("hook_open", []))
    card["hook_close"] = _coerce_logic_list(card.get("hook_close", []))
    card["required_coverage"] = _coerce_logic_list(card.get("required_coverage", []))
    card["resource_delta"] = _coerce_logic_list(card.get("resource_delta", []))
    card.setdefault("power_state", f"{max(1, int(chapter_count or 1))}章级事件强度")
    card.setdefault("identity_state", "")
    card.setdefault("relationship_delta", "")
    card.setdefault("timeline_stage", pacing_role or "主线推进")
    return card


def _merge_logic_cards(slot_logic: Dict[str, Any], node_logic: Dict[str, Any], pacing_role: str, chapter_count: int) -> Dict[str, Any]:
    slot_card = _normalize_logic_card(slot_logic, pacing_role, chapter_count)
    node_card = _normalize_logic_card(node_logic, pacing_role, chapter_count)
    return {
        "required_coverage": _dedupe_texts(slot_card.get("required_coverage", []) + node_card.get("required_coverage", [])),
        "template_name": str(node_card.get("template_name") or slot_card.get("template_name") or "").strip(),
        "preconditions": node_card.get("preconditions") or slot_card.get("preconditions", []),
        "state_outputs": node_card.get("state_outputs") or slot_card.get("state_outputs", []),
        "power_state": str(node_card.get("power_state") or slot_card.get("power_state") or f"{max(1, int(chapter_count or 1))}章级事件强度"),
        "identity_state": str(node_card.get("identity_state") or slot_card.get("identity_state") or ""),
        "relationship_delta": str(node_card.get("relationship_delta") or slot_card.get("relationship_delta") or ""),
        "timeline_stage": pacing_role or str(node_card.get("timeline_stage") or slot_card.get("timeline_stage") or "主线推进"),
        "hook_open": _dedupe_texts(node_card.get("hook_open", []) + slot_card.get("hook_open", [])),
        "hook_close": _dedupe_texts(node_card.get("hook_close", []) + slot_card.get("hook_close", [])),
        "resource_delta": _dedupe_texts(node_card.get("resource_delta", []) + slot_card.get("resource_delta", [])),
    }


def _build_slot_node(slot: Dict[str, Any], candidate: Dict[str, Any], index: int) -> SkeletonNode:
    node: SkeletonNode = candidate["node"]
    raw_blueprint = [dict(beat) for beat in (slot["chapter_blueprint"] or _fallback_chapter_blueprint(slot["chapter_count"]))]
    chapter_count = slot["chapter_count"] or max(1, node.chapter_count or 1)
    chapter_blueprint = _fit_blueprint_to_chapter_count(raw_blueprint, chapter_count)
    logic_card = _merge_logic_cards(slot.get("logic_card", {}), node.logic_card or {}, slot.get("pacing_role", ""), slot.get("chapter_count", chapter_count))
    return SkeletonNode(
        node_id=f"fused_event_{index + 1}",
        arc_name=slot["arc_name"],
        realm_level=max(int(slot["realm_level"]), int(node.realm_level)),
        pacing_role=slot["pacing_role"],
        original_summary=node.original_summary,
        conflict_hint=node.conflict_hint,
        function_hint=node.function_hint or slot["function_hint"],
        role_slots=_dedupe_texts(list(slot["role_slots"]) + list(node.role_slots))[:8],
        template_hint=slot["template_hint"] or node.template_hint,
        source_event_ids=_dedupe_texts(node.source_event_ids),
        chapter_start=1,
        chapter_end=chapter_count,
        chapter_count=chapter_count,
        chapter_blueprint=chapter_blueprint,
        character_keys=_dedupe_texts(list(slot.get("character_keys", [])) + list(node.character_keys))[:8],
        source_novels=_dedupe_texts(node.source_novels or [str(candidate["novel"])]),
        selection_ref=_candidate_ref(candidate),
        selection_source_novel=str(candidate["novel"]),
        selection_source_index=int(candidate["index"]),
        logic_card=logic_card,
    )


def _build_slot_fallback_node(slot: Dict[str, Any], base_node: SkeletonNode, index: int) -> SkeletonNode:
    raw_blueprint = [dict(beat) for beat in (slot["chapter_blueprint"] or _fallback_chapter_blueprint(slot["chapter_count"]))]
    chapter_count = slot["chapter_count"] or max(1, base_node.chapter_count or 1)
    chapter_blueprint = _fit_blueprint_to_chapter_count(raw_blueprint, chapter_count)
    logic_card = _merge_logic_cards(slot.get("logic_card", {}), base_node.logic_card or {}, slot.get("pacing_role", ""), slot.get("chapter_count", chapter_count))
    return SkeletonNode(
        node_id=f"fused_event_{index + 1}",
        arc_name=slot["arc_name"],
        realm_level=int(slot["realm_level"]),
        pacing_role=slot["pacing_role"],
        original_summary=base_node.original_summary,
        conflict_hint=base_node.conflict_hint,
        function_hint=base_node.function_hint or slot["function_hint"],
        role_slots=_dedupe_texts(list(slot["role_slots"]) + list(base_node.role_slots))[:8],
        template_hint=slot["template_hint"] or base_node.template_hint,
        source_event_ids=_dedupe_texts(base_node.source_event_ids),
        chapter_start=1,
        chapter_end=chapter_count,
        chapter_count=chapter_count,
        chapter_blueprint=chapter_blueprint,
        character_keys=_dedupe_texts(list(slot.get("character_keys", [])) + list(base_node.character_keys))[:8],
        source_novels=_dedupe_texts(base_node.source_novels or [slot["reference_novel"]]),
        selection_ref=f"{slot['reference_novel']}::{base_node.node_id}",
        selection_source_novel=slot["reference_novel"],
        selection_source_index=slot["slot_index"],
        logic_card=logic_card,
    )


def _slot_from_existing_node(
    reference_novel: str,
    node: SkeletonNode,
    slot_index: int,
    total_slots: int,
    source_novel_count: int,
) -> Dict[str, Any]:
    raw_blueprint = [dict(beat) for beat in (node.chapter_blueprint or _fallback_chapter_blueprint(node.chapter_count or 4))]
    chapter_count = max(1, node.chapter_count or 1)
    chapter_blueprint = _fit_blueprint_to_chapter_count(raw_blueprint, chapter_count)
    progress_ratio = slot_index / max(total_slots - 1, 1) if total_slots > 1 else 0.0
    return {
        "reference_novel": reference_novel,
        "slot_index": slot_index,
        "total_slots": total_slots,
        "progress_ratio": progress_ratio,
        "source_novel_count": source_novel_count,
        "arc_name": node.arc_name,
        "realm_level": node.realm_level,
        "pacing_role": node.pacing_role,
        "function_hint": node.function_hint,
        "conflict_hint": node.conflict_hint,
        "role_slots": node.role_slots[:],
        "template_hint": node.template_hint,
        "chapter_count": chapter_count,
        "chapter_blueprint": chapter_blueprint,
        "character_keys": node.character_keys[:],
        "logic_card": dict(node.logic_card or {}),
        "excluded_node_id": "",
        "excluded_event_ids": set(),
    }


def _top_replacement_candidates(
    slot: Dict[str, Any],
    candidate_pool: List[Dict[str, Any]],
    used_refs: set[str],
    recent_picks: List[Dict[str, Any]],
    usage_counts: Dict[str, int],
    source_limits: Dict[str, int],
    limit: int = 4,
) -> List[Dict[str, Any]]:
    ranked: List[tuple[float, Dict[str, Any]]] = []
    for candidate in candidate_pool:
        candidate_ref = _candidate_ref(candidate)
        if candidate_ref in used_refs:
            continue
        score = _score_candidate_for_slot(slot, candidate, recent_picks, usage_counts, source_limits)
        if score <= -1e8:
            continue
        ranked.append((score, candidate))
    ranked.sort(
        key=lambda item: (
            -item[0],
            usage_counts.get(item[1]["novel"], 0),
            item[1]["novel"],
            item[1]["node"].node_id,
        )
    )
    return [candidate for _score, candidate in ranked[:limit]]


def _refine_skeleton_sequence(
    base_novel: str,
    fused_nodes: List[SkeletonNode],
    per_novel_nodes: Dict[str, List[SkeletonNode]],
) -> List[SkeletonNode]:
    if not fused_nodes or not per_novel_nodes:
        return fused_nodes

    client = get_deepseek_client()
    candidate_pool = _flatten_node_candidates(per_novel_nodes)
    source_limits = _build_source_target_limits(base_novel, per_novel_nodes, len(fused_nodes))

    refined = [node for node in fused_nodes]
    review_recent: List[Dict[str, Any]] = []
    final_usage: Dict[str, int] = {}
    final_used_refs: set[str] = set()
    output_nodes: List[SkeletonNode] = []

    for idx, node in enumerate(refined):
        node.logic_card = _ensure_logic_card(client, node)
        next_node = refined[idx + 1] if idx + 1 < len(refined) else None
        assessment = _assess_skeleton_transition(client, output_nodes[-2:], node, next_node)
        chosen = node
        logic_notes = [str(item).strip() for item in assessment.get("issues", []) if str(item).strip()]

        if not assessment.get("is_consistent", True):
            if chosen.selection_ref:
                final_used_refs.discard(chosen.selection_ref)
            if chosen.selection_source_novel:
                final_usage[chosen.selection_source_novel] = max(0, final_usage.get(chosen.selection_source_novel, 0) - 1)

            slot = _slot_from_existing_node(
                reference_novel=base_novel,
                node=node,
                slot_index=idx,
                total_slots=len(refined),
                source_novel_count=len(per_novel_nodes),
            )
            replacement_recent = review_recent[-3:]
            replacement_candidates = _top_replacement_candidates(
                slot=slot,
                candidate_pool=candidate_pool,
                used_refs=final_used_refs | ({node.selection_ref} if node.selection_ref else set()),
                recent_picks=replacement_recent,
                usage_counts=final_usage,
                source_limits=source_limits,
                limit=4,
            )

            best_candidate_node = chosen
            best_assessment = assessment
            for candidate in replacement_candidates:
                replacement_node = _build_slot_node(slot, candidate, idx)
                replacement_node.logic_card = _ensure_logic_card(client, replacement_node)
                replacement_assessment = _assess_skeleton_transition(client, output_nodes[-2:], replacement_node, next_node)
                if _is_assessment_better(replacement_assessment, best_assessment):
                    best_candidate_node = replacement_node
                    best_assessment = replacement_assessment
            chosen = best_candidate_node
            logic_notes = [str(item).strip() for item in best_assessment.get("issues", []) if str(item).strip()]

        chosen.logic_notes = _dedupe_texts(logic_notes)
        output_nodes.append(chosen)
        if chosen.selection_ref:
            final_used_refs.add(chosen.selection_ref)
        if chosen.selection_source_novel:
            final_usage[chosen.selection_source_novel] = final_usage.get(chosen.selection_source_novel, 0) + 1
            review_recent.append(
                {
                    "novel": chosen.selection_source_novel,
                    "index": chosen.selection_source_index,
                    "progress_ratio": idx / max(len(refined) - 1, 1) if len(refined) > 1 else 0.0,
                    "node": chosen,
                }
            )
            review_recent[:] = review_recent[-3:]

    return output_nodes


def _assessment_rank(assessment: Dict[str, Any]) -> Tuple[int, int, int]:
    severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    is_consistent = 0 if assessment.get("is_consistent", False) else 1
    severity = severity_order.get(str(assessment.get("severity", "medium")).lower(), 1)
    issue_count = len(assessment.get("issues", []) or [])
    return (is_consistent, severity, issue_count)


def _is_assessment_better(candidate_assessment: Dict[str, Any], best_assessment: Dict[str, Any]) -> bool:
    return _assessment_rank(candidate_assessment) < _assessment_rank(best_assessment)


def _ensure_logic_card(client, node: SkeletonNode) -> Dict[str, Any]:
    if isinstance(node.logic_card, dict) and node.logic_card.get("timeline_stage"):
        node.logic_card = _normalize_logic_card(node.logic_card, node.pacing_role, node.chapter_count)
        return node.logic_card
    card = _derive_skeleton_logic_card(client, node)
    node.logic_card = _normalize_logic_card(card, node.pacing_role, node.chapter_count)
    return node.logic_card


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _derive_skeleton_logic_card(client, node: SkeletonNode) -> Dict[str, Any]:
    prompt = (
        f"你是剧情骨架分析器，请从下面的骨架节点中提炼最关键的逻辑约束。\n"
        f"摘要: {node.original_summary}\n"
        f"节奏角色: {node.pacing_role}\n"
        f"冲突/功能: {node.conflict_hint} / {node.function_hint}\n"
        f"角色槽位: {node.role_slots}\n"
        f"章数: {node.chapter_count}\n"
        "只返回 JSON，字段包含 preconditions, state_outputs, power_state, identity_state, relationship_delta, timeline_stage, hook_open, hook_close。"
    )
    raw = _quality_chat_completion_json(client, system="只返回合法 JSON。", user=prompt, json_mode=True, temperature=0.2)
    data = _safe_json_load(raw)
    if isinstance(data, dict):
        data.setdefault("preconditions", [])
        data.setdefault("state_outputs", [])
        data.setdefault("power_state", f"第{max(1, int(node.realm_level))}阶段附近")
        data.setdefault("identity_state", "")
        data.setdefault("relationship_delta", "")
        data.setdefault("timeline_stage", node.pacing_role or "主线推进")
        data.setdefault("hook_open", [])
        data.setdefault("hook_close", [])
        return data
    return {
        "preconditions": [],
        "state_outputs": [],
        "power_state": f"第{max(1, int(node.realm_level))}阶段附近",
        "identity_state": "",
        "relationship_delta": "",
        "timeline_stage": node.pacing_role or "主线推进",
        "hook_open": [],
        "hook_close": [],
    }


def _heuristic_transition_flags(previous_nodes: List[SkeletonNode], node: SkeletonNode, next_node: Optional[SkeletonNode]) -> List[str]:
    issues: List[str] = []
    prev_node = previous_nodes[-1] if previous_nodes else None
    if prev_node and prev_node.selection_source_novel and prev_node.selection_source_novel == node.selection_source_novel:
        if node.selection_source_index >= 0 and prev_node.selection_source_index >= 0 and node.selection_source_index < prev_node.selection_source_index:
            issues.append("同一来源中的事件顺序明显倒退。")
    if prev_node and prev_node.realm_level and node.realm_level and node.realm_level > prev_node.realm_level + 3:
        issues.append("前后节点的境界跨度过大，容易造成实力跳跃。")
    if prev_node and prev_node.function_hint and node.function_hint and prev_node.function_hint == node.function_hint and prev_node.conflict_hint == node.conflict_hint:
        issues.append("相邻节点的功能和冲突过于接近，可能造成剧情重复。")
    if next_node and node.realm_level and next_node.realm_level and node.realm_level > next_node.realm_level + 4:
        issues.append("当前节点强度明显高于后继节点，可能导致时间线倒挂。")
    if previous_nodes:
        available_state = _gather_available_state_outputs(previous_nodes)
        current_requirements = _extract_logic_terms((node.logic_card or {}).get("preconditions", []))
        if current_requirements:
            overlap = current_requirements & available_state
            if not overlap:
                issues.append("当前节点声明了前置条件，但前序节点的状态输出里几乎没有承接。")
    if prev_node:
        prev_identity = _extract_logic_terms([(prev_node.logic_card or {}).get("identity_state", "")])
        current_identity = _extract_logic_terms([(node.logic_card or {}).get("identity_state", "")])
        if prev_identity and current_identity and prev_identity.isdisjoint(current_identity):
            issues.append("前后节点的身份状态跳变过快，缺少过渡。")
        open_hooks = _extract_logic_terms((prev_node.logic_card or {}).get("hook_open", []))
        close_hooks = _extract_logic_terms((node.logic_card or {}).get("hook_close", []))
        if close_hooks and open_hooks and close_hooks.isdisjoint(open_hooks):
            issues.append("当前节点像是在回收钩子，但前一阶段没有对应铺垫。")
    return issues


def _gather_available_state_outputs(nodes: List[SkeletonNode]) -> set[str]:
    values: List[Any] = []
    for node in nodes:
        card = node.logic_card or {}
        values.extend(card.get("state_outputs", []))
        values.extend(card.get("hook_open", []))
        values.append(card.get("identity_state", ""))
        values.append(card.get("relationship_delta", ""))
        values.append(card.get("power_state", ""))
    return _extract_logic_terms(values)


def _extract_logic_terms(values: List[Any]) -> set[str]:
    terms: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        for chunk in re.split(r"[，。；;、/\s]+", text):
            token = chunk.strip()
            if len(token) >= 2:
                terms.add(token)
    return terms


def _render_logic_card_brief(node: Optional[SkeletonNode]) -> str:
    if not node:
        return "无"
    card = node.logic_card or {}
    return (
        f"摘要: {node.original_summary}\n"
        f"节奏角色: {node.pacing_role}\n"
        f"前置条件: {card.get('preconditions', [])}\n"
        f"状态输出: {card.get('state_outputs', [])}\n"
        f"打开钩子: {card.get('hook_open', [])}\n"
        f"关闭钩子: {card.get('hook_close', [])}\n"
        f"实力状态: {card.get('power_state', '')}\n"
        f"身份状态: {card.get('identity_state', '')}\n"
        f"关系变化: {card.get('relationship_delta', '')}"
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _assess_skeleton_transition(
    client,
    previous_nodes: List[SkeletonNode],
    node: SkeletonNode,
    next_node: Optional[SkeletonNode],
) -> Dict[str, Any]:
    heuristic_issues = _heuristic_transition_flags(previous_nodes, node, next_node)
    prev_text = "\n\n".join(_render_logic_card_brief(item) for item in previous_nodes) if previous_nodes else "无前序节点"
    next_text = _render_logic_card_brief(next_node)
    prompt = (
        "你是骨架阶段的连续性审校器。请判断当前骨架节点放在这个位置是否合理。"
        "重点检查：时间是否倒退、成长是否跳跃、身份是否错位、前后因果是否接得上、是否明显重复上一个大事件。"
        "如果不合理，后续系统会直接替换当前节点，而不是默认补桥。\n\n"
        f"前序节点:\n{prev_text}\n\n"
        f"当前节点:\n{_render_logic_card_brief(node)}\n\n"
        f"后继节点:\n{next_text}\n\n"
        f"启发式预警: {heuristic_issues}\n"
        "只返回 JSON，字段包含 is_consistent, issues, severity。"
    )
    raw = _quality_chat_completion_json(client, system="只返回合法 JSON。", user=prompt, json_mode=True, temperature=0.2)
    data = _safe_json_load(raw)
    if isinstance(data, dict):
        issues = [str(item).strip() for item in data.get("issues", []) if str(item).strip()]
        merged_issues = _dedupe_texts(heuristic_issues + issues)
        severity = str(data.get("severity", "low")).strip().lower() or "low"
        is_consistent = bool(data.get("is_consistent", not merged_issues))
        if heuristic_issues and severity in {"high", "critical"}:
            is_consistent = False
        return {"is_consistent": is_consistent, "issues": merged_issues, "severity": severity}
    return {"is_consistent": not heuristic_issues, "issues": heuristic_issues, "severity": "medium" if heuristic_issues else "low"}


def _compact_summary_fragment(summary: str) -> str:
    text = str(summary or "").strip()
    return text.strip(" ，。；;、")


def _build_fused_node_summary(contributors: List[SkeletonNode]) -> str:
    fragments = [_compact_summary_fragment(node.original_summary) for node in contributors]
    fragments = [fragment for fragment in _dedupe_texts(fragments) if fragment]
    if not fragments:
        return "主线围绕多方冲突与关系推进展开。"
    if len(fragments) == 1:
        return fragments[0][:280]
    if len(fragments) == 2:
        return f"{fragments[0]}；并牵出{fragments[1]}"[:280]
    return f"{fragments[0]}；并牵出{fragments[1]}；后续再压上{fragments[2]}"[:280]


def _top_fused_value(contributors: List[SkeletonNode], attr: str) -> str:
    values = [getattr(node, attr, "") for node in contributors]
    deduped = _dedupe_texts(_flatten_text_values(values))
    return deduped[0] if deduped else ""


def _blend_chapter_blueprints(contributors: List[SkeletonNode]) -> List[Dict[str, Any]]:
    anchor = contributors[0].chapter_blueprint or _fallback_chapter_blueprint(contributors[0].chapter_count)
    merged: List[Dict[str, Any]] = []
    support_summaries = [_compact_summary_fragment(node.original_summary) for node in contributors[1:] if _compact_summary_fragment(node.original_summary)]
    for idx, beat in enumerate(anchor):
        beat_copy = dict(beat)
        secondary = _dedupe_texts(beat_copy.get("secondary_functions", []) or [])
        if idx == len(anchor) - 1 and "下一事件钩子" not in secondary:
            secondary.append("下一事件钩子")
        if idx >= max(1, len(anchor) - 2) and "代价落地" not in secondary:
            secondary.append("代价落地")
        purpose_bits = [str(beat_copy.get("purpose", "")).strip()]
        if support_summaries:
            purpose_bits.append(support_summaries[min(idx, len(support_summaries) - 1)])
        beat_copy["secondary_functions"] = secondary
        beat_copy["purpose"] = "；".join(bit for bit in _dedupe_texts(purpose_bits) if bit)[:120]
        merged.append(beat_copy)
    return merged


def _build_event_role_plans(nodes: List[SkeletonNode]) -> List[EventRolePlan]:
    return [
        EventRolePlan(
            event_id=node.node_id,
            event_index=index,
            arc_name=node.arc_name,
            pacing_role=node.pacing_role,
            summary=node.original_summary,
            role_slots=node.role_slots[:4] or _DEFAULT_ROLE_SLOTS[:],
        )
        for index, node in enumerate(nodes)
    ]


def _cast_characters_multipass(client, fused_world: FusedWorld, nodes: List[SkeletonNode], event_role_plans: List[EventRolePlan]) -> CharacterSheet:
    events_text = _format_nodes_for_prompt(nodes)
    protagonist, core_cast = _generate_core_cast(client, fused_world, nodes, events_text)
    volume_plans = _build_volume_plans(nodes, event_role_plans, fused_world)
    volume_cast = _generate_volume_casts(volume_plans, fused_world, core_cast, protagonist.name)
    existing_cast = _dedupe_characters(core_cast + volume_cast)
    transient_cast = _generate_transient_casts(event_role_plans, nodes, fused_world, existing_cast, protagonist.name)
    supporting = _dedupe_characters(core_cast + volume_cast + transient_cast)
    updated_plans = _attach_candidate_names(event_role_plans, supporting)
    relationship_networks = _generate_relationship_networks(client, nodes, protagonist, supporting)
    return CharacterSheet(
        protagonist=protagonist,
        supporting=supporting,
        relationship_networks=relationship_networks,
        event_role_plans=updated_plans,
    )


def _build_volume_plans(nodes: List[SkeletonNode], event_role_plans: List[EventRolePlan], fused_world: FusedWorld) -> List[Dict[str, Any]]:
    plan_by_arc: Dict[str, Dict[str, Any]] = {}
    template_by_arc = {item.get("arc_name"): item for item in fused_world.volume_templates or []}
    for node, role_plan in zip(nodes, event_role_plans):
        current = plan_by_arc.setdefault(
            node.arc_name,
            {
                "arc_name": node.arc_name,
                "event_summaries": [],
                "role_slots": [],
                "recommended_new_characters": 4,
                "start_event_index": role_plan.event_index,
                "end_event_index": role_plan.event_index,
            },
        )
        current["event_summaries"].append(f"{_event_label(role_plan.event_index)}: {node.original_summary}")
        current["role_slots"].extend(role_plan.role_slots)
        current["end_event_index"] = role_plan.event_index
    volume_plans: List[Dict[str, Any]] = []
    for arc_name, plan in plan_by_arc.items():
        template = template_by_arc.get(arc_name, {})
        recommended = template.get("recommended_new_characters")
        if isinstance(recommended, int):
            plan["recommended_new_characters"] = recommended
        plan["role_slots"] = _dedupe_texts(plan["role_slots"])[:8]
        volume_plans.append(plan)
    return volume_plans


def _generate_core_cast(client, fused_world: FusedWorld, nodes: List[SkeletonNode], events_text: str) -> tuple[Character, List[Character]]:
    top_slots = _dedupe_texts(slot for node in nodes for slot in node.role_slots)[:10]
    prompt = (
        f"世界主题：{fused_world.global_theme}\n"
        f"关键事件骨架：\n{events_text}\n\n"
        f"优先覆盖的人物功能槽位：{top_slots}\n"
        f"只返回 JSON，schema: {json.dumps(_CORE_CAST_SCHEMA, ensure_ascii=False)}"
    )
    raw = _quality_chat_completion_json(client, system="只返回合法 JSON。", user=prompt, json_mode=True, temperature=0.7)
    data = _safe_json_load(raw)
    protagonist_payload = data.get("protagonist", {}) if isinstance(data, dict) else {}
    protagonist = Character(
        name=protagonist_payload.get("name", "苍玄"),
        role="主角",
        bond_depth="主角",
        entry_event="事件1",
        exit_event="大结局",
        dao_heart=protagonist_payload.get("dao_heart", ""),
        combat_style=protagonist_payload.get("combat_style", ""),
        personality_flaw=protagonist_payload.get("personality_flaw", ""),
        background=protagonist_payload.get("background", ""),
        tier="core",
        primary_arc=nodes[0].arc_name if nodes else "",
        role_slots=["主角"],
    )
    supporting = [_payload_to_character(item, "core", "") for item in (data.get("core_cast", []) if isinstance(data, dict) else [])]
    supporting = [item for item in supporting if item.name]
    if len(supporting) < 4:
        supporting.extend(_fallback_core_cast(nodes))
    return protagonist, _dedupe_characters(supporting)


def _generate_volume_casts(volume_plans: List[Dict[str, Any]], fused_world: FusedWorld, existing_cast: List[Character], protagonist_name: str) -> List[Character]:
    if not volume_plans:
        return []
    results: List[Character] = []
    with ThreadPoolExecutor(max_workers=min(4, len(volume_plans))) as executor:
        future_map = {
            executor.submit(_generate_single_volume_cast, plan, fused_world, [char.name for char in existing_cast], protagonist_name): plan["arc_name"]
            for plan in volume_plans
        }
        for future in as_completed(future_map):
            try:
                results.extend(future.result())
            except Exception:
                results.extend(_fallback_volume_cast(future_map[future], volume_plans))
    return results


def _generate_transient_casts(event_role_plans: List[EventRolePlan], nodes: List[SkeletonNode], fused_world: FusedWorld, existing_cast: List[Character], protagonist_name: str) -> List[Character]:
    if not event_role_plans:
        return []
    batches = [event_role_plans[i:i + 4] for i in range(0, len(event_role_plans), 4)]
    results: List[Character] = []
    existing_names = [character.name for character in existing_cast]
    with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
        future_map = {
            executor.submit(_generate_single_transient_batch, batch, nodes, fused_world, existing_names, protagonist_name): batch
            for batch in batches
        }
        for future in as_completed(future_map):
            try:
                results.extend(future.result())
            except Exception:
                results.extend(_fallback_transient_cast(future_map[future], nodes))
    return results


def _generate_single_volume_cast(volume_plan: Dict[str, Any], fused_world: FusedWorld, existing_names: List[str], protagonist_name: str) -> List[Character]:
    client = get_deepseek_client()
    events_text = "\n".join(volume_plan.get("event_summaries", []))
    prompt = (
        f"卷名：{volume_plan['arc_name']}\n"
        f"主要事件：\n{events_text}\n\n"
        f"需要补位的卷级槽位：{volume_plan['role_slots']}\n"
        f"已有角色：{existing_names + [protagonist_name]}\n"
        f"目标数量：{volume_plan['recommended_new_characters']}\n"
        f"只返回 JSON，schema: {json.dumps(_VOLUME_CAST_SCHEMA, ensure_ascii=False)}"
    )
    raw = _quality_chat_completion_json(client, system="只返回合法 JSON。", user=prompt, json_mode=True, temperature=0.8)
    data = _safe_json_load(raw)
    cast = [_payload_to_character(item, "volume", volume_plan["arc_name"]) for item in (data.get("volume_cast", []) if isinstance(data, dict) else [])]
    return [item for item in cast if item.name]


def _generate_single_transient_batch(batch: List[EventRolePlan], nodes: List[SkeletonNode], fused_world: FusedWorld, existing_names: List[str], protagonist_name: str) -> List[Character]:
    client = get_deepseek_client()
    node_map = {node.node_id: node for node in nodes}
    lines = []
    for plan in batch:
        node = node_map.get(plan.event_id)
        if node:
            lines.append(f"{_event_label(plan.event_index)} [{node.arc_name}] {plan.summary} | 槽位 {plan.role_slots}")
    prompt = (
        f"世界主题：{fused_world.global_theme}\n"
        f"待补的事件槽位：\n{chr(10).join(lines)}\n\n"
        f"已有角色：{existing_names + [protagonist_name]}\n"
        f"只返回 JSON，schema: {json.dumps(_TRANSIENT_CAST_SCHEMA, ensure_ascii=False)}"
    )
    raw = _quality_chat_completion_json(client, system="只返回合法 JSON。", user=prompt, json_mode=True, temperature=0.9)
    data = _safe_json_load(raw)
    primary_arc = batch[0].arc_name if batch else ""
    cast = [_payload_to_character(item, "transient", primary_arc) for item in (data.get("transient_cast", []) if isinstance(data, dict) else [])]
    return [item for item in cast if item.name]


def _generate_relationship_networks(client, nodes: List[SkeletonNode], protagonist: Character, supporting: List[Character]) -> List[StageNetwork]:
    arc_names = list(dict.fromkeys(node.arc_name for node in nodes))
    names = [protagonist.name] + [item.name for item in supporting[:20]]
    prompt = (
        f"事件阶段：{arc_names}\n"
        f"角色名单：{names}\n"
        f"只返回 JSON，schema: {json.dumps(_RELATIONSHIP_SCHEMA, ensure_ascii=False)}"
    )
    raw = _quality_chat_completion_json(client, system="只返回合法 JSON。", user=prompt, json_mode=True, temperature=0.5)
    data = _safe_json_load(raw)
    networks = [
        StageNetwork(stage=item.get("stage", f"阶段{idx + 1}"), active_characters=item.get("active_characters", []), relationship_status=item.get("relationship_status", ""))
        for idx, item in enumerate(data.get("relationship_networks", []) if isinstance(data, dict) else [])
    ]
    if networks:
        return networks
    return [
        StageNetwork(
            stage=arc_name,
            active_characters=[protagonist.name] + [item.name for item in supporting if item.primary_arc in ("", arc_name)][:6],
            relationship_status=f"{arc_name}阶段围绕资源、立场和旧怨持续拉扯。",
        )
        for arc_name in arc_names
    ]


def _attach_candidate_names(event_role_plans: List[EventRolePlan], supporting: List[Character]) -> List[EventRolePlan]:
    updated: List[EventRolePlan] = []
    for plan in event_role_plans:
        candidates = []
        for character in supporting:
            if not _is_char_active(character, plan.event_index):
                continue
            if set(plan.role_slots) & set(character.role_slots):
                candidates.append(character.name)
            elif any(slot in f"{character.role} {character.background}" for slot in plan.role_slots):
                candidates.append(character.name)
        plan.candidate_names = candidates[:4]
        updated.append(plan)
    return updated


def _is_char_active(character: Character, current_idx: int) -> bool:
    entry_match = re.search(r"\d+", character.entry_event or "")
    exit_match = re.search(r"\d+", character.exit_event or "")
    entry_idx = int(entry_match.group()) - 1 if entry_match else 0
    exit_idx = int(exit_match.group()) - 1 if exit_match else 9999
    return entry_idx <= current_idx <= exit_idx


def _payload_to_character(payload: Dict[str, Any], default_tier: str, primary_arc: str) -> Character:
    role_slots = payload.get("role_slots", [])
    if not isinstance(role_slots, list) or not role_slots:
        role_slots = _infer_slots_from_character_text(payload.get("role", ""), payload.get("background", ""))
    return Character(
        name=str(payload.get("name", "")).strip(),
        role=str(payload.get("role", "")).strip(),
        bond_depth=str(payload.get("bond_depth", "")).strip() or default_tier,
        entry_event=str(payload.get("entry_event", "事件1")).strip(),
        exit_event=str(payload.get("exit_event", "大结局")).strip(),
        return_event=str(payload.get("return_event", "")).strip(),
        dao_heart=str(payload.get("dao_heart", "")).strip(),
        combat_style=str(payload.get("combat_style", "")).strip(),
        personality_flaw=str(payload.get("personality_flaw", "")).strip(),
        background=str(payload.get("background", "")).strip(),
        tier=default_tier,
        primary_arc=primary_arc,
        role_slots=[str(item).strip() for item in role_slots if str(item).strip()],
    )


def _infer_slots_from_character_text(role: str, background: str) -> List[str]:
    text = f"{role} {background}"
    mapping = [
        ("师尊", "护道者"),
        ("前辈", "试探前辈"),
        ("掌柜", "掌柜主事"),
        ("主事", "维持秩序者"),
        ("宿敌", "宿敌"),
        ("追杀", "追杀者"),
        ("线人", "线人"),
        ("盟友", "临时盟友"),
        ("守卫", "守护者"),
    ]
    slots = [slot for keyword, slot in mapping if keyword in text]
    return slots or _DEFAULT_ROLE_SLOTS[:2]


def _dedupe_characters(characters: List[Character]) -> List[Character]:
    merged: Dict[str, Character] = {}
    for character in characters:
        name = character.name.strip()
        if not name:
            continue
        if name not in merged:
            merged[name] = character
            continue
        existing = merged[name]
        if len(existing.role_slots) < len(character.role_slots):
            existing.role_slots = character.role_slots
        if existing.tier == "transient" and character.tier != "transient":
            merged[name] = character
    return list(merged.values())


def _fallback_core_cast(nodes: List[SkeletonNode]) -> List[Character]:
    arc_name = nodes[0].arc_name if nodes else ""
    return [
        Character(name="百里清雪", role="师门护道者", bond_depth="核心", entry_event="事件1", exit_event="大结局", dao_heart="守住道统", combat_style="剑诀与阵法", personality_flaw="过度克制", background="出身宗门主脉", tier="core", primary_arc=arc_name, role_slots=["护道者", "规则维护者"]),
        Character(name="闻人策", role="长期宿敌", bond_depth="核心", entry_event="事件2", exit_event="大结局", dao_heart="压过主角", combat_style="诡道搏杀", personality_flaw="轻敌", background="敌对势力的年轻掌权者", tier="core", primary_arc=arc_name, role_slots=["宿敌", "阻碍者"]),
        Character(name="苏照微", role="危险盟友", bond_depth="核心", entry_event="事件3", exit_event="大结局", dao_heart="以交易换自由", combat_style="符箓和情报", personality_flaw="不轻信任何人", background="游走多方的情报商", tier="core", primary_arc=arc_name, role_slots=["消息提供者", "短期盟友"]),
    ]


def _fallback_volume_cast(arc_name: str, volume_plans: List[Dict[str, Any]]) -> List[Character]:
    plan = next((item for item in volume_plans if item["arc_name"] == arc_name), None)
    slots = plan.get("role_slots", []) if plan else _DEFAULT_ROLE_SLOTS
    start_event = _event_label(plan["start_event_index"]) if plan else "事件1"
    end_event = _event_label(plan["end_event_index"]) if plan else "事件3"
    return [
        Character(name=f"{arc_name}执事", role="规则维护者", bond_depth="卷级", entry_event=start_event, exit_event=end_event, dao_heart="维系秩序", combat_style="令牌与禁制", personality_flaw="刻板", background=f"{arc_name}的执事人物", tier="volume", primary_arc=arc_name, role_slots=slots[:2] or _DEFAULT_ROLE_SLOTS[:2])
    ]


def _fallback_transient_cast(batch: List[EventRolePlan], nodes: List[SkeletonNode]) -> List[Character]:
    node_map = {node.node_id: node for node in nodes}
    results: List[Character] = []
    for plan in batch[:3]:
        node = node_map.get(plan.event_id)
        role_slot = plan.role_slots[0] if plan.role_slots else "阻碍者"
        results.append(
            Character(
                name=f"{role_slot}{plan.event_index + 1}",
                role=role_slot,
                bond_depth="过客",
                entry_event=_event_label(plan.event_index),
                exit_event=_event_label(plan.event_index + 1),
                dao_heart="完成眼前目标",
                combat_style="功能性手段",
                personality_flaw="见利忘义",
                background=f"服务于{node.arc_name if node else '当前事件'}的功能角色",
                tier="transient",
                primary_arc=node.arc_name if node else "",
                role_slots=plan.role_slots[:2] or _DEFAULT_ROLE_SLOTS[:2],
            )
        )
    return results


def _format_nodes_for_prompt(nodes: List[SkeletonNode]) -> str:
    return "\n".join(
        f"{_event_label(index)} [{node.arc_name}] {node.original_summary} | 章数: {node.chapter_count} | 来源: {node.source_novels or ['unknown']} | 槽位: {node.role_slots}"
        for index, node in enumerate(nodes)
    )
