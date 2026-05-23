from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pipeline.core.skeleton_core import (
    _DEFAULT_ROLE_SLOTS,
    _atom_text,
    _dedupe_texts,
    _fallback_chapter_blueprint,
    _fit_blueprint_to_chapter_count,
    _flatten_text_values,
    _realm_level_from_arc,
    load_node_map,
    save_node_map,
)
from pipeline.core.story_models import SkeletonNode
from pipeline.core.story_state import (
    ProgressionStage,
    build_progression_stages,
    extract_max_stage_from_text,
    infer_stage_from_node,
    normalize_stage_name,
)
from pipeline.core.world_building_core import FusedWorld
from pipeline.step1_chunking import NarrativeEvent, VolumeArc
from pipeline.step2_extraction import PlotAtom
from pipeline.step3_event_induction import InducedEvent


_STEP8_FILENAME = "step8_source_skeletons.json"

_RESOLUTION_KEYWORDS: Sequence[str] = (
    "解决",
    "了结",
    "收束",
    "平息",
    "脱身",
    "翻盘",
    "揭露",
    "完成",
    "夺回",
)

_SETUP_KEYWORDS: Sequence[str] = (
    "接到",
    "进入",
    "启程",
    "调查",
    "开启",
    "发现",
    "任务",
    "试炼",
)

_LOCATION_HINTS: Sequence[str] = (
    "宗门",
    "学院",
    "基地",
    "城市",
    "村镇",
    "安全区",
    "营地",
    "城池",
    "秘境",
    "异界",
    "遗迹",
    "深渊",
    "禁区",
    "星域",
    "敌营",
    "总部",
)


def extract_source_skeletons(
    novel_arcs: Dict[str, List[VolumeArc]],
    all_atoms: Dict[str, List[PlotAtom]],
    fused_world: FusedWorld,
    induced_events_by_novel: Optional[Dict[str, List[InducedEvent]]] = None,
) -> Dict[str, List[SkeletonNode]]:
    per_novel_nodes: Dict[str, List[SkeletonNode]] = {}
    stages = build_progression_stages(fused_world)
    all_novel_names = sorted(set(novel_arcs) | set(all_atoms) | set((induced_events_by_novel or {}).keys()))

    for novel_name in all_novel_names:
        arcs = novel_arcs.get(novel_name, [])
        induced_events = (induced_events_by_novel or {}).get(novel_name, [])
        if induced_events:
            segments = _group_induced_events_to_segments_with_stages(
                induced_events=induced_events,
                stages=stages,
                min_events=3,
                max_events=6,
            )
            source_nodes = _segments_to_skeleton_nodes(segments, fused_world, stages)
            avg_size = round(len(induced_events) / len(segments), 2) if segments else 0.0
            print(
                f"[Step 8] Novel {novel_name}: induced events={len(induced_events)}, "
                f"segments={len(segments)}, avg events/segment={avg_size}"
            )
        else:
            source_nodes = _extract_skeleton_nodes(arcs, all_atoms.get(novel_name, []), fused_world, stages)
            for node in source_nodes:
                node.metadata["used_legacy_fallback"] = True
            print(
                f"[Step 8] Novel {novel_name}: induced events=0, segments={len(source_nodes)}, "
                f"avg events/segment=0.0 (legacy fallback)"
            )

        for node in source_nodes:
            node.source_novels = [novel_name]
            node.metadata.setdefault("used_legacy_fallback", False)
        if source_nodes:
            per_novel_nodes[novel_name] = source_nodes

    print(f"[Step 8] Extracted source skeletons for {len(per_novel_nodes)} novel(s).")
    return per_novel_nodes


def save_step8_output(per_novel_nodes: Dict[str, List[SkeletonNode]]):
    path = save_node_map(_STEP8_FILENAME, per_novel_nodes)
    print(f"[Step 8] Intermediate output saved -> {path.name}")
    return path


def load_step8_output() -> Dict[str, List[SkeletonNode]]:
    return load_node_map(_STEP8_FILENAME)


def group_induced_events_to_segments(
    induced_events: List[InducedEvent],
    min_events: int = 3,
    max_events: int = 6,
) -> List[Dict[str, Any]]:
    stages = _build_progression_stages_from_induced_events(induced_events)
    return _group_induced_events_to_segments_with_stages(induced_events, stages, min_events=min_events, max_events=max_events)


def _group_induced_events_to_segments_with_stages(
    induced_events: List[InducedEvent],
    stages: List[ProgressionStage],
    min_events: int,
    max_events: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[InducedEvent]] = defaultdict(list)
    for event in induced_events:
        grouped[(event.novel_source or "", event.arc_name or "Unnamed Arc")].append(event)

    segments: List[Dict[str, Any]] = []
    for (novel_source, arc_name), bucket in grouped.items():
        ordered = sorted(bucket, key=lambda item: (item.chapter_start, item.chapter_end, item.event_id))
        current: List[InducedEvent] = []
        for event in ordered:
            if not current:
                current = [event]
                continue
            if _should_split_before_event(current, event, stages, min_events=min_events, max_events=max_events):
                segments.append(_build_segment_record(current, novel_source, arc_name, len(segments)))
                current = [event]
                continue
            current.append(event)
        if current:
            segments.append(_build_segment_record(current, novel_source, arc_name, len(segments)))
    return segments


def _segments_to_skeleton_nodes(
    segments: List[Dict[str, Any]],
    fused_world: FusedWorld,
    stages: List[ProgressionStage],
) -> List[SkeletonNode]:
    per_arc_segments: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        per_arc_segments[segment["arc_name"]].append(segment)

    ordered_segments: List[Dict[str, Any]] = []
    for arc_name, bucket in per_arc_segments.items():
        ordered_segments.extend(
            sorted(bucket, key=lambda item: (item["chapter_start"], item["chapter_end"], item["segment_id"]))
        )

    total_segments = len(ordered_segments)
    nodes: List[SkeletonNode] = []
    for index, segment in enumerate(ordered_segments):
        nodes.append(_segment_to_skeleton_node(segment, fused_world, stages, index=index, total=total_segments))
    return nodes


def _segment_to_skeleton_node(
    segment: Dict[str, Any],
    fused_world: FusedWorld,
    stages: List[ProgressionStage],
    index: int,
    total: int,
) -> SkeletonNode:
    events: List[InducedEvent] = list(segment["events"])
    chapter_count = max(1, segment["chapter_end"] - segment["chapter_start"] + 1)
    dominant_function = _top_text_value(event.function_hint for event in events)
    dominant_conflict = _top_text_value(event.conflict_hint for event in events)
    summary = _build_segment_summary(events)
    matched_template = _match_flow_template_for_segment(chapter_count, dominant_conflict, fused_world)
    template_blueprint = matched_template.get("beat_blueprint") or _fallback_chapter_blueprint(chapter_count)
    blueprint = _fit_blueprint_to_chapter_count(template_blueprint, chapter_count)
    segment_stage = _infer_segment_stage(events, stages)
    pacing_role = _infer_segment_pacing_role(index, total, dominant_function)
    logic_card = _build_segment_logic_card(events, segment_stage, pacing_role)
    role_slots = _infer_segment_role_slots(events, fused_world, matched_template, summary, dominant_conflict, dominant_function)
    source_induced_event_ids = [event.event_id for event in events]
    source_atom_ids = _dedupe_texts(atom_id for event in events for atom_id in getattr(event, "source_atom_ids", []))
    source_refs = _build_segment_source_refs(events, source_atom_ids)

    return SkeletonNode(
        node_id=segment["segment_id"],
        arc_name=segment["arc_name"],
        realm_level=_realm_level_from_arc(segment["arc_name"]),
        pacing_role=pacing_role,
        original_summary=summary,
        conflict_hint=dominant_conflict,
        function_hint=dominant_function,
        role_slots=role_slots,
        template_hint=matched_template.get("name", f"segment_{chapter_count}_chapters"),
        source_event_ids=source_induced_event_ids[:],
        chapter_start=segment["chapter_start"],
        chapter_end=segment["chapter_end"],
        chapter_count=chapter_count,
        chapter_blueprint=blueprint,
        character_keys=_dedupe_texts(key for event in events for key in getattr(event, "character_keys", [])),
        source_novels=[segment["novel_source"]],
        logic_card=logic_card,
        source_induced_event_ids=source_induced_event_ids,
        source_atom_ids=source_atom_ids,
        source_refs=source_refs,
        stage=segment_stage,
        power_stage=segment_stage,
        metadata={
            "used_legacy_fallback": False,
            "segment_event_count": len(events),
            "segment_chapter_span": chapter_count,
        },
    )


def _select_node_summary(event: NarrativeEvent, atom: Optional[PlotAtom]) -> str:
    candidates = [atom.summary if atom else "", _compose_atom_summary(atom), event.summary, _fallback_event_summary(event)]
    for candidate in candidates:
        summary = str(candidate or "").strip()
        if summary:
            return summary[:240]
    return f"{event.arc_name} event"


def _compose_atom_summary(atom: Optional[PlotAtom]) -> str:
    if not atom:
        return ""
    fragments: List[str] = []
    if atom.core_action:
        fragments.append(atom.core_action.strip())
    if atom.conflict_type:
        fragments.append(f"冲突: {atom.conflict_type}")
    if atom.motivation:
        fragments.append(f"动机: {atom.motivation}")
    if atom.causality_consequence:
        fragments.append(f"后果: {atom.causality_consequence}")
    return "；".join(_dedupe_texts(fragments))


def _fallback_event_summary(event: NarrativeEvent) -> str:
    preview_lines: List[str] = []
    for chapter in event.chapters[:2]:
        for line in chapter.splitlines():
            clean = line.strip()
            if clean:
                preview_lines.append(clean)
                break
    if preview_lines:
        return " / ".join(dict.fromkeys(preview_lines))
    preview = " ".join(chapter.strip().replace("\n", " ") for chapter in event.chapters[:2]).strip()
    return preview[:240]


def _infer_pacing_role(global_idx: int, total_events: int, atom: Optional[PlotAtom]) -> str:
    if atom and atom.narrative_function:
        return _atom_text(atom.narrative_function)
    ratio = global_idx / total_events if total_events > 0 else 0
    if ratio < 0.08:
        return "开篇落点"
    if ratio > 0.92:
        return "卷末收束"
    if ratio < 0.35:
        return "铺垫与试探"
    if ratio < 0.72:
        return "冲突升级"
    return "高潮对决"


def _infer_role_slots(atom: Optional[PlotAtom], event: NarrativeEvent, fused_world: FusedWorld) -> List[str]:
    combined_text = " ".join(
        _flatten_text_values(
            [
                atom.conflict_type if atom else "",
                atom.narrative_function if atom else "",
                atom.summary if atom else "",
                atom.core_action if atom else "",
                event.summary,
            ]
        )
    ).lower()
    for template in fused_world.role_slot_templates or []:
        keywords = [str(item).lower() for item in template.get("keywords", [])]
        if keywords and any(keyword and keyword in combined_text for keyword in keywords):
            slots = template.get("role_slots", [])
            if isinstance(slots, list) and slots:
                return slots[:4]
    return _DEFAULT_ROLE_SLOTS[:]


def _extract_skeleton_nodes(
    arcs: List[VolumeArc],
    atoms: List[PlotAtom],
    fused_world: FusedWorld,
    stages: List[ProgressionStage],
) -> List[SkeletonNode]:
    atom_map = {atom.atom_id: atom for atom in atoms}
    atomic_items: List[Dict[str, Any]] = []
    total_raw_events = sum(len(arc.events) for arc in arcs)
    raw_index = 0
    for arc in arcs:
        realm_level = _realm_level_from_arc(arc.arc_name)
        for event in arc.events:
            atom = atom_map.get(event.event_id)
            atomic_items.append(
                {
                    "event": event,
                    "atom": atom,
                    "arc_name": arc.arc_name,
                    "realm_level": realm_level,
                    "summary": _select_node_summary(event, atom),
                    "role_slots": _infer_role_slots(atom, event, fused_world),
                    "conflict_hint": _atom_text(getattr(atom, "conflict_type", "")),
                    "function_hint": _atom_text(getattr(atom, "narrative_function", "")),
                    "raw_pacing_role": _infer_pacing_role(raw_index, total_raw_events, atom),
                    "character_keys": list(getattr(atom, "character_keys", []) or []),
                    "precondition": getattr(atom, "causality_precondition", ""),
                    "consequence": getattr(atom, "causality_consequence", ""),
                    "motivation": getattr(atom, "motivation", ""),
                    "core_action": getattr(atom, "core_action", ""),
                    "location": getattr(atom, "location", ""),
                    "cultivation_elements": list(getattr(atom, "cultivation_elements", []) or []),
                    "emotion": getattr(atom, "emotion", ""),
                    "chapter_count": max(
                        1,
                        int(getattr(atom, "chapter_end", getattr(event, "chapter_end", 0)) or 0)
                        - int(getattr(atom, "chapter_start", getattr(event, "chapter_start", 0)) or 0)
                        + 1,
                    ),
                    "chapter_start": int(getattr(atom, "chapter_start", getattr(event, "chapter_start", 0)) or 0),
                    "chapter_end": int(getattr(atom, "chapter_end", getattr(event, "chapter_end", 0)) or 0),
                }
            )
            raw_index += 1

    clusters = _cluster_atomic_items(atomic_items)
    nodes: List[SkeletonNode] = []
    total_clusters = len(clusters)
    for cluster_index, cluster in enumerate(clusters):
        summary = _build_cluster_summary(cluster)
        role_slots = _dedupe_texts(slot for item in cluster for slot in item["role_slots"])[:6] or _DEFAULT_ROLE_SLOTS[:]
        chapter_count = _cluster_chapter_total(cluster)
        primary_function = _top_cluster_value(cluster, "function_hint")
        stage = _infer_fallback_cluster_stage(cluster, stages)
        source_atom_ids = _dedupe_texts(item["atom"].atom_id for item in cluster if item.get("atom"))
        source_refs = _build_fallback_source_refs(cluster, source_atom_ids)
        pacing_role = _infer_cluster_pacing_role(cluster_index, total_clusters, cluster, primary_function)
        node = SkeletonNode(
            node_id=f"{cluster[0]['arc_name']}_story_event{cluster_index}",
            arc_name=cluster[0]["arc_name"],
            realm_level=max(item["realm_level"] for item in cluster),
            pacing_role=pacing_role,
            original_summary=summary,
            conflict_hint=_top_cluster_value(cluster, "conflict_hint"),
            function_hint=primary_function,
            role_slots=role_slots,
            template_hint=f"cluster_{chapter_count}_chapters",
            source_event_ids=[item["event"].event_id for item in cluster],
            chapter_start=min(item["chapter_start"] for item in cluster),
            chapter_end=max(item["chapter_end"] for item in cluster),
            chapter_count=chapter_count,
            chapter_blueprint=_fallback_blueprint_for_cluster(cluster),
            character_keys=_dedupe_texts(key for item in cluster for key in item.get("character_keys", [])),
            logic_card=_build_cluster_logic_card(cluster, chapter_count, pacing_role),
            source_atom_ids=source_atom_ids,
            source_refs=source_refs,
            stage=stage,
            power_stage=stage,
            metadata={"used_legacy_fallback": True},
        )
        nodes.append(node)
    return nodes


def _cluster_atomic_items(items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    clusters: List[List[Dict[str, Any]]] = []
    by_arc: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        by_arc.setdefault(item["arc_name"], []).append(item)

    for arc_items in by_arc.values():
        index = 0
        while index < len(arc_items):
            cluster: List[Dict[str, Any]] = []
            while index + len(cluster) < len(arc_items):
                candidate = arc_items[index + len(cluster)]
                cluster.append(candidate)
                chapter_total = _cluster_chapter_total(cluster)
                if chapter_total >= 4 and (
                    chapter_total >= 10
                    or _cluster_boundary_hit(candidate)
                    or _next_cluster_resets(arc_items, index + len(cluster) - 1)
                ):
                    break
                if chapter_total >= 12:
                    break
            clusters.append(cluster)
            index += len(cluster)
    return clusters


def _cluster_chapter_total(cluster: List[Dict[str, Any]]) -> int:
    covered = set()
    for item in cluster:
        start = int(item.get("chapter_start", 0) or 0)
        end = int(item.get("chapter_end", 0) or start or 0)
        if start and end and end >= start:
            covered.update(range(start, end + 1))
        else:
            covered.add(len(covered) + 1)
    return max(1, len(covered))


def _cluster_boundary_hit(item: Dict[str, Any]) -> bool:
    text = " ".join(_flatten_text_values([item.get("summary", ""), item.get("function_hint", ""), item.get("conflict_hint", "")]))
    if any(keyword in text for keyword in ["转折", "翻盘", "余波", "收束", "后果", "伏笔", "钩子"]):
        return True
    return any(keyword in text for keyword in ["高潮", "对决", "逆袭"])


def _next_cluster_resets(items: List[Dict[str, Any]], current_index: int) -> bool:
    if current_index + 1 >= len(items):
        return True
    next_item = items[current_index + 1]
    text = " ".join(_flatten_text_values([next_item.get("summary", ""), next_item.get("raw_pacing_role", "")]))
    return any(keyword in text for keyword in ["开篇", "铺垫", "试探", "入场"])


def _fallback_blueprint_for_cluster(cluster: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    relative_spans = _relative_chapter_spans_from_cluster(cluster)
    total = len(cluster)
    blueprint: List[Dict[str, Any]] = []
    for index, item in enumerate(cluster, start=1):
        if index == 1:
            primary = "铺垫入场"
            secondary = ["人物亮相"]
        elif index == total:
            primary = "余波收束"
            secondary = ["下一事件钩子"]
        elif index == max(2, total - 1):
            primary = "局势转折"
            secondary = ["代价落地"]
        else:
            primary = "冲突升级"
            secondary = []
        blueprint.append(
            {
                "beat_index": index,
                "chapter_span": relative_spans[index - 1],
                "primary_function": primary,
                "secondary_functions": secondary,
                "purpose": item["summary"],
            }
        )
    return blueprint or _fallback_chapter_blueprint(_cluster_chapter_total(cluster))


def _relative_chapter_spans_from_cluster(cluster: List[Dict[str, Any]]) -> List[List[int]]:
    covered = sorted(
        {
            chapter_no
            for item in cluster
            for chapter_no in range(
                int(item.get("chapter_start", 0) or 0),
                int(item.get("chapter_end", item.get("chapter_start", 0)) or item.get("chapter_start", 0) or 0) + 1,
            )
            if chapter_no
        }
    )
    if not covered:
        cursor = 1
        spans = []
        for item in cluster:
            span = max(1, int(item.get("chapter_count", 1) or 1))
            spans.append([cursor, cursor + span - 1])
            cursor += span
        return spans

    index_map = {chapter_no: index + 1 for index, chapter_no in enumerate(covered)}
    spans: List[List[int]] = []
    for item in cluster:
        start = int(item.get("chapter_start", 0) or 0)
        end = int(item.get("chapter_end", 0) or start or 0)
        if start and end and start in index_map and end in index_map:
            spans.append([index_map[start], index_map[end]])
        else:
            fallback = spans[-1][1] if spans else 0
            span = max(1, int(item.get("chapter_count", 1) or 1))
            spans.append([fallback + 1, fallback + span])
    return spans


def _top_cluster_value(cluster: List[Dict[str, Any]], key: str) -> str:
    values = [item.get(key, "") for item in cluster]
    deduped = _dedupe_texts(_flatten_text_values(values))
    return deduped[0] if deduped else ""


def _build_cluster_summary(cluster: List[Dict[str, Any]]) -> str:
    fragments = [item["summary"] for item in cluster[:3] if item.get("summary")]
    return "；".join(_dedupe_texts(fragments))[:280]


def _infer_cluster_pacing_role(
    cluster_index: int,
    total_clusters: int,
    cluster: List[Dict[str, Any]],
    primary_function: str,
) -> str:
    if primary_function:
        return primary_function
    ratio = cluster_index / total_clusters if total_clusters > 0 else 0
    if ratio < 0.08:
        return "开篇落点"
    if ratio > 0.92:
        return "卷末收束"
    if any("转折" in item.get("summary", "") for item in cluster):
        return "局势转折"
    if ratio < 0.35:
        return "铺垫与试探"
    if ratio < 0.72:
        return "冲突升级"
    return "高潮对决"


def _build_cluster_logic_card(cluster: List[Dict[str, Any]], chapter_count: int, pacing_role: str) -> Dict[str, Any]:
    inputs = _cluster_logic_terms(cluster, ["precondition", "motivation"], limit=8)
    outputs = _cluster_logic_terms(cluster, ["consequence", "core_action"], limit=8)
    relationship: List[str] = []
    for item in cluster:
        emotion = str(item.get("emotion", "") or "").strip()
        if emotion and len(item.get("character_keys", []) or []) >= 2:
            relationship.append(f"相关角色关系转向{emotion}")
    hooks_open = _cluster_keyword_terms(cluster[-2:], ["consequence", "summary"], ("伏笔", "线索", "后续", "隐患", "更大", "下一", "秘密", "真相"))
    hooks_close = _cluster_keyword_terms(cluster, ["core_action", "summary"], ("破局", "揭露", "翻盘", "和解", "救出", "斩杀", "夺得", "突破", "解决", "脱身"))
    power_terms: List[str] = []
    for item in cluster:
        for entry in item.get("cultivation_elements", []):
            text = str(entry or "").strip()
            if text:
                power_terms.append(text)
    identity_terms = _cluster_keyword_terms(
        cluster,
        ["summary", "core_action", "location"],
        ("弟子", "长老", "亲传", "外门", "内门", "掌柜", "少主", "客卿", "散修", "逃亡", "潜伏", "卧底", "供奉"),
    )
    return {
        "preconditions": inputs,
        "state_outputs": outputs,
        "power_state": _dedupe_texts(power_terms)[0] if power_terms else f"{chapter_count}章节事件强度",
        "identity_state": identity_terms[0] if identity_terms else "",
        "relationship_delta": "；".join(_dedupe_texts(relationship)[:3]),
        "timeline_stage": pacing_role or "主线推进",
        "hook_open": hooks_open,
        "hook_close": hooks_close,
    }


def _cluster_logic_terms(cluster: List[Dict[str, Any]], keys: List[str], limit: int) -> List[str]:
    values: List[str] = []
    for item in cluster:
        for key in keys:
            values.extend(_flatten_text_values([item.get(key, "")]))
    results: List[str] = []
    for value in values:
        normalized = str(value or "").replace("；", "。").replace(";", "。")
        for piece in normalized.split("。"):
            text = piece.strip("。；; \t\r\n")
            if len(text) >= 2 and text not in results:
                results.append(text[:32])
            if len(results) >= limit:
                return results
    return results


def _cluster_keyword_terms(cluster: List[Dict[str, Any]], keys: List[str], keywords: Tuple[str, ...]) -> List[str]:
    values: List[str] = []
    for item in cluster:
        text = " ".join(_flatten_text_values([item.get(key, "") for key in keys]))
        normalized = text.replace("；", "。").replace(";", "。")
        for piece in normalized.split("。"):
            cleaned = piece.strip("。；; \t\r\n")
            if cleaned and any(keyword in cleaned for keyword in keywords):
                values.append(cleaned[:32])
    return _dedupe_texts(values)[:4]


def _build_progression_stages_from_induced_events(induced_events: List[InducedEvent]) -> List[ProgressionStage]:
    stage_names = _dedupe_texts(event.power_state for event in induced_events if str(event.power_state or "").strip())
    if not stage_names:
        return build_progression_stages(None)
    pseudo_world = {"progression_stages": [{"name": name, "stage_index": index} for index, name in enumerate(stage_names, start=1)]}
    return build_progression_stages(pseudo_world)


def _should_split_before_event(
    current: List[InducedEvent],
    next_event: InducedEvent,
    stages: List[ProgressionStage],
    min_events: int,
    max_events: int,
) -> bool:
    if not current:
        return False
    if len(current) >= max_events:
        return True

    projected_start = min((event.chapter_start for event in current if event.chapter_start), default=next_event.chapter_start)
    projected_end = max([event.chapter_end for event in current if event.chapter_end] + [next_event.chapter_end])
    if projected_start and projected_end and projected_end - projected_start + 1 > 8:
        return True

    previous = current[-1]
    if (
        _stage_changed(previous, next_event, stages)
        or _identity_changed(previous, next_event)
        or _location_changed(previous, next_event)
        or _conflict_resolved(previous, next_event)
    ):
        return True

    similarity = _segment_similarity(current, next_event)
    if len(current) < min_events:
        return False
    return similarity < 0.35


def _build_segment_record(events: List[InducedEvent], novel_source: str, arc_name: str, global_index: int) -> Dict[str, Any]:
    return {
        "segment_id": f"{arc_name}_segment_{global_index}",
        "novel_source": novel_source,
        "arc_name": arc_name,
        "events": list(events),
        "chapter_start": min(event.chapter_start for event in events),
        "chapter_end": max(event.chapter_end for event in events),
    }


def _segment_similarity(current: List[InducedEvent], next_event: InducedEvent) -> float:
    function_score = _string_overlap(_top_text_value(event.function_hint for event in current), next_event.function_hint)
    conflict_score = _string_overlap(_top_text_value(event.conflict_hint for event in current), next_event.conflict_hint)
    hook_continuity = 1.0 if _hooks_connect(current[-1], next_event) else 0.0
    return function_score * 0.4 + conflict_score * 0.4 + hook_continuity * 0.2


def _hooks_connect(previous: InducedEvent, next_event: InducedEvent) -> bool:
    previous_terms = set(_dedupe_texts(list(previous.hook_open) + list(previous.hook_close)))
    next_terms = set(_dedupe_texts(list(next_event.state_inputs) + list(next_event.hook_open)))
    return bool(previous_terms and next_terms and previous_terms.intersection(next_terms))


def _stage_changed(previous: InducedEvent, next_event: InducedEvent, stages: List[ProgressionStage]) -> bool:
    previous_stage = infer_stage_from_node(previous, stages)
    next_stage = infer_stage_from_node(next_event, stages)
    if not previous_stage or not next_stage:
        return False
    return normalize_stage_name(previous_stage, stages) != normalize_stage_name(next_stage, stages)


def _identity_changed(previous: InducedEvent, next_event: InducedEvent) -> bool:
    prev_identity = str(getattr(previous, "identity_state", "") or "").strip()
    next_identity = str(getattr(next_event, "identity_state", "") or "").strip()
    if not prev_identity or not next_identity:
        return False
    if prev_identity == next_identity:
        return False
    return _string_overlap(prev_identity, next_identity) < 0.65


def _location_changed(previous: InducedEvent, next_event: InducedEvent) -> bool:
    previous_hint = _infer_event_location_hint(previous)
    next_hint = _infer_event_location_hint(next_event)
    if not previous_hint or not next_hint:
        return False
    return previous_hint != next_hint


def _conflict_resolved(previous: InducedEvent, next_event: InducedEvent) -> bool:
    previous_text = " ".join(_flatten_text_values([previous.summary, previous.conflict_hint, previous.hook_close]))
    next_text = " ".join(_flatten_text_values([next_event.summary, next_event.function_hint, next_event.hook_open]))
    if previous.hook_close and (next_event.hook_open or _has_any_keyword(next_text, _SETUP_KEYWORDS)):
        return True
    return _has_any_keyword(previous_text, _RESOLUTION_KEYWORDS) and _has_any_keyword(next_text, _SETUP_KEYWORDS)


def _infer_event_location_hint(event: InducedEvent) -> str:
    text = " ".join(_flatten_text_values([event.summary, event.raw_summary, event.conflict_hint]))
    for keyword in _LOCATION_HINTS:
        if keyword in text:
            return keyword
    return ""


def _infer_segment_stage(events: List[InducedEvent], stages: List[ProgressionStage]) -> str:
    for event in reversed(events):
        stage_name = infer_stage_from_node(event, stages)
        if stage_name:
            return stage_name
    text = " ".join(_flatten_text_values([event.summary for event in events] + [event.raw_summary for event in events]))
    return extract_max_stage_from_text(text, stages)


def _build_segment_summary(events: List[InducedEvent]) -> str:
    if not events:
        return ""
    lead = str(events[0].summary or events[0].raw_summary or "").strip()
    middle = str(events[len(events) // 2].summary or "").strip() if len(events) >= 3 else ""
    tail = str(events[-1].summary or events[-1].raw_summary or "").strip()
    fragments = [fragment for fragment in [lead, middle, tail] if fragment]
    deduped = _dedupe_texts(fragments)
    if not deduped:
        return "该段围绕连续冲突推进与状态转折展开。"
    return "；随后".join(deduped[:3])[:280]


def _match_flow_template_for_segment(
    chapter_count: int,
    dominant_conflict: str,
    fused_world: FusedWorld,
) -> Dict[str, Any]:
    templates = fused_world.event_flow_templates or []
    if not templates:
        return {
            "name": f"segment_{chapter_count}_chapters",
            "beat_blueprint": _fallback_chapter_blueprint(chapter_count),
            "required_coverage": ["铺垫入场", "冲突升级", "局势转折", "余波收束"],
        }
    best = templates[0]
    best_score = -999.0
    for template in templates:
        score = 0.0
        chapter_range = template.get("chapter_count_range", [3, 8])
        if isinstance(chapter_range, list) and len(chapter_range) == 2:
            low, high = int(chapter_range[0]), int(chapter_range[1])
            if low <= chapter_count <= high:
                score += 3.0
            else:
                score -= min(abs(chapter_count - low), abs(chapter_count - high)) * 0.2
        if dominant_conflict and dominant_conflict in template.get("dominant_conflicts", []):
            score += 4.0
        if score > best_score:
            best = template
            best_score = score
    return best


def _infer_segment_role_slots(
    events: List[InducedEvent],
    fused_world: FusedWorld,
    matched_template: Dict[str, Any],
    summary: str,
    dominant_conflict: str,
    dominant_function: str,
) -> List[str]:
    haystack = " ".join(
        _flatten_text_values([summary, dominant_conflict, dominant_function, matched_template.get("name", "")])
    ).lower()
    for item in fused_world.role_slot_templates or []:
        keywords = [str(keyword).lower() for keyword in item.get("keywords", [])]
        if keywords and any(keyword and keyword in haystack for keyword in keywords):
            slots = item.get("role_slots", [])
            if slots:
                return list(slots)[:6]
    for template in fused_world.event_templates or []:
        if dominant_conflict and template.get("conflict_type") == dominant_conflict:
            slots = template.get("role_slots", [])
            if slots:
                return list(slots)[:6]
    combined = _dedupe_texts(slot for event in events for slot in getattr(event, "character_keys", []))
    if combined:
        return combined[:4]
    return _DEFAULT_ROLE_SLOTS[:]


def _build_segment_logic_card(events: List[InducedEvent], stage_name: str, pacing_role: str) -> Dict[str, Any]:
    return {
        "preconditions": _dedupe_texts(item for event in events for item in getattr(event, "state_inputs", []))[:10],
        "state_outputs": _dedupe_texts(item for event in events for item in getattr(event, "state_outputs", []))[:10],
        "hook_open": _dedupe_texts(item for event in events for item in getattr(event, "hook_open", []))[:6],
        "hook_close": _dedupe_texts(item for event in events for item in getattr(event, "hook_close", []))[:6],
        "resource_delta": _dedupe_texts(item for event in events for item in getattr(event, "resource_delta", []))[:6],
        "relationship_delta": "；".join(
            _dedupe_texts(item for event in events for item in getattr(event, "relationship_delta", []))[:4]
        ),
        "power_state": stage_name or _top_text_value(event.power_state for event in events),
        "identity_state": _last_non_empty(event.identity_state for event in events),
        "timeline_stage": pacing_role or "主线推进",
    }


def _build_segment_source_refs(events: List[InducedEvent], source_atom_ids: List[str]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for event in events:
        key = ("induced_event", event.event_id, event.novel_source)
        if key not in seen:
            seen.add(key)
            refs.append({"ref_type": "induced_event", "ref_id": event.event_id, "source_novel": event.novel_source})
    novel_source = events[0].novel_source if events else ""
    for atom_id in source_atom_ids:
        key = ("atom", atom_id, novel_source)
        if key not in seen:
            seen.add(key)
            refs.append({"ref_type": "atom", "ref_id": atom_id, "source_novel": novel_source})
    return refs


def _build_fallback_source_refs(cluster: List[Dict[str, Any]], source_atom_ids: List[str]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    novel_source = ""
    for item in cluster:
        event = item.get("event")
        atom = item.get("atom")
        if atom:
            novel_source = getattr(atom, "novel_source", "") or novel_source
        event_id = getattr(event, "event_id", "")
        if event_id:
            key = ("legacy_event", event_id, novel_source)
            if key not in seen:
                seen.add(key)
                refs.append({"ref_type": "legacy_event", "ref_id": event_id, "source_novel": novel_source})
        chunk_id = getattr(event, "source_chunk_id", "")
        if chunk_id:
            key = ("chunk", chunk_id, novel_source)
            if key not in seen:
                seen.add(key)
                refs.append({"ref_type": "chunk", "ref_id": chunk_id, "source_novel": novel_source})
    for atom_id in source_atom_ids:
        key = ("atom", atom_id, novel_source)
        if key not in seen:
            seen.add(key)
            refs.append({"ref_type": "atom", "ref_id": atom_id, "source_novel": novel_source})
    return refs


def _infer_fallback_cluster_stage(cluster: List[Dict[str, Any]], stages: List[ProgressionStage]) -> str:
    texts: List[str] = []
    for item in cluster:
        atom = item.get("atom")
        if atom:
            stage_name = infer_stage_from_node(atom, stages)
            if stage_name:
                return stage_name
            texts.extend([getattr(atom, "summary", ""), getattr(atom, "raw_summary", ""), getattr(atom, "location", "")])
    return extract_max_stage_from_text(" ".join(_flatten_text_values(texts)), stages)


def _infer_segment_pacing_role(index: int, total: int, dominant_function: str) -> str:
    if dominant_function:
        return dominant_function
    ratio = index / max(total - 1, 1) if total > 1 else 0
    if ratio < 0.12:
        return "开篇落点"
    if ratio > 0.88:
        return "卷末收束"
    if ratio < 0.35:
        return "铺垫与试探"
    if ratio < 0.72:
        return "冲突升级"
    return "高潮对决"


def _string_overlap(left: str, right: str) -> float:
    left_tokens = set(_label_tokens(left))
    right_tokens = set(_label_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _label_tokens(text: str) -> List[str]:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return []
    pieces = [piece for piece in re.split(r"[\s/|,，、；。:：()（）\[\]<>]+", normalized) if piece]
    tokens: List[str] = []
    for piece in pieces:
        tokens.append(piece)
        if any("\u4e00" <= char <= "\u9fff" for char in piece) and len(piece) >= 2:
            tokens.extend(piece[index : index + 2] for index in range(len(piece) - 1))
    return _dedupe_texts(tokens)


def _top_text_value(values: Sequence[str] | Any) -> str:
    if isinstance(values, str):
        flattened = _flatten_text_values([values])
    elif hasattr(values, "__iter__"):
        flattened = _flatten_text_values(list(values))
    else:
        flattened = _flatten_text_values([values])
    if not flattened:
        return ""
    counter = Counter(flattened)
    return counter.most_common(1)[0][0]


def _last_non_empty(values: Any) -> str:
    candidate = ""
    for value in values:
        text = str(value or "").strip()
        if text:
            candidate = text
    return candidate


def _has_any_keyword(text: str, keywords: Sequence[str]) -> bool:
    haystack = str(text or "")
    return any(keyword in haystack for keyword in keywords)
