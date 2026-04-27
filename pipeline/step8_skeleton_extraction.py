from __future__ import annotations

from typing import Any, Dict, List, Optional

from pipeline.core.skeleton_core import (
    _DEFAULT_ROLE_SLOTS,
    _atom_text,
    _dedupe_texts,
    _fallback_chapter_blueprint,
    _flatten_text_values,
    _nodes_from_induced_events,
    _realm_level_from_arc,
    load_node_map,
    save_node_map,
)
from pipeline.core.story_models import SkeletonNode
from pipeline.core.world_building_core import FusedWorld
from pipeline.step1_chunking import NarrativeEvent, VolumeArc
from pipeline.step2_extraction import PlotAtom
from pipeline.step3_event_induction import InducedEvent


_STEP8_FILENAME = "step8_source_skeletons.json"


def extract_source_skeletons(
    novel_arcs: Dict[str, List[VolumeArc]],
    all_atoms: Dict[str, List[PlotAtom]],
    fused_world: FusedWorld,
    induced_events_by_novel: Optional[Dict[str, List[InducedEvent]]] = None,
) -> Dict[str, List[SkeletonNode]]:
    per_novel_nodes: Dict[str, List[SkeletonNode]] = {}
    for novel_name, arcs in novel_arcs.items():
        induced_events = (induced_events_by_novel or {}).get(novel_name, [])
        if induced_events:
            source_nodes = _nodes_from_induced_events(induced_events, fused_world)
        else:
            source_nodes = _extract_skeleton_nodes(arcs, all_atoms.get(novel_name, []), fused_world)
        for node in source_nodes:
            node.source_novels = [novel_name]
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
        _flatten_text_values([
            atom.conflict_type if atom else "",
            atom.narrative_function if atom else "",
            atom.summary if atom else "",
            atom.core_action if atom else "",
            event.summary,
        ])
    ).lower()
    for template in fused_world.role_slot_templates or []:
        keywords = [str(item).lower() for item in template.get("keywords", [])]
        if keywords and any(keyword and keyword in combined_text for keyword in keywords):
            slots = template.get("role_slots", [])
            if isinstance(slots, list) and slots:
                return slots[:4]
    return _DEFAULT_ROLE_SLOTS[:]


def _extract_skeleton_nodes(arcs: List[VolumeArc], atoms: List[PlotAtom], fused_world: FusedWorld) -> List[SkeletonNode]:
    atom_map = {atom.atom_id: atom for atom in atoms}
    atomic_items = []
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
                    "chapter_count": max(1, int(getattr(atom, "chapter_end", getattr(event, "chapter_end", 0)) or 0) - int(getattr(atom, "chapter_start", getattr(event, "chapter_start", 0)) or 0) + 1),
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
        nodes.append(
            SkeletonNode(
                node_id=f"{cluster[0]['arc_name']}_story_event{cluster_index}",
                arc_name=cluster[0]["arc_name"],
                realm_level=max(item["realm_level"] for item in cluster),
                pacing_role=_infer_cluster_pacing_role(cluster_index, total_clusters, cluster, primary_function),
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
                logic_card=_build_cluster_logic_card(cluster, chapter_count, _infer_cluster_pacing_role(cluster_index, total_clusters, cluster, primary_function)),
            )
        )
    return nodes


def _cluster_atomic_items(items):
    clusters = []
    by_arc: Dict[str, List[dict]] = {}
    for item in items:
        by_arc.setdefault(item["arc_name"], []).append(item)

    for arc_items in by_arc.values():
        index = 0
        while index < len(arc_items):
            cluster = []
            while index + len(cluster) < len(arc_items):
                candidate = arc_items[index + len(cluster)]
                cluster.append(candidate)
                chapter_total = _cluster_chapter_total(cluster)
                if chapter_total >= 4 and (
                    chapter_total >= 10 or _cluster_boundary_hit(candidate) or _next_cluster_resets(arc_items, index + len(cluster) - 1)
                ):
                    break
                if chapter_total >= 12:
                    break
            clusters.append(cluster)
            index += len(cluster)
    return clusters


def _cluster_chapter_total(cluster) -> int:
    covered = set()
    for item in cluster:
        start = int(item.get("chapter_start", 0) or 0)
        end = int(item.get("chapter_end", 0) or start or 0)
        if start and end and end >= start:
            covered.update(range(start, end + 1))
        else:
            covered.add(len(covered) + 1)
    return max(1, len(covered))


def _cluster_boundary_hit(item) -> bool:
    text = " ".join(_flatten_text_values([item.get("summary", ""), item.get("function_hint", ""), item.get("conflict_hint", "")]))
    if any(keyword in text for keyword in ["转折", "翻盘", "余波", "收束", "后果", "伏笔", "钩子"]):
        return True
    return any(keyword in text for keyword in ["高潮", "对决", "逆袭"])


def _next_cluster_resets(items, current_index: int) -> bool:
    if current_index + 1 >= len(items):
        return True
    next_item = items[current_index + 1]
    text = " ".join(_flatten_text_values([next_item.get("summary", ""), next_item.get("raw_pacing_role", "")]))
    return any(keyword in text for keyword in ["开篇", "铺垫", "试探", "入场"])


def _fallback_blueprint_for_cluster(cluster) -> List[Dict[str, Any]]:
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


def _relative_chapter_spans_from_cluster(cluster) -> List[List[int]]:
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


def _top_cluster_value(cluster, key: str) -> str:
    values = [item.get(key, "") for item in cluster]
    deduped = _dedupe_texts(_flatten_text_values(values))
    return deduped[0] if deduped else ""


def _build_cluster_summary(cluster) -> str:
    fragments = [item["summary"] for item in cluster[:3] if item.get("summary")]
    lead = "；".join(_dedupe_texts(fragments))
    return lead[:280]


def _infer_cluster_pacing_role(cluster_index: int, total_clusters: int, cluster, primary_function: str) -> str:
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


def _build_cluster_logic_card(cluster, chapter_count: int, pacing_role: str) -> Dict[str, Any]:
    inputs = _cluster_logic_terms(cluster, ["precondition", "motivation"], limit=8)
    outputs = _cluster_logic_terms(cluster, ["consequence", "core_action"], limit=8)
    relationship = []
    for item in cluster:
        emotion = str(item.get("emotion", "") or "").strip()
        if emotion and len(item.get("character_keys", []) or []) >= 2:
            relationship.append(f"相关角色关系转向{emotion}")
    hooks_open = _cluster_keyword_terms(cluster[-2:], ["consequence", "summary"], ("伏笔", "线索", "后续", "隐患", "更大", "下一", "秘密", "真相"))
    hooks_close = _cluster_keyword_terms(cluster, ["core_action", "summary"], ("破局", "揭露", "翻盘", "和解", "救出", "斩杀", "夺得", "突破", "解决", "脱身"))
    power_terms = []
    for item in cluster:
        for entry in item.get("cultivation_elements", []):
            text = str(entry or "").strip()
            if text and any(keyword in text for keyword in ("境", "期", "层", "重", "阶", "劫", "台")):
                power_terms.append(text)
    identity_terms = _cluster_keyword_terms(cluster, ["summary", "core_action", "location"], ("弟子", "长老", "亲传", "外门", "内门", "掌柜", "少主", "客卿", "散修", "逃亡", "潜伏", "卧底", "供奉"))
    return {
        "preconditions": inputs,
        "state_outputs": outputs,
        "power_state": _dedupe_texts(power_terms)[0] if power_terms else f"{chapter_count}章级事件强度",
        "identity_state": identity_terms[0] if identity_terms else "",
        "relationship_delta": "；".join(_dedupe_texts(relationship)[:3]),
        "timeline_stage": pacing_role or "主线推进",
        "hook_open": hooks_open,
        "hook_close": hooks_close,
    }


def _cluster_logic_terms(cluster, keys: List[str], limit: int) -> List[str]:
    values: List[str] = []
    for item in cluster:
        for key in keys:
            values.extend(_flatten_text_values([item.get(key, "")]))
    results: List[str] = []
    for value in values:
        for piece in str(value or "").replace("；", "，").replace("。", "，").split("，"):
            text = piece.strip("、/ \t\r\n")
            if len(text) >= 2 and text not in results:
                results.append(text[:32])
            if len(results) >= limit:
                return results
    return results


def _cluster_keyword_terms(cluster, keys: List[str], keywords: tuple[str, ...]) -> List[str]:
    values: List[str] = []
    for item in cluster:
        text = " ".join(_flatten_text_values([item.get(key, "") for key in keys]))
        for piece in text.replace("；", "，").replace("。", "，").split("，"):
            cleaned = piece.strip("、/ \t\r\n")
            if cleaned and any(keyword in cleaned for keyword in keywords):
                values.append(cleaned[:32])
    return _dedupe_texts(values)[:4]
