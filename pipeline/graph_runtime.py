"""Runtime graph reads and commit-gated patches for chapter generation.

The corpus graph remains evidence-only and immutable.  A generation run keeps
its own compact state ledger: candidates create a patch, validators decide
whether it is safe, and only then is that patch committed for the next chapter.
No helper in this module returns source quotes or complete source chapters.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import ChapterProgram, NarrativeGraph, ParagraphProgram, SceneProgram
from .jsonio import read_json, write_json


STATE_BEARING_FACT_KINDS = frozenset({
    "identity", "goal", "relationship", "location", "resource", "knowledge", "rule", "progression", "information",
})


def graph_fingerprint(graph: NarrativeGraph) -> str:
    body = json.dumps(graph.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _source_ref(chapter_id: str, kind: str, local_id: str) -> str:
    return f"{kind}:{chapter_id}:{local_id}"


def _node_by_source(graph: NarrativeGraph) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for node in graph.nodes:
        for source_id in node.source_ids:
            result.setdefault(source_id, node)
    return result


def _compact_node(node: Any, *, foreshadow_lifecycle: str | None = None, resolved_by_event_id: str = "") -> dict[str, Any]:
    """Return structured meaning only; evidence quotes never enter a prompt."""

    attributes = node.attributes if isinstance(node.attributes, dict) else {}
    item: dict[str, Any] = {"节点ID": node.node_id, "类型": node.kind}
    if node.kind == "entity":
        item.update({"名称": node.label, "实体类型": attributes.get("entity_kind", "")})
    elif node.kind == "fact":
        item.update({
            "事实类型": attributes.get("fact_kind", ""), "谓词": attributes.get("predicate", ""),
            "值": attributes.get("value", ""), "主体局部ID": attributes.get("subject_local_id", ""),
            "客体局部ID": attributes.get("object_local_id", ""),
        })
    elif node.kind == "event":
        item.update({"动作": attributes.get("action", ""), "阻力": attributes.get("obstacle", ""), "选择": attributes.get("decision", "")})
    elif node.kind == "scene":
        item.update({"紧张度": attributes.get("tension", 0)})
    elif node.kind == "time_anchor":
        item.update({"时间锚类型": attributes.get("anchor_kind", "")})
    elif node.kind == "foreshadow":
        # The immutable corpus graph can know a later resolution.  A writing
        # request must instead see only what has been committed before this
        # paragraph (plus a resolution event licensed in this paragraph).
        item.update({
            "生命周期": foreshadow_lifecycle or "open",
            "兑现事件": resolved_by_event_id if foreshadow_lifecycle == "resolved" else "",
        })
    return item


def _edge_compact(edge: Any) -> dict[str, Any]:
    return {
        "关系ID": edge.edge_id,
        "关系": edge.kind,
        "起点": edge.source_node_id,
        "终点": edge.target_node_id,
        "属性": edge.attributes if isinstance(edge.attributes, dict) else {},
    }


def initialize_runtime_graph(graph: NarrativeGraph | None, *, program: ChapterProgram | None = None) -> dict[str, Any]:
    """Create a mutable ledger separate from the immutable source graph."""

    return {
        "schema_version": "1.0",
        "mode": "narrative_graph" if graph is not None else "program_contract_fallback",
        "author_id": graph.author_id if graph is not None else "",
        "work_id": graph.work_id if graph is not None else "",
        "base_graph_fingerprint": graph_fingerprint(graph) if graph is not None else "",
        "program_source_hash": program.source_hash if program is not None else "",
        "committed_patches": [],
        "effective_state": {},
        "effective_foreshadows": {},
    }


def load_or_initialize_runtime_graph(
    path: Path | None,
    graph: NarrativeGraph | None,
    program: ChapterProgram,
) -> dict[str, Any]:
    if path is None or not path.exists():
        return initialize_runtime_graph(graph, program=program)
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("runtime graph state must be a JSON object")
    expected = graph_fingerprint(graph) if graph is not None else ""
    if str(payload.get("base_graph_fingerprint", "")) != expected:
        raise ValueError("runtime graph state belongs to a different base graph")
    if not isinstance(payload.get("committed_patches"), list) or not isinstance(payload.get("effective_state"), dict):
        raise ValueError("runtime graph state is incomplete")
    foreshadows = payload.get("effective_foreshadows", {})
    if not isinstance(foreshadows, dict):
        raise ValueError("runtime graph foreshadow state is incomplete")
    return payload


def _event_and_fact_ids(scene: SceneProgram, paragraph: ParagraphProgram) -> tuple[set[str], set[str]]:
    beats = {item.beat_id: item for item in scene.narrative_beats}
    fact_ids = set(paragraph.required_fact_ids)
    event_ids = set(paragraph.event_ids)
    for beat_id in paragraph.beat_ids:
        beat = beats.get(beat_id)
        if beat is not None:
            fact_ids.update(beat.required_fact_ids)
            fact_ids.update(beat.support_fact_ids)
            event_ids.update(beat.event_ids)
    events = {item.event_id: item for item in scene.event_programs}
    for event_id in tuple(event_ids):
        event = events.get(event_id)
        if event is not None:
            fact_ids.update(event.precondition_fact_ids)
            fact_ids.update(event.required_fact_ids)
    return fact_ids, event_ids


def _program_contract_subgraph(
    runtime: dict[str, Any],
    program: ChapterProgram,
    scene: SceneProgram,
    paragraph: ParagraphProgram,
) -> dict[str, Any]:
    fact_ids, event_ids = _event_and_fact_ids(scene, paragraph)
    facts = {item.fact_id: item for item in scene.fact_contracts}
    events = {item.event_id: item for item in scene.event_programs}
    names = {item.entity_id: item.canonical_name for item in program.entities}
    entity_ids = set(scene.participant_ids)
    for fact_id in fact_ids:
        fact = facts.get(fact_id)
        if fact is not None:
            entity_ids.add(fact.subject_id)
            if fact.object_id:
                entity_ids.add(fact.object_id)
    for event_id in event_ids:
        event = events.get(event_id)
        if event is not None:
            entity_ids.update(event.participant_ids)
    return {
        "模式": "程序契约子图",
        "章节ID": program.chapter_id,
        "场景ID": scene.scene_id,
        "段落ID": paragraph.paragraph_id,
        "实体": [{"实体ID": item, "名称": names.get(item, item)} for item in sorted(entity_ids)],
        "允许事实": [{
            "事实ID": item.fact_id, "类型": item.fact_kind, "主体": item.subject_id,
            "谓词": item.predicate, "值": item.value, "客体": item.object_id,
        } for item in scene.fact_contracts if item.fact_id in fact_ids],
        "允许事件": [{
            "事件ID": item.event_id, "动作": item.action, "参与者": list(item.participant_ids),
            "前提事实ID": list(item.precondition_fact_ids), "结果事实ID": list(item.outcome_fact_ids),
        } for item in scene.event_programs if item.event_id in event_ids],
        "关系与因果": [],
        "已提交状态": _state_for_entities(runtime, entity_ids),
        "硬约束": ["仅可使用本子图列出的实体、事实、事件与已提交状态", "不得添加未列出的关系、结果或后续事件"],
    }


def _state_for_entities(runtime: dict[str, Any], entity_ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(entity_ids)
    result: list[dict[str, Any]] = []
    for key, value in runtime.get("effective_state", {}).items():
        if not isinstance(value, dict):
            continue
        if str(value.get("subject_id", "")) in wanted:
            result.append({"状态键": key, **value})
    return result


def _foreshadow_participants(graph: NarrativeGraph, foreshadow_id: str) -> tuple[str, ...]:
    """Return only graph entity IDs structurally attached to a hint."""

    opened_by = {
        edge.source_node_id
        for edge in graph.edges
        if edge.kind == "foreshadow_open" and edge.target_node_id == foreshadow_id
    }
    participants = {
        edge.source_node_id
        for edge in graph.edges
        if edge.target_node_id in opened_by and edge.kind in {"asserts", "event_participant"}
    }
    return tuple(sorted(participants))


def _runtime_foreshadows_for_entities(runtime: dict[str, Any], entity_ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = set(entity_ids)
    result: list[dict[str, Any]] = []
    for item in runtime.get("effective_foreshadows", {}).values():
        if not isinstance(item, dict) or str(item.get("lifecycle", "")) != "open":
            continue
        participants = {str(value) for value in item.get("participant_ids", []) if str(value).strip()}
        if participants & wanted:
            result.append({
                "节点ID": str(item.get("foreshadow_id", "")),
                "类型": "foreshadow",
                "生命周期": "open",
                "来源": "已提交图谱补丁",
            })
    return result


def narrative_subgraph_for_paragraph(
    graph: NarrativeGraph | None,
    runtime: dict[str, Any],
    program: ChapterProgram,
    scene: SceneProgram,
    paragraph: ParagraphProgram,
) -> dict[str, Any]:
    """Read only the graph neighbourhood licensed for one prose paragraph."""

    if graph is None:
        return _program_contract_subgraph(runtime, program, scene, paragraph)
    sources = _node_by_source(graph)
    fact_ids, event_ids = _event_and_fact_ids(scene, paragraph)
    fact_nodes = {
        sources[_source_ref(program.chapter_id, "fact", item)].node_id
        for item in fact_ids if _source_ref(program.chapter_id, "fact", item) in sources
    }
    event_nodes = {
        sources[_source_ref(program.chapter_id, "event", item)].node_id
        for item in event_ids if _source_ref(program.chapter_id, "event", item) in sources
    }
    entity_local_ids = set(scene.participant_ids)
    facts = {item.fact_id: item for item in scene.fact_contracts}
    for fact_id in fact_ids:
        fact = facts.get(fact_id)
        if fact is not None:
            entity_local_ids.add(fact.subject_id)
            if fact.object_id:
                entity_local_ids.add(fact.object_id)
    entity_nodes = {
        sources[_source_ref(program.chapter_id, "entity", item)].node_id
        for item in entity_local_ids if _source_ref(program.chapter_id, "entity", item) in sources
    }
    scene_node = sources.get(_source_ref(program.chapter_id, "scene", scene.scene_id))
    allowed = set(fact_nodes) | set(event_nodes) | set(entity_nodes)
    if scene_node is not None:
        allowed.add(scene_node.node_id)
    permitted_edge_kinds = {
        "asserts", "fact_object", "event_participant", "event_trigger", "event_requires", "event_produces", "event_cost",
        "scene_contains", "scene_participant", "scene_entry_state", "scene_exit_state", "scene_at", "scene_time",
        "relationship", "causes", "spatial_relation", "state_change", "state_observed", "foreshadow_open", "foreshadow_resolved",
    }
    # A relationship or an open/closed foreshadow often has one endpoint just
    # outside the paragraph's direct facts.  Admit that *single* neighbour so
    # its meaning remains available, but never fan out into a chapter-wide
    # traversal.  This is deliberately performed before selecting edges.
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    one_hop_kinds = {"relationship", "foreshadow_open", "foreshadow_resolved"}
    graph_scoped_kinds = {"relationship", "spatial_relation", "location_transition", "foreshadow_open", "foreshadow_resolved"}
    for edge in graph.edges:
        if edge.kind not in one_hop_kinds:
            continue
        if edge.kind in graph_scoped_kinds and edge.chapter_id != program.chapter_id:
            continue
        if edge.source_node_id in allowed and edge.target_node_id in nodes_by_id:
            allowed.add(edge.target_node_id)
        elif edge.target_node_id in allowed and edge.source_node_id in nodes_by_id:
            allowed.add(edge.source_node_id)
    selected_edges = [
        edge for edge in graph.edges
        if edge.kind in permitted_edge_kinds
        and edge.source_node_id in allowed and edge.target_node_id in allowed
        and (edge.kind not in graph_scoped_kinds or edge.chapter_id == program.chapter_id)
    ]
    selected_nodes = [node for node in graph.nodes if node.node_id in allowed][:64]
    selected_edges = [edge for edge in selected_edges if edge.source_node_id in {node.node_id for node in selected_nodes} and edge.target_node_id in {node.node_id for node in selected_nodes}][:96]
    visible_foreshadows: dict[str, tuple[str, str]] = {}
    resolving_events = {edge.target_node_id: edge.source_node_id for edge in selected_edges if edge.kind == "foreshadow_resolved"}
    for node in selected_nodes:
        if node.kind != "foreshadow":
            continue
        resolved_by = resolving_events.get(node.node_id, "")
        visible_foreshadows[node.node_id] = ("resolved" if resolved_by else "open", resolved_by)
    runtime_foreshadows = _runtime_foreshadows_for_entities(runtime, entity_nodes)
    selected_node_payloads = [
        _compact_node(node, foreshadow_lifecycle=visible_foreshadows.get(node.node_id, ("", ""))[0], resolved_by_event_id=visible_foreshadows.get(node.node_id, ("", ""))[1])
        for node in selected_nodes
    ]
    known_node_ids = {item["节点ID"] for item in selected_node_payloads}
    selected_node_payloads.extend(item for item in runtime_foreshadows if item["节点ID"] and item["节点ID"] not in known_node_ids)
    return {
        "模式": "剧情事实子图",
        "图谱指纹": graph_fingerprint(graph),
        "章节ID": program.chapter_id,
        "场景ID": scene.scene_id,
        "段落ID": paragraph.paragraph_id,
        "节点": selected_node_payloads,
        "关系与因果": [_edge_compact(edge) for edge in selected_edges],
        "已提交状态": _state_for_entities(runtime, entity_nodes | entity_local_ids),
        "硬约束": ["本子图不含原文引文；不得索取或补写整章原文", "仅可使用节点、关系、因果和已提交状态中的信息", "未出现的伏笔仍为未解状态，不得提前兑现"],
    }


def _entity_graph_id(graph: NarrativeGraph | None, chapter_id: str, local_id: str) -> str:
    if graph is None:
        return local_id
    node = _node_by_source(graph).get(_source_ref(chapter_id, "entity", local_id))
    return node.node_id if node is not None else local_id


def _state_slot(fact: Any, object_id: str) -> str:
    if fact.fact_kind == "location":
        return "location:current"
    if fact.fact_kind in {"identity", "goal"}:
        return f"{fact.fact_kind}:{fact.predicate}"
    if fact.fact_kind == "progression":
        level_cues = ("境界", "修为", "修炼", "突破", "层", "阶", "级")
        if any(cue in f"{fact.predicate} {fact.value}" for cue in level_cues):
            return "progression:level"
    return f"{fact.fact_kind}:{fact.predicate}:{object_id or fact.value}"


def _scene_validation_snapshot(scene_results: list[dict[str, Any]]) -> tuple[set[str], set[str], list[str]]:
    facts: set[str] = set()
    events: set[str] = set()
    issues: list[str] = []
    for result in scene_results:
        validations = result.get("validations", []) if isinstance(result, dict) else []
        latest = validations[-1] if isinstance(validations, list) and validations and isinstance(validations[-1], dict) else {}
        if not bool(result.get("passed")) or not bool(result.get("prose_passed", True)) or not bool(result.get("role_passed", True)):
            issues.append(f"scene:{result.get('scene_id', '')}:validation_failed")
        facts.update(str(item) for item in latest.get("realized_fact_ids", []) if str(item).strip())
        events.update(str(item) for item in latest.get("realized_event_ids", []) if str(item).strip())
    return facts, events, issues


def _foreshadow_patch_updates(
    graph: NarrativeGraph | None,
    program: ChapterProgram,
    realized_fact_ids: set[str],
    realized_event_ids: set[str],
) -> list[dict[str, Any]]:
    """Record only hints opened or resolved by this accepted chapter.

    The source graph may include future chapters, therefore its final
    lifecycle value is never copied into runtime state.  A resolution becomes
    visible only if the resolving event has been validated in this patch.
    """

    if graph is None:
        return []
    realized_sources = {
        *(_source_ref(program.chapter_id, "fact", item) for item in realized_fact_ids),
        *(_source_ref(program.chapter_id, "event", item) for item in realized_event_ids),
    }
    updates: list[dict[str, Any]] = []
    for node in graph.nodes:
        if node.kind != "foreshadow" or node.chapter_id != program.chapter_id:
            continue
        if not set(node.source_ids) & realized_sources:
            continue
        resolution_edges = [
            edge for edge in graph.edges
            if edge.kind == "foreshadow_resolved" and edge.target_node_id == node.node_id
            and set(edge.source_ids) & realized_sources
        ]
        resolved_by_event_id = ""
        if resolution_edges:
            source_ids = set(resolution_edges[0].source_ids) & realized_sources
            for source_id in source_ids:
                prefix = f"event:{program.chapter_id}:"
                if source_id.startswith(prefix):
                    resolved_by_event_id = source_id[len(prefix):]
                    break
        updates.append({
            "foreshadow_id": node.node_id,
            "lifecycle": "resolved" if resolution_edges else "open",
            "participant_ids": list(_foreshadow_participants(graph, node.node_id)),
            "opened_in_chapter": program.chapter_id,
            "resolved_by_event_id": resolved_by_event_id,
        })
    return updates


def build_chapter_graph_patch(
    graph: NarrativeGraph | None,
    runtime: dict[str, Any],
    program: ChapterProgram,
    scene_results: list[dict[str, Any]],
    *,
    prose_passed: bool,
    style_passed: bool,
    length_passed: bool,
) -> dict[str, Any]:
    """Build a candidate patch from already-validated program claims only."""

    realized_facts, realized_events, issues = _scene_validation_snapshot(scene_results)
    # Exit-state facts are mandatory even when the compiler classified their
    # prose expression as optional.  A state patch may never claim an
    # un-realised change merely because its fact was not a headline beat.
    all_facts = {
        item.fact_id
        for scene in program.scene_programs
        for item in scene.fact_contracts
        if item.must_realize
    }
    all_facts.update(
        fact_id
        for scene in program.scene_programs
        for fact_id in scene.exit_state_fact_ids
    )
    all_events = {item.event_id for scene in program.scene_programs for item in scene.event_programs}
    missing_facts = sorted(all_facts - realized_facts)
    missing_events = sorted(all_events - realized_events)
    if missing_facts:
        issues.append("missing_realized_facts:" + ",".join(missing_facts))
    if missing_events:
        issues.append("missing_realized_events:" + ",".join(missing_events))
    fact_by_id = {item.fact_id: item for scene in program.scene_programs for item in scene.fact_contracts}
    event_by_id = {item.event_id: item for scene in program.scene_programs for item in scene.event_programs}
    producer_by_fact = {
        fact_id: event.event_id
        for event in event_by_id.values()
        for fact_id in event.outcome_fact_ids
    }
    updates: list[dict[str, Any]] = []
    effective = {str(key): dict(value) for key, value in runtime.get("effective_state", {}).items() if isinstance(value, dict)}
    causal_links: list[dict[str, Any]] = []
    for scene in program.scene_programs:
        for event in scene.event_programs:
            for fact_id in event.precondition_fact_ids:
                producer = producer_by_fact.get(fact_id)
                if producer and producer != event.event_id:
                    causal_links.append({"起因事件ID": producer, "结果事件ID": event.event_id, "依据事实ID": fact_id})
                elif fact_id not in fact_by_id:
                    issues.append(f"unknown_causal_precondition:{event.event_id}:{fact_id}")
        for fact_id in scene.exit_state_fact_ids:
            fact = fact_by_id.get(fact_id)
            if fact is None or fact.fact_kind not in STATE_BEARING_FACT_KINDS:
                continue
            subject_id = _entity_graph_id(graph, program.chapter_id, fact.subject_id)
            object_id = _entity_graph_id(graph, program.chapter_id, fact.object_id) if fact.object_id else ""
            slot = _state_slot(fact, object_id)
            state_key = f"{subject_id}|{slot}"
            value = object_id or fact.value
            previous = effective.get(state_key)
            operation = "add" if previous is None else ("observe" if str(previous.get("value", "")) == value else "update")
            update = {
                "state_key": state_key, "subject_id": subject_id, "slot": slot, "value": value,
                "object_id": object_id, "operation": operation, "fact_id": fact_id, "scene_id": scene.scene_id,
            }
            effective[state_key] = update
            updates.append(update)
    foreshadow_updates = _foreshadow_patch_updates(graph, program, realized_facts, realized_events)
    checks = {
        "scene_validation_passed": not any(item.startswith("scene:") for item in issues),
        "prose_passed": prose_passed,
        "style_passed": style_passed,
        "length_passed": length_passed,
        "missing_fact_ids": missing_facts,
        "missing_event_ids": missing_events,
        "causal_link_count": len(causal_links),
        "foreshadow_update_count": len(foreshadow_updates),
        "issues": issues,
    }
    patch_id = hashlib.sha256((program.chapter_id + program.source_hash + json.dumps({"state": updates, "foreshadows": foreshadow_updates}, ensure_ascii=False, sort_keys=True)).encode("utf-8")).hexdigest()[:20]
    return {
        "schema_version": "1.0", "patch_id": f"patch:{patch_id}", "status": "candidate",
        "base_graph_fingerprint": graph_fingerprint(graph) if graph is not None else "",
        "chapter_id": program.chapter_id, "program_source_hash": program.source_hash,
        "realized_fact_ids": sorted(realized_facts), "realized_event_ids": sorted(realized_events),
        "state_updates": updates, "causal_links": causal_links, "foreshadow_updates": foreshadow_updates, "checks": checks,
    }


def validate_graph_patch(graph: NarrativeGraph | None, runtime: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    expected = graph_fingerprint(graph) if graph is not None else ""
    if str(patch.get("status", "")) != "candidate":
        issues.append("patch_not_candidate")
    if str(patch.get("base_graph_fingerprint", "")) != expected:
        issues.append("base_graph_mismatch")
    checks = patch.get("checks", {}) if isinstance(patch.get("checks"), dict) else {}
    if checks.get("issues") or not all(bool(checks.get(item)) for item in ("scene_validation_passed", "prose_passed", "style_passed", "length_passed")):
        issues.append("generation_validation_not_passed")
    active = {str(key): dict(value) for key, value in runtime.get("effective_state", {}).items() if isinstance(value, dict)}
    for update in patch.get("state_updates", []):
        if not isinstance(update, dict):
            issues.append("invalid_state_update")
            continue
        key = str(update.get("state_key", ""))
        operation = str(update.get("operation", ""))
        if not key or operation not in {"add", "update", "observe"}:
            issues.append("invalid_state_operation")
            continue
        previous = active.get(key)
        if operation == "add" and previous is not None:
            issues.append(f"state_add_conflict:{key}")
        if operation == "update" and previous is None:
            issues.append(f"state_update_without_prior:{key}")
        active[key] = update
    foreshadow_updates = patch.get("foreshadow_updates", [])
    if not isinstance(foreshadow_updates, list):
        issues.append("invalid_foreshadow_updates")
        foreshadow_updates = []
    known_foreshadows = {
        node.node_id: node
        for node in graph.nodes
        if node.kind == "foreshadow"
    } if graph is not None else {}
    realized_sources = {
        *(_source_ref(str(patch.get("chapter_id", "")), "fact", str(item)) for item in patch.get("realized_fact_ids", []) if str(item).strip()),
        *(_source_ref(str(patch.get("chapter_id", "")), "event", str(item)) for item in patch.get("realized_event_ids", []) if str(item).strip()),
    }
    seen_foreshadows: set[str] = set()
    for update in foreshadow_updates:
        if not isinstance(update, dict):
            issues.append("invalid_foreshadow_update")
            continue
        foreshadow_id = str(update.get("foreshadow_id", ""))
        lifecycle = str(update.get("lifecycle", ""))
        if not foreshadow_id or lifecycle not in {"open", "resolved"}:
            issues.append("invalid_foreshadow_update")
            continue
        if foreshadow_id in seen_foreshadows:
            issues.append(f"duplicate_foreshadow_update:{foreshadow_id}")
            continue
        seen_foreshadows.add(foreshadow_id)
        if graph is None:
            issues.append(f"foreshadow_update_without_graph:{foreshadow_id}")
            continue
        node = known_foreshadows.get(foreshadow_id)
        if node is None or node.chapter_id != str(patch.get("chapter_id", "")):
            issues.append(f"unknown_foreshadow:{foreshadow_id}")
            continue
        if not set(node.source_ids) & realized_sources:
            issues.append(f"unrealised_foreshadow_opening:{foreshadow_id}")
        if lifecycle == "resolved":
            resolved = any(
                edge.kind == "foreshadow_resolved"
                and edge.target_node_id == foreshadow_id
                and set(edge.source_ids) & realized_sources
                for edge in graph.edges
            )
            if not resolved:
                issues.append(f"unrealised_foreshadow_resolution:{foreshadow_id}")
    return {
        "passed": not issues,
        "issues": issues,
        "state_update_count": len(patch.get("state_updates", [])),
        "foreshadow_update_count": len(foreshadow_updates),
    }


def commit_graph_patch(runtime: dict[str, Any], patch: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    if not bool(validation.get("passed")):
        raise ValueError("cannot commit a graph patch that failed validation")
    committed = json.loads(json.dumps(patch, ensure_ascii=False))
    committed["status"] = "committed"
    committed["commit_validation"] = validation
    existing = runtime.get("committed_patches", [])
    if any(isinstance(item, dict) and item.get("patch_id") == committed["patch_id"] for item in existing):
        return runtime
    state = {str(key): dict(value) for key, value in runtime.get("effective_state", {}).items() if isinstance(value, dict)}
    for update in committed.get("state_updates", []):
        if isinstance(update, dict):
            state[str(update["state_key"])] = update
    foreshadows = {
        str(key): dict(value)
        for key, value in runtime.get("effective_foreshadows", {}).items()
        if isinstance(value, dict)
    }
    for update in committed.get("foreshadow_updates", []):
        if isinstance(update, dict) and str(update.get("foreshadow_id", "")).strip():
            foreshadows[str(update["foreshadow_id"])] = update
    return {
        **runtime,
        "committed_patches": [*existing, committed],
        "effective_state": state,
        "effective_foreshadows": foreshadows,
    }


def write_runtime_graph(path: Path, runtime: dict[str, Any]) -> Path:
    write_json(path, runtime)
    return path
