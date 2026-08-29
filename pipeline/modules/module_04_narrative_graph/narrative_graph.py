"""Build the local evidence-first narrative fact graph.

This module never asks a model to infer extra setting.  It turns accepted
annotations plus deterministic cross-chapter identity resolution into the one
persisted graph that later generation stages can query for facts and state.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable

from .contracts import ChapterAnnotation, NarrativeGraph, NarrativeGraphEdge, NarrativeGraphNode, SourceSpan, WorkContinuity, validate_annotation
from .jsonio import read_json, read_jsonl, write_json
from .paths import corpus_dir


DEFAULT_GRAPH_LIMIT = 20
_FORESHADOW_OPEN_CUES = (
    "伏笔", "预兆", "暗示", "疑团", "谜团", "疑云", "未解", "日后",
    # These are unresolved-information signals rather than ordinary negative
    # sentiment.  They let the graph retain concrete threads such as a
    # mysterious object or a concealed motive without guessing a resolution.
    "神秘", "异常", "秘密", "隐瞒", "未知", "猜测", "推测", "疑惑",
)
_FORESHADOW_RESOLUTION_CUES = ("揭开", "揭示", "真相", "应验", "兑现", "解开", "解密")


@dataclass(frozen=True)
class NarrativeGraphQualityIssue:
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class NarrativeGraphQualityReport:
    author_id: str
    work_id: str
    chapter_count: int
    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    issues: tuple[NarrativeGraphQualityIssue, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not any(item.severity == "error" for item in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "author_id": self.author_id,
            "work_id": self.work_id,
            "chapter_count": self.chapter_count,
            "passed": self.passed,
            "node_counts": self.node_counts,
            "edge_counts": self.edge_counts,
            "issues": [item.to_dict() for item in self.issues],
        }


def _source_ref(chapter_id: str, kind: str, local_id: str) -> str:
    return f"{kind}:{chapter_id}:{local_id}"


def _node_ref(kind: str, chapter_id: str, local_id: str) -> str:
    return _source_ref(chapter_id, kind, local_id)


def _entity_node_ref(global_id: str) -> str:
    return f"entity:{global_id}"


def _dedupe_spans(spans: Iterable[SourceSpan]) -> tuple[SourceSpan, ...]:
    seen: set[tuple[str, str, int, int, str]] = set()
    result: list[SourceSpan] = []
    for span in spans:
        key = (span.chapter_id, span.unit_id, span.start, span.end, span.quote)
        if key not in seen:
            seen.add(key)
            result.append(span)
    return tuple(result)


def _event_fact_ids(event: object) -> set[str]:
    return {
        *getattr(event, "trigger_fact_ids", ()),
        *getattr(event, "precondition_fact_ids", ()),
        *getattr(event, "outcome_fact_ids", ()),
        *getattr(event, "cost_fact_ids", ()),
        *getattr(event, "basis_fact_ids", ()),
    }


def _has_cue(value: str, cues: tuple[str, ...]) -> bool:
    text = str(value or "")
    return any(cue in text for cue in cues)


def _event_position(chapter_index: int, order: int, evidence: tuple[SourceSpan, ...]) -> tuple[int, int, int]:
    return chapter_index, order, min((span.start for span in evidence), default=10**12)


def _evidence_overlaps(left: tuple[SourceSpan, ...], right: tuple[SourceSpan, ...]) -> bool:
    if not left or not right:
        return False
    return min(span.start for span in left) <= max(span.end for span in right) and min(span.start for span in right) <= max(span.end for span in left)


def _edge_id(kind: str, source: str, target: str, source_id: str, ordinal: int) -> str:
    return f"edge:{kind}:{ordinal:05d}:{source_id}:{source}->{target}"


def build_narrative_graph(author_id: str, work_id: str, annotations: list[ChapterAnnotation], continuity: WorkContinuity) -> NarrativeGraph:
    """Project validated annotations into nodes and fully traceable relations."""

    if not annotations:
        raise ValueError("cannot build a narrative graph without annotations")
    if tuple(item.chapter_id for item in annotations) != continuity.chapter_ids:
        raise ValueError("continuity chapters must exactly match graph annotations")

    local_to_global = {
        member_id: entity.global_entity_id
        for entity in continuity.global_entities
        for member_id in entity.member_entity_ids
    }
    nodes: list[NarrativeGraphNode] = []
    edges: list[NarrativeGraphEdge] = []
    ordinal = 0

    def add_edge(kind: str, source: str, target: str, source_ids: Iterable[str], evidence: Iterable[SourceSpan], *, chapter_id: str = "", chapter_index: int = -1, order: int = -1, attributes: dict[str, object] | None = None) -> None:
        nonlocal ordinal
        ordinal += 1
        source_ids_tuple = tuple(dict.fromkeys(source_ids))
        edges.append(NarrativeGraphEdge(
            _edge_id(kind, source, target, source_ids_tuple[0], ordinal), kind, source, target,
            source_ids_tuple, _dedupe_spans(evidence), chapter_id, chapter_index, order, attributes or {},
        ))

    local_entities = {entity.entity_id: entity for annotation in annotations for entity in annotation.entities}
    for global_entity in continuity.global_entities:
        member_entities = [local_entities[item] for item in global_entity.member_entity_ids]
        nodes.append(NarrativeGraphNode(
            _entity_node_ref(global_entity.global_entity_id), "entity", global_entity.canonical_name,
            tuple(_source_ref(entity.evidence[0].chapter_id, "entity", entity.entity_id) for entity in member_entities),
            _dedupe_spans(span for entity in member_entities for span in entity.evidence),
            attributes={
                "entity_kind": global_entity.kind,
                "aliases": list(global_entity.aliases),
                "global_entity_id": global_entity.global_entity_id,
            },
        ))

    event_nodes: dict[tuple[str, str], str] = {}
    fact_nodes: dict[tuple[str, str], str] = {}
    scene_nodes: dict[tuple[str, str], str] = {}
    anchor_nodes: dict[tuple[str, str], str] = {}
    fact_producers: dict[tuple[str, str], str] = {}
    event_records: list[dict[str, object]] = []
    foreshadow_records: list[dict[str, object]] = []

    def has_equivalent_foreshadow(
        participants: set[str],
        evidence: tuple[SourceSpan, ...],
        chapter_index: int,
    ) -> bool:
        """Avoid opening two lifecycle nodes for one annotated narrative cue.

        Extractors can represent the very same wording as both a supporting
        fact and the event that carries it.  Matching participant overlap plus
        overlapping evidence is strict enough to collapse only that duplicate,
        while preserving separate hints involving the same character later in
        the chapter.
        """

        return any(
            int(record["position"][0]) == chapter_index
            and bool(set(record["participants"]) & participants)
            and _evidence_overlaps(tuple(record["evidence"]), evidence)
            for record in foreshadow_records
        )

    for chapter_index, annotation in enumerate(annotations):
        facts = {item.fact_id: item for item in annotation.facts}
        events = {item.event_id: item for item in annotation.events}
        for fact in annotation.facts:
            node_id = _node_ref("fact", annotation.chapter_id, fact.fact_id)
            fact_nodes[(annotation.chapter_id, fact.fact_id)] = node_id
            nodes.append(NarrativeGraphNode(
                node_id, "fact", f"{fact.predicate}: {fact.value or fact.object_id}",
                (_source_ref(annotation.chapter_id, "fact", fact.fact_id),), fact.evidence,
                annotation.chapter_id, chapter_index, attributes={
                    "fact_kind": fact.kind, "predicate": fact.predicate, "value": fact.value,
                    "certainty": fact.certainty, "subject_local_id": fact.subject_id, "object_local_id": fact.object_id,
                },
            ))
            add_edge("asserts", _entity_node_ref(local_to_global[fact.subject_id]), node_id,
                     (_source_ref(annotation.chapter_id, "fact", fact.fact_id),), fact.evidence,
                     chapter_id=annotation.chapter_id, chapter_index=chapter_index)
            if fact.object_id:
                add_edge("fact_object", node_id, _entity_node_ref(local_to_global[fact.object_id]),
                         (_source_ref(annotation.chapter_id, "fact", fact.fact_id),), fact.evidence,
                         chapter_id=annotation.chapter_id, chapter_index=chapter_index)
                if fact.kind == "relationship":
                    add_edge("relationship", _entity_node_ref(local_to_global[fact.subject_id]), _entity_node_ref(local_to_global[fact.object_id]),
                             (_source_ref(annotation.chapter_id, "fact", fact.fact_id),), fact.evidence,
                             chapter_id=annotation.chapter_id, chapter_index=chapter_index,
                             attributes={"predicate": fact.predicate, "value": fact.value, "fact_id": fact.fact_id})
            if _has_cue(f"{fact.predicate} {fact.value}", _FORESHADOW_OPEN_CUES):
                foreshadow_id = _node_ref("foreshadow", annotation.chapter_id, f"fact-{fact.fact_id}")
                node_index = len(nodes)
                nodes.append(NarrativeGraphNode(
                    foreshadow_id, "foreshadow", "未解叙事线索",
                    (_source_ref(annotation.chapter_id, "fact", fact.fact_id),), fact.evidence,
                    annotation.chapter_id, chapter_index,
                    attributes={"lifecycle": "open", "opened_by": "fact", "fact_id": fact.fact_id},
                ))
                add_edge("foreshadow_open", node_id, foreshadow_id, (_source_ref(annotation.chapter_id, "fact", fact.fact_id),), fact.evidence,
                         chapter_id=annotation.chapter_id, chapter_index=chapter_index)
                foreshadow_records.append({
                    "node_id": foreshadow_id, "node_index": node_index,
                    "position": _event_position(chapter_index, -1, fact.evidence),
                    "participants": {_entity_node_ref(local_to_global[fact.subject_id]), *({_entity_node_ref(local_to_global[fact.object_id])} if fact.object_id else set())},
                    "evidence": fact.evidence,
                })
        for anchor in annotation.time_anchors:
            node_id = _node_ref("time_anchor", annotation.chapter_id, anchor.anchor_id)
            anchor_nodes[(annotation.chapter_id, anchor.anchor_id)] = node_id
            nodes.append(NarrativeGraphNode(
                node_id, "time_anchor", anchor.label, (_source_ref(annotation.chapter_id, "time_anchor", anchor.anchor_id),),
                anchor.evidence, annotation.chapter_id, chapter_index,
                attributes={"anchor_kind": anchor.kind},
            ))
        for event in sorted(annotation.events, key=lambda item: (item.order, item.event_id)):
            node_id = _node_ref("event", annotation.chapter_id, event.event_id)
            event_nodes[(annotation.chapter_id, event.event_id)] = node_id
            nodes.append(NarrativeGraphNode(
                node_id, "event", event.summary, (_source_ref(annotation.chapter_id, "event", event.event_id),),
                event.evidence, annotation.chapter_id, chapter_index, event.order,
                attributes={
                    "action": event.action, "obstacle": event.obstacle, "decision": event.decision,
                    "action_type": event.action_type,
                    "actor_entity_id": local_to_global.get(event.actor_id, ""),
                    "target_entity_ids": [local_to_global[item] for item in event.target_ids],
                    "basis_fact_ids": list(event.basis_fact_ids),
                },
            ))
            event_ref = _source_ref(annotation.chapter_id, "event", event.event_id)
            for participant_id in event.participant_ids:
                add_edge("event_participant", _entity_node_ref(local_to_global[participant_id]), node_id, (event_ref,), event.evidence,
                         chapter_id=annotation.chapter_id, chapter_index=chapter_index, order=event.order)
            for edge_kind, ids in (
                ("event_trigger", event.trigger_fact_ids), ("event_requires", event.precondition_fact_ids),
                ("event_produces", event.outcome_fact_ids), ("event_cost", event.cost_fact_ids),
            ):
                for fact_id in ids:
                    add_edge(edge_kind, node_id, fact_nodes[(annotation.chapter_id, fact_id)], (event_ref,), event.evidence,
                             chapter_id=annotation.chapter_id, chapter_index=chapter_index, order=event.order)
            for fact_id in (*event.trigger_fact_ids, *event.precondition_fact_ids):
                producer = fact_producers.get((annotation.chapter_id, fact_id))
                if producer and producer != node_id:
                    add_edge("causes", producer, node_id, (event_ref, _source_ref(annotation.chapter_id, "fact", fact_id)), event.evidence,
                             chapter_id=annotation.chapter_id, chapter_index=chapter_index, order=event.order,
                             attributes={"via_fact_id": fact_id})
            for fact_id in event.outcome_fact_ids:
                fact_producers[(annotation.chapter_id, fact_id)] = node_id
            event_participants = {_entity_node_ref(local_to_global[item]) for item in event.participant_ids}
            event_records.append({
                "node_id": node_id, "event_id": event.event_id, "chapter_id": annotation.chapter_id,
                "position": _event_position(chapter_index, event.order, event.evidence),
                "participants": event_participants, "text": f"{event.summary} {event.action}", "evidence": event.evidence,
            })
            if _has_cue(f"{event.summary} {event.action}", _FORESHADOW_OPEN_CUES) and not has_equivalent_foreshadow(event_participants, event.evidence, chapter_index):
                foreshadow_id = _node_ref("foreshadow", annotation.chapter_id, f"event-{event.event_id}")
                node_index = len(nodes)
                nodes.append(NarrativeGraphNode(
                    foreshadow_id, "foreshadow", "未解叙事线索",
                    (event_ref,), event.evidence, annotation.chapter_id, chapter_index, event.order,
                    attributes={"lifecycle": "open", "opened_by": "event", "event_id": event.event_id},
                ))
                add_edge("foreshadow_open", node_id, foreshadow_id, (event_ref,), event.evidence,
                         chapter_id=annotation.chapter_id, chapter_index=chapter_index, order=event.order)
                foreshadow_records.append({
                    "node_id": foreshadow_id, "node_index": node_index,
                    "position": _event_position(chapter_index, event.order, event.evidence),
                    "participants": event_participants,
                    "evidence": event.evidence,
                })
        for scene in sorted(annotation.scenes, key=lambda item: (item.order, item.scene_id)):
            node_id = _node_ref("scene", annotation.chapter_id, scene.scene_id)
            scene_nodes[(annotation.chapter_id, scene.scene_id)] = node_id
            nodes.append(NarrativeGraphNode(
                node_id, "scene", scene.objective, (_source_ref(annotation.chapter_id, "scene", scene.scene_id),),
                scene.evidence, annotation.chapter_id, chapter_index, scene.order,
                attributes={"tension": scene.tension},
            ))
            scene_ref = _source_ref(annotation.chapter_id, "scene", scene.scene_id)
            for event_id in scene.event_ids:
                add_edge("scene_contains", node_id, event_nodes[(annotation.chapter_id, event_id)], (scene_ref,), scene.evidence,
                         chapter_id=annotation.chapter_id, chapter_index=chapter_index, order=scene.order)
            for participant_id in scene.participant_ids:
                add_edge("scene_participant", _entity_node_ref(local_to_global[participant_id]), node_id, (scene_ref,), scene.evidence,
                         chapter_id=annotation.chapter_id, chapter_index=chapter_index, order=scene.order)
            for fact_id in scene.entry_fact_ids:
                add_edge("scene_entry_state", node_id, fact_nodes[(annotation.chapter_id, fact_id)], (scene_ref,), scene.evidence,
                         chapter_id=annotation.chapter_id, chapter_index=chapter_index, order=scene.order)
            for fact_id in scene.exit_fact_ids:
                add_edge("scene_exit_state", node_id, fact_nodes[(annotation.chapter_id, fact_id)], (scene_ref,), scene.evidence,
                         chapter_id=annotation.chapter_id, chapter_index=chapter_index, order=scene.order)
            for location_id in scene.location_ids:
                add_edge("scene_at", node_id, _entity_node_ref(local_to_global[location_id]), (scene_ref,), scene.evidence,
                         chapter_id=annotation.chapter_id, chapter_index=chapter_index, order=scene.order)
            for anchor_id in scene.time_anchor_ids:
                add_edge("scene_time", node_id, anchor_nodes[(annotation.chapter_id, anchor_id)], (scene_ref,), scene.evidence,
                         chapter_id=annotation.chapter_id, chapter_index=chapter_index, order=scene.order)
        for relation in annotation.temporal_relations:
            edge_kind = "temporal_before" if relation.kind == "before" else "temporal_simultaneous"
            add_edge(edge_kind, event_nodes[(annotation.chapter_id, relation.before_event_id)], event_nodes[(annotation.chapter_id, relation.after_event_id)],
                     (_source_ref(annotation.chapter_id, "temporal_relation", relation.relation_id),), relation.evidence,
                     chapter_id=annotation.chapter_id, chapter_index=chapter_index,
                     attributes={"relation_kind": relation.kind, "basis": relation.basis, "anchor_id": relation.anchor_id})
        for relation in annotation.spatial_relations:
            add_edge("spatial_relation", _entity_node_ref(local_to_global[relation.subject_id]), _entity_node_ref(local_to_global[relation.location_id]),
                     (_source_ref(annotation.chapter_id, "spatial_relation", relation.relation_id),), relation.evidence,
                     chapter_id=annotation.chapter_id, chapter_index=chapter_index,
                     attributes={"relation_kind": relation.kind, "fact_id": relation.fact_id})
        for change in annotation.state_changes:
            event = events[change.event_id]
            source = event_nodes[(annotation.chapter_id, change.event_id)]
            target = fact_nodes[(annotation.chapter_id, change.after_fact_id or change.before_fact_id)]
            add_edge("state_change", source, target, (_source_ref(annotation.chapter_id, "state_change", change.change_id),), event.evidence,
                     chapter_id=annotation.chapter_id, chapter_index=chapter_index, order=event.order,
                     attributes={"operation": change.operation, "before_fact_id": change.before_fact_id, "after_fact_id": change.after_fact_id})

    # A thread is only marked resolved when a later, explicitly resolving event
    # touches one of the same participants.  This deliberately leaves most
    # hints open rather than guessing that a similar later event paid them off.
    for record in foreshadow_records:
        for event in event_records:
            if event["position"] <= record["position"]:
                continue
            if not set(event["participants"]) & set(record["participants"]):
                continue
            if not _has_cue(str(event["text"]), _FORESHADOW_RESOLUTION_CUES):
                continue
            node_index = int(record["node_index"])
            current = nodes[node_index]
            nodes[node_index] = replace(current, attributes={
                **current.attributes,
                "lifecycle": "resolved",
                "resolved_by_event_id": str(event["event_id"]),
                "resolved_in_chapter": str(event["chapter_id"]),
            })
            add_edge("foreshadow_resolved", str(event["node_id"]), str(record["node_id"]),
                     (f"foreshadow_resolution:{record['node_id']}", _source_ref(str(event["chapter_id"]), "event", str(event["event_id"]))),
                     event["evidence"], chapter_id=str(event["chapter_id"]),
                     chapter_index=int(event["position"][0]), order=int(event["position"][1]))
            break

    for previous, current in zip(continuity.timeline_events, continuity.timeline_events[1:]):
        add_edge("timeline_precedes", event_nodes[(previous.chapter_id, previous.event_id)], event_nodes[(current.chapter_id, current.event_id)],
                 (f"timeline:{previous.timeline_event_id}", f"timeline:{current.timeline_event_id}"), current.evidence,
                 chapter_id=current.chapter_id, chapter_index=current.chapter_index, order=current.event_order)
    for entry in continuity.state_ledger:
        add_edge("state_observed", event_nodes[(entry.chapter_id, entry.event_id)], fact_nodes[(entry.chapter_id, entry.fact_id)],
                 (f"state_ledger:{entry.entry_id}",), entry.evidence, chapter_id=entry.chapter_id,
                 chapter_index=entry.chapter_index, attributes={"operation": entry.operation, "slot": entry.slot})
    for transition in continuity.location_transitions:
        target_id = transition.to_location_global_id or transition.from_location_global_id
        add_edge("location_transition", _entity_node_ref(transition.global_subject_id), _entity_node_ref(target_id),
                 (f"location_transition:{transition.transition_id}",), transition.evidence,
                 chapter_id=transition.chapter_id, chapter_index=transition.chapter_index,
                 attributes={"event_id": transition.event_id, "relation_kind": transition.relation_kind,
                             "from_location_global_id": transition.from_location_global_id,
                             "to_location_global_id": transition.to_location_global_id})

    return NarrativeGraph(
        author_id, work_id, tuple(item.chapter_id for item in annotations),
        {item.chapter_id: item.source_hash for item in annotations}, tuple(nodes), tuple(edges),
    )


def _temporal_cycle_exists(graph: NarrativeGraph) -> bool:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.kind in {"temporal_before", "timeline_precedes"}:
            adjacency[edge.source_node_id].append(edge.target_node_id)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        cycle = any(visit(target) for target in adjacency[node_id])
        visiting.remove(node_id)
        visited.add(node_id)
        return cycle

    return any(visit(node_id) for node_id in tuple(adjacency))


def _cross_chapter_state_issues(continuity: WorkContinuity) -> list[NarrativeGraphQualityIssue]:
    """Reject state replacement that is neither explicit nor chronologically valid."""

    active: dict[tuple[str, str], object] = {}
    issues: list[NarrativeGraphQualityIssue] = []
    timeline_order = {
        (item.chapter_id, item.event_id): (item.chapter_index, item.event_order)
        for item in continuity.timeline_events
    }
    ordered = sorted(
        continuity.state_ledger,
        key=lambda item: (*timeline_order.get((item.chapter_id, item.event_id), (item.chapter_index, 10**9)), item.entry_id),
    )
    for entry in ordered:
        key = (entry.global_subject_id, entry.slot)
        previous = active.get(key)
        if entry.operation == "remove":
            if previous is None:
                issues.append(NarrativeGraphQualityIssue(
                    "state_remove_without_active_value",
                    f"state entry {entry.entry_id} removes a value that is not active",
                ))
            active.pop(key, None)
            continue
        if entry.operation == "update" and previous is None:
            issues.append(NarrativeGraphQualityIssue(
                "state_update_without_active_value",
                f"state entry {entry.entry_id} updates a value that has no prior state",
            ))
        if entry.operation == "add" and previous is not None and getattr(previous, "value", "") != entry.value:
            issues.append(NarrativeGraphQualityIssue(
                "state_add_overwrites_active_value",
                f"state entry {entry.entry_id} overwrites an active state without an update operation",
            ))
        if entry.operation == "observe" and previous is not None and getattr(previous, "value", "") != entry.value:
            issues.append(NarrativeGraphQualityIssue(
                "state_change_missing_operation",
                f"state entry {entry.entry_id} changes an active state but is only marked observe",
            ))
        active[key] = entry
    return issues


def assess_narrative_graph(graph: NarrativeGraph, annotations: list[ChapterAnnotation], continuity: WorkContinuity) -> NarrativeGraphQualityReport:
    issues: list[NarrativeGraphQualityIssue] = []
    try:
        graph.validate()
    except (TypeError, ValueError) as exc:
        issues.append(NarrativeGraphQualityIssue("invalid_contract", str(exc)))
    if graph.chapter_ids != tuple(item.chapter_id for item in annotations):
        issues.append(NarrativeGraphQualityIssue("chapter_order_mismatch", "graph chapters differ from accepted annotations"))
    by_kind = Counter(node.kind for node in graph.nodes)
    expected = {
        "entity": len(continuity.global_entities),
        "fact": sum(len(item.facts) for item in annotations),
        "event": sum(len(item.events) for item in annotations),
        "scene": sum(len(item.scenes) for item in annotations),
        "time_anchor": sum(len(item.time_anchors) for item in annotations),
    }
    for kind, count in expected.items():
        if by_kind[kind] != count:
            issues.append(NarrativeGraphQualityIssue("node_coverage", f"{kind} nodes: expected {count}, found {by_kind[kind]}"))
    edge_counts = Counter(edge.kind for edge in graph.edges)
    if edge_counts["event_participant"] != sum(len(item.participant_ids) for annotation in annotations for item in annotation.events):
        issues.append(NarrativeGraphQualityIssue("event_participant_coverage", "every event participant must be represented in the graph"))
    if edge_counts["scene_contains"] != sum(len(item.event_ids) for annotation in annotations for item in annotation.scenes):
        issues.append(NarrativeGraphQualityIssue("scene_event_coverage", "every scene-event link must be represented in the graph"))
    if edge_counts["state_change"] != sum(len(item.state_changes) for item in annotations):
        issues.append(NarrativeGraphQualityIssue("state_change_coverage", "every state change must be represented in the graph"))
    if edge_counts["spatial_relation"] != sum(len(item.spatial_relations) for item in annotations):
        issues.append(NarrativeGraphQualityIssue("spatial_relation_coverage", "every spatial relation must be represented in the graph"))
    if edge_counts["temporal_before"] + edge_counts["temporal_simultaneous"] != sum(len(item.temporal_relations) for item in annotations):
        issues.append(NarrativeGraphQualityIssue("temporal_relation_coverage", "every temporal relation must be represented in the graph"))
    if edge_counts["location_transition"] != len(continuity.location_transitions):
        issues.append(NarrativeGraphQualityIssue("location_transition_coverage", "every accepted location transition must be represented in the graph"))
    if edge_counts["state_observed"] != len(continuity.state_ledger):
        issues.append(NarrativeGraphQualityIssue("state_ledger_coverage", "every accepted state ledger entry must be represented in the graph"))
    expected_relationships = sum(1 for annotation in annotations for fact in annotation.facts if fact.kind == "relationship" and fact.object_id)
    if edge_counts["relationship"] != expected_relationships:
        issues.append(NarrativeGraphQualityIssue("relationship_coverage", "every object-backed relationship fact must become a typed relationship edge"))
    foreshadows = [node for node in graph.nodes if node.kind == "foreshadow"]
    opened_ids = {edge.target_node_id for edge in graph.edges if edge.kind == "foreshadow_open"}
    resolved_ids = {edge.target_node_id for edge in graph.edges if edge.kind == "foreshadow_resolved"}
    for node in foreshadows:
        lifecycle = str(node.attributes.get("lifecycle", ""))
        if node.node_id not in opened_ids:
            issues.append(NarrativeGraphQualityIssue("foreshadow_opening_missing", f"foreshadow {node.node_id} has no opening edge"))
        if lifecycle == "resolved" and node.node_id not in resolved_ids:
            issues.append(NarrativeGraphQualityIssue("foreshadow_resolution_missing", f"foreshadow {node.node_id} is resolved without a resolving event"))
        if lifecycle == "open" and node.node_id in resolved_ids:
            issues.append(NarrativeGraphQualityIssue("foreshadow_lifecycle_mismatch", f"foreshadow {node.node_id} has a resolution edge but remains open"))
    issues.extend(_cross_chapter_state_issues(continuity))
    if _temporal_cycle_exists(graph):
        issues.append(NarrativeGraphQualityIssue("temporal_cycle", "temporal graph contains a directed cycle"))
    return NarrativeGraphQualityReport(graph.author_id, graph.work_id, len(graph.chapter_ids), dict(sorted(by_kind.items())), dict(sorted(edge_counts.items())), tuple(issues))


def _annotation_path(folder: Path, limit: int) -> Path:
    exact = folder / f"chapter_annotations.sample-{limit}.jsonl"
    if exact.exists():
        return exact
    candidates = sorted(folder.glob("chapter_annotations.sample-*.jsonl"))
    if not candidates:
        raise FileNotFoundError("no evidence-first chapter annotation file found")
    return candidates[-1]


def _load_documents(author_id: str, work_id: str):
    return {
        row["chapter_id"]: row
        for row in read_jsonl(corpus_dir(author_id, work_id) / "chapter_documents.jsonl")
        if str(row.get("chapter_id", "")).strip()
    }


def load_narrative_graph(author_id: str, work_id: str) -> NarrativeGraph:
    path = corpus_dir(author_id, work_id) / "narrative_graph.json"
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise FileNotFoundError("narrative_graph.json is missing; run the graph stage after continuity")
    graph = NarrativeGraph.from_dict(payload)
    graph.validate()
    return graph


def build_narrative_graph_work(author_id: str, work_id: str, *, limit: int = DEFAULT_GRAPH_LIMIT) -> tuple[Path, Path]:
    """Persist ``narrative_graph.json`` only after full source coverage passes."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    folder = corpus_dir(author_id, work_id)
    documents = _load_documents(author_id, work_id)
    annotations = [ChapterAnnotation.from_dict(row) for row in read_jsonl(_annotation_path(folder, limit))][:limit]
    if not annotations:
        raise FileNotFoundError("no chapter annotations found")
    for annotation in annotations:
        document_payload = documents.get(annotation.chapter_id)
        if not isinstance(document_payload, dict):
            raise ValueError(f"missing source document for {annotation.chapter_id}")
        from .contracts import ChapterDocument
        report = validate_annotation(annotation, ChapterDocument.from_dict(document_payload))
        if not report.passed:
            raise ValueError(f"annotation is invalid for graph: {annotation.chapter_id}")
    continuity_path = folder / f"work_continuity.sample-{len(annotations)}.json"
    continuity_payload = read_json(continuity_path)
    if not isinstance(continuity_payload, dict):
        raise FileNotFoundError("matching continuity graph is missing; run continuity before graph")
    continuity = WorkContinuity.from_dict(continuity_payload)
    continuity.validate()
    graph = build_narrative_graph(author_id, work_id, annotations, continuity)
    quality = assess_narrative_graph(graph, annotations, continuity)
    if not quality.passed:
        messages = "; ".join(f"{item.code}: {item.message}" for item in quality.issues)
        raise ValueError(f"narrative graph quality checks failed: {messages}")
    target = folder / "narrative_graph.json"
    quality_target = folder / "narrative_graph.quality.json"
    write_json(target, graph.to_dict())
    write_json(quality_target, quality.to_dict())
    return target, quality_target
