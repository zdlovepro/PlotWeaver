"""Stage 4: deterministic cross-chapter continuity fusion.

This stage consumes only the evidence-first chapter annotations.  It does not
call a model and it deliberately resolves identities only through exact,
high-confidence names or aliases already present in those annotations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import re
from pathlib import Path
from typing import Iterable

from .contracts import (
    ChapterAnnotation,
    ChapterDocument,
    ContinuityIssue,
    GlobalEntity,
    LocationTransition,
    SourceSpan,
    StateLedgerEntry,
    TimelineEvent,
    WorkContinuity,
    validate_annotation,
)
from .jsonio import read_jsonl, write_json
from .paths import corpus_dir


DEFAULT_CONTINUITY_LIMIT = 20
STATE_BEARING_FACT_KINDS = frozenset({
    "identity", "goal", "relationship", "location", "resource", "knowledge", "rule", "progression", "information",
})

# Role labels are meaningful within a chapter but are not stable identities
# across a work.  Excluding them makes the resolver conservative by default.
_GENERIC_PERSON_LABELS = frozenset({
    "父亲", "母亲", "爷爷", "奶奶", "外公", "外婆", "师父", "师兄", "师姐", "师弟", "师妹",
    "掌门", "长老", "族长", "家主", "村长", "少年", "青年", "中年人", "老者", "老人", "弟子",
})
_PARENTHETICAL = re.compile(r"[（(]([^（）()]{1,32})[）)]")


@dataclass(frozen=True)
class ContinuityQualityIssue:
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuityQualityReport:
    author_id: str
    work_id: str
    chapter_count: int
    issues: tuple[ContinuityQualityIssue, ...] = field(default_factory=tuple)
    global_entity_count: int = 0
    timeline_event_count: int = 0
    state_entry_count: int = 0
    location_transition_count: int = 0
    unlinked_spatial_relation_count: int = 0

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "author_id": self.author_id,
            "work_id": self.work_id,
            "chapter_count": self.chapter_count,
            "passed": self.passed,
            "global_entity_count": self.global_entity_count,
            "timeline_event_count": self.timeline_event_count,
            "state_entry_count": self.state_entry_count,
            "location_transition_count": self.location_transition_count,
            "unlinked_spatial_relation_count": self.unlinked_spatial_relation_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _normalise_name(value: str) -> str:
    return re.sub(r"[\s·•.。]", "", str(value or "")).casefold()


def _name_variants(name: str, aliases: Iterable[str]) -> set[str]:
    raw = [str(name or ""), *(str(alias or "") for alias in aliases)]
    result: set[str] = set()
    for value in raw:
        normalized = _normalise_name(value)
        if normalized:
            result.add(normalized)
        base = _normalise_name(_PARENTHETICAL.sub("", value))
        if base:
            result.add(base)
        for inner in _PARENTHETICAL.findall(value):
            normalized_inner = _normalise_name(inner)
            if normalized_inner:
                result.add(normalized_inner)
    return result


def _is_safe_variant(kind: str, variant: str) -> bool:
    if len(variant) < 2:
        return False
    if kind == "person" and variant in _GENERIC_PERSON_LABELS:
        return False
    return True


def _dedupe_spans(spans: Iterable[SourceSpan]) -> tuple[SourceSpan, ...]:
    seen: set[tuple[int, int, str, str]] = set()
    output: list[SourceSpan] = []
    for span in spans:
        key = (span.start, span.end, span.quote, span.unit_id)
        if key not in seen:
            seen.add(key)
            output.append(span)
    return tuple(output)


def _first_evidence_start(item: object) -> int:
    return min((span.start for span in getattr(item, "evidence", ())), default=10**12)


def _event_fact_ids(event: object) -> set[str]:
    return {
        *getattr(event, "trigger_fact_ids", ()),
        *getattr(event, "precondition_fact_ids", ()),
        *getattr(event, "outcome_fact_ids", ()),
        *getattr(event, "cost_fact_ids", ()),
    }


def _evidence_overlaps(left: tuple[SourceSpan, ...], right: tuple[SourceSpan, ...]) -> bool:
    if not left or not right:
        return False
    left_start, left_end = min(span.start for span in left), max(span.end for span in left)
    right_start, right_end = min(span.start for span in right), max(span.end for span in right)
    return left_start <= right_end and right_start <= left_end


def _build_global_entities(annotations: list[ChapterAnnotation]) -> tuple[tuple[GlobalEntity, ...], dict[str, str], list[ContinuityIssue]]:
    ordered_entities = [entity for annotation in annotations for entity in annotation.entities]
    entity_by_id = {entity.entity_id: entity for entity in ordered_entities}
    union_find = _UnionFind(entity_by_id)
    by_variant: dict[tuple[str, str], list[str]] = defaultdict(list)
    skipped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entity in ordered_entities:
        for variant in _name_variants(entity.canonical_name, entity.aliases):
            key = (entity.kind, variant)
            if _is_safe_variant(*key):
                by_variant[key].append(entity.entity_id)
            else:
                skipped[key].add(entity.entity_id)
    for entity_ids in by_variant.values():
        for entity_id in entity_ids[1:]:
            union_find.union(entity_ids[0], entity_id)

    groups: dict[str, list[str]] = defaultdict(list)
    for entity in ordered_entities:
        groups[union_find.find(entity.entity_id)].append(entity.entity_id)
    chapter_position = {annotation.chapter_id: index for index, annotation in enumerate(annotations)}
    ordered_groups = sorted(
        groups.values(),
        key=lambda ids: min((chapter_position[entity_by_id[item].evidence[0].chapter_id], item) for item in ids),
    )
    global_entities: list[GlobalEntity] = []
    local_to_global: dict[str, str] = {}
    for index, member_ids in enumerate(ordered_groups, start=1):
        members = [entity_by_id[item] for item in member_ids]
        kind = members[0].kind
        # A parenthetical source form often carries an alias rather than the
        # preferred cross-chapter display name.  Prefer an exact non-parenthetic
        # canonical form when one is available; no semantic guess is involved.
        plain_names = [entity.canonical_name for entity in members if not _PARENTHETICAL.search(entity.canonical_name)]
        name_counts = Counter(plain_names or [entity.canonical_name for entity in members])
        canonical_name = sorted(name_counts, key=lambda name: (-name_counts[name], -len(name), name))[0]
        aliases = tuple(sorted({alias for entity in members for alias in (entity.canonical_name, *entity.aliases) if alias != canonical_name}))
        global_id = f"global:{kind}:{index:04d}"
        global_entities.append(GlobalEntity(global_id, kind, canonical_name, tuple(member_ids), aliases))
        local_to_global.update({member_id: global_id for member_id in member_ids})

    issues: list[ContinuityIssue] = []
    for (kind, label), members in sorted(skipped.items()):
        if len(members) > 1:
            issues.append(ContinuityIssue(
                "generic_alias_not_merged", "info",
                f"通用{kind}称谓“{label}”未跨章合并，以避免错误认同",
                related_ids=tuple(sorted(members)),
            ))
    return tuple(global_entities), local_to_global, issues


def _anchor_ids_for_event(annotation: ChapterAnnotation, event_id: str, evidence: tuple[SourceSpan, ...]) -> tuple[str, ...]:
    scene_ids = {scene.scene_id for scene in annotation.scenes if event_id in scene.event_ids}
    from_scenes = [anchor_id for scene in annotation.scenes if scene.scene_id in scene_ids for anchor_id in scene.time_anchor_ids]
    if not evidence:
        return tuple(dict.fromkeys(from_scenes))
    start, end = min(span.start for span in evidence), max(span.end for span in evidence)
    from_events = [
        anchor.anchor_id for anchor in annotation.time_anchors
        if min(span.start for span in anchor.evidence) <= end and start <= max(span.end for span in anchor.evidence)
    ]
    return tuple(dict.fromkeys((*from_scenes, *from_events)))


def _event_spatial_relations(annotation: ChapterAnnotation, event: object) -> tuple[str, ...]:
    fact_ids = _event_fact_ids(event)
    return tuple(
        relation.relation_id for relation in annotation.spatial_relations
        if relation.fact_id in fact_ids or _evidence_overlaps(relation.evidence, event.evidence)
    )


def _timeline(
    annotations: list[ChapterAnnotation],
    local_to_global: dict[str, str],
) -> tuple[tuple[TimelineEvent, ...], dict[tuple[str, str], TimelineEvent]]:
    timeline: list[TimelineEvent] = []
    indexed: dict[tuple[str, str], TimelineEvent] = {}
    for chapter_index, annotation in enumerate(annotations):
        for event in sorted(annotation.events, key=lambda item: (item.order, _first_evidence_start(item), item.event_id)):
            timeline_event = TimelineEvent(
                f"{annotation.chapter_id}:timeline-{len(timeline) + 1:04d}",
                annotation.chapter_id,
                event.event_id,
                chapter_index,
                event.order,
                tuple(dict.fromkeys(local_to_global[item] for item in event.participant_ids)),
                _anchor_ids_for_event(annotation, event.event_id, event.evidence),
                _event_spatial_relations(annotation, event),
                event.evidence,
            )
            timeline.append(timeline_event)
            indexed[(annotation.chapter_id, event.event_id)] = timeline_event
    return tuple(timeline), indexed


def _state_slot(kind: str, predicate: str, value: str, object_global_id: str) -> str:
    """Choose a semantic state dimension without conflating accumulated facts.

    A location is one current position regardless of whether the source says
    “arrived at” or “was in”.  Knowledge and ordinary progression facts are
    additive unless they name the same object or a recognised level dimension;
    otherwise two independent discoveries would look like an impossible state
    replacement merely because they share a verb such as “发现”.
    """

    if kind == "location":
        return "location:current"
    if kind in {"identity", "goal"}:
        return f"{kind}:{predicate}"
    if kind == "progression":
        level_cues = ("境界", "修为", "修炼", "突破", "层", "阶", "级")
        if any(cue in f"{predicate} {value}" for cue in level_cues):
            return "progression:level"
    return f"{kind}:{predicate}:{object_global_id or value}"


def _state_ledger(
    annotations: list[ChapterAnnotation],
    local_to_global: dict[str, str],
) -> tuple[tuple[StateLedgerEntry, ...], list[ContinuityIssue]]:
    entries: list[StateLedgerEntry] = []
    issues: list[ContinuityIssue] = []
    active: dict[tuple[str, str], StateLedgerEntry] = {}
    for chapter_index, annotation in enumerate(annotations):
        facts = {fact.fact_id: fact for fact in annotation.facts}
        changes_by_fact: dict[str, tuple[str, str]] = {}
        for change in annotation.state_changes:
            if change.after_fact_id:
                changes_by_fact[change.after_fact_id] = (change.operation, change.event_id)
            if change.before_fact_id:
                changes_by_fact[change.before_fact_id] = (change.operation, change.event_id)
        for event in sorted(annotation.events, key=lambda item: (item.order, _first_evidence_start(item), item.event_id)):
            for fact_id in sorted(_event_fact_ids(event)):
                fact = facts[fact_id]
                if fact.kind not in STATE_BEARING_FACT_KINDS:
                    continue
                operation, changed_event_id = changes_by_fact.get(fact_id, ("observe", event.event_id))
                if changed_event_id != event.event_id:
                    continue
                global_subject = local_to_global[fact.subject_id]
                global_object = local_to_global.get(fact.object_id, "")
                slot = _state_slot(fact.kind, fact.predicate, fact.value, global_object)
                value = fact.value or global_object
                previous = active.get((global_subject, slot))
                raw_operation = operation
                entry_id = f"{annotation.chapter_id}:state-{len(entries) + 1:04d}"
                if operation != "remove" and previous is None and operation == "update":
                    operation = "add"
                elif operation != "remove" and previous is not None:
                    if previous.value == value and operation in {"add", "update"}:
                        operation = "observe"
                    elif previous.value != value and operation in {"add", "observe"}:
                        operation = "update"
                if raw_operation != operation:
                    issues.append(ContinuityIssue(
                        "state_operation_normalized", "warning",
                        f"状态槽位 {slot} 的操作由 {raw_operation} 规范为 {operation}",
                        annotation.chapter_id,
                        (previous.entry_id,) if previous is not None else (),
                    ))
                entry = StateLedgerEntry(
                    entry_id,
                    annotation.chapter_id,
                    event.event_id,
                    chapter_index,
                    global_subject,
                    fact.fact_id,
                    operation,
                    slot,
                    value,
                    global_object,
                    fact.evidence,
                )
                if operation == "remove":
                    active.pop((global_subject, slot), None)
                else:
                    active[(global_subject, slot)] = entry
                entries.append(entry)
    return tuple(entries), issues


def _location_transitions(
    annotations: list[ChapterAnnotation],
    local_to_global: dict[str, str],
) -> tuple[tuple[LocationTransition, ...], tuple[str, ...], list[ContinuityIssue]]:
    transitions: list[LocationTransition] = []
    unlinked_ids: list[str] = []
    issues: list[ContinuityIssue] = []
    current_locations: dict[str, str] = {}
    for chapter_index, annotation in enumerate(annotations):
        facts_to_events: dict[str, list[object]] = defaultdict(list)
        for event in annotation.events:
            for fact_id in _event_fact_ids(event):
                facts_to_events[fact_id].append(event)
        for relation in annotation.spatial_relations:
            candidates = sorted(facts_to_events.get(relation.fact_id, []), key=lambda item: (item.order, _first_evidence_start(item)))
            if not candidates:
                candidates = sorted(
                    (event for event in annotation.events if _evidence_overlaps(relation.evidence, event.evidence)),
                    key=lambda item: (item.order, _first_evidence_start(item)),
                )
            if not candidates:
                unlinked_ids.append(relation.relation_id)
                issues.append(ContinuityIssue(
                    "unlinked_spatial_relation", "warning",
                    "空间关系未被任何事件引用，无法放入跨章时间线",
                    annotation.chapter_id,
                    (relation.relation_id,),
                ))
                continue
            event = candidates[0]
            subject = local_to_global[relation.subject_id]
            location = local_to_global[relation.location_id]
            previous = current_locations.get(subject, "")
            if relation.kind == "leaves":
                transition = LocationTransition(
                    f"{annotation.chapter_id}:move-{len(transitions) + 1:04d}", annotation.chapter_id, event.event_id,
                    chapter_index, subject, relation.relation_id, relation.kind, location, "", relation.evidence,
                )
                if previous == location:
                    current_locations.pop(subject, None)
            else:
                transition = LocationTransition(
                    f"{annotation.chapter_id}:move-{len(transitions) + 1:04d}", annotation.chapter_id, event.event_id,
                    chapter_index, subject, relation.relation_id, relation.kind, previous, location, relation.evidence,
                )
                if relation.kind == "at" and previous and previous != location:
                    issues.append(ContinuityIssue(
                        "location_jump_without_movement", "warning",
                        "角色位置改变仅被标为静态位置，后续应复核是否遗漏移动事件",
                        annotation.chapter_id,
                        (transition.transition_id,),
                    ))
                current_locations[subject] = location
            transitions.append(transition)
    return tuple(transitions), tuple(unlinked_ids), issues


def build_work_continuity(author_id: str, work_id: str, annotations: list[ChapterAnnotation]) -> WorkContinuity:
    if not annotations:
        raise ValueError("cannot build continuity without annotations")
    global_entities, local_to_global, issues = _build_global_entities(annotations)
    timeline_events, _ = _timeline(annotations, local_to_global)
    state_ledger, state_issues = _state_ledger(annotations, local_to_global)
    transitions, unlinked_spatial_ids, location_issues = _location_transitions(annotations, local_to_global)
    return WorkContinuity(
        author_id,
        work_id,
        tuple(annotation.chapter_id for annotation in annotations),
        "2.1",
        global_entities,
        timeline_events,
        state_ledger,
        transitions,
        unlinked_spatial_ids,
        tuple((*issues, *state_issues, *location_issues)),
    )


def assess_continuity(continuity: WorkContinuity, annotations: list[ChapterAnnotation]) -> ContinuityQualityReport:
    issues: list[ContinuityQualityIssue] = []
    try:
        continuity.validate()
    except (TypeError, ValueError) as exc:
        issues.append(ContinuityQualityIssue("invalid_contract", str(exc)))
    chapter_ids = tuple(annotation.chapter_id for annotation in annotations)
    if continuity.chapter_ids != chapter_ids:
        issues.append(ContinuityQualityIssue("chapter_order_mismatch", "continuity chapter order differs from source annotations"))
    entity_ids = {entity.entity_id for annotation in annotations for entity in annotation.entities}
    global_ids = {entity.global_entity_id for entity in continuity.global_entities}
    event_keys = {(annotation.chapter_id, event.event_id) for annotation in annotations for event in annotation.events}
    timeline_keys = {(item.chapter_id, item.event_id) for item in continuity.timeline_events}
    if timeline_keys != event_keys:
        issues.append(ContinuityQualityIssue("timeline_coverage", "not every chapter event appears exactly once in the work timeline"))
    member_ids = [member for entity in continuity.global_entities for member in entity.member_entity_ids]
    if set(member_ids) != entity_ids or len(member_ids) != len(set(member_ids)):
        issues.append(ContinuityQualityIssue("entity_coverage", "chapter entities must belong to exactly one global entity"))
    for item in continuity.timeline_events:
        if not set(item.participant_global_ids) <= global_ids:
            issues.append(ContinuityQualityIssue("unknown_global_participant", f"timeline event {item.timeline_event_id} has an unknown global participant"))
    for item in continuity.state_ledger:
        if item.global_subject_id not in global_ids or (item.chapter_id, item.event_id) not in timeline_keys:
            issues.append(ContinuityQualityIssue("invalid_state_reference", f"state entry {item.entry_id} has an unknown event or global entity"))
    for item in continuity.location_transitions:
        ids = {item.global_subject_id, item.from_location_global_id, item.to_location_global_id} - {""}
        if not ids <= global_ids or (item.chapter_id, item.event_id) not in timeline_keys:
            issues.append(ContinuityQualityIssue("invalid_location_reference", f"location transition {item.transition_id} has an unknown event or global entity"))
    input_spatial_ids = {relation.relation_id for annotation in annotations for relation in annotation.spatial_relations}
    accounted_spatial_ids = {item.relation_id for item in continuity.location_transitions} | set(continuity.unlinked_spatial_relation_ids)
    if accounted_spatial_ids != input_spatial_ids:
        issues.append(ContinuityQualityIssue("spatial_relation_coverage", "every source spatial relation must be linked to an event or explicitly queued for review"))
    for previous, current in zip(continuity.timeline_events, continuity.timeline_events[1:]):
        if (current.chapter_index, current.event_order) < (previous.chapter_index, previous.event_order):
            issues.append(ContinuityQualityIssue("non_monotonic_timeline", "timeline order is not monotonic"))
            break
    return ContinuityQualityReport(
        continuity.author_id,
        continuity.work_id,
        len(continuity.chapter_ids),
        tuple(issues),
        len(continuity.global_entities),
        len(continuity.timeline_events),
        len(continuity.state_ledger),
        len(continuity.location_transitions),
        len(continuity.unlinked_spatial_relation_ids),
    )


def _load_documents(author_id: str, work_id: str) -> dict[str, ChapterDocument]:
    folder = corpus_dir(author_id, work_id)
    return {
        row["chapter_id"]: ChapterDocument.from_dict(row)
        for row in read_jsonl(folder / "chapter_documents.jsonl")
        if str(row.get("chapter_id", "")).strip()
    }


def _annotation_path(folder: Path, limit: int) -> Path:
    target = folder / f"chapter_annotations.sample-{limit}.jsonl"
    if target.exists():
        return target
    candidates = sorted(folder.glob("chapter_annotations.sample-*.jsonl"))
    if not candidates:
        raise FileNotFoundError("no evidence-first chapter annotation file found")
    return candidates[-1]


def build_continuity_work(author_id: str, work_id: str, *, limit: int = DEFAULT_CONTINUITY_LIMIT) -> tuple[Path, Path]:
    """Build and persist a continuity graph only when contracts and coverage pass."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    folder = corpus_dir(author_id, work_id)
    documents = _load_documents(author_id, work_id)
    annotations = [ChapterAnnotation.from_dict(row) for row in read_jsonl(_annotation_path(folder, limit))][:limit]
    if not annotations:
        raise FileNotFoundError("no chapter annotations found")
    for annotation in annotations:
        document = documents.get(annotation.chapter_id)
        if document is None:
            raise ValueError(f"missing source document for {annotation.chapter_id}")
        report = validate_annotation(annotation, document)
        if not report.passed:
            messages = "; ".join(f"{issue.path}: {issue.message}" for issue in report.issues)
            raise ValueError(f"annotation is invalid for continuity: {messages}")
        if annotation.schema_version < "2.1":
            raise ValueError(f"annotation {annotation.chapter_id} lacks stage-3 spatiotemporal fields")
    continuity = build_work_continuity(author_id, work_id, annotations)
    quality = assess_continuity(continuity, annotations)
    if not quality.passed:
        messages = "; ".join(f"{issue.code}: {issue.message}" for issue in quality.issues)
        raise ValueError(f"continuity quality checks failed: {messages}")
    target = folder / f"work_continuity.sample-{len(annotations)}.json"
    quality_target = folder / f"work_continuity.sample-{len(annotations)}.quality.json"
    write_json(target, continuity.to_dict())
    write_json(quality_target, {
        **quality.to_dict(),
        "continuity_issue_counts": dict(Counter(issue.code for issue in continuity.issues)),
        "continuity_issues": [issue.to_dict() for issue in continuity.issues],
    })
    return target, quality_target
