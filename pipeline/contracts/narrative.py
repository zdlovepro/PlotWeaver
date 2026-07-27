"""Evidence-first narrative contracts for facts, events, scenes and state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .source import SourceSpan


ENTITY_KINDS = frozenset({"person", "organization", "location", "item", "concept", "creature"})
FACT_KINDS = frozenset({"identity", "goal", "emotion", "relationship", "location", "resource", "knowledge", "rule", "progression", "information"})
STATE_OPERATIONS = frozenset({"add", "update", "remove"})
TIME_ANCHOR_KINDS = frozenset({"explicit", "relative"})
TEMPORAL_RELATION_KINDS = frozenset({"before", "simultaneous"})
TEMPORAL_RELATION_BASES = frozenset({"document_order", "explicit_anchor"})
SPATIAL_RELATION_KINDS = frozenset({"at", "moves_to", "enters", "leaves"})


def _required(value: str, field_name: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


@dataclass(frozen=True)
class Entity:
    entity_id: str
    canonical_name: str
    kind: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    evidence: tuple[SourceSpan, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.entity_id, "entity_id")
        _required(self.canonical_name, "canonical_name")
        if self.kind not in ENTITY_KINDS:
            raise ValueError(f"unsupported entity kind: {self.kind}")
        if self.canonical_name in self.aliases:
            raise ValueError("canonical name must not be repeated in aliases")
        _unique(self.aliases, "aliases")
        if not self.evidence:
            raise ValueError("entity must have source evidence")
        for span in self.evidence:
            span.validate()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "aliases": list(self.aliases), "evidence": [span.to_dict() for span in self.evidence]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Entity":
        return cls(
            entity_id=str(payload.get("entity_id", "")),
            canonical_name=str(payload.get("canonical_name", "")),
            kind=str(payload.get("kind", "")),
            aliases=tuple(str(item) for item in payload.get("aliases", []) if str(item).strip()),
            evidence=tuple(SourceSpan.from_dict(item) for item in payload.get("evidence", []) if isinstance(item, dict)),
        )


@dataclass(frozen=True)
class Fact:
    fact_id: str
    chapter_id: str
    kind: str
    subject_id: str
    predicate: str
    value: str
    object_id: str = ""
    certainty: float = 1.0
    evidence: tuple[SourceSpan, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.fact_id, "fact_id")
        _required(self.chapter_id, "chapter_id")
        if self.kind not in FACT_KINDS:
            raise ValueError(f"unsupported fact kind: {self.kind}")
        _required(self.subject_id, "subject_id")
        _required(self.predicate, "predicate")
        if not self.value and not self.object_id:
            raise ValueError("fact requires value or object_id")
        if not 0.0 < self.certainty <= 1.0:
            raise ValueError("fact certainty must be in (0, 1]")
        if not self.evidence:
            raise ValueError("fact must have source evidence")
        for span in self.evidence:
            span.validate()
            if span.chapter_id != self.chapter_id:
                raise ValueError("fact evidence chapter does not match fact")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "evidence": [span.to_dict() for span in self.evidence]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Fact":
        return cls(
            fact_id=str(payload.get("fact_id", "")),
            chapter_id=str(payload.get("chapter_id", "")),
            kind=str(payload.get("kind", "")),
            subject_id=str(payload.get("subject_id", "")),
            predicate=str(payload.get("predicate", "")),
            value=str(payload.get("value", "")),
            object_id=str(payload.get("object_id", "")),
            certainty=float(payload.get("certainty", 1.0)),
            evidence=tuple(SourceSpan.from_dict(item) for item in payload.get("evidence", []) if isinstance(item, dict)),
        )


@dataclass(frozen=True)
class EventAtom:
    event_id: str
    chapter_id: str
    order: int
    summary: str
    participant_ids: tuple[str, ...]
    trigger_fact_ids: tuple[str, ...]
    precondition_fact_ids: tuple[str, ...]
    action: str
    obstacle: str
    decision: str
    outcome_fact_ids: tuple[str, ...]
    cost_fact_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence: tuple[SourceSpan, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.event_id, "event_id")
        _required(self.chapter_id, "chapter_id")
        if self.order < 0:
            raise ValueError("event order must be non-negative")
        _required(self.summary, "event summary")
        if not self.participant_ids:
            raise ValueError("event must have participants")
        _unique(self.participant_ids, "event participants")
        _required(self.action, "event action")
        if not self.outcome_fact_ids:
            raise ValueError("event must declare outcome facts")
        if not self.evidence:
            raise ValueError("event must have source evidence")
        for span in self.evidence:
            span.validate()
            if span.chapter_id != self.chapter_id:
                raise ValueError("event evidence chapter does not match event")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "participant_ids": list(self.participant_ids), "trigger_fact_ids": list(self.trigger_fact_ids), "precondition_fact_ids": list(self.precondition_fact_ids), "outcome_fact_ids": list(self.outcome_fact_ids), "cost_fact_ids": list(self.cost_fact_ids), "evidence": [span.to_dict() for span in self.evidence]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventAtom":
        fields = ("participant_ids", "trigger_fact_ids", "precondition_fact_ids", "outcome_fact_ids", "cost_fact_ids")
        values = {field: tuple(str(item) for item in payload.get(field, []) if str(item).strip()) for field in fields}
        return cls(
            event_id=str(payload.get("event_id", "")),
            chapter_id=str(payload.get("chapter_id", "")),
            order=int(payload.get("order", -1)),
            summary=str(payload.get("summary", "")),
            action=str(payload.get("action", "")),
            obstacle=str(payload.get("obstacle", "")),
            decision=str(payload.get("decision", "")),
            evidence=tuple(SourceSpan.from_dict(item) for item in payload.get("evidence", []) if isinstance(item, dict)),
            **values,
        )


@dataclass(frozen=True)
class TimeAnchor:
    """An exact textual time cue, kept apart from inferred event ordering."""

    anchor_id: str
    chapter_id: str
    label: str
    kind: str
    evidence: tuple[SourceSpan, ...]

    def validate(self) -> None:
        _required(self.anchor_id, "anchor_id")
        _required(self.chapter_id, "chapter_id")
        _required(self.label, "time anchor label")
        if self.kind not in TIME_ANCHOR_KINDS:
            raise ValueError(f"unsupported time anchor kind: {self.kind}")
        if not self.evidence:
            raise ValueError("time anchor must have source evidence")
        for span in self.evidence:
            span.validate()
            if span.chapter_id != self.chapter_id:
                raise ValueError("time anchor evidence chapter does not match anchor")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "evidence": [span.to_dict() for span in self.evidence]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TimeAnchor":
        return cls(
            anchor_id=str(payload.get("anchor_id", "")),
            chapter_id=str(payload.get("chapter_id", "")),
            label=str(payload.get("label", "")),
            kind=str(payload.get("kind", "")),
            evidence=tuple(SourceSpan.from_dict(item) for item in payload.get("evidence", []) if isinstance(item, dict)),
        )


@dataclass(frozen=True)
class TemporalRelation:
    """A traceable relation between two event atoms in a single chapter."""

    relation_id: str
    chapter_id: str
    before_event_id: str
    after_event_id: str
    kind: str
    basis: str
    evidence: tuple[SourceSpan, ...]
    anchor_id: str = ""

    def validate(self) -> None:
        _required(self.relation_id, "relation_id")
        _required(self.chapter_id, "chapter_id")
        _required(self.before_event_id, "before_event_id")
        _required(self.after_event_id, "after_event_id")
        if self.before_event_id == self.after_event_id:
            raise ValueError("temporal relation cannot reference the same event twice")
        if self.kind not in TEMPORAL_RELATION_KINDS:
            raise ValueError(f"unsupported temporal relation kind: {self.kind}")
        if self.basis not in TEMPORAL_RELATION_BASES:
            raise ValueError(f"unsupported temporal relation basis: {self.basis}")
        if self.basis == "explicit_anchor" and not self.anchor_id:
            raise ValueError("explicit-anchor temporal relation requires anchor_id")
        if not self.evidence:
            raise ValueError("temporal relation must have source evidence")
        for span in self.evidence:
            span.validate()
            if span.chapter_id != self.chapter_id:
                raise ValueError("temporal relation evidence chapter does not match relation")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "evidence": [span.to_dict() for span in self.evidence]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemporalRelation":
        return cls(
            relation_id=str(payload.get("relation_id", "")),
            chapter_id=str(payload.get("chapter_id", "")),
            before_event_id=str(payload.get("before_event_id", "")),
            after_event_id=str(payload.get("after_event_id", "")),
            kind=str(payload.get("kind", "")),
            basis=str(payload.get("basis", "")),
            evidence=tuple(SourceSpan.from_dict(item) for item in payload.get("evidence", []) if isinstance(item, dict)),
            anchor_id=str(payload.get("anchor_id", "")),
        )


@dataclass(frozen=True)
class SpatialRelation:
    """A fact-backed placement or movement relation to a location entity."""

    relation_id: str
    chapter_id: str
    subject_id: str
    location_id: str
    kind: str
    fact_id: str
    evidence: tuple[SourceSpan, ...]

    def validate(self) -> None:
        _required(self.relation_id, "relation_id")
        _required(self.chapter_id, "chapter_id")
        _required(self.subject_id, "spatial relation subject_id")
        _required(self.location_id, "spatial relation location_id")
        _required(self.fact_id, "spatial relation fact_id")
        if self.kind not in SPATIAL_RELATION_KINDS:
            raise ValueError(f"unsupported spatial relation kind: {self.kind}")
        if not self.evidence:
            raise ValueError("spatial relation must have source evidence")
        for span in self.evidence:
            span.validate()
            if span.chapter_id != self.chapter_id:
                raise ValueError("spatial relation evidence chapter does not match relation")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "evidence": [span.to_dict() for span in self.evidence]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpatialRelation":
        return cls(
            relation_id=str(payload.get("relation_id", "")),
            chapter_id=str(payload.get("chapter_id", "")),
            subject_id=str(payload.get("subject_id", "")),
            location_id=str(payload.get("location_id", "")),
            kind=str(payload.get("kind", "")),
            fact_id=str(payload.get("fact_id", "")),
            evidence=tuple(SourceSpan.from_dict(item) for item in payload.get("evidence", []) if isinstance(item, dict)),
        )


@dataclass(frozen=True)
class SceneCard:
    scene_id: str
    chapter_id: str
    order: int
    participant_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    objective: str
    entry_fact_ids: tuple[str, ...]
    exit_fact_ids: tuple[str, ...]
    tension: int
    evidence: tuple[SourceSpan, ...]
    location_ids: tuple[str, ...] = field(default_factory=tuple)
    time_anchor_ids: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.scene_id, "scene_id")
        _required(self.chapter_id, "chapter_id")
        if self.order < 0:
            raise ValueError("scene order must be non-negative")
        if not self.participant_ids or not self.event_ids:
            raise ValueError("scene needs participants and events")
        _unique(self.participant_ids, "scene participants")
        _unique(self.event_ids, "scene events")
        _unique(self.location_ids, "scene locations")
        _unique(self.time_anchor_ids, "scene time anchors")
        _required(self.objective, "scene objective")
        if not self.exit_fact_ids:
            raise ValueError("scene must declare exit facts")
        if not 0 <= self.tension <= 5:
            raise ValueError("scene tension must be in [0, 5]")
        if not self.evidence:
            raise ValueError("scene must have source evidence")
        for span in self.evidence:
            span.validate()
            if span.chapter_id != self.chapter_id:
                raise ValueError("scene evidence chapter does not match scene")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "participant_ids": list(self.participant_ids), "event_ids": list(self.event_ids), "entry_fact_ids": list(self.entry_fact_ids), "exit_fact_ids": list(self.exit_fact_ids), "location_ids": list(self.location_ids), "time_anchor_ids": list(self.time_anchor_ids), "evidence": [span.to_dict() for span in self.evidence]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SceneCard":
        return cls(
            scene_id=str(payload.get("scene_id", "")),
            chapter_id=str(payload.get("chapter_id", "")),
            order=int(payload.get("order", -1)),
            participant_ids=tuple(str(item) for item in payload.get("participant_ids", []) if str(item).strip()),
            event_ids=tuple(str(item) for item in payload.get("event_ids", []) if str(item).strip()),
            objective=str(payload.get("objective", "")),
            entry_fact_ids=tuple(str(item) for item in payload.get("entry_fact_ids", []) if str(item).strip()),
            exit_fact_ids=tuple(str(item) for item in payload.get("exit_fact_ids", []) if str(item).strip()),
            tension=int(payload.get("tension", -1)),
            evidence=tuple(SourceSpan.from_dict(item) for item in payload.get("evidence", []) if isinstance(item, dict)),
            location_ids=tuple(str(item) for item in payload.get("location_ids", []) if str(item).strip()),
            time_anchor_ids=tuple(str(item) for item in payload.get("time_anchor_ids", []) if str(item).strip()),
        )


@dataclass(frozen=True)
class StateChange:
    change_id: str
    chapter_id: str
    event_id: str
    operation: str
    before_fact_id: str = ""
    after_fact_id: str = ""

    def validate(self) -> None:
        _required(self.change_id, "change_id")
        _required(self.chapter_id, "chapter_id")
        _required(self.event_id, "event_id")
        if self.operation not in STATE_OPERATIONS:
            raise ValueError(f"unsupported state operation: {self.operation}")
        if self.operation == "add" and not self.after_fact_id:
            raise ValueError("add state change requires after_fact_id")
        if self.operation == "remove" and not self.before_fact_id:
            raise ValueError("remove state change requires before_fact_id")
        if self.operation == "update" and (not self.before_fact_id or not self.after_fact_id):
            raise ValueError("update state change requires before_fact_id and after_fact_id")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StateChange":
        return cls(
            change_id=str(payload.get("change_id", "")),
            chapter_id=str(payload.get("chapter_id", "")),
            event_id=str(payload.get("event_id", "")),
            operation=str(payload.get("operation", "")),
            before_fact_id=str(payload.get("before_fact_id", "")),
            after_fact_id=str(payload.get("after_fact_id", "")),
        )


@dataclass(frozen=True)
class ChapterAnnotation:
    """Complete structural annotation for one chapter, excluding style metrics."""

    chapter_id: str
    source_hash: str
    entities: tuple[Entity, ...]
    facts: tuple[Fact, ...]
    events: tuple[EventAtom, ...]
    scenes: tuple[SceneCard, ...]
    state_changes: tuple[StateChange, ...]
    schema_version: str = "2.1"
    time_anchors: tuple[TimeAnchor, ...] = field(default_factory=tuple)
    temporal_relations: tuple[TemporalRelation, ...] = field(default_factory=tuple)
    spatial_relations: tuple[SpatialRelation, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "source_hash": self.source_hash,
            "entities": [item.to_dict() for item in self.entities],
            "facts": [item.to_dict() for item in self.facts],
            "events": [item.to_dict() for item in self.events],
            "scenes": [item.to_dict() for item in self.scenes],
            "state_changes": [item.to_dict() for item in self.state_changes],
            "time_anchors": [item.to_dict() for item in self.time_anchors],
            "temporal_relations": [item.to_dict() for item in self.temporal_relations],
            "spatial_relations": [item.to_dict() for item in self.spatial_relations],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterAnnotation":
        return cls(
            chapter_id=str(payload.get("chapter_id", "")),
            source_hash=str(payload.get("source_hash", "")),
            entities=tuple(Entity.from_dict(item) for item in payload.get("entities", []) if isinstance(item, dict)),
            facts=tuple(Fact.from_dict(item) for item in payload.get("facts", []) if isinstance(item, dict)),
            events=tuple(EventAtom.from_dict(item) for item in payload.get("events", []) if isinstance(item, dict)),
            scenes=tuple(SceneCard.from_dict(item) for item in payload.get("scenes", []) if isinstance(item, dict)),
            state_changes=tuple(StateChange.from_dict(item) for item in payload.get("state_changes", []) if isinstance(item, dict)),
            schema_version=str(payload.get("schema_version", "2.0")),
            time_anchors=tuple(TimeAnchor.from_dict(item) for item in payload.get("time_anchors", []) if isinstance(item, dict)),
            temporal_relations=tuple(TemporalRelation.from_dict(item) for item in payload.get("temporal_relations", []) if isinstance(item, dict)),
            spatial_relations=tuple(SpatialRelation.from_dict(item) for item in payload.get("spatial_relations", []) if isinstance(item, dict)),
        )
