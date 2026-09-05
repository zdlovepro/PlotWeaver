"""Typed, evidence-linked contracts for cross-chapter narrative continuity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .source import SourceSpan


CONTINUITY_SEVERITIES = frozenset({"error", "warning", "info"})
STATE_OPERATIONS = frozenset({"add", "update", "remove", "observe"})


def _required(value: str, field_name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


def _validate_evidence(spans: tuple[SourceSpan, ...], chapter_id: str, field_name: str) -> None:
    if not spans:
        raise ValueError(f"{field_name} must have source evidence")
    for span in spans:
        span.validate()
        if span.chapter_id != chapter_id:
            raise ValueError(f"{field_name} evidence chapter does not match")


@dataclass(frozen=True)
class GlobalEntity:
    global_entity_id: str
    kind: str
    canonical_name: str
    member_entity_ids: tuple[str, ...]
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.global_entity_id, "global_entity_id")
        _required(self.kind, "global entity kind")
        _required(self.canonical_name, "global entity canonical_name")
        if not self.member_entity_ids:
            raise ValueError("global entity needs at least one chapter entity")
        _unique(self.member_entity_ids, "global entity members")
        _unique(self.aliases, "global entity aliases")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "member_entity_ids": list(self.member_entity_ids), "aliases": list(self.aliases)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GlobalEntity":
        return cls(
            global_entity_id=str(payload.get("global_entity_id", "")),
            kind=str(payload.get("kind", "")),
            canonical_name=str(payload.get("canonical_name", "")),
            member_entity_ids=tuple(str(item) for item in payload.get("member_entity_ids", []) if str(item).strip()),
            aliases=tuple(str(item) for item in payload.get("aliases", []) if str(item).strip()),
        )


@dataclass(frozen=True)
class TimelineEvent:
    timeline_event_id: str
    chapter_id: str
    event_id: str
    chapter_index: int
    event_order: int
    participant_global_ids: tuple[str, ...]
    time_anchor_ids: tuple[str, ...] = field(default_factory=tuple)
    spatial_relation_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence: tuple[SourceSpan, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.timeline_event_id, "timeline_event_id")
        _required(self.chapter_id, "timeline event chapter_id")
        _required(self.event_id, "timeline event event_id")
        if self.chapter_index < 0 or self.event_order < 0:
            raise ValueError("timeline event indexes must be non-negative")
        if not self.participant_global_ids:
            raise ValueError("timeline event needs participants")
        _unique(self.participant_global_ids, "timeline event participants")
        _unique(self.time_anchor_ids, "timeline event time anchors")
        _unique(self.spatial_relation_ids, "timeline event spatial relations")
        _validate_evidence(self.evidence, self.chapter_id, "timeline event")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "participant_global_ids": list(self.participant_global_ids),
            "time_anchor_ids": list(self.time_anchor_ids),
            "spatial_relation_ids": list(self.spatial_relation_ids),
            "evidence": [span.to_dict() for span in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TimelineEvent":
        return cls(
            timeline_event_id=str(payload.get("timeline_event_id", "")),
            chapter_id=str(payload.get("chapter_id", "")),
            event_id=str(payload.get("event_id", "")),
            chapter_index=int(payload.get("chapter_index", -1)),
            event_order=int(payload.get("event_order", -1)),
            participant_global_ids=tuple(str(item) for item in payload.get("participant_global_ids", []) if str(item).strip()),
            time_anchor_ids=tuple(str(item) for item in payload.get("time_anchor_ids", []) if str(item).strip()),
            spatial_relation_ids=tuple(str(item) for item in payload.get("spatial_relation_ids", []) if str(item).strip()),
            evidence=tuple(SourceSpan.from_dict(item) for item in payload.get("evidence", []) if isinstance(item, dict)),
        )


@dataclass(frozen=True)
class StateLedgerEntry:
    entry_id: str
    chapter_id: str
    event_id: str
    chapter_index: int
    global_subject_id: str
    fact_id: str
    operation: str
    slot: str
    value: str
    object_global_id: str = ""
    evidence: tuple[SourceSpan, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.entry_id, "state entry_id")
        _required(self.chapter_id, "state entry chapter_id")
        _required(self.event_id, "state entry event_id")
        _required(self.global_subject_id, "state entry subject")
        _required(self.fact_id, "state entry fact_id")
        _required(self.slot, "state entry slot")
        if self.chapter_index < 0:
            raise ValueError("state entry chapter_index must be non-negative")
        if self.operation not in STATE_OPERATIONS:
            raise ValueError(f"unsupported state entry operation: {self.operation}")
        _validate_evidence(self.evidence, self.chapter_id, "state entry")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "evidence": [span.to_dict() for span in self.evidence]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StateLedgerEntry":
        return cls(
            entry_id=str(payload.get("entry_id", "")),
            chapter_id=str(payload.get("chapter_id", "")),
            event_id=str(payload.get("event_id", "")),
            chapter_index=int(payload.get("chapter_index", -1)),
            global_subject_id=str(payload.get("global_subject_id", "")),
            fact_id=str(payload.get("fact_id", "")),
            operation=str(payload.get("operation", "")),
            slot=str(payload.get("slot", "")),
            value=str(payload.get("value", "")),
            object_global_id=str(payload.get("object_global_id", "")),
            evidence=tuple(SourceSpan.from_dict(item) for item in payload.get("evidence", []) if isinstance(item, dict)),
        )


@dataclass(frozen=True)
class LocationTransition:
    transition_id: str
    chapter_id: str
    event_id: str
    chapter_index: int
    global_subject_id: str
    relation_id: str
    relation_kind: str
    from_location_global_id: str = ""
    to_location_global_id: str = ""
    evidence: tuple[SourceSpan, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.transition_id, "location transition_id")
        _required(self.chapter_id, "location transition chapter_id")
        _required(self.event_id, "location transition event_id")
        _required(self.global_subject_id, "location transition subject")
        _required(self.relation_id, "location transition relation_id")
        _required(self.relation_kind, "location transition relation_kind")
        if self.chapter_index < 0:
            raise ValueError("location transition chapter_index must be non-negative")
        if not self.from_location_global_id and not self.to_location_global_id:
            raise ValueError("location transition needs a source or destination")
        _validate_evidence(self.evidence, self.chapter_id, "location transition")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "evidence": [span.to_dict() for span in self.evidence]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocationTransition":
        return cls(
            transition_id=str(payload.get("transition_id", "")),
            chapter_id=str(payload.get("chapter_id", "")),
            event_id=str(payload.get("event_id", "")),
            chapter_index=int(payload.get("chapter_index", -1)),
            global_subject_id=str(payload.get("global_subject_id", "")),
            relation_id=str(payload.get("relation_id", "")),
            relation_kind=str(payload.get("relation_kind", "")),
            from_location_global_id=str(payload.get("from_location_global_id", "")),
            to_location_global_id=str(payload.get("to_location_global_id", "")),
            evidence=tuple(SourceSpan.from_dict(item) for item in payload.get("evidence", []) if isinstance(item, dict)),
        )


@dataclass(frozen=True)
class ContinuityIssue:
    code: str
    severity: str
    message: str
    chapter_id: str = ""
    related_ids: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.code, "continuity issue code")
        _required(self.message, "continuity issue message")
        if self.severity not in CONTINUITY_SEVERITIES:
            raise ValueError(f"unsupported continuity issue severity: {self.severity}")
        _unique(self.related_ids, "continuity issue related_ids")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "related_ids": list(self.related_ids)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContinuityIssue":
        return cls(
            code=str(payload.get("code", "")),
            severity=str(payload.get("severity", "")),
            message=str(payload.get("message", "")),
            chapter_id=str(payload.get("chapter_id", "")),
            related_ids=tuple(str(item) for item in payload.get("related_ids", []) if str(item).strip()),
        )


@dataclass(frozen=True)
class WorkContinuity:
    author_id: str
    work_id: str
    chapter_ids: tuple[str, ...]
    annotation_schema_version: str
    global_entities: tuple[GlobalEntity, ...]
    timeline_events: tuple[TimelineEvent, ...]
    state_ledger: tuple[StateLedgerEntry, ...]
    location_transitions: tuple[LocationTransition, ...]
    unlinked_spatial_relation_ids: tuple[str, ...] = field(default_factory=tuple)
    issues: tuple[ContinuityIssue, ...] = field(default_factory=tuple)
    schema_version: str = "1.0"

    def validate(self) -> None:
        _required(self.author_id, "continuity author_id")
        _required(self.work_id, "continuity work_id")
        if not self.chapter_ids:
            raise ValueError("continuity needs chapters")
        _unique(self.chapter_ids, "continuity chapter_ids")
        _unique(self.unlinked_spatial_relation_ids, "continuity unlinked spatial relations")
        for group in (self.global_entities, self.timeline_events, self.state_ledger, self.location_transitions, self.issues):
            for item in group:
                item.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "author_id": self.author_id,
            "work_id": self.work_id,
            "chapter_ids": list(self.chapter_ids),
            "annotation_schema_version": self.annotation_schema_version,
            "global_entities": [item.to_dict() for item in self.global_entities],
            "timeline_events": [item.to_dict() for item in self.timeline_events],
            "state_ledger": [item.to_dict() for item in self.state_ledger],
            "location_transitions": [item.to_dict() for item in self.location_transitions],
            "unlinked_spatial_relation_ids": list(self.unlinked_spatial_relation_ids),
            "issues": [item.to_dict() for item in self.issues],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkContinuity":
        return cls(
            author_id=str(payload.get("author_id", "")),
            work_id=str(payload.get("work_id", "")),
            chapter_ids=tuple(str(item) for item in payload.get("chapter_ids", []) if str(item).strip()),
            annotation_schema_version=str(payload.get("annotation_schema_version", "")),
            global_entities=tuple(GlobalEntity.from_dict(item) for item in payload.get("global_entities", []) if isinstance(item, dict)),
            timeline_events=tuple(TimelineEvent.from_dict(item) for item in payload.get("timeline_events", []) if isinstance(item, dict)),
            state_ledger=tuple(StateLedgerEntry.from_dict(item) for item in payload.get("state_ledger", []) if isinstance(item, dict)),
            location_transitions=tuple(LocationTransition.from_dict(item) for item in payload.get("location_transitions", []) if isinstance(item, dict)),
            unlinked_spatial_relation_ids=tuple(str(item) for item in payload.get("unlinked_spatial_relation_ids", []) if str(item).strip()),
            issues=tuple(ContinuityIssue.from_dict(item) for item in payload.get("issues", []) if isinstance(item, dict)),
            schema_version=str(payload.get("schema_version", "1.0")),
        )
