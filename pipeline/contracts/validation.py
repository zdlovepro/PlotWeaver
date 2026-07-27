"""Deterministic contract validation.  No model call is allowed here."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .narrative import ChapterAnnotation
from .source import ChapterDocument
from .style import ChapterStyleCard


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    subject_id: str
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {"subject_id": self.subject_id, "passed": self.passed, "issues": [issue.to_dict() for issue in self.issues]}


class ContractValidationError(ValueError):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__("; ".join(f"{issue.path}: {issue.message}" for issue in report.issues) or "contract validation failed")


def _check(issues: list[ValidationIssue], path: str, action: Callable[[], None]) -> None:
    try:
        action()
    except (TypeError, ValueError) as exc:
        issues.append(ValidationIssue("invalid_contract", path, str(exc)))


def validate_document(document: ChapterDocument, *, raise_on_error: bool = False) -> ValidationReport:
    issues: list[ValidationIssue] = []
    _check(issues, "document", document.validate)
    report = ValidationReport(document.chapter_id, tuple(issues))
    if raise_on_error and not report.passed:
        raise ContractValidationError(report)
    return report


def validate_annotation(
    annotation: ChapterAnnotation,
    document: ChapterDocument,
    style: ChapterStyleCard | None = None,
    *,
    raise_on_error: bool = False,
) -> ValidationReport:
    """Validate IDs, references and exact evidence spans against source text."""

    issues: list[ValidationIssue] = []
    _check(issues, "document", document.validate)
    if annotation.chapter_id != document.chapter_id:
        issues.append(ValidationIssue("chapter_mismatch", "annotation.chapter_id", "annotation chapter_id differs from source document"))
    if annotation.source_hash != document.source_hash:
        issues.append(ValidationIssue("source_hash_mismatch", "annotation.source_hash", "annotation source_hash differs from source document"))
    for name, items in (
        ("entities", annotation.entities),
        ("facts", annotation.facts),
        ("events", annotation.events),
        ("scenes", annotation.scenes),
    ):
        if not items:
            issues.append(ValidationIssue("missing_required_content", name, f"{name} must not be empty"))

    def validate_items(items: tuple[Any, ...], prefix: str) -> None:
        for index, item in enumerate(items):
            _check(issues, f"{prefix}[{index}]", item.validate)

    validate_items(annotation.entities, "entities")
    validate_items(annotation.facts, "facts")
    validate_items(annotation.events, "events")
    validate_items(annotation.scenes, "scenes")
    validate_items(annotation.state_changes, "state_changes")
    validate_items(annotation.time_anchors, "time_anchors")
    validate_items(annotation.temporal_relations, "temporal_relations")
    validate_items(annotation.spatial_relations, "spatial_relations")

    def duplicate_ids(items: tuple[Any, ...], field: str) -> set[str]:
        values = [str(getattr(item, field, "")) for item in items]
        return {value for value in values if value and values.count(value) > 1}

    for name, items, field in (
        ("entities", annotation.entities, "entity_id"),
        ("facts", annotation.facts, "fact_id"),
        ("events", annotation.events, "event_id"),
        ("scenes", annotation.scenes, "scene_id"),
        ("state_changes", annotation.state_changes, "change_id"),
        ("time_anchors", annotation.time_anchors, "anchor_id"),
        ("temporal_relations", annotation.temporal_relations, "relation_id"),
        ("spatial_relations", annotation.spatial_relations, "relation_id"),
    ):
        for duplicate in duplicate_ids(items, field):
            issues.append(ValidationIssue("duplicate_id", name, f"duplicate id: {duplicate}"))
    for name, items in (("events", annotation.events), ("scenes", annotation.scenes)):
        orders = [item.order for item in items]
        if len(orders) != len(set(orders)):
            issues.append(ValidationIssue("duplicate_order", name, f"{name} contains duplicate order values"))

    entity_ids = {item.entity_id for item in annotation.entities}
    fact_ids = {item.fact_id for item in annotation.facts}
    event_ids = {item.event_id for item in annotation.events}
    time_anchor_ids = {item.anchor_id for item in annotation.time_anchors}
    facts_by_id = {item.fact_id: item for item in annotation.facts}
    events_by_id = {item.event_id: item for item in annotation.events}
    unit_ids = {unit.unit_id for unit in document.units}

    for name, items in (
        ("facts", annotation.facts),
        ("events", annotation.events),
        ("scenes", annotation.scenes),
        ("state_changes", annotation.state_changes),
    ):
        for index, item in enumerate(items):
            if item.chapter_id != annotation.chapter_id:
                issues.append(ValidationIssue("chapter_mismatch", f"{name}[{index}].chapter_id", "item chapter_id differs from annotation"))

    def check_spans(path: str, spans: tuple[Any, ...]) -> None:
        for index, span in enumerate(spans):
            _check(issues, f"{path}.evidence[{index}]", lambda span=span: span.validate(document.text))
            if span.chapter_id != document.chapter_id:
                issues.append(ValidationIssue("chapter_mismatch", f"{path}.evidence[{index}]", "evidence references a different chapter"))
            if span.unit_id and span.unit_id not in unit_ids:
                issues.append(ValidationIssue("unknown_unit", f"{path}.evidence[{index}]", f"unknown source unit: {span.unit_id}"))

    for index, item in enumerate(annotation.entities):
        check_spans(f"entities[{index}]", item.evidence)
    for index, item in enumerate(annotation.facts):
        check_spans(f"facts[{index}]", item.evidence)
        if item.subject_id not in entity_ids:
            issues.append(ValidationIssue("unknown_entity", f"facts[{index}].subject_id", f"unknown entity: {item.subject_id}"))
        if item.object_id and item.object_id not in entity_ids:
            issues.append(ValidationIssue("unknown_entity", f"facts[{index}].object_id", f"unknown entity: {item.object_id}"))
    for index, item in enumerate(annotation.events):
        check_spans(f"events[{index}]", item.evidence)
        for entity_id in item.participant_ids:
            if entity_id not in entity_ids:
                issues.append(ValidationIssue("unknown_entity", f"events[{index}].participant_ids", f"unknown entity: {entity_id}"))
        for fact_id in (*item.trigger_fact_ids, *item.precondition_fact_ids, *item.outcome_fact_ids, *item.cost_fact_ids):
            if fact_id not in fact_ids:
                issues.append(ValidationIssue("unknown_fact", f"events[{index}]", f"unknown fact: {fact_id}"))
    for index, item in enumerate(annotation.scenes):
        check_spans(f"scenes[{index}]", item.evidence)
        for entity_id in item.participant_ids:
            if entity_id not in entity_ids:
                issues.append(ValidationIssue("unknown_entity", f"scenes[{index}].participant_ids", f"unknown entity: {entity_id}"))
        for event_id in item.event_ids:
            if event_id not in event_ids:
                issues.append(ValidationIssue("unknown_event", f"scenes[{index}].event_ids", f"unknown event: {event_id}"))
        for fact_id in (*item.entry_fact_ids, *item.exit_fact_ids):
            if fact_id not in fact_ids:
                issues.append(ValidationIssue("unknown_fact", f"scenes[{index}]", f"unknown fact: {fact_id}"))
        for location_id in item.location_ids:
            if location_id not in entity_ids:
                issues.append(ValidationIssue("unknown_entity", f"scenes[{index}].location_ids", f"unknown location entity: {location_id}"))
            elif next(entity for entity in annotation.entities if entity.entity_id == location_id).kind != "location":
                issues.append(ValidationIssue("invalid_location", f"scenes[{index}].location_ids", f"entity is not a location: {location_id}"))
        for anchor_id in item.time_anchor_ids:
            if anchor_id not in time_anchor_ids:
                issues.append(ValidationIssue("unknown_time_anchor", f"scenes[{index}].time_anchor_ids", f"unknown time anchor: {anchor_id}"))
    for index, item in enumerate(annotation.time_anchors):
        check_spans(f"time_anchors[{index}]", item.evidence)
    for index, item in enumerate(annotation.temporal_relations):
        check_spans(f"temporal_relations[{index}]", item.evidence)
        if item.before_event_id not in event_ids:
            issues.append(ValidationIssue("unknown_event", f"temporal_relations[{index}].before_event_id", f"unknown event: {item.before_event_id}"))
        if item.after_event_id not in event_ids:
            issues.append(ValidationIssue("unknown_event", f"temporal_relations[{index}].after_event_id", f"unknown event: {item.after_event_id}"))
        if item.anchor_id and item.anchor_id not in time_anchor_ids:
            issues.append(ValidationIssue("unknown_time_anchor", f"temporal_relations[{index}].anchor_id", f"unknown time anchor: {item.anchor_id}"))
        before = events_by_id.get(item.before_event_id)
        after = events_by_id.get(item.after_event_id)
        if item.basis == "document_order" and before is not None and after is not None:
            before_start = min((span.start for span in before.evidence), default=10**12)
            after_start = min((span.start for span in after.evidence), default=10**12)
            if before_start >= after_start:
                issues.append(ValidationIssue("invalid_temporal_order", f"temporal_relations[{index}]", "document-order relation does not follow source offsets"))
    for index, item in enumerate(annotation.spatial_relations):
        check_spans(f"spatial_relations[{index}]", item.evidence)
        if item.subject_id not in entity_ids:
            issues.append(ValidationIssue("unknown_entity", f"spatial_relations[{index}].subject_id", f"unknown entity: {item.subject_id}"))
        if item.location_id not in entity_ids:
            issues.append(ValidationIssue("unknown_entity", f"spatial_relations[{index}].location_id", f"unknown location entity: {item.location_id}"))
        elif next(entity for entity in annotation.entities if entity.entity_id == item.location_id).kind != "location":
            issues.append(ValidationIssue("invalid_location", f"spatial_relations[{index}].location_id", f"entity is not a location: {item.location_id}"))
        if item.fact_id not in fact_ids:
            issues.append(ValidationIssue("unknown_fact", f"spatial_relations[{index}].fact_id", f"unknown fact: {item.fact_id}"))
            continue
        fact = facts_by_id[item.fact_id]
        if fact.kind != "location" or fact.subject_id != item.subject_id or fact.object_id != item.location_id:
            issues.append(ValidationIssue("invalid_spatial_basis", f"spatial_relations[{index}]", "spatial relation must match its location fact subject and object"))
        fact_spans = {(span.start, span.end, span.quote, span.unit_id) for span in fact.evidence}
        if not all((span.start, span.end, span.quote, span.unit_id) in fact_spans for span in item.evidence):
            issues.append(ValidationIssue("invalid_spatial_evidence", f"spatial_relations[{index}]", "spatial relation evidence must come from its supporting fact"))
    for index, item in enumerate(annotation.state_changes):
        if item.event_id not in event_ids:
            issues.append(ValidationIssue("unknown_event", f"state_changes[{index}].event_id", f"unknown event: {item.event_id}"))
        for fact_id in (item.before_fact_id, item.after_fact_id):
            if fact_id and fact_id not in fact_ids:
                issues.append(ValidationIssue("unknown_fact", f"state_changes[{index}]", f"unknown fact: {fact_id}"))
    scene_event_ids = {event_id for scene in annotation.scenes for event_id in scene.event_ids}
    for event_id in event_ids - scene_event_ids:
        issues.append(ValidationIssue("uncovered_event", "scenes", f"event is not assigned to any scene: {event_id}"))

    if style is not None:
        if style.chapter_id != document.chapter_id:
            issues.append(ValidationIssue("chapter_mismatch", "style.chapter_id", "style chapter_id differs from source document"))
        if style.source_hash != document.source_hash:
            issues.append(ValidationIssue("source_hash_mismatch", "style.source_hash", "style source_hash differs from source document"))
        validate_items(style.metrics, "style.metrics")
        validate_items(style.observations, "style.observations")
        for index, item in enumerate(style.observations):
            check_spans(f"style.observations[{index}]", item.evidence)

    report = ValidationReport(annotation.chapter_id, tuple(issues))
    if raise_on_error and not report.passed:
        raise ContractValidationError(report)
    return report
