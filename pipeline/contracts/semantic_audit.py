"""Typed results for content-level semantic audits of accepted annotations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SEMANTIC_AUDIT_CATEGORIES = frozenset({
    "entity_identity", "relationship", "fact_semantics", "event_frame", "state_transition", "causality",
    "temporal", "spatial", "clue_lifecycle", "coverage", "audit_contract", "other",
})
SEMANTIC_AUDIT_SEVERITIES = frozenset({"error", "warning", "info"})
SEMANTIC_AUDIT_VERDICTS = frozenset({"pass", "review"})


def _required(value: str, field_name: str) -> None:
    if not str(value or "").strip():
        raise ValueError(f"{field_name} must not be empty")


def _unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


@dataclass(frozen=True)
class SemanticAuditIssue:
    issue_id: str
    category: str
    severity: str
    object_ids: tuple[str, ...]
    evidence_unit_ids: tuple[str, ...]
    diagnosis: str
    proposed_action: str
    confidence: float

    def validate(self) -> None:
        _required(self.issue_id, "semantic audit issue_id")
        if self.category not in SEMANTIC_AUDIT_CATEGORIES:
            raise ValueError(f"unsupported semantic audit category: {self.category}")
        if self.severity not in SEMANTIC_AUDIT_SEVERITIES:
            raise ValueError(f"unsupported semantic audit severity: {self.severity}")
        if not self.object_ids or not self.evidence_unit_ids:
            raise ValueError("semantic audit issue needs object and evidence IDs")
        _unique(self.object_ids, "semantic audit object IDs")
        _unique(self.evidence_unit_ids, "semantic audit evidence unit IDs")
        _required(self.diagnosis, "semantic audit diagnosis")
        _required(self.proposed_action, "semantic audit proposed action")
        if not 0 <= self.confidence <= 1:
            raise ValueError("semantic audit confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "object_ids": list(self.object_ids),
            "evidence_unit_ids": list(self.evidence_unit_ids),
        }


@dataclass(frozen=True)
class SemanticAuditBatch:
    batch_id: str
    chapter_id: str
    source_unit_ids: tuple[str, ...]
    audited_entity_ids: tuple[str, ...]
    audited_fact_ids: tuple[str, ...]
    audited_event_ids: tuple[str, ...]
    verdict: str
    issues: tuple[SemanticAuditIssue, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.batch_id, "semantic audit batch_id")
        _required(self.chapter_id, "semantic audit chapter_id")
        if not self.source_unit_ids:
            raise ValueError("semantic audit batch needs source units")
        for values, name in (
            (self.source_unit_ids, "source units"),
            (self.audited_entity_ids, "audited entities"),
            (self.audited_fact_ids, "audited facts"),
            (self.audited_event_ids, "audited events"),
        ):
            _unique(values, f"semantic audit {name}")
        if self.verdict not in SEMANTIC_AUDIT_VERDICTS:
            raise ValueError(f"unsupported semantic audit verdict: {self.verdict}")
        for issue in self.issues:
            issue.validate()
            if not set(issue.evidence_unit_ids) <= set(self.source_unit_ids):
                raise ValueError("semantic audit issue references evidence outside its batch")
        _unique(tuple(issue.issue_id for issue in self.issues), "semantic audit issue IDs")
        if self.verdict == "pass" and self.issues:
            raise ValueError("semantic audit pass verdict cannot contain issues")
        if self.verdict == "review" and not self.issues:
            raise ValueError("semantic audit review verdict needs issues")

    @property
    def passed(self) -> bool:
        return self.verdict == "pass" and not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "chapter_id": self.chapter_id,
            "source_unit_ids": list(self.source_unit_ids),
            "audited_entity_ids": list(self.audited_entity_ids),
            "audited_fact_ids": list(self.audited_fact_ids),
            "audited_event_ids": list(self.audited_event_ids),
            "verdict": self.verdict,
            "passed": self.passed,
            "issues": [item.to_dict() for item in self.issues],
        }


@dataclass(frozen=True)
class SemanticAuditReport:
    author_id: str
    work_id: str
    chapter_ids: tuple[str, ...]
    batches: tuple[SemanticAuditBatch, ...]
    model_id: str
    prompt_version: str
    schema_version: str = "1.0"

    def validate(self) -> None:
        _required(self.author_id, "semantic audit author_id")
        _required(self.work_id, "semantic audit work_id")
        _required(self.model_id, "semantic audit model_id")
        _required(self.prompt_version, "semantic audit prompt_version")
        if not self.chapter_ids or not self.batches:
            raise ValueError("semantic audit report needs chapters and batches")
        _unique(self.chapter_ids, "semantic audit chapter IDs")
        _unique(tuple(item.batch_id for item in self.batches), "semantic audit batch IDs")
        for batch in self.batches:
            batch.validate()
            if batch.chapter_id not in self.chapter_ids:
                raise ValueError("semantic audit batch references unknown chapter")

    @property
    def passed(self) -> bool:
        return all(batch.passed for batch in self.batches)

    def to_dict(self) -> dict[str, Any]:
        counts = {category: 0 for category in sorted(SEMANTIC_AUDIT_CATEGORIES)}
        severities = {severity: 0 for severity in sorted(SEMANTIC_AUDIT_SEVERITIES)}
        for batch in self.batches:
            for issue in batch.issues:
                counts[issue.category] += 1
                severities[issue.severity] += 1
        return {
            "schema_version": self.schema_version,
            "author_id": self.author_id,
            "work_id": self.work_id,
            "chapter_ids": list(self.chapter_ids),
            "chapter_count": len(self.chapter_ids),
            "batch_count": len(self.batches),
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "passed": self.passed,
            "issue_counts_by_category": counts,
            "issue_counts_by_severity": severities,
            "batches": [item.to_dict() for item in self.batches],
        }
