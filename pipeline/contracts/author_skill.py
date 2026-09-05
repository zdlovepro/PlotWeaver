"""Contracts for a portable, evidence-qualified novel-author Skill bundle.

Package validity and author-distillation validity are deliberately separate.
A structurally valid single-work package is useful for controlled experiments,
but it remains ``draft`` until cross-work and held-out evidence exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


SKILL_RELEASE_STATUSES = frozenset({"draft", "candidate", "validated"})
TRAIT_VALIDATION_STATUSES = frozenset({"single_work_candidate", "cross_work_candidate", "validated"})
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _required(value: str, field_name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


@dataclass(frozen=True)
class DistilledAuthorTrait:
    """One evidence-backed writing observation with explicit validation scope."""

    trait_id: str
    family: str
    rule: str
    application: str
    avoid: str
    source_work_ids: tuple[str, ...]
    evidence_chapter_ids: tuple[str, ...]
    support_count: int
    confidence: float
    validation_status: str = "single_work_candidate"
    applicable_contexts: tuple[str, ...] = field(default_factory=tuple)
    counter_contexts: tuple[str, ...] = field(default_factory=tuple)
    within_author_stability: float | None = None
    genre_contrast_score: float | None = None

    def validate(self) -> None:
        _required(self.trait_id, "author trait_id")
        _required(self.family, "author trait family")
        _required(self.rule, "author trait rule")
        _required(self.application, "author trait application")
        _required(self.avoid, "author trait avoid")
        if not self.source_work_ids or not self.evidence_chapter_ids:
            raise ValueError("author trait needs source works and evidence chapters")
        _unique(self.source_work_ids, "author trait source works")
        _unique(self.evidence_chapter_ids, "author trait evidence chapters")
        _unique(self.applicable_contexts, "author trait applicable contexts")
        _unique(self.counter_contexts, "author trait counter contexts")
        if self.support_count != len(self.evidence_chapter_ids):
            raise ValueError("author trait support_count must equal evidence chapter count")
        if not 0 < self.confidence <= 1:
            raise ValueError("author trait confidence must be in (0, 1]")
        if self.validation_status not in TRAIT_VALIDATION_STATUSES:
            raise ValueError(f"unsupported author trait validation status: {self.validation_status}")
        for name, value in (
            ("within_author_stability", self.within_author_stability),
            ("genre_contrast_score", self.genre_contrast_score),
        ):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.validation_status == "validated":
            if len(self.source_work_ids) < 2:
                raise ValueError("validated author trait needs evidence from at least two works")
            if self.within_author_stability is None or self.genre_contrast_score is None:
                raise ValueError("validated author trait needs stability and genre contrast scores")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "source_work_ids": list(self.source_work_ids),
            "evidence_chapter_ids": list(self.evidence_chapter_ids),
            "applicable_contexts": list(self.applicable_contexts),
            "counter_contexts": list(self.counter_contexts),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DistilledAuthorTrait":
        return cls(
            trait_id=str(payload.get("trait_id", "")),
            family=str(payload.get("family", "")),
            rule=str(payload.get("rule", "")),
            application=str(payload.get("application", "")),
            avoid=str(payload.get("avoid", "")),
            source_work_ids=tuple(str(item) for item in payload.get("source_work_ids", []) if str(item).strip()),
            evidence_chapter_ids=tuple(str(item) for item in payload.get("evidence_chapter_ids", []) if str(item).strip()),
            support_count=int(payload.get("support_count", 0)),
            confidence=float(payload.get("confidence", 0)),
            validation_status=str(payload.get("validation_status", "single_work_candidate")),
            applicable_contexts=tuple(str(item) for item in payload.get("applicable_contexts", []) if str(item).strip()),
            counter_contexts=tuple(str(item) for item in payload.get("counter_contexts", []) if str(item).strip()),
            within_author_stability=(
                float(payload["within_author_stability"])
                if payload.get("within_author_stability") is not None else None
            ),
            genre_contrast_score=(
                float(payload["genre_contrast_score"])
                if payload.get("genre_contrast_score") is not None else None
            ),
        )


@dataclass(frozen=True)
class SkillQualification:
    """Evidence gates that determine the highest honest release status."""

    source_work_count: int
    content_audit_chapter_count: int = 0
    semantic_audit_passed: bool = False
    cross_work_distillation_passed: bool = False
    held_out_evaluation_passed: bool = False
    long_form_evaluation_passed: bool = False
    style_discrimination_passed: bool = False
    copy_risk_passed: bool = False
    portable_self_check_passed: bool = False

    def validate(self) -> None:
        if self.source_work_count <= 0:
            raise ValueError("skill qualification source_work_count must be positive")
        if self.content_audit_chapter_count < 0:
            raise ValueError("content audit chapter count must not be negative")

    @property
    def highest_release_status(self) -> str:
        candidate = (
            self.source_work_count >= 2
            and self.content_audit_chapter_count >= 5
            and self.semantic_audit_passed
            and self.cross_work_distillation_passed
            and self.portable_self_check_passed
        )
        validated = candidate and all((
            self.held_out_evaluation_passed,
            self.long_form_evaluation_passed,
            self.style_discrimination_passed,
            self.copy_risk_passed,
        ))
        if validated:
            return "validated"
        if candidate:
            return "candidate"
        return "draft"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "highest_release_status": self.highest_release_status}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillQualification":
        return cls(
            source_work_count=int(payload.get("source_work_count", 0)),
            content_audit_chapter_count=int(payload.get("content_audit_chapter_count", 0)),
            semantic_audit_passed=bool(payload.get("semantic_audit_passed", False)),
            cross_work_distillation_passed=bool(payload.get("cross_work_distillation_passed", False)),
            held_out_evaluation_passed=bool(payload.get("held_out_evaluation_passed", False)),
            long_form_evaluation_passed=bool(payload.get("long_form_evaluation_passed", False)),
            style_discrimination_passed=bool(payload.get("style_discrimination_passed", False)),
            copy_risk_passed=bool(payload.get("copy_risk_passed", False)),
            portable_self_check_passed=bool(payload.get("portable_self_check_passed", False)),
        )


@dataclass(frozen=True)
class AuthorSkillBundle:
    """Manifest-level contract for one delivered author Skill folder."""

    author_id: str
    skill_name: str
    release_status: str
    source_work_ids: tuple[str, ...]
    chapter_ids: tuple[str, ...]
    candidate_trait_count: int
    validated_author_trait_count: int
    narrative_policy_count: int
    generated_files: tuple[str, ...]
    limitations: tuple[str, ...]
    qualification: SkillQualification
    schema_version: str = "2.0"

    def validate(self) -> None:
        _required(self.author_id, "author skill author_id")
        if not _SKILL_NAME_RE.fullmatch(self.skill_name) or len(self.skill_name) > 64:
            raise ValueError("skill_name must be lowercase hyphen-case and at most 64 characters")
        if self.release_status not in SKILL_RELEASE_STATUSES:
            raise ValueError(f"unsupported skill release status: {self.release_status}")
        if not self.source_work_ids or not self.chapter_ids or not self.generated_files:
            raise ValueError("author skill needs source works, chapters and generated files")
        _unique(self.source_work_ids, "author skill source works")
        _unique(self.chapter_ids, "author skill chapters")
        _unique(self.generated_files, "author skill generated files")
        _unique(self.limitations, "author skill limitations")
        for value, name in (
            (self.candidate_trait_count, "candidate_trait_count"),
            (self.validated_author_trait_count, "validated_author_trait_count"),
            (self.narrative_policy_count, "narrative_policy_count"),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        self.qualification.validate()
        rank = {"draft": 0, "candidate": 1, "validated": 2}
        if rank[self.release_status] > rank[self.qualification.highest_release_status]:
            raise ValueError("release_status exceeds available qualification evidence")
        if self.release_status == "validated" and self.validated_author_trait_count <= 0:
            raise ValueError("validated author skill needs validated author traits")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "author_id": self.author_id,
            "skill_name": self.skill_name,
            "release_status": self.release_status,
            "source_work_ids": list(self.source_work_ids),
            "chapter_ids": list(self.chapter_ids),
            "candidate_trait_count": self.candidate_trait_count,
            "validated_author_trait_count": self.validated_author_trait_count,
            "narrative_policy_count": self.narrative_policy_count,
            "generated_files": list(self.generated_files),
            "limitations": list(self.limitations),
            "qualification": self.qualification.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AuthorSkillBundle":
        qualification = payload.get("qualification", {})
        return cls(
            author_id=str(payload.get("author_id", "")),
            skill_name=str(payload.get("skill_name", "")),
            release_status=str(payload.get("release_status", "draft")),
            source_work_ids=tuple(str(item) for item in payload.get("source_work_ids", []) if str(item).strip()),
            chapter_ids=tuple(str(item) for item in payload.get("chapter_ids", []) if str(item).strip()),
            candidate_trait_count=int(payload.get("candidate_trait_count", 0)),
            validated_author_trait_count=int(payload.get("validated_author_trait_count", 0)),
            narrative_policy_count=int(payload.get("narrative_policy_count", 0)),
            generated_files=tuple(str(item) for item in payload.get("generated_files", []) if str(item).strip()),
            limitations=tuple(str(item) for item in payload.get("limitations", []) if str(item).strip()),
            qualification=SkillQualification.from_dict(qualification if isinstance(qualification, dict) else {}),
            schema_version=str(payload.get("schema_version", "2.0")),
        )
