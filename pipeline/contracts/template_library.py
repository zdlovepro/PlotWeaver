"""Contracts for generic, evidence-supported narrative template libraries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TEMPLATE_LEVELS = frozenset({"micro", "event", "scene", "chapter", "macro"})


def _required(value: str, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True)
class NarrativeTemplate:
    template_id: str
    level: str
    purpose: str
    role_slots: tuple[str, ...]
    beat_sequence: tuple[str, ...]
    state_effects: tuple[str, ...]
    selection_tags: tuple[str, ...]
    variation_axes: tuple[str, ...]
    evidence_chapter_ids: tuple[str, ...]
    support_count: int
    confidence: float

    def validate(self) -> None:
        _required(self.template_id, "template_id")
        if self.level not in TEMPLATE_LEVELS:
            raise ValueError(f"unsupported template level: {self.level}")
        _required(self.purpose, "template purpose")
        if not self.beat_sequence or not self.selection_tags or not self.evidence_chapter_ids:
            raise ValueError("template needs beats, tags and evidence chapters")
        _unique(self.role_slots, "template role slots")
        _unique(self.beat_sequence, "template beats")
        _unique(self.state_effects, "template state effects")
        _unique(self.selection_tags, "template tags")
        _unique(self.variation_axes, "template variation axes")
        _unique(self.evidence_chapter_ids, "template evidence chapters")
        if self.support_count != len(self.evidence_chapter_ids):
            raise ValueError("template support_count must equal evidence chapter count")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("template confidence must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NarrativeTemplate":
        fields = ("role_slots", "beat_sequence", "state_effects", "selection_tags", "variation_axes", "evidence_chapter_ids")
        values = {field: tuple(str(item) for item in payload.get(field, []) if str(item).strip()) for field in fields}
        return cls(
            template_id=str(payload.get("template_id", "")),
            level=str(payload.get("level", "")),
            purpose=str(payload.get("purpose", "")),
            support_count=int(payload.get("support_count", 0)),
            confidence=float(payload.get("confidence", 0.0)),
            **values,
        )


@dataclass(frozen=True)
class NarrativeTemplateLibrary:
    author_id: str
    work_id: str
    chapter_ids: tuple[str, ...]
    continuity_schema_version: str
    templates: tuple[NarrativeTemplate, ...]
    schema_version: str = "1.0"

    def validate(self) -> None:
        _required(self.author_id, "template library author_id")
        _required(self.work_id, "template library work_id")
        if not self.chapter_ids:
            raise ValueError("template library needs source chapters")
        _unique(self.chapter_ids, "template library chapter_ids")
        if not self.templates:
            raise ValueError("template library needs templates")
        identifiers = [item.template_id for item in self.templates]
        _unique(tuple(identifiers), "template identifiers")
        for item in self.templates:
            item.validate()
            unknown = set(item.evidence_chapter_ids) - set(self.chapter_ids)
            if unknown:
                raise ValueError(f"template has unknown evidence chapters: {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "author_id": self.author_id,
            "work_id": self.work_id,
            "chapter_ids": list(self.chapter_ids),
            "continuity_schema_version": self.continuity_schema_version,
            "templates": [item.to_dict() for item in self.templates],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NarrativeTemplateLibrary":
        return cls(
            author_id=str(payload.get("author_id", "")),
            work_id=str(payload.get("work_id", "")),
            chapter_ids=tuple(str(item) for item in payload.get("chapter_ids", []) if str(item).strip()),
            continuity_schema_version=str(payload.get("continuity_schema_version", "")),
            templates=tuple(NarrativeTemplate.from_dict(item) for item in payload.get("templates", []) if isinstance(item, dict)),
            schema_version=str(payload.get("schema_version", "1.0")),
        )
