"""Contracts for high-level, evidence-supported author style constraints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


STYLE_FAMILIES = frozenset({
    "sentence_rhythm", "paragraph_pacing", "dialogue", "transition",
    "focalization", "imagery", "emotion",
})


def _required(value: str, field_name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


@dataclass(frozen=True)
class StyleBaseline:
    metric_id: str
    mean: float
    minimum: float
    maximum: float
    unit: str

    def validate(self) -> None:
        _required(self.metric_id, "style baseline metric_id")
        if self.minimum < 0 or self.maximum < 0 or self.mean < 0:
            raise ValueError("style baseline values must be non-negative")
        if self.minimum > self.maximum or not self.minimum <= self.mean <= self.maximum:
            raise ValueError("style baseline must satisfy minimum <= mean <= maximum")
        if self.unit not in {"count", "chars", "ratio", "mean_chars"}:
            raise ValueError(f"unsupported style baseline unit: {self.unit}")
        if self.unit == "ratio" and self.maximum > 1:
            raise ValueError("style baseline ratio must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StyleBaseline":
        return cls(
            metric_id=str(payload.get("metric_id", "")),
            mean=float(payload.get("mean", -1)),
            minimum=float(payload.get("minimum", -1)),
            maximum=float(payload.get("maximum", -1)),
            unit=str(payload.get("unit", "")),
        )


@dataclass(frozen=True)
class StyleConstraint:
    constraint_id: str
    family: str
    rule: str
    application: str
    avoid: str
    evidence_chapter_ids: tuple[str, ...]
    support_count: int
    confidence: float
    evidence_metric_ids: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.constraint_id, "style constraint_id")
        if self.family not in STYLE_FAMILIES:
            raise ValueError(f"unsupported style family: {self.family}")
        _required(self.rule, "style constraint rule")
        _required(self.application, "style constraint application")
        _required(self.avoid, "style constraint avoid")
        if len(self.evidence_chapter_ids) < 2:
            raise ValueError("style constraint needs at least two evidence chapters")
        _unique(self.evidence_chapter_ids, "style constraint evidence chapters")
        if self.support_count != len(self.evidence_chapter_ids):
            raise ValueError("style constraint support_count must equal evidence chapter count")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("style constraint confidence must be in (0, 1]")
        if not self.evidence_metric_ids:
            raise ValueError("style constraint needs at least one evidence metric")
        _unique(self.evidence_metric_ids, "style constraint evidence metrics")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "evidence_chapter_ids": list(self.evidence_chapter_ids),
            "evidence_metric_ids": list(self.evidence_metric_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StyleConstraint":
        return cls(
            constraint_id=str(payload.get("constraint_id", "")),
            family=str(payload.get("family", "")),
            rule=str(payload.get("rule", "")),
            application=str(payload.get("application", "")),
            avoid=str(payload.get("avoid", "")),
            evidence_chapter_ids=tuple(str(item) for item in payload.get("evidence_chapter_ids", []) if str(item).strip()),
            support_count=int(payload.get("support_count", 0)),
            confidence=float(payload.get("confidence", 0)),
            evidence_metric_ids=tuple(str(item) for item in payload.get("evidence_metric_ids", []) if str(item).strip()),
        )


@dataclass(frozen=True)
class AuthorStyleProfile:
    author_id: str
    work_id: str
    chapter_ids: tuple[str, ...]
    baselines: tuple[StyleBaseline, ...]
    constraints: tuple[StyleConstraint, ...]
    guardrails: tuple[str, ...]
    schema_version: str = "1.0"

    def validate(self) -> None:
        _required(self.author_id, "style profile author_id")
        _required(self.work_id, "style profile work_id")
        if not self.chapter_ids or not self.baselines or not self.constraints or not self.guardrails:
            raise ValueError("style profile needs chapters, baselines, constraints and guardrails")
        _unique(self.chapter_ids, "style profile chapter_ids")
        _unique(tuple(item.metric_id for item in self.baselines), "style profile baseline metrics")
        _unique(tuple(item.constraint_id for item in self.constraints), "style profile constraints")
        _unique(self.guardrails, "style profile guardrails")
        source_ids = set(self.chapter_ids)
        for item in self.baselines:
            item.validate()
        for item in self.constraints:
            item.validate()
            if not set(item.evidence_chapter_ids) <= source_ids:
                raise ValueError("style constraint references unknown chapter")

    def to_dict(self) -> dict[str, Any]:
        return {
            "author_id": self.author_id,
            "work_id": self.work_id,
            "chapter_ids": list(self.chapter_ids),
            "baselines": [item.to_dict() for item in self.baselines],
            "constraints": [item.to_dict() for item in self.constraints],
            "guardrails": list(self.guardrails),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AuthorStyleProfile":
        return cls(
            author_id=str(payload.get("author_id", "")),
            work_id=str(payload.get("work_id", "")),
            chapter_ids=tuple(str(item) for item in payload.get("chapter_ids", []) if str(item).strip()),
            baselines=tuple(StyleBaseline.from_dict(item) for item in payload.get("baselines", []) if isinstance(item, dict)),
            constraints=tuple(StyleConstraint.from_dict(item) for item in payload.get("constraints", []) if isinstance(item, dict)),
            guardrails=tuple(str(item) for item in payload.get("guardrails", []) if str(item).strip()),
            schema_version=str(payload.get("schema_version", "1.0")),
        )
