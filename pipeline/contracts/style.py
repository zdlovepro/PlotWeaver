"""Quantified style observations, deliberately separated from plot facts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .source import SourceSpan


STYLE_SCOPES = frozenset({"chapter", "scene", "paragraph", "dialogue"})


@dataclass(frozen=True)
class StyleMetric:
    metric_id: str
    value: float
    unit: str
    denominator: str = ""

    def validate(self) -> None:
        if not str(self.metric_id).strip():
            raise ValueError("style metric_id must not be empty")
        if self.value < 0:
            raise ValueError("style metric value must be non-negative")
        if self.unit not in {"count", "chars", "ratio", "mean_chars", "score_0_5"}:
            raise ValueError(f"unsupported style metric unit: {self.unit}")
        if self.unit == "ratio" and self.value > 1:
            raise ValueError("style ratio must be in [0, 1]")
        if self.unit == "score_0_5" and self.value > 5:
            raise ValueError("style score must be in [0, 5]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StyleMetric":
        return cls(
            metric_id=str(payload.get("metric_id", "")),
            value=float(payload.get("value", -1)),
            unit=str(payload.get("unit", "")),
            denominator=str(payload.get("denominator", "")),
        )


@dataclass(frozen=True)
class StylePatternObservation:
    pattern_id: str
    scope: str
    family: str
    label: str
    frequency: int
    evidence: tuple[SourceSpan, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if not str(self.pattern_id).strip() or not str(self.family).strip() or not str(self.label).strip():
            raise ValueError("style pattern requires id, family and label")
        if self.scope not in STYLE_SCOPES:
            raise ValueError(f"unsupported style scope: {self.scope}")
        if self.frequency <= 0:
            raise ValueError("style pattern frequency must be positive")
        if not self.evidence:
            raise ValueError("style pattern must have source evidence")
        for span in self.evidence:
            span.validate()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "evidence": [span.to_dict() for span in self.evidence]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StylePatternObservation":
        return cls(
            pattern_id=str(payload.get("pattern_id", "")),
            scope=str(payload.get("scope", "")),
            family=str(payload.get("family", "")),
            label=str(payload.get("label", "")),
            frequency=int(payload.get("frequency", 0)),
            evidence=tuple(SourceSpan.from_dict(item) for item in payload.get("evidence", []) if isinstance(item, dict)),
        )


@dataclass(frozen=True)
class ChapterStyleCard:
    chapter_id: str
    source_hash: str
    metrics: tuple[StyleMetric, ...]
    observations: tuple[StylePatternObservation, ...]
    schema_version: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "source_hash": self.source_hash,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "observations": [item.to_dict() for item in self.observations],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterStyleCard":
        return cls(
            chapter_id=str(payload.get("chapter_id", "")),
            source_hash=str(payload.get("source_hash", "")),
            metrics=tuple(StyleMetric.from_dict(item) for item in payload.get("metrics", []) if isinstance(item, dict)),
            observations=tuple(StylePatternObservation.from_dict(item) for item in payload.get("observations", []) if isinstance(item, dict)),
            schema_version=str(payload.get("schema_version", "2.0")),
        )
