"""第三模块的按需事实补全契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .source import SourceSpan


FACT_HYDRATION_SCHEMA_VERSION = "1.0"
FACT_PRIORITIES = frozenset({"core", "supporting"})
FACT_CATEGORIES = frozenset({
    "identity",
    "goal",
    "constraint",
    "action",
    "result",
    "state",
    "relationship",
    "time",
    "space",
    "resource",
    "knowledge",
    "causal_support",
    "unresolved_thread_support",
})
CERTAINTY_MODES = frozenset({"observed", "asserted", "believed", "rumored", "inferred"})


def _strings(payload: dict[str, Any], name: str) -> tuple[str, ...]:
    raw = payload.get(name, [])
    if not isinstance(raw, list):
        return ()
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _required(value: str, name: str) -> None:
    if not str(value or "").strip():
        raise ValueError(f"{name} must not be empty")


def _unique(values: tuple[str, ...], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True)
class FactRequirement:
    requirement_id: str
    outline_node_ids: tuple[str, ...]
    chapter_ids: tuple[str, ...]
    category: str
    question: str
    why_needed: str
    priority: str

    def validate(self) -> None:
        _required(self.requirement_id, "fact requirement_id")
        if self.category not in FACT_CATEGORIES:
            raise ValueError(f"unsupported fact requirement category: {self.category}")
        if self.priority not in FACT_PRIORITIES:
            raise ValueError(f"unsupported fact requirement priority: {self.priority}")
        if not self.outline_node_ids or not self.chapter_ids:
            raise ValueError("fact requirement must point to outline nodes and chapters")
        _required(self.question, "fact requirement question")
        _required(self.why_needed, "fact requirement why_needed")
        _unique(self.outline_node_ids, "fact requirement outline_node_ids")
        _unique(self.chapter_ids, "fact requirement chapter_ids")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["outline_node_ids"] = list(self.outline_node_ids)
        result["chapter_ids"] = list(self.chapter_ids)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactRequirement":
        return cls(
            requirement_id=str(payload.get("requirement_id", "")),
            outline_node_ids=_strings(payload, "outline_node_ids"),
            chapter_ids=_strings(payload, "chapter_ids"),
            category=str(payload.get("category", "")),
            question=str(payload.get("question", "")),
            why_needed=str(payload.get("why_needed", "")),
            priority=str(payload.get("priority", "")),
        )


@dataclass(frozen=True)
class HydratedFact:
    fact_id: str
    requirement_ids: tuple[str, ...]
    outline_node_ids: tuple[str, ...]
    chapter_id: str
    category: str
    statement: str
    subject_mentions: tuple[str, ...]
    related_mentions: tuple[str, ...]
    certainty: str
    evidence: tuple[SourceSpan, ...]
    confidence: float = 1.0

    def validate(self, source_text: str | None = None) -> None:
        _required(self.fact_id, "hydrated fact_id")
        _required(self.chapter_id, "hydrated fact chapter_id")
        _required(self.statement, "hydrated fact statement")
        if self.category not in FACT_CATEGORIES:
            raise ValueError(f"unsupported hydrated fact category: {self.category}")
        if self.certainty not in CERTAINTY_MODES:
            raise ValueError(f"unsupported hydrated fact certainty: {self.certainty}")
        if not self.requirement_ids or not self.outline_node_ids or not self.evidence:
            raise ValueError("hydrated fact must retain requirement, outline and evidence links")
        for name, values in (
            ("hydrated fact requirement_ids", self.requirement_ids),
            ("hydrated fact outline_node_ids", self.outline_node_ids),
            ("hydrated fact subject_mentions", self.subject_mentions),
            ("hydrated fact related_mentions", self.related_mentions),
        ):
            _unique(values, name)
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("hydrated fact confidence must be in (0, 1]")
        for span in self.evidence:
            span.validate(source_text)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "requirement_ids", "outline_node_ids", "subject_mentions", "related_mentions"
        ):
            result[key] = list(result[key])
        result["evidence"] = [item.to_dict() for item in self.evidence]
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HydratedFact":
        return cls(
            fact_id=str(payload.get("fact_id", "")),
            requirement_ids=_strings(payload, "requirement_ids"),
            outline_node_ids=_strings(payload, "outline_node_ids"),
            chapter_id=str(payload.get("chapter_id", "")),
            category=str(payload.get("category", "")),
            statement=str(payload.get("statement", "")),
            subject_mentions=_strings(payload, "subject_mentions"),
            related_mentions=_strings(payload, "related_mentions"),
            certainty=str(payload.get("certainty", "")),
            evidence=tuple(
                SourceSpan.from_dict(item)
                for item in payload.get("evidence", []) if isinstance(item, dict)
            ),
            confidence=float(payload.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class FactHydrationBundle:
    author_id: str
    work_id: str
    outline_fingerprint: str
    requirements: tuple[FactRequirement, ...]
    facts: tuple[HydratedFact, ...]
    unanswered_requirement_ids: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = FACT_HYDRATION_SCHEMA_VERSION

    def validate(self) -> None:
        _required(self.author_id, "fact bundle author_id")
        _required(self.work_id, "fact bundle work_id")
        _required(self.outline_fingerprint, "fact bundle outline_fingerprint")
        if not self.requirements:
            raise ValueError("fact hydration requires outline-driven requirements")
        requirement_ids = tuple(item.requirement_id for item in self.requirements)
        _unique(requirement_ids, "fact requirement_ids")
        fact_ids = tuple(item.fact_id for item in self.facts)
        _unique(fact_ids, "hydrated fact_ids")
        for item in self.requirements:
            item.validate()
        known = set(requirement_ids)
        for item in self.facts:
            item.validate()
            if not set(item.requirement_ids).issubset(known):
                raise ValueError("hydrated fact refers to an unknown requirement")
        if not set(self.unanswered_requirement_ids).issubset(known):
            raise ValueError("unknown unanswered fact requirement")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "author_id": self.author_id,
            "work_id": self.work_id,
            "outline_fingerprint": self.outline_fingerprint,
            "requirements": [item.to_dict() for item in self.requirements],
            "facts": [item.to_dict() for item in self.facts],
            "unanswered_requirement_ids": list(self.unanswered_requirement_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactHydrationBundle":
        return cls(
            author_id=str(payload.get("author_id", "")),
            work_id=str(payload.get("work_id", "")),
            outline_fingerprint=str(payload.get("outline_fingerprint", "")),
            requirements=tuple(
                FactRequirement.from_dict(item)
                for item in payload.get("requirements", []) if isinstance(item, dict)
            ),
            facts=tuple(
                HydratedFact.from_dict(item)
                for item in payload.get("facts", []) if isinstance(item, dict)
            ),
            unanswered_requirement_ids=_strings(payload, "unanswered_requirement_ids"),
            schema_version=str(payload.get("schema_version", FACT_HYDRATION_SCHEMA_VERSION)),
        )

