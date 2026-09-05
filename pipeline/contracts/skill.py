"""Contracts for the final abstract author skill and prompt program."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TEMPLATE_LEVELS = frozenset({"micro", "scene", "chapter", "thread", "arc"})
PROMPT_STAGES = frozenset({"planning", "drafting", "validation", "repair"})


@dataclass(frozen=True)
class Template:
    template_id: str
    level: str
    name: str
    applicability: tuple[str, ...]
    role_slots: tuple[str, ...]
    beat_sequence: tuple[str, ...]
    state_change_signatures: tuple[str, ...]
    variation_axes: tuple[str, ...]
    anti_patterns: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    support_count: int
    confidence: float

    def validate(self) -> None:
        if not str(self.template_id).strip() or not str(self.name).strip():
            raise ValueError("template requires id and name")
        if self.level not in TEMPLATE_LEVELS:
            raise ValueError(f"unsupported template level: {self.level}")
        if not self.applicability or not self.beat_sequence or not self.evidence_ids:
            raise ValueError("template requires applicability, beats and evidence")
        if self.support_count != len(set(self.evidence_ids)):
            raise ValueError("template support_count must equal unique evidence_ids")
        if not 0 < self.confidence <= 1:
            raise ValueError("template confidence must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Template":
        tuple_fields = ("applicability", "role_slots", "beat_sequence", "state_change_signatures", "variation_axes", "anti_patterns", "evidence_ids")
        values = {field: tuple(str(item) for item in payload.get(field, []) if str(item).strip()) for field in tuple_fields}
        return cls(
            template_id=str(payload.get("template_id", "")),
            level=str(payload.get("level", "")),
            name=str(payload.get("name", "")),
            support_count=int(payload.get("support_count", 0)),
            confidence=float(payload.get("confidence", 0)),
            **values,
        )


@dataclass(frozen=True)
class AuthorProfile:
    profile_id: str
    author_id: str
    source_work_ids: tuple[str, ...]
    template_ids: tuple[str, ...]
    narrative_rules: tuple[str, ...]
    style_targets: tuple[str, ...]
    confidence: float
    schema_version: str = "2.0"

    def validate(self) -> None:
        if not str(self.profile_id).strip() or not str(self.author_id).strip():
            raise ValueError("profile requires id and author_id")
        if not self.source_work_ids or not self.template_ids:
            raise ValueError("profile requires source works and templates")
        if not 0 < self.confidence <= 1:
            raise ValueError("profile confidence must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AuthorProfile":
        return cls(
            profile_id=str(payload.get("profile_id", "")),
            author_id=str(payload.get("author_id", "")),
            source_work_ids=tuple(str(item) for item in payload.get("source_work_ids", []) if str(item).strip()),
            template_ids=tuple(str(item) for item in payload.get("template_ids", []) if str(item).strip()),
            narrative_rules=tuple(str(item) for item in payload.get("narrative_rules", []) if str(item).strip()),
            style_targets=tuple(str(item) for item in payload.get("style_targets", []) if str(item).strip()),
            confidence=float(payload.get("confidence", 0)),
            schema_version=str(payload.get("schema_version", "2.0")),
        )


@dataclass(frozen=True)
class PromptProgram:
    program_id: str
    author_profile_id: str
    stage: str
    required_inputs: tuple[str, ...]
    output_contract: str
    rules: tuple[str, ...]
    evaluation_metrics: tuple[str, ...]
    version: str

    def validate(self) -> None:
        if not str(self.program_id).strip() or not str(self.author_profile_id).strip():
            raise ValueError("prompt program requires program_id and author_profile_id")
        if self.stage not in PROMPT_STAGES:
            raise ValueError(f"unsupported prompt stage: {self.stage}")
        if not self.required_inputs or not str(self.output_contract).strip() or not self.rules:
            raise ValueError("prompt program must define inputs, output contract and rules")
        if not str(self.version).strip():
            raise ValueError("prompt program version must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptProgram":
        return cls(
            program_id=str(payload.get("program_id", "")),
            author_profile_id=str(payload.get("author_profile_id", "")),
            stage=str(payload.get("stage", "")),
            required_inputs=tuple(str(item) for item in payload.get("required_inputs", []) if str(item).strip()),
            output_contract=str(payload.get("output_contract", "")),
            rules=tuple(str(item) for item in payload.get("rules", []) if str(item).strip()),
            evaluation_metrics=tuple(str(item) for item in payload.get("evaluation_metrics", []) if str(item).strip()),
            version=str(payload.get("version", "")),
        )
