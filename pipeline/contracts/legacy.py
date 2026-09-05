"""Compatibility records used by the pre-contract extraction pipeline.

They are intentionally isolated here so new stages can migrate to the
evidence-first contracts without breaking the current command path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ChapterRecord:
    chapter_id: str
    author_id: str
    work_id: str
    chapter_no: int
    source_chapter_no: int
    title: str
    text: str
    char_count: int
    source_hash: str
    previous_chapter_id: str = ""
    next_chapter_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChapterState:
    chapter_id: str
    chapter_function: str = ""
    entities: list[dict[str, Any]] = field(default_factory=list)
    event_chain: list[dict[str, str]] = field(default_factory=list)
    scene_beats: list[dict[str, Any]] = field(default_factory=list)
    character_deltas: list[dict[str, Any]] = field(default_factory=list)
    relation_deltas: list[dict[str, Any]] = field(default_factory=list)
    temporal_spatial_state: dict[str, Any] = field(default_factory=dict)
    cultivation_state: dict[str, Any] = field(default_factory=dict)
    information_state: dict[str, Any] = field(default_factory=dict)
    pacing_curve: dict[str, Any] = field(default_factory=dict)
    prose_metrics: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, str]] = field(default_factory=list)
    extraction_mode: str = ""
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], chapter_id: str) -> "ChapterState":
        fields = {item.name for item in cls.__dataclass_fields__.values()}
        normalized = {key: value for key, value in dict(payload or {}).items() if key in fields}
        normalized["chapter_id"] = chapter_id
        for key in ("entities", "event_chain", "scene_beats", "character_deltas", "relation_deltas", "evidence"):
            value = normalized.get(key, [])
            normalized[key] = value if isinstance(value, list) else []
        for key in ("temporal_spatial_state", "cultivation_state", "information_state", "pacing_curve", "prose_metrics"):
            value = normalized.get(key, {})
            normalized[key] = value if isinstance(value, dict) else {}
        return cls(**normalized)


@dataclass
class StoryState:
    chapter_id: str
    characters: dict[str, dict[str, Any]] = field(default_factory=dict)
    relations: dict[str, dict[str, Any]] = field(default_factory=dict)
    locations: dict[str, str] = field(default_factory=dict)
    temporal_anchor: str = ""
    cultivation: dict[str, Any] = field(default_factory=dict)
    open_foreshadows: list[dict[str, Any]] = field(default_factory=list)
    entity_registry: dict[str, dict[str, Any]] = field(default_factory=dict)
    alias_index: dict[str, str] = field(default_factory=dict)
    knowledge: dict[str, list[str]] = field(default_factory=dict)
    last_scene: dict[str, Any] = field(default_factory=dict)
    issues: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
