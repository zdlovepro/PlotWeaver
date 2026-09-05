"""第二模块的分层大纲契约。该层不包含原文或事实表。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


OUTLINE_SCHEMA_VERSION = "1.0"
OUTLINE_LEVELS = frozenset({"story_arc", "volume", "book"})
OUTLINE_PROFILES = frozenset({"short_validation", "full_book"})


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
class CharacterArcOutline:
    character_name: str
    entry_state: str
    pursuit: str
    key_choices: tuple[str, ...]
    change: str
    exit_state: str

    def validate(self) -> None:
        _required(self.character_name, "character arc name")
        _required(self.entry_state, "character arc entry_state")
        _required(self.change, "character arc change")
        _required(self.exit_state, "character arc exit_state")
        _unique(self.key_choices, "character arc key_choices")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["key_choices"] = list(self.key_choices)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CharacterArcOutline":
        return cls(
            character_name=str(payload.get("character_name", "")),
            entry_state=str(payload.get("entry_state", "")),
            pursuit=str(payload.get("pursuit", "")),
            key_choices=_strings(payload, "key_choices"),
            change=str(payload.get("change", "")),
            exit_state=str(payload.get("exit_state", "")),
        )


@dataclass(frozen=True)
class OutlineNode:
    outline_id: str
    level: str
    order: int
    title: str
    chapter_ids: tuple[str, ...]
    child_outline_ids: tuple[str, ...]
    summary: str
    opening_situation: str
    central_goal: str
    central_conflict: str
    causal_chain: tuple[str, ...]
    turning_points: tuple[str, ...]
    ending_change: str
    open_threads: tuple[str, ...]
    character_arcs: tuple[CharacterArcOutline, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.outline_id, "outline_id")
        if self.level not in OUTLINE_LEVELS:
            raise ValueError(f"unsupported outline level: {self.level}")
        if self.order < 0:
            raise ValueError("outline order must be non-negative")
        _required(self.title, "outline title")
        _required(self.summary, "outline summary")
        _required(self.opening_situation, "outline opening_situation")
        _required(self.central_conflict, "outline central_conflict")
        _required(self.ending_change, "outline ending_change")
        if not self.chapter_ids or not self.causal_chain:
            raise ValueError("outline node requires chapters and causal chain")
        for name, values in (
            ("outline chapter_ids", self.chapter_ids),
            ("outline child_outline_ids", self.child_outline_ids),
            ("outline causal_chain", self.causal_chain),
            ("outline turning_points", self.turning_points),
            ("outline open_threads", self.open_threads),
        ):
            _unique(values, name)
        for item in self.character_arcs:
            item.validate()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "chapter_ids", "child_outline_ids", "causal_chain", "turning_points", "open_threads"
        ):
            result[key] = list(result[key])
        result["character_arcs"] = [item.to_dict() for item in self.character_arcs]
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OutlineNode":
        return cls(
            outline_id=str(payload.get("outline_id", "")),
            level=str(payload.get("level", "")),
            order=int(payload.get("order", -1)),
            title=str(payload.get("title", "")),
            chapter_ids=_strings(payload, "chapter_ids"),
            child_outline_ids=_strings(payload, "child_outline_ids"),
            summary=str(payload.get("summary", "")),
            opening_situation=str(payload.get("opening_situation", "")),
            central_goal=str(payload.get("central_goal", "")),
            central_conflict=str(payload.get("central_conflict", "")),
            causal_chain=_strings(payload, "causal_chain"),
            turning_points=_strings(payload, "turning_points"),
            ending_change=str(payload.get("ending_change", "")),
            open_threads=_strings(payload, "open_threads"),
            character_arcs=tuple(
                CharacterArcOutline.from_dict(item)
                for item in payload.get("character_arcs", []) if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True)
class HierarchicalOutlineBundle:
    author_id: str
    work_id: str
    profile: str
    source_chapter_ids: tuple[str, ...]
    source_synopsis_hashes: tuple[str, ...]
    nodes: tuple[OutlineNode, ...]
    root_outline_ids: tuple[str, ...]
    aggregation_ceiling: str
    schema_version: str = OUTLINE_SCHEMA_VERSION

    def validate(self) -> None:
        _required(self.author_id, "outline author_id")
        _required(self.work_id, "outline work_id")
        if self.profile not in OUTLINE_PROFILES:
            raise ValueError(f"unsupported outline profile: {self.profile}")
        if self.aggregation_ceiling not in OUTLINE_LEVELS:
            raise ValueError(f"unsupported aggregation ceiling: {self.aggregation_ceiling}")
        if not self.source_chapter_ids or not self.nodes or not self.root_outline_ids:
            raise ValueError("hierarchical outline bundle is incomplete")
        _unique(self.source_chapter_ids, "outline source_chapter_ids")
        node_ids = tuple(item.outline_id for item in self.nodes)
        _unique(node_ids, "outline node_ids")
        if not set(self.root_outline_ids).issubset(node_ids):
            raise ValueError("outline root refers to an unknown node")
        known = set(node_ids)
        for node in self.nodes:
            node.validate()
            if not set(node.child_outline_ids).issubset(known):
                raise ValueError("outline node refers to an unknown child")
            if not set(node.chapter_ids).issubset(self.source_chapter_ids):
                raise ValueError("outline node refers to an unknown chapter")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "author_id": self.author_id,
            "work_id": self.work_id,
            "profile": self.profile,
            "source_chapter_ids": list(self.source_chapter_ids),
            "source_synopsis_hashes": list(self.source_synopsis_hashes),
            "nodes": [item.to_dict() for item in self.nodes],
            "root_outline_ids": list(self.root_outline_ids),
            "aggregation_ceiling": self.aggregation_ceiling,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HierarchicalOutlineBundle":
        return cls(
            author_id=str(payload.get("author_id", "")),
            work_id=str(payload.get("work_id", "")),
            profile=str(payload.get("profile", "")),
            source_chapter_ids=_strings(payload, "source_chapter_ids"),
            source_synopsis_hashes=_strings(payload, "source_synopsis_hashes"),
            nodes=tuple(
                OutlineNode.from_dict(item)
                for item in payload.get("nodes", []) if isinstance(item, dict)
            ),
            root_outline_ids=_strings(payload, "root_outline_ids"),
            aggregation_ceiling=str(payload.get("aggregation_ceiling", "")),
            schema_version=str(payload.get("schema_version", OUTLINE_SCHEMA_VERSION)),
        )

