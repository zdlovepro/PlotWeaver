"""第八模块使用的无原文生成规划契约。

这些合同在叙事图谱建立之后生成，用于安排新正文的事件依赖、章节窗口和线索
生命周期。它们不是第二模块从原作提取的分层大纲，也不保存来源原文。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _required(value: str, field_name: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


def _cycle_exists(events: dict[str, "EventGraphEvent"]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(event_id: str) -> bool:
        if event_id in visiting:
            return True
        if event_id in visited:
            return False
        visiting.add(event_id)
        if any(visit(parent) for parent in events[event_id].causal_parent_event_ids):
            return True
        visiting.remove(event_id)
        visited.add(event_id)
        return False

    return any(visit(event_id) for event_id in events if event_id not in visited)


@dataclass(frozen=True)
class EventGraphEvent:
    """One cross-chapter event and its structural obligations."""

    event_id: str
    chapter_id: str
    chapter_index: int
    event_order: int
    summary: str
    action: str
    participant_ids: tuple[str, ...]
    precondition_fact_ids: tuple[str, ...]
    outcome_fact_ids: tuple[str, ...]
    cost_fact_ids: tuple[str, ...] = field(default_factory=tuple)
    causal_parent_event_ids: tuple[str, ...] = field(default_factory=tuple)
    causal_child_event_ids: tuple[str, ...] = field(default_factory=tuple)
    opens_foreshadow_ids: tuple[str, ...] = field(default_factory=tuple)
    resolves_foreshadow_ids: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.event_id, "event_id")
        _required(self.chapter_id, "event chapter_id")
        if self.chapter_index < 0 or self.event_order < 0:
            raise ValueError("event positions must be non-negative")
        _required(self.summary, "event summary")
        _required(self.action, "event action")
        if not self.participant_ids:
            raise ValueError("event graph event requires participants")
        for field_name, values in (
            ("event participant_ids", self.participant_ids),
            ("event precondition_fact_ids", self.precondition_fact_ids),
            ("event outcome_fact_ids", self.outcome_fact_ids),
            ("event cost_fact_ids", self.cost_fact_ids),
            ("event causal_parent_event_ids", self.causal_parent_event_ids),
            ("event causal_child_event_ids", self.causal_child_event_ids),
            ("event opens_foreshadow_ids", self.opens_foreshadow_ids),
            ("event resolves_foreshadow_ids", self.resolves_foreshadow_ids),
        ):
            _unique(values, field_name)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), **{
            "participant_ids": list(self.participant_ids),
            "precondition_fact_ids": list(self.precondition_fact_ids),
            "outcome_fact_ids": list(self.outcome_fact_ids),
            "cost_fact_ids": list(self.cost_fact_ids),
            "causal_parent_event_ids": list(self.causal_parent_event_ids),
            "causal_child_event_ids": list(self.causal_child_event_ids),
            "opens_foreshadow_ids": list(self.opens_foreshadow_ids),
            "resolves_foreshadow_ids": list(self.resolves_foreshadow_ids),
        }}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventGraphEvent":
        fields = (
            "participant_ids", "precondition_fact_ids", "outcome_fact_ids", "cost_fact_ids",
            "causal_parent_event_ids", "causal_child_event_ids", "opens_foreshadow_ids", "resolves_foreshadow_ids",
        )
        values = {
            field: tuple(str(item) for item in payload.get(field, []) if str(item).strip())
            for field in fields
        }
        return cls(
            event_id=str(payload.get("event_id", "")), chapter_id=str(payload.get("chapter_id", "")),
            chapter_index=int(payload.get("chapter_index", -1)), event_order=int(payload.get("event_order", -1)),
            summary=str(payload.get("summary", "")), action=str(payload.get("action", "")), **values,
        )


@dataclass(frozen=True)
class StoryStage:
    """A bounded story phase, equivalent to a partial volume when sampled."""

    stage_id: str
    order: int
    chapter_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    entry_event_ids: tuple[str, ...]
    exit_event_ids: tuple[str, ...]
    summary: str = ""
    open_foreshadow_ids: tuple[str, ...] = field(default_factory=tuple)
    resolve_foreshadow_ids: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.stage_id, "stage_id")
        if self.order < 0:
            raise ValueError("stage order must be non-negative")
        if not self.chapter_ids or not self.event_ids:
            raise ValueError("story stage requires chapters and events")
        for field_name, values in (
            ("stage chapter_ids", self.chapter_ids), ("stage event_ids", self.event_ids),
            ("stage entry_event_ids", self.entry_event_ids), ("stage exit_event_ids", self.exit_event_ids),
            ("stage open_foreshadow_ids", self.open_foreshadow_ids),
            ("stage resolve_foreshadow_ids", self.resolve_foreshadow_ids),
        ):
            _unique(values, field_name)
        if not set(self.entry_event_ids).issubset(self.event_ids) or not set(self.exit_event_ids).issubset(self.event_ids):
            raise ValueError("stage boundary events must belong to the stage")

    def to_dict(self) -> dict[str, Any]:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StoryStage":
        keys = ("chapter_ids", "event_ids", "entry_event_ids", "exit_event_ids", "open_foreshadow_ids", "resolve_foreshadow_ids")
        values = {key: tuple(str(item) for item in payload.get(key, []) if str(item).strip()) for key in keys}
        return cls(
            stage_id=str(payload.get("stage_id", "")), order=int(payload.get("order", -1)),
            summary=str(payload.get("summary", "")), **values,
        )


@dataclass(frozen=True)
class EventGraph:
    """The dependency graph used to unlock chapter-level work."""

    author_id: str
    work_id: str
    base_graph_fingerprint: str
    chapter_ids: tuple[str, ...]
    events: tuple[EventGraphEvent, ...]
    schema_version: str = "1.0"

    def validate(self) -> None:
        _required(self.author_id, "event graph author_id")
        _required(self.work_id, "event graph work_id")
        _required(self.base_graph_fingerprint, "event graph base_graph_fingerprint")
        if not self.chapter_ids or not self.events:
            raise ValueError("event graph requires chapters and events")
        _unique(self.chapter_ids, "event graph chapter_ids")
        event_ids = tuple(item.event_id for item in self.events)
        _unique(event_ids, "event graph event_ids")
        events = {item.event_id: item for item in self.events}
        for event in self.events:
            event.validate()
            if event.chapter_id not in self.chapter_ids:
                raise ValueError("event graph event belongs to an unknown chapter")
            references = set(event.causal_parent_event_ids) | set(event.causal_child_event_ids)
            if event.event_id in references or not references.issubset(events):
                raise ValueError("event graph causal reference is invalid")
        for event in self.events:
            for parent_id in event.causal_parent_event_ids:
                if event.event_id not in events[parent_id].causal_child_event_ids:
                    raise ValueError("event graph parent/child edges must be reciprocal")
        if _cycle_exists(events):
            raise ValueError("event graph causal dependencies contain a cycle")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "author_id": self.author_id, "work_id": self.work_id,
            "base_graph_fingerprint": self.base_graph_fingerprint, "chapter_ids": list(self.chapter_ids),
            "events": [item.to_dict() for item in self.events],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventGraph":
        return cls(
            author_id=str(payload.get("author_id", "")), work_id=str(payload.get("work_id", "")),
            base_graph_fingerprint=str(payload.get("base_graph_fingerprint", "")),
            chapter_ids=tuple(str(item) for item in payload.get("chapter_ids", []) if str(item).strip()),
            events=tuple(EventGraphEvent.from_dict(item) for item in payload.get("events", []) if isinstance(item, dict)),
            schema_version=str(payload.get("schema_version", "1.0")),
        )


@dataclass(frozen=True)
class WorkStoryPlan:
    """A stage plan that gives long-form generation a bounded direction."""

    author_id: str
    work_id: str
    base_graph_fingerprint: str
    stages: tuple[StoryStage, ...]
    schema_version: str = "1.0"

    def validate(self, event_graph: EventGraph | None = None) -> None:
        _required(self.author_id, "story plan author_id")
        _required(self.work_id, "story plan work_id")
        _required(self.base_graph_fingerprint, "story plan base_graph_fingerprint")
        if not self.stages:
            raise ValueError("story plan requires stages")
        ids = tuple(item.stage_id for item in self.stages)
        _unique(ids, "story plan stage_ids")
        if tuple(item.order for item in self.stages) != tuple(range(len(self.stages))):
            raise ValueError("story plan stage orders must be consecutive")
        chapters: list[str] = []
        event_ids: list[str] = []
        for stage in self.stages:
            stage.validate()
            chapters.extend(stage.chapter_ids)
            event_ids.extend(stage.event_ids)
        _unique(tuple(chapters), "story plan chapter_ids")
        _unique(tuple(event_ids), "story plan event_ids")
        if event_graph is not None:
            event_graph.validate()
            if self.author_id != event_graph.author_id or self.work_id != event_graph.work_id or self.base_graph_fingerprint != event_graph.base_graph_fingerprint:
                raise ValueError("story plan belongs to a different event graph")
            if tuple(chapters) != event_graph.chapter_ids or set(event_ids) != {item.event_id for item in event_graph.events}:
                raise ValueError("story plan must cover every graph chapter and event exactly once")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "author_id": self.author_id, "work_id": self.work_id,
            "base_graph_fingerprint": self.base_graph_fingerprint,
            "stages": [item.to_dict() for item in self.stages],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkStoryPlan":
        return cls(
            author_id=str(payload.get("author_id", "")), work_id=str(payload.get("work_id", "")),
            base_graph_fingerprint=str(payload.get("base_graph_fingerprint", "")),
            stages=tuple(StoryStage.from_dict(item) for item in payload.get("stages", []) if isinstance(item, dict)),
            schema_version=str(payload.get("schema_version", "1.0")),
        )


@dataclass(frozen=True)
class ChapterContract:
    """The events and lifecycle work unlocked for one generated chapter."""

    chapter_id: str
    chapter_index: int
    source_hash: str
    stage_id: str
    base_graph_fingerprint: str
    active_event_ids: tuple[str, ...]
    required_prior_event_ids: tuple[str, ...]
    entry_fact_ids: tuple[str, ...]
    exit_fact_ids: tuple[str, ...]
    open_foreshadow_ids: tuple[str, ...]
    resolve_foreshadow_ids: tuple[str, ...]
    forbidden_event_ids: tuple[str, ...]
    schema_version: str = "1.0"

    def validate(self, event_graph: EventGraph | None = None) -> None:
        _required(self.chapter_id, "chapter contract chapter_id")
        _required(self.source_hash, "chapter contract source_hash")
        _required(self.stage_id, "chapter contract stage_id")
        _required(self.base_graph_fingerprint, "chapter contract base_graph_fingerprint")
        if self.chapter_index < 0 or not self.active_event_ids:
            raise ValueError("chapter contract requires a chapter index and active events")
        for field_name, values in (
            ("chapter contract active_event_ids", self.active_event_ids),
            ("chapter contract required_prior_event_ids", self.required_prior_event_ids),
            ("chapter contract entry_fact_ids", self.entry_fact_ids), ("chapter contract exit_fact_ids", self.exit_fact_ids),
            ("chapter contract open_foreshadow_ids", self.open_foreshadow_ids),
            ("chapter contract resolve_foreshadow_ids", self.resolve_foreshadow_ids),
            ("chapter contract forbidden_event_ids", self.forbidden_event_ids),
        ):
            _unique(values, field_name)
        if set(self.active_event_ids) & set(self.forbidden_event_ids):
            raise ValueError("chapter contract cannot activate and forbid the same event")
        if event_graph is not None:
            event_graph.validate()
            if self.base_graph_fingerprint != event_graph.base_graph_fingerprint:
                raise ValueError("chapter contract belongs to a different event graph")
            events = {item.event_id: item for item in event_graph.events}
            if not (set(self.active_event_ids) | set(self.required_prior_event_ids) | set(self.forbidden_event_ids)).issubset(events):
                raise ValueError("chapter contract refers to an unknown event")
            if any(events[event_id].chapter_id != self.chapter_id for event_id in self.active_event_ids):
                raise ValueError("chapter contract may activate only its own chapter events")
            if any(events[event_id].chapter_index >= self.chapter_index for event_id in self.required_prior_event_ids):
                raise ValueError("chapter contract prior event is not prior")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, tuple):
                result[key] = list(value)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterContract":
        fields = (
            "active_event_ids", "required_prior_event_ids", "entry_fact_ids", "exit_fact_ids",
            "open_foreshadow_ids", "resolve_foreshadow_ids", "forbidden_event_ids",
        )
        values = {field: tuple(str(item) for item in payload.get(field, []) if str(item).strip()) for field in fields}
        return cls(
            chapter_id=str(payload.get("chapter_id", "")), chapter_index=int(payload.get("chapter_index", -1)),
            source_hash=str(payload.get("source_hash", "")), stage_id=str(payload.get("stage_id", "")),
            base_graph_fingerprint=str(payload.get("base_graph_fingerprint", "")),
            schema_version=str(payload.get("schema_version", "1.0")), **values,
        )
