"""第一模块的章节剧情压缩契约。

只保存可追溯的局部梗概、完整章节梗概和章首/章末状态。事实表、实体关系、
跨章线程、功能分类与文风分析均不属于第一模块。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

SYNOPSIS_SCHEMA_VERSION = "5.0"


def _required(value: str, name: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _unique(values: Iterable[str], name: str) -> None:
    values = tuple(values)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


def _strings(payload: dict[str, Any], name: str) -> tuple[str, ...]:
    raw = payload.get(name, [])
    if not isinstance(raw, list):
        return ()
    return tuple(str(item).strip() for item in raw if str(item).strip())


@dataclass(frozen=True)
class LocalPlotSegment:
    segment_id: str
    order: int
    summary: str
    source_unit_ids: tuple[str, ...]

    def validate(self) -> None:
        _required(self.segment_id, "local segment_id")
        if self.order < 0:
            raise ValueError("local segment order must be non-negative")
        _required(self.summary, "local segment summary")
        if not self.source_unit_ids:
            raise ValueError("local segment requires source_unit_ids")
        _unique(self.source_unit_ids, "local segment source_unit_ids")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_unit_ids"] = list(self.source_unit_ids)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalPlotSegment":
        return cls(str(payload.get("segment_id", "")), int(payload.get("order", -1)),
                   str(payload.get("summary", "")), _strings(payload, "source_unit_ids"))


@dataclass(frozen=True)
class LocalSynopsis:
    window_id: str
    chapter_id: str
    reviewed_source_unit_ids: tuple[str, ...]
    segments: tuple[LocalPlotSegment, ...]

    def validate(self) -> None:
        _required(self.window_id, "window_id")
        _required(self.chapter_id, "chapter_id")
        if not self.reviewed_source_unit_ids:
            raise ValueError("local synopsis requires reviewed_source_unit_ids")
        _unique(self.reviewed_source_unit_ids, "reviewed_source_unit_ids")
        if len(self.segments) != 1:
            raise ValueError("each navigation block requires exactly one local plot segment")
        ids = tuple(item.segment_id for item in self.segments)
        _unique(ids, "local segment_ids")
        if tuple(item.order for item in self.segments) != tuple(range(len(self.segments))):
            raise ValueError("local segment orders must be consecutive")
        allowed = set(self.reviewed_source_unit_ids)
        for item in self.segments:
            item.validate()
            if not set(item.source_unit_ids).issubset(allowed):
                raise ValueError("local segment refers to a source unit outside its window")

    def to_dict(self) -> dict[str, Any]:
        return {"window_id": self.window_id, "chapter_id": self.chapter_id,
                "reviewed_source_unit_ids": list(self.reviewed_source_unit_ids),
                "segments": [item.to_dict() for item in self.segments]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalSynopsis":
        return cls(str(payload.get("window_id", "")), str(payload.get("chapter_id", "")),
                   _strings(payload, "reviewed_source_unit_ids"),
                   tuple(LocalPlotSegment.from_dict(item) for item in payload.get("segments", []) if isinstance(item, dict)))


@dataclass(frozen=True)
class BoundaryFrame:
    text: str
    source_unit_ids: tuple[str, ...]

    def validate(self, known_source_ids: set[str], name: str) -> None:
        _required(self.text, name)
        if not self.source_unit_ids:
            raise ValueError(f"{name} requires source_unit_ids")
        _unique(self.source_unit_ids, f"{name} source_unit_ids")
        if not set(self.source_unit_ids).issubset(known_source_ids):
            raise ValueError(f"{name} refers to unknown source units")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "source_unit_ids": list(self.source_unit_ids)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BoundaryFrame":
        return cls(str(payload.get("text", "")), _strings(payload, "source_unit_ids"))


@dataclass(frozen=True)
class ChapterSummary:
    """完整章节梗概及其确定性来源链。

    章节梗概允许由多个句子组成。第一模块不在局部梗概与章节梗概之间增加节点层，
    因而这里直接引用按原文顺序排列的局部梗概。
    """

    text: str
    local_segment_ids: tuple[str, ...]

    def validate(self, known_segment_ids: set[str], name: str = "chapter_summary") -> None:
        _required(self.text, name)
        if not self.local_segment_ids:
            raise ValueError(f"{name} requires local_segment_ids")
        _unique(self.local_segment_ids, f"{name} local_segment_ids")
        if not set(self.local_segment_ids).issubset(known_segment_ids):
            raise ValueError(f"{name} refers to an unknown local segment")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "local_segment_ids": list(self.local_segment_ids)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterSummary":
        return cls(str(payload.get("text", "")), _strings(payload, "local_segment_ids"))


@dataclass(frozen=True)
class ChapterSynopsis:
    chapter_id: str
    opening_state: BoundaryFrame
    chapter_summary: ChapterSummary
    ending_state: BoundaryFrame

    def validate(self, local_synopses: tuple[LocalSynopsis, ...] | None = None) -> None:
        _required(self.chapter_id, "chapter synopsis chapter_id")
        ordered_segments = tuple(segment.segment_id for local in (local_synopses or ()) for segment in local.segments)
        known_segments = set(ordered_segments) if local_synopses is not None else set(self.chapter_summary.local_segment_ids)
        self.chapter_summary.validate(known_segments)
        if local_synopses is not None:
            if self.chapter_summary.local_segment_ids != ordered_segments:
                raise ValueError("chapter summary must reference every local segment exactly once in source order")
            known_sources = {source_id for local in local_synopses for source_id in local.reviewed_source_unit_ids}
            self.opening_state.validate(known_sources, "opening_state")
            self.ending_state.validate(known_sources, "ending_state")

    def to_dict(self) -> dict[str, Any]:
        return {"chapter_id": self.chapter_id, "opening_state": self.opening_state.to_dict(),
                "chapter_summary": self.chapter_summary.to_dict(),
                "ending_state": self.ending_state.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterSynopsis":
        return cls(str(payload.get("chapter_id", "")), BoundaryFrame.from_dict(payload.get("opening_state", {})),
                   ChapterSummary.from_dict(payload.get("chapter_summary", {})),
                   BoundaryFrame.from_dict(payload.get("ending_state", {})))


@dataclass(frozen=True)
class SynopsisIssue:
    severity: str
    code: str
    message: str
    def to_dict(self) -> dict[str, str]: return asdict(self)


@dataclass(frozen=True)
class SynopsisQuality:
    passed: bool
    content_status: str
    source_window_coverage: float
    local_synopsis_count: int
    local_segment_count: int
    issues: tuple[SynopsisIssue, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "issues": [item.to_dict() for item in self.issues]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SynopsisQuality":
        return cls(bool(payload.get("passed", False)), str(payload.get("content_status", "not_run")),
                   float(payload.get("source_window_coverage", 0.0)), int(payload.get("local_synopsis_count", 0)),
                   int(payload.get("local_segment_count", 0)),
                   tuple(SynopsisIssue(str(item.get("severity", "error")), str(item.get("code", "")), str(item.get("message", ""))) for item in payload.get("issues", []) if isinstance(item, dict)))


@dataclass(frozen=True)
class ChapterSynopsisBundle:
    chapter_id: str
    source_hash: str
    local_synopses: tuple[LocalSynopsis, ...]
    chapter_synopsis: ChapterSynopsis
    quality: SynopsisQuality
    schema_version: str = SYNOPSIS_SCHEMA_VERSION

    def validate(self) -> None:
        _required(self.chapter_id, "bundle chapter_id")
        _required(self.source_hash, "bundle source_hash")
        if self.schema_version != SYNOPSIS_SCHEMA_VERSION:
            raise ValueError(f"unsupported synopsis schema version: {self.schema_version}")
        if not self.local_synopses:
            raise ValueError("chapter synopsis bundle requires local synopses")
        for item in self.local_synopses:
            item.validate()
            if item.chapter_id != self.chapter_id:
                raise ValueError("local synopsis belongs to another chapter")
        self.chapter_synopsis.validate(self.local_synopses)
        if self.chapter_synopsis.chapter_id != self.chapter_id:
            raise ValueError("chapter synopsis belongs to another chapter")
        if not self.quality.passed or self.quality.content_status != "verified":
            raise ValueError("formal chapter synopsis bundle requires verified content")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "chapter_id": self.chapter_id,
                "source_hash": self.source_hash, "local_synopses": [item.to_dict() for item in self.local_synopses],
                "chapter_synopsis": self.chapter_synopsis.to_dict(), "quality": self.quality.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterSynopsisBundle":
        return cls(str(payload.get("chapter_id", "")), str(payload.get("source_hash", "")),
                   tuple(LocalSynopsis.from_dict(item) for item in payload.get("local_synopses", []) if isinstance(item, dict)),
                   ChapterSynopsis.from_dict(payload.get("chapter_synopsis", {})),
                   SynopsisQuality.from_dict(payload.get("quality", {})),
                   str(payload.get("schema_version", SYNOPSIS_SCHEMA_VERSION)))
