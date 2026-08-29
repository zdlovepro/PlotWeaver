"""第一模块的中等粒度剧情梗概契约。

第一模块只压缩剧情，不承担事实、实体、关系、时空或文风提取。原文段落 ID
仅用于追溯梗概依据，不代表逐段事实抽取。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


SYNOPSIS_SCHEMA_VERSION = "3.3"

PLOT_FUNCTIONS = frozenset({
    "setup", "goal", "pressure", "decision", "action",
    "revelation", "turn", "outcome", "transition",
})
CHAPTER_FUNCTIONS = frozenset({
    "setup", "development", "escalation", "turning_point", "climax",
    "aftermath", "transition", "mixed",
})


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
    """一个语义分块中的主要剧情推进。"""

    segment_id: str
    order: int
    summary: str
    narrative_function: str
    story_change: str
    source_unit_ids: tuple[str, ...]

    def validate(self) -> None:
        _required(self.segment_id, "local segment_id")
        if self.order < 0:
            raise ValueError("local segment order must be non-negative")
        _required(self.summary, "local segment summary")
        if self.narrative_function not in PLOT_FUNCTIONS:
            raise ValueError(f"unsupported plot function: {self.narrative_function}")
        if not self.source_unit_ids:
            raise ValueError("local segment requires source_unit_ids")
        _unique(self.source_unit_ids, "local segment source_unit_ids")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_unit_ids"] = list(self.source_unit_ids)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalPlotSegment":
        return cls(
            segment_id=str(payload.get("segment_id", "")),
            order=int(payload.get("order", -1)),
            summary=str(payload.get("summary", "")),
            narrative_function=str(payload.get("narrative_function", "")),
            story_change=str(payload.get("story_change", "")),
            source_unit_ids=_strings(payload, "source_unit_ids"),
        )


@dataclass(frozen=True)
class LocalSynopsis:
    """语义分块的压缩结果；不是章节最终粒度，也不是事实表。"""

    window_id: str
    chapter_id: str
    reviewed_source_unit_ids: tuple[str, ...]
    opening_frame: str
    ending_frame: str
    segments: tuple[LocalPlotSegment, ...]

    def validate(self) -> None:
        _required(self.window_id, "window_id")
        _required(self.chapter_id, "chapter_id")
        if not self.reviewed_source_unit_ids:
            raise ValueError("local synopsis requires reviewed_source_unit_ids")
        _unique(self.reviewed_source_unit_ids, "reviewed_source_unit_ids")
        if "\n" in self.opening_frame.strip():
            raise ValueError("opening_frame must be one concise paragraph")
        if "\n" in self.ending_frame.strip():
            raise ValueError("ending_frame must be one concise paragraph")
        # 分块内剧情段数量由原文实际推进决定，不把详略偏好作为契约失败。
        segment_ids = tuple(item.segment_id for item in self.segments)
        _unique(segment_ids, "local segment_ids")
        if tuple(item.order for item in self.segments) != tuple(range(len(self.segments))):
            raise ValueError("local segment orders must be consecutive")
        allowed = set(self.reviewed_source_unit_ids)
        for item in self.segments:
            item.validate()
            if not set(item.source_unit_ids).issubset(allowed):
                raise ValueError("local segment refers to a source unit outside its window")

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "chapter_id": self.chapter_id,
            "reviewed_source_unit_ids": list(self.reviewed_source_unit_ids),
            "opening_frame": self.opening_frame,
            "ending_frame": self.ending_frame,
            "segments": [item.to_dict() for item in self.segments],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalSynopsis":
        return cls(
            window_id=str(payload.get("window_id", "")),
            chapter_id=str(payload.get("chapter_id", "")),
            reviewed_source_unit_ids=_strings(payload, "reviewed_source_unit_ids"),
            opening_frame=str(payload.get("opening_frame", "")),
            ending_frame=str(payload.get("ending_frame", "")),
            segments=tuple(
                LocalPlotSegment.from_dict(item)
                for item in payload.get("segments", [])
                if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True)
class SynopsisStatement:
    """由章级核心剧情节点支持的压缩判断。"""

    text: str
    node_ids: tuple[str, ...]

    def validate(self, known_node_ids: set[str], name: str, *, required: bool = True) -> None:
        text = str(self.text or "").strip()
        if required and not text:
            raise ValueError(f"{name} must not be empty")
        if not text:
            if self.node_ids:
                raise ValueError(f"empty {name} must not cite nodes")
            return
        if not self.node_ids:
            raise ValueError(f"{name} requires supporting node_ids")
        _unique(self.node_ids, f"{name} node_ids")
        if not set(self.node_ids).issubset(known_node_ids):
            raise ValueError(f"{name} refers to unknown chapter nodes")

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "node_ids": list(self.node_ids)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SynopsisStatement":
        if not isinstance(payload, dict):
            return cls(text="", node_ids=())
        return cls(text=str(payload.get("text", "")), node_ids=_strings(payload, "node_ids"))


@dataclass(frozen=True)
class ChapterPlotNode:
    """章级核心剧情节点；一章通常只保留 2—6 个，硬上限为 8。"""

    node_id: str
    order: int
    summary: str
    narrative_function: str
    story_change: str
    local_segment_ids: tuple[str, ...]

    def validate(self, known_segment_ids: set[str]) -> None:
        _required(self.node_id, "chapter node_id")
        if self.order < 0:
            raise ValueError("chapter node order must be non-negative")
        _required(self.summary, "chapter node summary")
        if self.narrative_function not in PLOT_FUNCTIONS:
            raise ValueError(f"unsupported chapter node function: {self.narrative_function}")
        if not self.local_segment_ids:
            raise ValueError("chapter node requires local_segment_ids")
        _unique(self.local_segment_ids, "chapter node local_segment_ids")
        if not set(self.local_segment_ids).issubset(known_segment_ids):
            raise ValueError("chapter node refers to an unknown local segment")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["local_segment_ids"] = list(self.local_segment_ids)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterPlotNode":
        return cls(
            node_id=str(payload.get("node_id", "")),
            order=int(payload.get("order", -1)),
            summary=str(payload.get("summary", "")),
            narrative_function=str(payload.get("narrative_function", "")),
            story_change=str(payload.get("story_change", "")),
            local_segment_ids=_strings(payload, "local_segment_ids"),
        )


@dataclass(frozen=True)
class ChapterPhaseSynopsis:
    """比单个场景更粗的章节阶段，通常覆盖若干连续核心节点。"""

    phase_id: str
    order: int
    title: str
    summary: str
    node_ids: tuple[str, ...]

    def validate(self, known_node_ids: set[str]) -> None:
        _required(self.phase_id, "chapter phase_id")
        if self.order < 0:
            raise ValueError("chapter phase order must be non-negative")
        _required(self.title, "chapter phase title")
        _required(self.summary, "chapter phase summary")
        if not self.node_ids:
            raise ValueError("chapter phase requires node_ids")
        _unique(self.node_ids, "chapter phase node_ids")
        if not set(self.node_ids).issubset(known_node_ids):
            raise ValueError("chapter phase refers to an unknown chapter node")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["node_ids"] = list(self.node_ids)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterPhaseSynopsis":
        return cls(
            phase_id=str(payload.get("phase_id", "")),
            order=int(payload.get("order", -1)),
            title=str(payload.get("title", "")),
            summary=str(payload.get("summary", "")),
            node_ids=_strings(payload, "node_ids"),
        )


@dataclass(frozen=True)
class ChapterSynopsis:
    chapter_id: str
    one_sentence_summary: SynopsisStatement
    chapter_function: str
    opening_state: SynopsisStatement
    core_nodes: tuple[ChapterPlotNode, ...]
    ending_state: SynopsisStatement
    open_threads: tuple[SynopsisStatement, ...]
    phases: tuple[ChapterPhaseSynopsis, ...]

    def validate(self, local_synopses: tuple[LocalSynopsis, ...] | None = None) -> None:
        _required(self.chapter_id, "chapter synopsis chapter_id")
        if self.chapter_function not in CHAPTER_FUNCTIONS:
            raise ValueError(f"unsupported chapter function: {self.chapter_function}")
        if not self.core_nodes or len(self.core_nodes) > 8:
            raise ValueError("chapter synopsis requires one to eight core nodes")
        if not self.phases or len(self.phases) > 6:
            raise ValueError("chapter synopsis requires one to six phases")

        node_ids = tuple(item.node_id for item in self.core_nodes)
        known_nodes = set(node_ids)
        _unique(node_ids, "chapter node_ids")
        if tuple(item.order for item in self.core_nodes) != tuple(range(len(self.core_nodes))):
            raise ValueError("chapter node orders must be consecutive")
        known_segments = {
            segment.segment_id
            for local in (local_synopses or ())
            for segment in local.segments
        }
        for item in self.core_nodes:
            item.validate(known_segments if local_synopses is not None else set(item.local_segment_ids))
        if local_synopses is not None:
            referenced_segments = {
                segment_id
                for item in self.core_nodes
                for segment_id in item.local_segment_ids
            }
            if referenced_segments != known_segments:
                raise ValueError("chapter nodes must cover every local segment at least once")

        self.one_sentence_summary.validate(known_nodes, "one_sentence_summary")
        self.opening_state.validate(known_nodes, "opening_state")
        self.ending_state.validate(known_nodes, "ending_state")
        for index, item in enumerate(self.open_threads):
            item.validate(known_nodes, f"open_threads[{index}]")

        phase_ids = tuple(item.phase_id for item in self.phases)
        _unique(phase_ids, "chapter phase_ids")
        if tuple(item.order for item in self.phases) != tuple(range(len(self.phases))):
            raise ValueError("chapter phase orders must be consecutive")
        referenced_nodes: list[str] = []
        for item in self.phases:
            item.validate(known_nodes)
            referenced_nodes.extend(item.node_ids)
        if len(referenced_nodes) != len(set(referenced_nodes)) or set(referenced_nodes) != known_nodes:
            raise ValueError("chapter phases must cover every core node exactly once")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "one_sentence_summary": self.one_sentence_summary.to_dict(),
            "chapter_function": self.chapter_function,
            "opening_state": self.opening_state.to_dict(),
            "core_nodes": [item.to_dict() for item in self.core_nodes],
            "ending_state": self.ending_state.to_dict(),
            "open_threads": [item.to_dict() for item in self.open_threads],
            "phases": [item.to_dict() for item in self.phases],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterSynopsis":
        return cls(
            chapter_id=str(payload.get("chapter_id", "")),
            one_sentence_summary=SynopsisStatement.from_dict(payload.get("one_sentence_summary", {})),
            chapter_function=str(payload.get("chapter_function", "")),
            opening_state=SynopsisStatement.from_dict(payload.get("opening_state", {})),
            core_nodes=tuple(
                ChapterPlotNode.from_dict(item)
                for item in payload.get("core_nodes", [])
                if isinstance(item, dict)
            ),
            ending_state=SynopsisStatement.from_dict(payload.get("ending_state", {})),
            open_threads=tuple(
                SynopsisStatement.from_dict(item)
                for item in payload.get("open_threads", [])
                if isinstance(item, dict)
            ),
            phases=tuple(
                ChapterPhaseSynopsis.from_dict(item)
                for item in payload.get("phases", [])
                if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True)
class SynopsisIssue:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SynopsisQuality:
    passed: bool
    source_coverage: float
    local_synopsis_count: int
    local_segment_count: int
    chapter_node_count: int
    phase_count: int
    issues: tuple[SynopsisIssue, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "issues": [item.to_dict() for item in self.issues]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SynopsisQuality":
        return cls(
            passed=bool(payload.get("passed", False)),
            source_coverage=float(payload.get("source_coverage", 0.0)),
            local_synopsis_count=int(payload.get("local_synopsis_count", 0)),
            local_segment_count=int(payload.get("local_segment_count", 0)),
            chapter_node_count=int(payload.get("chapter_node_count", 0)),
            phase_count=int(payload.get("phase_count", 0)),
            issues=tuple(
                SynopsisIssue(
                    severity=str(item.get("severity", "error")),
                    code=str(item.get("code", "")),
                    message=str(item.get("message", "")),
                )
                for item in payload.get("issues", [])
                if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True)
class ChapterSynopsisBundle:
    chapter_id: str
    source_hash: str
    local_synopses: tuple[LocalSynopsis, ...]
    chapter_synopsis: ChapterSynopsis
    quality: SynopsisQuality
    content_verified: bool = True
    evidence_verified: bool = True
    schema_version: str = SYNOPSIS_SCHEMA_VERSION

    def validate(self) -> None:
        _required(self.chapter_id, "bundle chapter_id")
        _required(self.source_hash, "bundle source_hash")
        if not self.local_synopses:
            raise ValueError("chapter synopsis bundle requires local synopses")
        for item in self.local_synopses:
            item.validate()
            if item.chapter_id != self.chapter_id:
                raise ValueError("local synopsis belongs to another chapter")
        self.chapter_synopsis.validate(self.local_synopses)
        if self.chapter_synopsis.chapter_id != self.chapter_id:
            raise ValueError("chapter synopsis belongs to another chapter")
        if not self.quality.passed:
            raise ValueError("chapter synopsis quality did not pass")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "chapter_id": self.chapter_id,
            "source_hash": self.source_hash,
            "local_synopses": [item.to_dict() for item in self.local_synopses],
            "chapter_synopsis": self.chapter_synopsis.to_dict(),
            "quality": self.quality.to_dict(),
            "content_verified": self.content_verified,
            "evidence_verified": self.evidence_verified,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterSynopsisBundle":
        return cls(
            chapter_id=str(payload.get("chapter_id", "")),
            source_hash=str(payload.get("source_hash", "")),
            local_synopses=tuple(
                LocalSynopsis.from_dict(item)
                for item in payload.get("local_synopses", [])
                if isinstance(item, dict)
            ),
            chapter_synopsis=ChapterSynopsis.from_dict(payload.get("chapter_synopsis", {})),
            quality=SynopsisQuality.from_dict(payload.get("quality", {})),
            content_verified=bool(payload.get("content_verified", True)),
            evidence_verified=bool(payload.get("evidence_verified", True)),
            schema_version=str(payload.get("schema_version", SYNOPSIS_SCHEMA_VERSION)),
        )
