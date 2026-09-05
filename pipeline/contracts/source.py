"""不可变原文、精确证据和保守的自然段切分契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import re
from typing import Any


SOURCE_UNIT_KINDS = frozenset({
    "paragraph", "dialogue", "narration", "heading", "boilerplate",
    "editorial_note", "separator", "publisher_note",
})
CHAPTER_CONTENT_KINDS = frozenset({"narrative", "mixed", "editorial_notice"})


def _required(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


@dataclass(frozen=True)
class SourceSpan:
    """章节正文中的半开字符区间，短引必须与原文完全一致。"""

    chapter_id: str
    start: int
    end: int
    quote: str
    unit_id: str = ""

    def validate(self, chapter_text: str | None = None) -> None:
        _required(self.chapter_id, "chapter_id")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("source span must satisfy 0 <= start < end")
        if not self.quote:
            raise ValueError("source span quote must not be empty")
        if chapter_text is not None:
            if self.end > len(chapter_text):
                raise ValueError("source span exceeds chapter text")
            if chapter_text[self.start:self.end] != self.quote:
                raise ValueError("source span quote does not match chapter text")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceSpan":
        return cls(
            chapter_id=str(payload.get("chapter_id", "")),
            start=int(payload.get("start", -1)),
            end=int(payload.get("end", -1)),
            quote=str(payload.get("quote", "")),
            unit_id=str(payload.get("unit_id", "")),
        )


@dataclass(frozen=True)
class SourceUnit:
    """确定性原文单元；模型产物通过 ``unit_id``引用它。"""

    unit_id: str
    chapter_id: str
    index: int
    kind: str
    start: int
    end: int
    text: str
    quote_depth_before: int = 0
    quote_depth_after: int = 0

    def validate(self, chapter_text: str | None = None) -> None:
        _required(self.unit_id, "unit_id")
        _required(self.chapter_id, "chapter_id")
        if self.kind not in SOURCE_UNIT_KINDS:
            raise ValueError(f"unsupported source unit kind: {self.kind}")
        if self.index < 0:
            raise ValueError("source unit index must be non-negative")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("source unit must satisfy 0 <= start < end")
        if not self.text:
            raise ValueError("source unit text must not be empty")
        if self.quote_depth_before < 0 or self.quote_depth_after < 0:
            raise ValueError("source unit quote depth must be non-negative")
        if chapter_text is not None:
            if self.end > len(chapter_text) or chapter_text[self.start:self.end] != self.text:
                raise ValueError("source unit does not match chapter text")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceUnit":
        kind = str(payload.get("kind", ""))
        if kind == "author_note":
            kind = "editorial_note"
        return cls(
            unit_id=str(payload.get("unit_id", "")),
            chapter_id=str(payload.get("chapter_id", "")),
            index=int(payload.get("index", -1)),
            kind=kind,
            start=int(payload.get("start", -1)),
            end=int(payload.get("end", -1)),
            text=str(payload.get("text", "")),
            quote_depth_before=int(payload.get("quote_depth_before", 0)),
            quote_depth_after=int(payload.get("quote_depth_after", 0)),
        )

    @property
    def is_annotation_eligible(self) -> bool:
        return self.kind not in {"boilerplate", "editorial_note", "separator", "publisher_note", "heading"}


@dataclass(frozen=True)
class ChapterDocument:
    """不可变章节正文以及精确定位的原文单元。"""

    chapter_id: str
    source_hash: str
    text: str
    units: tuple[SourceUnit, ...] = field(default_factory=tuple)
    content_kind: str = "narrative"

    def validate(self) -> None:
        _required(self.chapter_id, "chapter_id")
        _required(self.source_hash, "source_hash")
        if not self.text:
            raise ValueError("chapter text must not be empty")
        if self.content_kind not in CHAPTER_CONTENT_KINDS:
            raise ValueError(f"unsupported chapter content kind: {self.content_kind}")
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.source_hash:
            raise ValueError("source_hash does not match chapter text")
        seen_ids: set[str] = set()
        last_start = -1
        quote_depth = 0
        for unit in self.units:
            unit.validate(self.text)
            if unit.chapter_id != self.chapter_id:
                raise ValueError("source unit chapter_id does not match document")
            if unit.unit_id in seen_ids:
                raise ValueError(f"duplicate source unit id: {unit.unit_id}")
            if unit.start < last_start:
                raise ValueError("source units must be ordered by start offset")
            if unit.quote_depth_before != quote_depth:
                raise ValueError("source unit quote_depth_before is not continuous")
            quote_depth = _advance_quote_depth(unit.text, quote_depth)
            if unit.quote_depth_after != quote_depth:
                raise ValueError("source unit quote_depth_after does not match source text")
            seen_ids.add(unit.unit_id)
            last_start = unit.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "source_hash": self.source_hash,
            "text": self.text,
            "units": [unit.to_dict() for unit in self.units],
            "content_kind": self.content_kind,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterDocument":
        content_kind = str(payload.get("content_kind", "narrative"))
        if content_kind == "author_notice":
            content_kind = "editorial_notice"
        return cls(
            chapter_id=str(payload.get("chapter_id", "")),
            source_hash=str(payload.get("source_hash", "")),
            text=str(payload.get("text", "")),
            units=tuple(SourceUnit.from_dict(item) for item in payload.get("units", []) if isinstance(item, dict)),
            content_kind=content_kind,
        )

    @property
    def annotation_units(self) -> tuple[SourceUnit, ...]:
        if self.content_kind == "editorial_notice":
            return ()
        return tuple(unit for unit in self.units if unit.is_annotation_eligible)

    @property
    def is_narrative(self) -> bool:
        return bool(self.annotation_units)


def _advance_quote_depth(text: str, initial_depth: int = 0) -> int:
    depth = initial_depth
    for char in text:
        if char in "“「『":
            depth += 1
        elif char in "”」』" and depth:
            depth -= 1
    return depth


def _is_boilerplate(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return normalized in {"(本章完)", "（本章完）", "【本章完】"}


def _is_separator(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return len(normalized) >= 2 and all(char in "-—*_~=·" for char in normalized)


_EDITORIAL_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"(?:求|跪求|还请).{0,12}(?:月票|推荐票|订阅|收藏|打赏|书评|评论|追读)",
    r"(?:月票|推荐票|订阅|收藏|打赏|书评|评论|追读).{0,12}(?:支持|感谢|加更|更新|拜托)",
    r"(?:上架感言|完本感言|更新说明|请假(?:条|说明)?|作者的话|单章公告)",
    r"(?:今天|明天|本周|本月|累计).{0,12}(?:加更|更新|欠更|补更)",
))
_EDITORIAL_TITLE_MARKERS = ("公告", "感言", "请假", "单章", "上架", "更新说明", "作者的话", "完本")


def _is_editorial_note(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return any(pattern.search(normalized) for pattern in _EDITORIAL_PATTERNS)


def _is_publisher_note(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text).lower()
    return any(marker in normalized for marker in (
        "http://", "https://", "www.", "更多精彩小说", "请访问", "最新网址",
    ))


def _mark_terminal_editorial_blocks(units: list[SourceUnit]) -> list[SourceUnit]:
    marked: list[SourceUnit] = []
    in_terminal_note = False
    for index, unit in enumerate(units):
        if unit.kind == "editorial_note":
            in_terminal_note = index >= len(units) - 4 or any(
                previous.kind == "separator" for previous in units[max(0, index - 2):index]
            )
        elif in_terminal_note and unit.kind == "paragraph":
            unit = replace(unit, kind="editorial_note")
        elif unit.kind == "separator":
            in_terminal_note = False
        marked.append(unit)
    return marked


def build_paragraph_units(chapter_id: str, chapter_text: str) -> tuple[SourceUnit, ...]:
    """不改动任何正文字符，只为每个非空自然段建立稳定位置。"""

    _required(chapter_id, "chapter_id")
    if not chapter_text:
        raise ValueError("chapter text must not be empty")
    units: list[SourceUnit] = []
    quote_depth = 0
    for match in re.finditer(r"[^\n]+", chapter_text):
        raw = match.group(0)
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.end() - (len(raw) - len(raw.rstrip()))
        if start >= end:
            continue
        unit_text = chapter_text[start:end]
        after_depth = _advance_quote_depth(unit_text, quote_depth)
        kind = "paragraph"
        if _is_boilerplate(unit_text):
            kind = "boilerplate"
        elif _is_separator(unit_text):
            kind = "separator"
        elif _is_publisher_note(unit_text):
            kind = "publisher_note"
        elif _is_editorial_note(unit_text):
            kind = "editorial_note"
        elif unit_text.lstrip().startswith(("“", "「", "『")):
            kind = "dialogue"
        units.append(SourceUnit(
            unit_id=f"{chapter_id}:p{len(units):04d}",
            chapter_id=chapter_id,
            index=len(units),
            kind=kind,
            start=start,
            end=end,
            text=unit_text,
            quote_depth_before=quote_depth,
            quote_depth_after=after_depth,
        ))
        quote_depth = after_depth
    if not units:
        raise ValueError("chapter text contains no non-whitespace paragraph")
    return tuple(_mark_terminal_editorial_blocks(units))


def classify_chapter_content(title: str, units: tuple[SourceUnit, ...]) -> str:
    eligible = [unit for unit in units if unit.is_annotation_eligible]
    if not eligible:
        return "editorial_notice"
    editorial = [unit for unit in units if unit.kind in {"editorial_note", "publisher_note"}]
    if not editorial:
        return "narrative"
    normalized_title = re.sub(r"\s+", "", str(title or ""))
    if any(marker in normalized_title for marker in _EDITORIAL_TITLE_MARKERS) and len(eligible) <= len(editorial) + 1:
        return "editorial_notice"
    return "mixed"


def build_chapter_document(chapter_id: str, source_hash: str, chapter_text: str, title: str = "") -> ChapterDocument:
    units = build_paragraph_units(chapter_id, chapter_text)
    return ChapterDocument(
        chapter_id=chapter_id,
        source_hash=source_hash,
        text=chapter_text,
        units=units,
        content_kind=classify_chapter_content(title, units),
    )

