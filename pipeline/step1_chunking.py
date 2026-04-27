"""
step1_chunking.py – Physical Chunking & Arc Anchoring

Responsibilities:
  - Read raw .txt files from input_dir.
  - Use regex to split text into chapters (e.g., matching "第.*章").
  - Group every 100 chapters into a physical chunk (NarrativeEvent) representing a Volume (卷01, 卷02).
  - No LLM calls – this step is purely deterministic Python logic.

Output:
  List[VolumeArc], where each VolumeArc holds a list of NarrativeEvent objects.
  The result is also persisted to ``intermediate_dir/step1_chunks.json`` for
  pipeline resume capability.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

import config
from pipeline.core.common_json import read_json_file, write_json_file

@dataclass
class NarrativeEvent:
    event_id: str
    arc_name: str
    chapters: List[str]          # raw chapter texts merged together
    summary: str = ""            # filled in during chunking
    chapter_start: int = 0
    chapter_end: int = 0
    subchunk_index: int = 1
    subchunk_total: int = 1


@dataclass
class VolumeArc:
    arc_name: str
    realm_start: str
    realm_end: str
    events: List[NarrativeEvent] = field(default_factory=list)


# ── Public API ──────────────────────────────────────────────────────────────

def process_novel(novel_path: str | Path) -> List[VolumeArc]:
    """
    Read a single novel file, chunk into chapters, and return a list of VolumeArcs.
    Groups every 100 chapters into a Volume (卷01, 卷02...).
    """
    novel_path = Path(novel_path)
    text = novel_path.read_text(encoding="utf-8")
    chapters = _split_into_chapters(text)

    arcs = _build_volume_arcs(chapters, chapters_per_volume=100)

    for arc in arcs:
        arc.events = _physical_chunk_arc(arc.arc_name, arc.events)

    return arcs


def process_all_novels(input_dir: str | Path) -> dict[str, List[VolumeArc]]:
    """
    Process every .txt file in *input_dir* and return a mapping
    {filename: List[VolumeArc]}.
    """
    input_dir = Path(input_dir)
    results: dict[str, List[VolumeArc]] = {}
    for novel_file in sorted(input_dir.glob("*.txt")):
        print(f"[Step 1] Processing: {novel_file.name}")
        results[novel_file.name] = process_novel(novel_file)
    if not results:
        print("[Step 1] Warning: no .txt files found in input directory.")
    else:
        save_step1_output(results)
    return results


# ── Intermediate I/O ────────────────────────────────────────────────────────

_STEP1_FILENAME = "step1_chunks.json"

def save_step1_output(novel_arcs: dict[str, List[VolumeArc]]) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _STEP1_FILENAME
    serialisable = {
        novel_name: [asdict(arc) for arc in arcs]
        for novel_name, arcs in novel_arcs.items()
    }
    write_json_file(out_path, serialisable)
    print(f"[Step 1] Intermediate output saved → {out_path}")
    return out_path


def load_step1_output(intermediate_dir: str | Path | None = None) -> dict[str, List[VolumeArc]]:
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)
    in_path = inter_dir / _STEP1_FILENAME
    if not in_path.exists():
        raise FileNotFoundError(
            f"Step 1 intermediate file not found: {in_path}\n"
            "Run the pipeline from Step 1 first to generate it."
        )
    raw = read_json_file(in_path)
    result: dict[str, List[VolumeArc]] = {}
    for novel_name, arcs_data in raw.items():
        arcs: List[VolumeArc] = []
        for arc_d in arcs_data:
            events = [NarrativeEvent(**ev) for ev in arc_d.get("events", [])]
            arc = VolumeArc(
                arc_name=arc_d["arc_name"],
                realm_start=arc_d["realm_start"],
                realm_end=arc_d["realm_end"],
                events=events,
            )
            arcs.append(arc)
        result[novel_name] = arcs
    print(f"[Step 1] Loaded intermediate output from {in_path}")
    return result


# ── Internal helpers ────────────────────────────────────────────────────────
def _split_into_chapters(text: str) -> List[str]:
    """
    Split a novel text into individual chapters.
    Automatically filters out Table of Contents (目录) and empty duplicate headings.
    """
    # 使用正则表达式匹配章节标题行（匹配整行）
    pattern = re.compile(r"^(第[零一二三四五六七八九十百千万\d]+章.*|Chapter\s*\d+.*)", re.MULTILINE | re.IGNORECASE)
    parts = pattern.split(text)

    # 如果完全没有匹配到章节，按双换行符回退处理
    if len(parts) <= 1:
        return [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 100]

    chapters: List[str] = []

    # parts[0] 是第一章之前的文本（比如书籍简介、作者信息等）
    preamble = parts[0].strip()
    # 只有当简介内容比较长（大于100字）时才保留，否则（如只有书名作者）直接丢弃
    if len(preamble) > 100:
        chapters.append(preamble)

    i = 1
    while i < len(parts) - 1:
        heading = parts[i].strip()
        body = parts[i + 1].strip()

        chapter_content = f"{heading}\n{body}".strip()

        # 核心去重逻辑：如果这一“章”去除了空格和换行后，总字数不到 100 字，
        # 说明它绝对不是真正的章节（通常是目录里的一行，或者是排版重复的标题行）
        # 直接丢弃它，不加入最终列表。
        if len(chapter_content.replace(" ", "").replace("\n", "")) > 100:
            chapters.append(chapter_content)

        i += 2

    return chapters



def _build_volume_arcs(chapters: List[str], chapters_per_volume: int = 100) -> List[VolumeArc]:
    """
    Group chapters into VolumeArcs evenly based on chapters_per_volume.
    """
    arcs: List[VolumeArc] = []
    for i in range(0, len(chapters), chapters_per_volume):
        vol_num = (i // chapters_per_volume) + 1
        arc_name = f"卷{vol_num:02d}"
        arc_chapters = chapters[i:i + chapters_per_volume]

        # 构造临时的 NarrativeEvent：默认以较小原子段为单位，长章节会先按段落切开。
        raw_events: List[NarrativeEvent] = []
        for j, ch in enumerate(arc_chapters):
            chapter_no = i + j + 1
            raw_events.extend(_chapter_to_atomic_events(arc_name, ch, chapter_no))

        arcs.append(
            VolumeArc(
                arc_name=arc_name,
                realm_start=arc_name,  # 不再使用境界，以卷名替代
                realm_end=f"卷{vol_num + 1:02d}",
                events=raw_events,
            )
        )
    return arcs


def _physical_chunk_arc(
    arc_name: str, raw_events: List[NarrativeEvent]
) -> List[NarrativeEvent]:
    """
    Merge consecutive raw chapter events into physical NarrativeEvent chunks.
    Groups between MIN_CHAPTERS and MAX_CHAPTERS chapters together.
    """
    min_ch = config.SEMANTIC_CHUNK_MIN_CHAPTERS
    max_ch = config.SEMANTIC_CHUNK_MAX_CHAPTERS
    target_chars = max(1800, int(config.MAX_TEXT_CHUNK_LENGTH * 0.85))
    same_chapter_limit = max(3600, int(config.MAX_TEXT_CHUNK_LENGTH * 1.6))

    merged: List[NarrativeEvent] = []
    i = 0
    event_counter = 0
    while i < len(raw_events):
        group: List[NarrativeEvent] = []
        char_count = 0
        while i + len(group) < len(raw_events):
            candidate = raw_events[i + len(group)]
            candidate_chars = _event_char_count(candidate)
            next_group = group + [candidate]
            next_chapter_span = _group_chapter_span(next_group)
            if next_chapter_span > max_ch:
                break

            projected_chars = char_count + candidate_chars
            current_chapter_span = _group_chapter_span(group)
            same_chapter_continuation = bool(group) and _is_same_chapter_continuation(group[-1], candidate)
            char_limit = same_chapter_limit if same_chapter_continuation and next_chapter_span == current_chapter_span else target_chars

            if group and current_chapter_span >= min_ch and projected_chars > char_limit:
                break

            group.append(candidate)
            char_count = projected_chars

            if _group_should_close(
                group,
                raw_events,
                current_index=i + len(group) - 1,
                min_ch=min_ch,
                target_chars=target_chars,
                same_chapter_limit=same_chapter_limit,
                char_count=char_count,
            ):
                break

        if _group_chapter_span(group) < min_ch and merged:
            # Merge tiny trailing group into the last event
            merged[-1].chapters.extend(
                ch for ev in group for ch in ev.chapters
            )
            merged[-1].chapter_end = group[-1].chapter_end
            merged[-1].subchunk_index = 1
            merged[-1].subchunk_total = 1
            i += len(group)
            continue

        merged.append(
            NarrativeEvent(
                event_id=f"{arc_name}_event{event_counter}",
                arc_name=arc_name,
                chapters=[ch for ev in group for ch in ev.chapters],
                summary="",
                chapter_start=group[0].chapter_start if group else 0,
                chapter_end=group[-1].chapter_end if group else 0,
                subchunk_index=1,
                subchunk_total=1,
            )
        )
        event_counter += 1
        i += len(group)

    return merged


def _chapter_to_atomic_events(
    arc_name: str,
    chapter_text: str,
    chapter_no: int,
) -> List[NarrativeEvent]:
    segments = _split_chapter_into_segments(chapter_text)
    total = len(segments)
    events: List[NarrativeEvent] = []
    for idx, segment in enumerate(segments, start=1):
        suffix = f"_seg{idx}" if total > 1 else ""
        events.append(
            NarrativeEvent(
                event_id=f"{arc_name}_ch{chapter_no}{suffix}",
                arc_name=arc_name,
                chapters=[segment],
                chapter_start=chapter_no,
                chapter_end=chapter_no,
                subchunk_index=idx,
                subchunk_total=total,
            )
        )
    return events


def _split_chapter_into_segments(chapter_text: str) -> List[str]:
    clean_text = str(chapter_text or "").strip()
    if not clean_text:
        return []

    target_chars = max(2200, int(config.MAX_TEXT_CHUNK_LENGTH * 0.95))
    hard_limit = max(3200, int(config.MAX_TEXT_CHUNK_LENGTH * 1.35))
    if len(clean_text) <= hard_limit:
        return [clean_text]

    heading, body = _split_heading_and_body(clean_text)
    paragraphs = _split_body_into_paragraphs(body or clean_text)
    segments: List[str] = []
    buffer: List[str] = []
    buffer_len = 0

    for paragraph in paragraphs:
        parts = _split_oversized_paragraph(paragraph, target_chars, hard_limit)
        for part in parts:
            part_len = len(part)
            if buffer and buffer_len + part_len > hard_limit:
                segments.append("\n\n".join(buffer).strip())
                buffer = []
                buffer_len = 0
            buffer.append(part)
            buffer_len += part_len
            if buffer_len >= target_chars:
                segments.append("\n\n".join(buffer).strip())
                buffer = []
                buffer_len = 0

    if buffer:
        segments.append("\n\n".join(buffer).strip())

    if not segments:
        segments = [clean_text]

    if len(segments) == 1:
        return [clean_text]

    rendered: List[str] = []
    total = len(segments)
    for idx, segment in enumerate(segments, start=1):
        prefix_parts = []
        if heading:
            prefix_parts.append(heading)
        prefix_parts.append(f"【本章片段{idx}/{total}】")
        prefix_parts.append(segment)
        rendered.append("\n".join(part for part in prefix_parts if part).strip())
    return rendered


def _split_heading_and_body(chapter_text: str) -> tuple[str, str]:
    lines = [line.rstrip() for line in chapter_text.splitlines()]
    if not lines:
        return "", ""
    heading = lines[0].strip()
    if re.match(r"^(第[零一二三四五六七八九十百千万\d]+章.*|Chapter\s*\d+.*)$", heading, re.IGNORECASE):
        body = "\n".join(lines[1:]).strip()
        return heading, body
    return "", chapter_text


def _split_body_into_paragraphs(body: str) -> List[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", body) if part.strip()]
    if paragraphs:
        return paragraphs
    return [line.strip() for line in body.splitlines() if line.strip()]


def _split_oversized_paragraph(paragraph: str, target_chars: int, hard_limit: int) -> List[str]:
    clean = paragraph.strip()
    if len(clean) <= hard_limit:
        return [clean]

    sentences = re.split(r"(?<=[。！？!?；;])", clean)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not sentences:
        return [clean[i:i + target_chars] for i in range(0, len(clean), target_chars)]

    segments: List[str] = []
    buffer = ""
    for sentence in sentences:
        if buffer and len(buffer) + len(sentence) > hard_limit:
            segments.append(buffer.strip())
            buffer = ""
        buffer += sentence
        if len(buffer) >= target_chars:
            segments.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        segments.append(buffer.strip())
    return segments or [clean]


def _event_char_count(event: NarrativeEvent) -> int:
    return sum(len(chapter or "") for chapter in event.chapters)


def _group_chapter_span(events: List[NarrativeEvent]) -> int:
    covered: set[int] = set()
    for event in events:
        start = int(getattr(event, "chapter_start", 0) or 0)
        end = int(getattr(event, "chapter_end", 0) or start or 0)
        if start and end and end >= start:
            covered.update(range(start, end + 1))
    return len(covered)


def _is_same_chapter_continuation(left: NarrativeEvent, right: NarrativeEvent) -> bool:
    return (
        left.chapter_start == right.chapter_start
        and left.chapter_end == right.chapter_end
        and left.chapter_start != 0
    )


def _group_should_close(
    group: List[NarrativeEvent],
    raw_events: List[NarrativeEvent],
    current_index: int,
    min_ch: int,
    target_chars: int,
    same_chapter_limit: int,
    char_count: int,
) -> bool:
    chapter_span = _group_chapter_span(group)
    if chapter_span < min_ch:
        return False
    if current_index + 1 >= len(raw_events):
        return True

    next_event = raw_events[current_index + 1]
    if _is_same_chapter_continuation(group[-1], next_event):
        return char_count >= same_chapter_limit

    return char_count >= target_chars
