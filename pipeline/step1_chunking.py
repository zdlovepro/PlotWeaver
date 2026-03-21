"""
step1_chunking.py – Semantic Chunking & Arc Anchoring

Responsibilities:
  - Identify major cultivation realm transitions to split text into Volume Arcs.
  - Merge 3-5 related chapters into a single "Narrative Event" using DeepSeek.

Output:
  List[VolumeArc], where each VolumeArc holds a list of NarrativeEvent objects.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

import config
from pipeline.utils import get_deepseek_client, chat_completion_json

# ── Cultivation realm keywords used to detect arc boundaries ─────────────────
REALM_KEYWORDS: List[str] = [
    "炼气", "筑基", "金丹", "元婴", "化神", "炼虚", "合体", "大乘", "渡劫",
    "真仙", "金仙", "太乙", "大罗", "混元",
    # English / romanised equivalents (for translated texts)
    "Qi Condensation", "Foundation Establishment", "Core Formation",
    "Nascent Soul", "Deity Transformation", "Void Refinement",
    "Body Integration", "Mahayana", "Tribulation Transcendence",
]


@dataclass
class NarrativeEvent:
    event_id: str
    arc_name: str
    chapters: List[str]          # raw chapter texts merged together
    summary: str = ""            # filled in during chunking


@dataclass
class VolumeArc:
    arc_name: str
    realm_start: str
    realm_end: str
    events: List[NarrativeEvent] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def process_novel(novel_path: str | Path) -> List[VolumeArc]:
    """
    Read a single novel file, detect realm arcs, and return a list of VolumeArcs
    each containing semantically chunked NarrativeEvents.
    """
    novel_path = Path(novel_path)
    text = novel_path.read_text(encoding="utf-8")
    chapters = _split_into_chapters(text)

    arc_boundaries = _detect_arc_boundaries(chapters)
    arcs = _build_volume_arcs(chapters, arc_boundaries)

    client = get_deepseek_client()
    for arc in arcs:
        arc.events = _semantic_chunk_arc(client, arc.arc_name, arc.events)

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
    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_into_chapters(text: str) -> List[str]:
    """
    Split a novel text into individual chapters.
    Detects common Chinese chapter headers such as:
      第一章, 第1章, 第001章, Chapter 1, …
    Falls back to double-newline paragraph splitting if no headers are found.
    """
    pattern = re.compile(
        r"(第[零一二三四五六七八九十百千万\d]+章|Chapter\s*\d+)",
        re.IGNORECASE,
    )
    parts = pattern.split(text)
    if len(parts) <= 1:
        # No chapter headings found – split on blank lines
        return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chapters: List[str] = []
    # parts alternates: [pre, heading, body, heading, body, ...]
    i = 1
    while i < len(parts) - 1:
        heading = parts[i].strip()
        body = parts[i + 1].strip()
        chapters.append(f"{heading}\n{body}")
        i += 2
    return chapters


def _detect_arc_boundaries(chapters: List[str]) -> List[tuple[int, str]]:
    """
    Return a list of (chapter_index, realm_name) pairs where a realm transition
    is detected. Uses simple keyword matching.
    """
    boundaries: List[tuple[int, str]] = [(0, "开始")]
    seen: set[str] = set()
    for idx, chapter in enumerate(chapters):
        for realm in REALM_KEYWORDS:
            if realm in chapter and realm not in seen:
                boundaries.append((idx, realm))
                seen.add(realm)
                break  # one realm per chapter transition
    return boundaries


def _build_volume_arcs(
    chapters: List[str], boundaries: List[tuple[int, str]]
) -> List[VolumeArc]:
    """
    Given chapter list and arc boundary positions, group chapters into VolumeArcs.
    Each VolumeArc initially holds raw NarrativeEvents (one per chapter group).
    """
    arcs: List[VolumeArc] = []
    for i, (start_idx, realm_name) in enumerate(boundaries):
        end_idx = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(chapters)
        next_realm = boundaries[i + 1][1] if i + 1 < len(boundaries) else "终境"

        arc_chapters = chapters[start_idx:end_idx]
        # Create one raw NarrativeEvent per chapter (will be merged in next step)
        raw_events = [
            NarrativeEvent(
                event_id=f"{realm_name}_ch{start_idx + j}",
                arc_name=realm_name,
                chapters=[ch],
            )
            for j, ch in enumerate(arc_chapters)
        ]
        arcs.append(
            VolumeArc(
                arc_name=realm_name,
                realm_start=realm_name,
                realm_end=next_realm,
                events=raw_events,
            )
        )
    return arcs


def _semantic_chunk_arc(
    client, arc_name: str, raw_events: List[NarrativeEvent]
) -> List[NarrativeEvent]:
    """
    Merge consecutive raw chapter events into semantic Narrative Events
    (groups of MIN_CHAPTERS to MAX_CHAPTERS) and generate a summary for each.
    """
    min_ch = config.SEMANTIC_CHUNK_MIN_CHAPTERS
    max_ch = config.SEMANTIC_CHUNK_MAX_CHAPTERS

    merged: List[NarrativeEvent] = []
    i = 0
    event_counter = 0
    while i < len(raw_events):
        group = raw_events[i : i + max_ch]
        if len(group) < min_ch and merged:
            # Merge tiny trailing group into the last event
            merged[-1].chapters.extend(
                ch for ev in group for ch in ev.chapters
            )
            i += len(group)
            continue

        combined_text = "\n\n".join(
            ch for ev in group for ch in ev.chapters
        )
        summary = _summarise_event(client, arc_name, combined_text)
        merged.append(
            NarrativeEvent(
                event_id=f"{arc_name}_event{event_counter}",
                arc_name=arc_name,
                chapters=[ch for ev in group for ch in ev.chapters],
                summary=summary,
            )
        )
        event_counter += 1
        i += len(group)

    return merged


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _summarise_event(client, arc_name: str, text: str) -> str:
    """Call DeepSeek to produce a concise summary for a narrative event."""
    prompt = (
        f"你是一位修仙小说情节分析师。\n"
        f"当前境界弧：{arc_name}\n\n"
        f"请为以下章节内容撰写一段简洁的情节摘要（200字以内），"
        f"重点保留：境界突破、核心机缘、主要冲突、关键人物行动。\n\n"
        f"章节内容：\n{text[:config.MAX_TEXT_CHUNK_LENGTH]}"
    )
    result = chat_completion_json(
        client,
        system="你是专业的修仙小说情节摘要生成助手。",
        user=prompt,
        json_mode=False,
    )
    return result.strip()
