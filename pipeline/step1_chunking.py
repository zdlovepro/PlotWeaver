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
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, ensure_ascii=False, indent=2)
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
    with open(in_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
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

        # 构造临时的 NarrativeEvent（每章一个，随后会根据配置块大小合并）
        raw_events = [
            NarrativeEvent(
                event_id=f"{arc_name}_ch{i + j + 1}",
                arc_name=arc_name,
                chapters=[ch],
            )
            for j, ch in enumerate(arc_chapters)
        ]

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

        merged.append(
            NarrativeEvent(
                event_id=f"{arc_name}_event{event_counter}",
                arc_name=arc_name,
                chapters=[ch for ev in group for ch in ev.chapters],
                summary="",
            )
        )
        event_counter += 1
        i += len(group)

    return merged