"""
step1_chunking.py – Physical Chunking & Arc Anchoring

Responsibilities:
  - Read raw .txt files from input_dir.
  - Use regex to split text into chapters (e.g., matching "第.*章").
  - Detect major cultivation realm transitions to split text into Volume Arcs.
  - Group every N chapters (based on semantic_chunk_min_chapters from config)
    into a physical chunk (NarrativeEvent).
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
    each containing physically chunked NarrativeEvents.

    No LLM calls are made. Chapters are split by regex and grouped into chunks
    of size between ``SEMANTIC_CHUNK_MIN_CHAPTERS`` and
    ``SEMANTIC_CHUNK_MAX_CHAPTERS`` (from config) using pure Python logic.
    """
    novel_path = Path(novel_path)
    text = novel_path.read_text(encoding="utf-8")
    chapters = _split_into_chapters(text)

    arc_boundaries = _detect_arc_boundaries(chapters)
    arcs = _build_volume_arcs(chapters, arc_boundaries)

    for arc in arcs:
        arc.events = _physical_chunk_arc(arc.arc_name, arc.events)

    return arcs


def process_all_novels(input_dir: str | Path) -> dict[str, List[VolumeArc]]:
    """
    Process every .txt file in *input_dir* and return a mapping
    {filename: List[VolumeArc]}.

    The result is saved to ``intermediate_dir/step1_chunks.json``.
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


# ── Intermediate I/O ──────────────────────────────────────────────────────────

_STEP1_FILENAME = "step1_chunks.json"


def save_step1_output(novel_arcs: dict[str, List[VolumeArc]]) -> Path:
    """Serialise *novel_arcs* to ``intermediate_dir/step1_chunks.json``."""
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
    """Load previously saved Step 1 output from ``intermediate_dir/step1_chunks.json``."""
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


def _physical_chunk_arc(
    arc_name: str, raw_events: List[NarrativeEvent]
) -> List[NarrativeEvent]:
    """
    Merge consecutive raw chapter events into physical NarrativeEvent chunks.
    Groups between MIN_CHAPTERS and MAX_CHAPTERS chapters together.
    No LLM is used – the summary field is left empty so that Step 2 reads
    the raw chapter text directly.
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
