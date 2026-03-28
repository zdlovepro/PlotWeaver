"""
step2_extraction.py – Dual-stage Plot Extraction

Uses DeepSeek API (JSON mode) in two passes:
  Pass 1 – Objective elements: characters, core actions, cultivation elements.
  Pass 2 – Subjective logic:   causality, motivations, conflict types, tension.

Output:
  List[PlotAtom] – enriched, structured plot atoms ready for ChromaDB ingestion.
  The result is also persisted to ``intermediate_dir/step2_extracted_plots.json``
  for pipeline resume capability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

import config
from pipeline.step1_chunking import NarrativeEvent, VolumeArc
from pipeline.utils import get_deepseek_client, chat_completion_json


@dataclass
class PlotAtom:
    atom_id: str
    arc_name: str
    novel_source: str
    # Pass-1 fields
    characters: List[str] = field(default_factory=list)
    core_action: str = ""
    cultivation_elements: List[str] = field(default_factory=list)
    location: str = ""
    # Pass-2 fields
    causality_precondition: str = ""
    causality_consequence: str = ""
    motivation: str = ""
    conflict_type: str = ""      # e.g. 宗门争斗 / 天劫 / 机缘争夺
    narrative_function: str = "" # e.g. 打脸 / 传承 / 复仇
    emotion: str = ""
    tension_level: int = 5       # 1-10
    # Original text summary
    summary: str = ""


# ── Public API ────────────────────────────────────────────────────────────────

def extract_all(
    novel_arcs: dict[str, List[VolumeArc]]
) -> dict[str, List[PlotAtom]]:
    """
    Run dual-stage extraction for every novel's events.

    Args:
        novel_arcs: {novel_filename: [VolumeArc, ...]}

    Returns:
        {novel_filename: [PlotAtom, ...]}

    The result is saved to ``intermediate_dir/step2_extracted_plots.json``.
    """
    client = get_deepseek_client()
    result: dict[str, List[PlotAtom]] = {}
    for novel_name, arcs in novel_arcs.items():
        print(f"[Step 2] Extracting plot atoms from: {novel_name}", flush=True)
        atoms: List[PlotAtom] = []
        all_events = [event for arc in arcs for event in arc.events]
        for event in tqdm(all_events, desc=f"[Step 2] {novel_name}", unit="event"):
            atom = _extract_event(client, novel_name, event)
            atoms.append(atom)
        result[novel_name] = atoms
        print(f"[Step 2]   → {len(atoms)} atoms extracted from {novel_name}", flush=True)
    save_step2_output(result)
    return result


# ── Intermediate I/O ──────────────────────────────────────────────────────────

_STEP2_FILENAME = "step2_extracted_plots.json"


def save_step2_output(all_atoms: dict[str, List[PlotAtom]]) -> Path:
    """Serialise *all_atoms* to ``intermediate_dir/step2_extracted_plots.json``."""
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _STEP2_FILENAME
    serialisable = {
        novel_name: [asdict(atom) for atom in atoms]
        for novel_name, atoms in all_atoms.items()
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, ensure_ascii=False, indent=2)
    print(f"[Step 2] Intermediate output saved → {out_path}", flush=True)
    return out_path


def load_step2_output(intermediate_dir: str | Path | None = None) -> dict[str, List[PlotAtom]]:
    """Load previously saved Step 2 output from ``intermediate_dir/step2_extracted_plots.json``."""
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)
    in_path = inter_dir / _STEP2_FILENAME
    if not in_path.exists():
        raise FileNotFoundError(
            f"Step 2 intermediate file not found: {in_path}\n"
            "Run the pipeline from Step 1 or Step 2 first to generate it."
        )
    with open(in_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    result: dict[str, List[PlotAtom]] = {}
    for novel_name, atoms_data in raw.items():
        result[novel_name] = [PlotAtom(**d) for d in atoms_data]
    print(f"[Step 2] Loaded intermediate output from {in_path}")
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_event(client, novel_name: str, event: NarrativeEvent) -> PlotAtom:
    text = event.summary or "\n".join(event.chapters[:1])[:config.MAX_TEXT_CHUNK_LENGTH]

    pass1 = _pass1_objective(client, event.arc_name, text)
    pass2 = _pass2_subjective(client, event.arc_name, text, pass1)

    atom = PlotAtom(
        atom_id=event.event_id,
        arc_name=event.arc_name,
        novel_source=novel_name,
        summary=event.summary,
    )

    # Populate from Pass 1
    atom.characters = pass1.get("characters", [])
    atom.core_action = pass1.get("core_action", "")
    atom.cultivation_elements = pass1.get("cultivation_elements", [])
    atom.location = pass1.get("location", "")

    # Populate from Pass 2
    atom.causality_precondition = pass2.get("causality_precondition", "")
    atom.causality_consequence = pass2.get("causality_consequence", "")
    atom.motivation = pass2.get("motivation", "")
    atom.conflict_type = pass2.get("conflict_type", "")
    atom.narrative_function = pass2.get("narrative_function", "")
    atom.emotion = pass2.get("emotion", "")
    atom.tension_level = int(pass2.get("tension_level", 5))

    return atom


_PASS1_SCHEMA = """\
{
  "characters": ["角色名列表"],
  "core_action": "本事件最核心的一个动作/行为（字符串）",
  "cultivation_elements": ["涉及的境界/功法/灵宝/丹药列表"],
  "location": "事件发生地点"
}"""

_PASS2_SCHEMA = """\
{
  "causality_precondition": "该事件发生的前置原因",
  "causality_consequence": "该事件导致的后续结果",
  "motivation": "主要角色的核心动机",
  "conflict_type": "冲突类型（如：宗门争斗/天劫/机缘争夺/复仇/传承/情感纠葛）",
  "narrative_function": "叙事功能（如：打脸/机缘/天劫/逆袭/伏笔/世界观展示）",
  "emotion": "主要情感基调（如：热血/悲壮/紧张/轻松/压抑）",
  "tension_level": 7
}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _pass1_objective(client, arc_name: str, text: str) -> Dict[str, Any]:
    prompt = (
        f"你是修仙小说情节客观要素抽取专家。\n"
        f"当前境界弧：{arc_name}\n\n"
        f"请仅从以下情节摘要中，提取客观事实性信息，严格按照JSON Schema输出：\n"
        f"Schema:\n{_PASS1_SCHEMA}\n\n"
        f"情节内容：\n{text}"
    )
    raw = chat_completion_json(
        client,
        system="你是专业的修仙小说情节结构化抽取助手，只输出合法JSON。",
        user=prompt,
        json_mode=True,
    )
    return _safe_parse(raw)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _pass2_subjective(
    client, arc_name: str, text: str, pass1_result: Dict[str, Any]
) -> Dict[str, Any]:
    prompt = (
        f"你是修仙小说情节逻辑分析专家。\n"
        f"当前境界弧：{arc_name}\n\n"
        f"已知客观要素：{json.dumps(pass1_result, ensure_ascii=False)}\n\n"
        f"请结合以下原文，分析主观逻辑信息，严格按照JSON Schema输出：\n"
        f"Schema:\n{_PASS2_SCHEMA}\n\n"
        f"情节内容：\n{text}"
    )
    raw = chat_completion_json(
        client,
        system="你是专业的修仙小说叙事逻辑分析助手，只输出合法JSON。",
        user=prompt,
        json_mode=True,
    )
    return _safe_parse(raw)


def _safe_parse(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to extract JSON block from markdown code fences
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    return {}
