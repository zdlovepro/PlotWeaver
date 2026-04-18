"""
step2_extraction.py – Dual-stage Plot Extraction

Uses DeepSeek API (JSON mode) in two passes:
  Pass 1 – Objective elements: characters, core actions, cultivation elements.
  Pass 2 – Subjective logic:   causality, motivations, conflict types, tension.

Features Long-Context full text processing and Entity State Tracking (Character Memory).
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
    conflict_type: str = ""
    narrative_function: str = ""
    emotion: str = ""
    tension_level: int = 5
    summary: str = ""


# ── Public API ──────────────────────────────────────────────────────────────

def extract_all(
    novel_arcs: dict[str, List[VolumeArc]]
) -> dict[str, List[PlotAtom]]:
    client = get_deepseek_client()
    result: dict[str, List[PlotAtom]] = {}
    for novel_name, arcs in novel_arcs.items():
        print(f"[Step 2] Extracting plot atoms from: {novel_name}", flush=True)
        atoms: List[PlotAtom] = []
        all_events = [event for arc in arcs for event in arc.events]

        # 核心改动：初始化一本小说的全局人物记忆字典 { "角色名": "关系" }
        character_memory: dict[str, str] = {}

        for event in tqdm(all_events, desc=f"[Step 2] {novel_name}", unit="event"):
            atom = _extract_event(client, novel_name, event, character_memory)
            atoms.append(atom)

            # 动态更新人物记忆库
            for char_str in atom.characters:
                # 尝试解析 "姓名(关系)" 的格式
                if "(" in char_str and ")" in char_str:
                    name = char_str.split("(")[0].strip()
                    rel = char_str.split("(")[1].split(")")[0].strip()

                    # 如果是一个新人物，或者之前是"陌生人"但现在关系明确了，就更新记忆
                    if name not in character_memory or (rel != "陌生人" and "陌生人" in character_memory.get(name, "")):
                        character_memory[name] = rel

        result[novel_name] = atoms
        print(f"[Step 2]   → {len(atoms)} atoms extracted from {novel_name}", flush=True)
    save_step2_output(result)
    return result


# ── Intermediate I/O ────────────────────────────────────────────────────────

_STEP2_FILENAME = "step2_extracted_plots.json"

def save_step2_output(all_atoms: dict[str, List[PlotAtom]]) -> Path:
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


# ── Internal helpers ────────────────────────────────────────────────────────

def _extract_event(client, novel_name: str, event: NarrativeEvent, character_memory: dict[str, str]) -> PlotAtom:
    # 完整读取事件内的所有章节文本
    text = event.summary or "\n".join(event.chapters)

    # Pass 1 传入人物记忆库
    pass1 = _pass1_objective(client, event.arc_name, text, character_memory)
    pass2 = _pass2_subjective(client, event.arc_name, text, pass1)
    atom_summary = _build_atom_summary(event, pass1, pass2)

    atom = PlotAtom(
        atom_id=event.event_id,
        arc_name=event.arc_name,
        novel_source=novel_name,
        summary=atom_summary,
    )

    atom.characters = pass1.get("characters", [])
    atom.core_action = pass1.get("core_action", "")
    atom.cultivation_elements = pass1.get("cultivation_elements", [])
    atom.location = pass1.get("location", "")

    atom.causality_precondition = pass2.get("causality_precondition", "")
    atom.causality_consequence = pass2.get("causality_consequence", "")
    atom.motivation = pass2.get("motivation", "")
    atom.conflict_type = pass2.get("conflict_type", "")
    atom.narrative_function = pass2.get("narrative_function", "")
    atom.emotion = pass2.get("emotion", "")
    atom.tension_level = int(pass2.get("tension_level", 5))

    return atom


def _build_atom_summary(
    event: NarrativeEvent,
    pass1_result: Dict[str, Any],
    pass2_result: Dict[str, Any],
) -> str:
    existing = (event.summary or "").strip()
    if existing:
        return existing[:240]

    core_action = str(pass1_result.get("core_action", "")).strip()
    location = str(pass1_result.get("location", "")).strip()
    motivation = str(pass2_result.get("motivation", "")).strip()
    conflict_type = str(pass2_result.get("conflict_type", "")).strip()
    consequence = str(pass2_result.get("causality_consequence", "")).strip()
    emotion = str(pass2_result.get("emotion", "")).strip()
    characters = _clean_character_names(pass1_result.get("characters", []), limit=3)

    fragments: List[str] = []
    if characters and core_action:
        fragments.append(f"{'、'.join(characters)}卷入{core_action}")
    elif core_action:
        fragments.append(core_action)

    if location:
        fragments.append(f"地点在{location}")
    if motivation:
        fragments.append(f"动机是{motivation}")
    if conflict_type:
        fragments.append(f"冲突集中在{conflict_type}")
    if consequence:
        fragments.append(f"结果导致{consequence}")
    if emotion:
        fragments.append(f"基调偏{emotion}")

    summary = "；".join(_dedupe_text_fragments(fragments))
    if summary:
        return summary[:240]
    return _fallback_event_summary(event)


def _clean_character_names(raw_characters: Any, limit: int = 3) -> List[str]:
    if not isinstance(raw_characters, list):
        return []

    cleaned: List[str] = []
    for raw_name in raw_characters:
        name = str(raw_name or "").strip()
        if not name:
            continue
        for bracket in ("(", "（"):
            if bracket in name:
                name = name.split(bracket, 1)[0].strip()
        if name and name not in cleaned:
            cleaned.append(name)
        if len(cleaned) >= limit:
            break
    return cleaned


def _dedupe_text_fragments(fragments: List[str]) -> List[str]:
    unique: List[str] = []
    for fragment in fragments:
        text = fragment.strip(" ；;，,")
        if text and text not in unique:
            unique.append(text)
    return unique


def _fallback_event_summary(event: NarrativeEvent) -> str:
    preview_lines: List[str] = []
    for chapter in event.chapters[:2]:
        for line in chapter.splitlines():
            line = line.strip()
            if line:
                preview_lines.append(line)
                break

    if preview_lines:
        return " / ".join(dict.fromkeys(preview_lines))[:240]

    merged_preview = " ".join(chapter.strip().replace("\n", " ") for chapter in event.chapters[:2]).strip()
    if merged_preview:
        return merged_preview[:240]
    return f"{event.arc_name} event"

# 修改 Schema：要求输出格式必须为 姓名(关系)
_PASS1_SCHEMA = """\
{
  "characters": ["角色名(与主角的关系，如：主角/师尊/敌人/朋友/陌生人/路人等。若前文已有，请尽量保持一致，若本章出现新身份则更新)"],
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
def _pass1_objective(client, arc_name: str, text: str, character_memory: dict[str, str]) -> Dict[str, Any]:
    # 将前文记忆格式化输出到提示词中
    mem_str = json.dumps(character_memory, ensure_ascii=False) if character_memory else "暂无，本卷为起始阶段。"

    prompt = (
        f"你是修仙小说情节客观要素抽取专家。\n"
        f"当前卷目进度：{arc_name}\n"
        f"前文已积累的人物关系记忆库：{mem_str}\n\n"
        f"请仅从以下情节摘要中，提取客观事实性信息，严格按照JSON Schema输出：\n"
        f"Schema:\n{_PASS1_SCHEMA}\n\n"
        f"注意：提取人物时，请尽量参考前文关系。如果本章看不出人物关系，可填'陌生人'。\n\n"
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
        f"当前卷目进度：{arc_name}\n\n"
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
