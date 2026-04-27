from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

import config
from pipeline.core.common_json import read_json_file, write_json_file
from pipeline.core.entity_aliasing import NovelAliasRegistry, sanitize_text_entities, sanitize_text_list
from pipeline.step1_chunking import NarrativeEvent, VolumeArc
from pipeline.core.utils import chat_completion_json, get_deepseek_client


@dataclass
class PlotAtom:
    atom_id: str
    arc_name: str
    novel_source: str
    chapter_count: int = 1
    chapter_start: int = 0
    chapter_end: int = 0
    raw_characters: List[str] = field(default_factory=list)
    character_keys: List[str] = field(default_factory=list)
    raw_location: str = ""
    raw_core_action: str = ""
    raw_summary: str = ""
    characters: List[str] = field(default_factory=list)
    core_action: str = ""
    cultivation_elements: List[str] = field(default_factory=list)
    location: str = ""
    causality_precondition: str = ""
    causality_consequence: str = ""
    motivation: str = ""
    conflict_type: str = ""
    narrative_function: str = ""
    emotion: str = ""
    tension_level: int = 5
    summary: str = ""


_STEP2_FILENAME = "step2_extracted_plots.json"

_PASS1_SCHEMA = {
    "characters": ["角色名(关系)"],
    "core_action": "一句核心动作",
    "cultivation_elements": ["境界/法宝/丹药/秘境"],
    "location": "事件地点",
}

_PASS2_SCHEMA = {
    "causality_precondition": "前置原因",
    "causality_consequence": "后续结果",
    "motivation": "主要动机",
    "conflict_type": "冲突类型",
    "narrative_function": "叙事功能",
    "emotion": "情绪基调",
    "tension_level": 5,
}


def extract_all(novel_arcs: Dict[str, List[VolumeArc]]) -> Dict[str, List[PlotAtom]]:
    client = get_deepseek_client()
    result: Dict[str, List[PlotAtom]] = {}
    for novel_name, arcs in novel_arcs.items():
        print(f"[Step 2] Extracting plot atoms from: {novel_name}", flush=True)
        atoms: List[PlotAtom] = []
        character_memory: Dict[str, str] = {}
        alias_registry = NovelAliasRegistry(novel_name)
        all_events = [event for arc in arcs for event in arc.events]
        for event in tqdm(all_events, desc=f"[Step 2] {novel_name}", unit="event"):
            atoms.append(_extract_event(client, novel_name, event, character_memory, alias_registry))
        result[novel_name] = atoms
        print(f"[Step 2] -> {len(atoms)} atoms extracted from {novel_name}", flush=True)
    save_step2_output(result)
    return result


def save_step2_output(all_atoms: Dict[str, List[PlotAtom]]) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _STEP2_FILENAME
    payload = {novel_name: [asdict(atom) for atom in atoms] for novel_name, atoms in all_atoms.items()}
    write_json_file(out_path, payload)
    print(f"[Step 2] Intermediate output saved -> {out_path}", flush=True)
    return out_path


def load_step2_output(intermediate_dir: str | Path | None = None) -> Dict[str, List[PlotAtom]]:
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)
    in_path = inter_dir / _STEP2_FILENAME
    if not in_path.exists():
        raise FileNotFoundError(f"Step 2 intermediate file not found: {in_path}")
    raw = read_json_file(in_path)
    result: Dict[str, List[PlotAtom]] = {}
    for novel_name, atoms_data in raw.items():
        alias_registry = NovelAliasRegistry(novel_name)
        loaded_atoms = [_normalize_loaded_atom(PlotAtom(**item), alias_registry) for item in atoms_data]
        _backfill_legacy_chapter_ranges(loaded_atoms)
        result[novel_name] = loaded_atoms
    print(f"[Step 2] Loaded intermediate output from {in_path}")
    return result


def _extract_event(
    client,
    novel_name: str,
    event: NarrativeEvent,
    character_memory: Dict[str, str],
    alias_registry: NovelAliasRegistry,
) -> PlotAtom:
    text = event.summary or "\n".join(event.chapters)
    pass1 = _pass1_objective(client, event.arc_name, text, character_memory)
    pass2 = _pass2_subjective(client, event.arc_name, text, pass1)
    _update_character_memory(character_memory, pass1.get("characters", []))

    raw_characters = _coerce_list(pass1.get("characters", []))
    raw_location = str(pass1.get("location", "") or "").strip()
    raw_core_action = str(pass1.get("core_action", "") or "").strip()

    atom = PlotAtom(
        atom_id=event.event_id,
        arc_name=event.arc_name,
        novel_source=novel_name,
        chapter_count=max(1, int(getattr(event, "chapter_end", 0) or 0) - int(getattr(event, "chapter_start", 0) or 0) + 1),
        chapter_start=int(getattr(event, "chapter_start", 0) or 0),
        chapter_end=int(getattr(event, "chapter_end", 0) or 0),
    )
    atom.raw_characters = raw_characters
    atom.character_keys = alias_registry.character_keys_for(raw_characters)
    atom.raw_location = raw_location
    atom.raw_core_action = raw_core_action
    atom.raw_summary = _build_raw_atom_summary(event, pass1, pass2)
    atom.characters = alias_registry.alias_characters(raw_characters)
    atom.core_action = sanitize_text_entities(raw_core_action, raw_characters, raw_location, alias_registry)
    atom.cultivation_elements = sanitize_text_list(pass1.get("cultivation_elements", []), raw_characters, raw_location, alias_registry)
    atom.location = alias_registry.alias_location(raw_location)
    atom.causality_precondition = sanitize_text_entities(pass2.get("causality_precondition", ""), raw_characters, raw_location, alias_registry)
    atom.causality_consequence = sanitize_text_entities(pass2.get("causality_consequence", ""), raw_characters, raw_location, alias_registry)
    atom.motivation = sanitize_text_entities(pass2.get("motivation", ""), raw_characters, raw_location, alias_registry)
    atom.conflict_type = sanitize_text_entities(pass2.get("conflict_type", ""), raw_characters, raw_location, alias_registry)
    atom.narrative_function = sanitize_text_entities(pass2.get("narrative_function", ""), raw_characters, raw_location, alias_registry)
    atom.emotion = sanitize_text_entities(pass2.get("emotion", ""), raw_characters, raw_location, alias_registry)
    atom.tension_level = _clamp_int(pass2.get("tension_level", 5), 1, 10, default=5)
    atom.summary = _build_atom_summary(event, pass1, pass2, alias_registry)
    return atom


def _normalize_loaded_atom(atom: PlotAtom, alias_registry: NovelAliasRegistry) -> PlotAtom:
    raw_characters = atom.raw_characters[:] if atom.raw_characters else atom.characters[:]
    raw_location = atom.raw_location or atom.location
    atom.raw_characters = [str(item or "").strip() for item in raw_characters if str(item or "").strip()]
    atom.character_keys = atom.character_keys or alias_registry.character_keys_for(atom.raw_characters)
    atom.raw_location = str(raw_location or "").strip()
    atom.raw_core_action = atom.raw_core_action or atom.core_action
    atom.raw_summary = atom.raw_summary or atom.summary
    atom.characters = alias_registry.alias_characters(atom.raw_characters)
    atom.location = alias_registry.alias_location(atom.raw_location)
    atom.core_action = sanitize_text_entities(atom.core_action, atom.raw_characters, atom.raw_location, alias_registry)
    atom.cultivation_elements = sanitize_text_list(atom.cultivation_elements, atom.raw_characters, atom.raw_location, alias_registry)
    atom.causality_precondition = sanitize_text_entities(atom.causality_precondition, atom.raw_characters, atom.raw_location, alias_registry)
    atom.causality_consequence = sanitize_text_entities(atom.causality_consequence, atom.raw_characters, atom.raw_location, alias_registry)
    atom.motivation = sanitize_text_entities(atom.motivation, atom.raw_characters, atom.raw_location, alias_registry)
    atom.conflict_type = sanitize_text_entities(atom.conflict_type, atom.raw_characters, atom.raw_location, alias_registry)
    atom.narrative_function = sanitize_text_entities(atom.narrative_function, atom.raw_characters, atom.raw_location, alias_registry)
    atom.emotion = sanitize_text_entities(atom.emotion, atom.raw_characters, atom.raw_location, alias_registry)
    atom.summary = sanitize_text_entities(atom.summary, atom.raw_characters, atom.raw_location, alias_registry)
    if atom.chapter_start > 0 and atom.chapter_end >= atom.chapter_start:
        atom.chapter_count = max(1, atom.chapter_end - atom.chapter_start + 1)
    return atom


def _backfill_legacy_chapter_ranges(atoms: List[PlotAtom]) -> None:
    arc_cursors: Dict[str, int] = {}
    for atom in atoms:
        arc_name = str(atom.arc_name or "unknown_arc")
        cursor = arc_cursors.get(arc_name, 1)
        has_valid_range = atom.chapter_start > 0 and atom.chapter_end >= atom.chapter_start
        if has_valid_range:
            atom.chapter_count = max(1, atom.chapter_end - atom.chapter_start + 1)
            arc_cursors[arc_name] = max(cursor, atom.chapter_end + 1)
            continue

        span = max(1, int(getattr(atom, "chapter_count", 0) or 1))
        atom.chapter_start = cursor
        atom.chapter_end = cursor + span - 1
        atom.chapter_count = span
        arc_cursors[arc_name] = atom.chapter_end + 1


def _update_character_memory(character_memory: Dict[str, str], raw_characters: Any) -> None:
    if not isinstance(raw_characters, list):
        return
    for char_text in raw_characters:
        text = str(char_text or "").strip()
        if not text:
            continue
        match = re.match(r"^(.*?)[(（]([^()（）]+)[)）]$", text)
        if not match:
            continue
        name = match.group(1).strip()
        relation = match.group(2).strip()
        if not name:
            continue
        if name not in character_memory or (character_memory[name] in {"陌生人", "路人"} and relation):
            character_memory[name] = relation


def _build_raw_atom_summary(event: NarrativeEvent, pass1_result: Dict[str, Any], pass2_result: Dict[str, Any]) -> str:
    existing = str(event.summary or "").strip()
    if existing:
        return existing[:240]
    core_action = str(pass1_result.get("core_action", "") or "").strip()
    motivation = str(pass2_result.get("motivation", "") or "").strip()
    conflict_type = str(pass2_result.get("conflict_type", "") or "").strip()
    consequence = str(pass2_result.get("causality_consequence", "") or "").strip()
    emotion = str(pass2_result.get("emotion", "") or "").strip()
    fragments: List[str] = []
    if core_action:
        fragments.append(f"事件围绕{core_action}展开")
    if motivation:
        fragments.append(f"主要动因是{motivation}")
    if conflict_type:
        fragments.append(f"冲突落在{conflict_type}")
    if consequence:
        fragments.append(f"后续将引出{consequence}")
    if emotion:
        fragments.append(f"整体氛围偏向{emotion}")
    summary = "，".join(_dedupe_text_fragments(fragments))
    return (summary or _fallback_event_summary(event))[:240]


def _build_atom_summary(
    event: NarrativeEvent,
    pass1_result: Dict[str, Any],
    pass2_result: Dict[str, Any],
    alias_registry: NovelAliasRegistry,
) -> str:
    raw_characters = _coerce_list(pass1_result.get("characters", []))
    raw_location = str(pass1_result.get("location", "") or "").strip()
    raw_summary = _build_raw_atom_summary(event, pass1_result, pass2_result)
    return sanitize_text_entities(raw_summary, raw_characters, raw_location, alias_registry)[:240]


def _fallback_event_summary(event: NarrativeEvent) -> str:
    preview_lines: List[str] = []
    for chapter in event.chapters[:2]:
        for line in chapter.splitlines():
            clean = line.strip()
            if clean:
                preview_lines.append(clean)
                break
    if preview_lines:
        return " / ".join(dict.fromkeys(preview_lines))[:240]
    merged_preview = " ".join(chapter.strip().replace("\n", " ") for chapter in event.chapters[:2]).strip()
    return merged_preview[:240] if merged_preview else f"{event.arc_name} event"


def _dedupe_text_fragments(fragments: List[str]) -> List[str]:
    unique: List[str] = []
    for fragment in fragments:
        text = str(fragment or "").strip(" ，。；;、")
        if text and text not in unique:
            unique.append(text)
    return unique


def _coerce_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item or "").strip() for item in value if str(item or "").strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _clamp_int(value: Any, low: int, high: int, default: int = 0) -> int:
    try:
        num = int(value)
    except Exception:
        num = default
    return max(low, min(high, num))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _pass1_objective(client, arc_name: str, text: str, character_memory: Dict[str, str]) -> Dict[str, Any]:
    memory_text = json.dumps(character_memory, ensure_ascii=False) if character_memory else "暂无"
    prompt = (
        f"你是修仙小说事件抽取助手。\n"
        f"当前卷：{arc_name}\n"
        f"已有角色关系记忆：{memory_text}\n\n"
        f"请从下面文本中抽取客观要素，只返回 JSON。\n"
        f"Schema: {json.dumps(_PASS1_SCHEMA, ensure_ascii=False)}\n\n"
        f"文本：\n{text}"
    )
    raw = chat_completion_json(
        client,
        system="只返回合法 JSON。",
        user=prompt,
        json_mode=True,
    )
    data = _safe_parse(raw)
    return data if isinstance(data, dict) else {}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _pass2_subjective(client, arc_name: str, text: str, pass1_result: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        f"你是修仙小说事件逻辑分析助手。\n"
        f"当前卷：{arc_name}\n"
        f"已知客观要素：{json.dumps(pass1_result, ensure_ascii=False)}\n\n"
        f"请分析因果、动机、冲突与叙事功能，只返回 JSON。\n"
        f"Schema: {json.dumps(_PASS2_SCHEMA, ensure_ascii=False)}\n\n"
        f"文本：\n{text}"
    )
    raw = chat_completion_json(
        client,
        system="只返回合法 JSON。",
        user=prompt,
        json_mode=True,
    )
    data = _safe_parse(raw)
    return data if isinstance(data, dict) else {}


def _safe_parse(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        data = json.loads(text.strip())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
