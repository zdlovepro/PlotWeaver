from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

import config
from pipeline.core.common_json import read_json_file, safe_json_load, write_json_file
from pipeline.core.entity_aliasing import NovelAliasRegistry, sanitize_text_entities, sanitize_text_list
from pipeline.core.utils import chat_completion_json, get_deepseek_client
from pipeline.step1_chunking import NarrativeEvent, VolumeArc


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
    semantic_window_id: str = ""
    atom_index_in_window: int = 0
    atom_type: str = ""
    actor: str = ""
    goal: str = ""
    obstacle: str = ""
    action: str = ""
    outcome: str = ""
    state_delta: Dict[str, Any] = field(default_factory=dict)
    is_complete: bool = True
    source_spans: List[Dict[str, Any]] = field(default_factory=list)
    boundary_reason: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class SemanticWindow:
    semantic_window_id: str
    novel_source: str
    arc_name: str
    source_chunk_ids: List[str] = field(default_factory=list)
    source_spans: List[Dict[str, Any]] = field(default_factory=list)
    text: str = ""
    window_type: str = "normal"
    contains_tail_from_previous: bool = False
    carried_tail_id: str = ""
    has_incomplete_tail: bool = False
    incomplete_tail_id: str = ""
    notes: str = ""


@dataclass
class IncompleteTail:
    tail_id: str
    novel_source: str
    arc_name: str
    from_semantic_window_id: str
    from_source_chunk_ids: List[str] = field(default_factory=list)
    source_spans: List[Dict[str, Any]] = field(default_factory=list)
    text: str = ""
    summary: str = ""
    reason: str = ""
    missing_parts: List[str] = field(default_factory=list)
    carry_to_next_chunk: bool = True
    carried_to_semantic_window_id: str = ""
    status: str = "open"


_STEP2_FILENAME = "step2_extracted_plots.json"
_STEP2_PLOT_ATOMS_FILENAME = "step2_plot_atoms.json"
_STEP2_WINDOWS_FILENAME = "step2_semantic_windows.json"
_STEP2_TAILS_FILENAME = "step2_tail_carry_log.json"

_ATOMIZATION_SCHEMA = {
    "complete_atoms": [
        {
            "atom_type": "setup|trigger|conflict|choice|attempt|reversal|reveal|gain|loss|relationship|hook_open|hook_close|consequence|transition",
            "chapter_start": 1,
            "chapter_end": 2,
            "summary": "一句话摘要",
            "raw_characters": ["角色名(关系)"],
            "raw_location": "地点",
            "core_action": "核心动作",
            "cultivation_elements": ["境界/法宝/丹药/秘境/功法"],
            "actor": "主要行动者",
            "goal": "行动目标",
            "obstacle": "主要阻碍",
            "action": "采取的行动",
            "outcome": "明确结果",
            "state_delta": {
                "reputation": "+/-/optional",
                "resource": "+/-/optional",
                "relationship": "+/-/optional",
                "injury": "+/-/optional",
                "secret_exposure": "+/-/optional",
                "enemy_attention": "+/-/optional",
                "open_hook": "+/optional",
            },
            "causality_precondition": "前置原因",
            "causality_consequence": "后续结果",
            "motivation": "动机",
            "conflict_type": "冲突类型",
            "narrative_function": "叙事功能",
            "emotion": "情绪基调",
            "tension_level": 1,
            "boundary_reason": "为什么它是完整 PlotAtom",
        }
    ],
    "incomplete_tail": {
        "exists": True,
        "chapter_start": 7,
        "chapter_end": 8,
        "summary": "尾部未闭合剧情摘要",
        "text_excerpt": "尾部原文摘录，不超过1200字",
        "reason": "为什么它不是完整 PlotAtom",
        "missing_parts": ["outcome", "state_delta"],
        "carry_to_next_chunk": True,
    },
    "boundary_notes": ["说明哪些章节被识别为完整原子，哪些章节需要顺延"],
}

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

_FALLBACK_TAIL_KEYWORDS = (
    "忽然",
    "就在这时",
    "正要",
    "刚刚",
    "才发现",
    "只见",
    "一道身影",
    "新的任务",
    "刚进入",
    "敌人出现",
    "悬念",
    "未完",
    "线索",
    "伏笔",
    "下一刻",
)

_SETUP_KEYWORDS = ("开局", "铺垫", "初到", "登场", "入门")
_TRIGGER_KEYWORDS = ("触发", "接到", "发现", "遭遇", "收到")
_CONFLICT_KEYWORDS = ("冲突", "争夺", "追杀", "围攻", "对峙", "压迫")
_CHOICE_KEYWORDS = ("选择", "决定", "取舍", "站队")
_ATTEMPT_KEYWORDS = ("尝试", "潜入", "试探", "行动", "布局")
_REVERSAL_KEYWORDS = ("反转", "翻盘", "逆袭", "破局", "打脸")
_REVEAL_KEYWORDS = ("揭露", "真相", "身份暴露", "识破")
_GAIN_KEYWORDS = ("得到", "获得", "夺得", "拜入", "突破", "收获")
_LOSS_KEYWORDS = ("失去", "损失", "受伤", "牺牲", "折损")
_RELATIONSHIP_KEYWORDS = ("关系", "和解", "误会", "结盟", "决裂")
_HOOK_OPEN_KEYWORDS = ("伏笔", "线索", "后续", "隐患", "更大", "钩子")
_HOOK_CLOSE_KEYWORDS = ("收束", "解决", "脱身", "斩杀", "平息")
_CONSEQUENCE_KEYWORDS = ("后果", "影响", "代价", "余波")


def extract_all(novel_arcs: Dict[str, List[VolumeArc]]) -> Dict[str, List[PlotAtom]]:
    client = get_deepseek_client()
    all_atoms: Dict[str, List[PlotAtom]] = {}
    all_windows: Dict[str, List[SemanticWindow]] = {}
    all_tails: Dict[str, List[IncompleteTail]] = {}

    for novel_name, arcs in novel_arcs.items():
        print(f"[Step 2] Processing novel: {novel_name}", flush=True)
        alias_registry = NovelAliasRegistry(novel_name)
        atoms, windows, tails, stats = _build_semantic_windows_and_extract_atoms(
            client=client,
            novel_name=novel_name,
            arcs=arcs,
            alias_registry=alias_registry,
        )
        all_atoms[novel_name] = atoms
        all_windows[novel_name] = windows
        all_tails[novel_name] = tails
        print(f"[Step 2] Physical chunks: {stats['physical_chunks']}", flush=True)
        print(f"[Step 2] Semantic windows: {stats['semantic_windows']}", flush=True)
        print(f"[Step 2] Plot atoms extracted: {stats['plot_atoms']}", flush=True)
        print(f"[Step 2] Incomplete tails detected: {stats['tails_detected']}", flush=True)
        print(f"[Step 2] Tails merged: {stats['tails_merged']}", flush=True)
        print(f"[Step 2] Tails dropped at end: {stats['tails_dropped']}", flush=True)

    save_step2_output(all_atoms)
    _save_plot_atoms_output(all_atoms)
    _save_semantic_windows_output(all_windows)
    _save_tail_carry_log_output(all_tails)
    return all_atoms


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
        loaded_atoms = [
            _normalize_loaded_atom(PlotAtom(**_coerce_loaded_atom_record(item)), alias_registry)
            for item in atoms_data
        ]
        _backfill_legacy_chapter_ranges(loaded_atoms)
        result[novel_name] = loaded_atoms
    print(f"[Step 2] Loaded intermediate output from {in_path}")
    return result


def load_step2_semantic_windows(intermediate_dir: str | Path | None = None) -> Dict[str, List[SemanticWindow]]:
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)
    in_path = inter_dir / _STEP2_WINDOWS_FILENAME
    if not in_path.exists():
        raise FileNotFoundError(f"Step 2 semantic windows file not found: {in_path}")
    raw = read_json_file(in_path)
    result: Dict[str, List[SemanticWindow]] = {}
    for novel_name, windows_data in raw.items():
        result[novel_name] = [SemanticWindow(**item) for item in windows_data]
    print(f"[Step 2] Loaded semantic windows from {in_path}")
    return result


def load_step2_tail_carry_log(intermediate_dir: str | Path | None = None) -> Dict[str, List[IncompleteTail]]:
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)
    in_path = inter_dir / _STEP2_TAILS_FILENAME
    if not in_path.exists():
        raise FileNotFoundError(f"Step 2 tail carry log not found: {in_path}")
    raw = read_json_file(in_path)
    result: Dict[str, List[IncompleteTail]] = {}
    for novel_name, tails_data in raw.items():
        result[novel_name] = [IncompleteTail(**item) for item in tails_data]
    print(f"[Step 2] Loaded tail carry log from {in_path}")
    return result


def _build_semantic_windows_and_extract_atoms(
    client,
    novel_name: str,
    arcs: List[VolumeArc],
    alias_registry: NovelAliasRegistry,
) -> Tuple[List[PlotAtom], List[SemanticWindow], List[IncompleteTail], Dict[str, int]]:
    physical_chunks = [event for arc in arcs for event in arc.events]
    atoms: List[PlotAtom] = []
    semantic_windows: List[SemanticWindow] = []
    tails: List[IncompleteTail] = []
    carry_tail: Optional[IncompleteTail] = None
    character_memory: Dict[str, str] = {}
    tails_merged = 0
    tails_dropped = 0

    for window_index, chunk in enumerate(
        tqdm(physical_chunks, desc=f"[Step 2] {novel_name}", unit="chunk"),
        start=1,
    ):
        semantic_window = _create_semantic_window(
            novel_name=novel_name,
            chunk=chunk,
            window_index=window_index,
            carry_tail=carry_tail,
        )
        if carry_tail:
            carry_tail.status = "merged"
            carry_tail.carried_to_semantic_window_id = semantic_window.semantic_window_id
            tails_merged += 1

        try:
            ai_result = _atomize_semantic_window(
                client=client,
                semantic_window=semantic_window,
                character_memory=character_memory,
            )
            window_atoms, next_tail, notes = _parse_atomization_result(
                ai_result=ai_result,
                semantic_window=semantic_window,
                novel_name=novel_name,
                alias_registry=alias_registry,
                character_memory=character_memory,
            )
        except RetryError:
            window_atoms, next_tail, notes = _fallback_atomize_semantic_window(
                client=client,
                semantic_window=semantic_window,
                novel_name=novel_name,
                alias_registry=alias_registry,
                character_memory=character_memory,
                failure_reason="atomizer_retry_exhausted",
            )
        except Exception as exc:
            window_atoms, next_tail, notes = _fallback_atomize_semantic_window(
                client=client,
                semantic_window=semantic_window,
                novel_name=novel_name,
                alias_registry=alias_registry,
                character_memory=character_memory,
                failure_reason=f"atomizer_failure:{exc}",
            )

        semantic_window.has_incomplete_tail = next_tail is not None
        semantic_window.incomplete_tail_id = next_tail.tail_id if next_tail else ""
        semantic_window.notes = " | ".join(note for note in notes if note) or semantic_window.notes
        semantic_windows.append(semantic_window)
        atoms.extend(window_atoms)
        if next_tail:
            tails.append(next_tail)
        carry_tail = next_tail

    if carry_tail:
        carry_tail.status = "dropped_at_end"
        carry_tail.reason = _append_note(carry_tail.reason, "源文本结束，tail 未闭合")
        carry_tail.carry_to_next_chunk = False
        tails_dropped += 1

    stats = {
        "physical_chunks": len(physical_chunks),
        "semantic_windows": len(semantic_windows),
        "plot_atoms": len(atoms),
        "tails_detected": len(tails),
        "tails_merged": tails_merged,
        "tails_dropped": tails_dropped,
    }
    return atoms, semantic_windows, tails, stats


def _create_semantic_window(
    novel_name: str,
    chunk: NarrativeEvent,
    window_index: int,
    carry_tail: Optional[IncompleteTail],
) -> SemanticWindow:
    source_chunk_ids: List[str] = []
    source_spans: List[Dict[str, Any]] = []
    notes: List[str] = []

    if carry_tail:
        source_chunk_ids.extend(carry_tail.from_source_chunk_ids)
        source_spans.extend(
            {
                "chunk_id": span.get("chunk_id", ""),
                "chapter_start": int(span.get("chapter_start", 0) or 0),
                "chapter_end": int(span.get("chapter_end", 0) or 0),
                "span_role": "carried_tail",
            }
            for span in carry_tail.source_spans
        )
        notes.append(f"包含上一个 tail: {carry_tail.tail_id}")

    source_chunk_ids.append(chunk.event_id)
    source_spans.append(
        {
            "chunk_id": chunk.event_id,
            "chapter_start": int(getattr(chunk, "chapter_start", 0) or 0),
            "chapter_end": int(getattr(chunk, "chapter_end", 0) or 0),
            "span_role": "current_chunk",
        }
    )

    arc_name = carry_tail.arc_name if carry_tail and carry_tail.arc_name == chunk.arc_name else chunk.arc_name
    if carry_tail and carry_tail.arc_name != chunk.arc_name:
        notes.append(f"tail 跨卷拼接: {carry_tail.arc_name} -> {chunk.arc_name}")

    semantic_window_id = f"{_novel_prefix(novel_name)}_window_{window_index:04d}"
    return SemanticWindow(
        semantic_window_id=semantic_window_id,
        novel_source=novel_name,
        arc_name=arc_name,
        source_chunk_ids=_dedupe_list(source_chunk_ids),
        source_spans=source_spans,
        text=_build_semantic_window_text(carry_tail, chunk),
        window_type="reflowed" if carry_tail else "normal",
        contains_tail_from_previous=carry_tail is not None,
        carried_tail_id=carry_tail.tail_id if carry_tail else "",
        has_incomplete_tail=False,
        incomplete_tail_id="",
        notes=" | ".join(notes) if notes else f"基于物理 chunk {chunk.event_id}",
    )


def _build_semantic_window_text(carry_tail: Optional[IncompleteTail], chunk: NarrativeEvent) -> str:
    parts: List[str] = []
    if carry_tail and carry_tail.text.strip():
        parts.append(
            "[承接上一段未闭合剧情]\n"
            f"tail_id: {carry_tail.tail_id}\n"
            f"tail_summary: {carry_tail.summary}\n"
            f"tail_text:\n{carry_tail.text.strip()}"
        )
    parts.append(
        "[当前物理 chunk]\n"
        f"chunk_id: {chunk.event_id}\n"
        f"chapter_range: {int(getattr(chunk, 'chapter_start', 0) or 0)}-{int(getattr(chunk, 'chapter_end', 0) or 0)}\n"
        f"text:\n{_event_text(chunk)}"
    )
    return "\n\n".join(parts).strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _atomize_semantic_window(
    client,
    semantic_window: SemanticWindow,
    character_memory: Dict[str, str],
) -> Dict[str, Any]:
    source_span_text = json.dumps(semantic_window.source_spans, ensure_ascii=False)
    memory_text = json.dumps(character_memory, ensure_ascii=False) if character_memory else "暂无"
    prompt = (
        "你是小说情节原子识别器。\n"
        "请把输入文本拆分为多个完整 PlotAtom，并识别尾部未闭合剧情。\n"
        "每个完整 PlotAtom 必须同时具备：actor、goal、obstacle、action、outcome、state_delta。\n"
        "如果某段内容只是新事件的开头，例如接到新任务、刚进入新地点、敌人刚出现、冲突刚触发、悬念刚抛出，"
        "但还没有 outcome 和 state_delta，则不要强行总结成 PlotAtom，必须放入 incomplete_tail。\n\n"
        "硬性规则：\n"
        "1. 没有 outcome 的内容不能成为完整 PlotAtom。\n"
        "2. 没有 state_delta 的内容不能成为完整 PlotAtom。\n"
        "3. 纯场景开启、任务开启、敌人刚登场、悬念句不能单独成为完整 PlotAtom。\n"
        "4. 一个语义窗口中可以包含多个 complete_atoms。\n"
        "5. 章节范围必须尽量准确。\n"
        "6. 如果不确定某尾部是否完整，宁可放入 incomplete_tail，也不要强行生成 PlotAtom。\n\n"
        f"当前小说: {semantic_window.novel_source}\n"
        f"当前卷: {semantic_window.arc_name}\n"
        f"semantic_window_id: {semantic_window.semantic_window_id}\n"
        f"source_chunk_ids: {semantic_window.source_chunk_ids}\n"
        f"source_spans: {source_span_text}\n"
        f"contains_tail_from_previous: {semantic_window.contains_tail_from_previous}\n"
        f"carried_tail_id: {semantic_window.carried_tail_id or 'none'}\n"
        f"角色关系记忆: {memory_text}\n\n"
        f"返回 JSON Schema: {json.dumps(_ATOMIZATION_SCHEMA, ensure_ascii=False)}\n\n"
        "下面是待识别文本：\n"
        f"{semantic_window.text}"
    )
    raw = chat_completion_json(
        client,
        system="只返回合法 JSON，不要输出额外解释。",
        user=prompt,
        json_mode=True,
        temperature=0.2,
    )
    data = safe_json_load(raw)
    if not isinstance(data, dict):
        raise ValueError("Atomizer did not return a JSON object.")
    if not isinstance(data.get("complete_atoms", []), list):
        raise ValueError("Atomizer result missing complete_atoms list.")
    if not isinstance(data.get("incomplete_tail", {}), dict):
        raise ValueError("Atomizer result missing incomplete_tail object.")
    return data


def _parse_atomization_result(
    ai_result: Dict[str, Any],
    semantic_window: SemanticWindow,
    novel_name: str,
    alias_registry: NovelAliasRegistry,
    character_memory: Dict[str, str],
) -> Tuple[List[PlotAtom], Optional[IncompleteTail], List[str]]:
    raw_atoms = ai_result.get("complete_atoms", [])
    raw_tail = ai_result.get("incomplete_tail", {})
    boundary_notes = [str(item).strip() for item in ai_result.get("boundary_notes", []) if str(item).strip()]

    if not raw_atoms and not bool(raw_tail.get("exists", False)):
        raise ValueError("Atomizer returned neither complete atoms nor incomplete tail.")

    parsed_atoms: List[PlotAtom] = []
    deferred_tail_from_atom: Optional[IncompleteTail] = None

    for atom_index, raw_atom in enumerate(raw_atoms, start=1):
        atom = _coerce_plot_atom_from_ai(
            raw_atom=raw_atom,
            semantic_window=semantic_window,
            novel_name=novel_name,
            alias_registry=alias_registry,
            atom_index=atom_index,
        )
        issues = _validate_atom_completeness(atom)
        is_last_atom = atom_index == len(raw_atoms)
        if issues:
            if is_last_atom:
                deferred_tail_from_atom = _tail_from_incomplete_atom(atom, semantic_window, issues)
                boundary_notes.append(
                    f"最后一个 atom 因缺失 {','.join(issues)} 被转为 incomplete_tail"
                )
                continue
            atom = _repair_incomplete_middle_atom(atom, issues)
            boundary_notes.append(
                f"中间 atom {atom.atom_id} 缺失 {','.join(issues)}，已按兼容模式补全并记录 warning"
            )

        parsed_atoms.append(atom)
        _update_character_memory(character_memory, atom.raw_characters)

    parsed_tail = _coerce_incomplete_tail_from_ai(
        raw_tail=raw_tail,
        semantic_window=semantic_window,
        fallback_tail=deferred_tail_from_atom,
    )
    return parsed_atoms, parsed_tail, boundary_notes


def _coerce_plot_atom_from_ai(
    raw_atom: Dict[str, Any],
    semantic_window: SemanticWindow,
    novel_name: str,
    alias_registry: NovelAliasRegistry,
    atom_index: int,
) -> PlotAtom:
    window_start, window_end = _window_chapter_range(semantic_window)
    raw_characters = _coerce_list(raw_atom.get("raw_characters", []))
    raw_location = str(raw_atom.get("raw_location", "") or "").strip()
    raw_core_action = str(raw_atom.get("core_action", "") or "").strip()
    raw_summary = str(raw_atom.get("summary", "") or "").strip()
    chapter_start = _clamp_int(raw_atom.get("chapter_start", window_start), window_start, window_end, default=window_start)
    chapter_end = _clamp_int(raw_atom.get("chapter_end", chapter_start), chapter_start, window_end, default=chapter_start)
    state_delta = _coerce_state_delta(raw_atom.get("state_delta", {}))
    atom_type = str(raw_atom.get("atom_type", "") or "").strip() or _infer_atom_type(
        raw_summary,
        raw_atom.get("narrative_function", ""),
        raw_atom.get("conflict_type", ""),
        raw_core_action,
        raw_atom.get("outcome", ""),
    )

    atom = PlotAtom(
        atom_id=f"{semantic_window.semantic_window_id}_atom{atom_index}",
        arc_name=semantic_window.arc_name,
        novel_source=novel_name,
        chapter_count=max(1, chapter_end - chapter_start + 1),
        chapter_start=chapter_start,
        chapter_end=chapter_end,
        raw_characters=raw_characters,
        character_keys=alias_registry.character_keys_for(raw_characters),
        raw_location=raw_location,
        raw_core_action=raw_core_action,
        raw_summary=raw_summary,
        characters=alias_registry.alias_characters(raw_characters),
        core_action=sanitize_text_entities(raw_core_action, raw_characters, raw_location, alias_registry),
        cultivation_elements=sanitize_text_list(raw_atom.get("cultivation_elements", []), raw_characters, raw_location, alias_registry),
        location=alias_registry.alias_location(raw_location),
        causality_precondition=sanitize_text_entities(raw_atom.get("causality_precondition", ""), raw_characters, raw_location, alias_registry),
        causality_consequence=sanitize_text_entities(raw_atom.get("causality_consequence", ""), raw_characters, raw_location, alias_registry),
        motivation=sanitize_text_entities(raw_atom.get("motivation", ""), raw_characters, raw_location, alias_registry),
        conflict_type=sanitize_text_entities(raw_atom.get("conflict_type", ""), raw_characters, raw_location, alias_registry),
        narrative_function=sanitize_text_entities(raw_atom.get("narrative_function", ""), raw_characters, raw_location, alias_registry),
        emotion=sanitize_text_entities(raw_atom.get("emotion", ""), raw_characters, raw_location, alias_registry),
        tension_level=_clamp_int(raw_atom.get("tension_level", 5), 1, 10, default=5),
        summary=sanitize_text_entities(raw_summary, raw_characters, raw_location, alias_registry)[:240],
        semantic_window_id=semantic_window.semantic_window_id,
        atom_index_in_window=atom_index,
        atom_type=atom_type,
        actor=sanitize_text_entities(raw_atom.get("actor", ""), raw_characters, raw_location, alias_registry),
        goal=sanitize_text_entities(raw_atom.get("goal", ""), raw_characters, raw_location, alias_registry),
        obstacle=sanitize_text_entities(raw_atom.get("obstacle", ""), raw_characters, raw_location, alias_registry),
        action=sanitize_text_entities(raw_atom.get("action", ""), raw_characters, raw_location, alias_registry),
        outcome=sanitize_text_entities(raw_atom.get("outcome", ""), raw_characters, raw_location, alias_registry),
        state_delta=state_delta,
        is_complete=True,
        source_spans=_derive_source_spans_from_window(semantic_window, chapter_start, chapter_end, "complete_part"),
        boundary_reason=str(raw_atom.get("boundary_reason", "") or "").strip(),
        warnings=[],
    )

    if not atom.summary:
        atom.summary = _build_atom_summary_from_fields(atom)
    if not atom.raw_summary:
        atom.raw_summary = atom.summary
    if not atom.core_action:
        atom.core_action = atom.action
    if not atom.causality_consequence and atom.outcome:
        atom.causality_consequence = atom.outcome
    if not atom.actor:
        atom.actor = sanitize_text_entities(_first_non_empty(raw_characters), raw_characters, raw_location, alias_registry)
    if not atom.goal:
        atom.goal = atom.motivation
    if not atom.obstacle:
        atom.obstacle = atom.conflict_type
    return atom


def _coerce_incomplete_tail_from_ai(
    raw_tail: Dict[str, Any],
    semantic_window: SemanticWindow,
    fallback_tail: Optional[IncompleteTail] = None,
) -> Optional[IncompleteTail]:
    exists = bool(raw_tail.get("exists", False))
    if not exists and fallback_tail is None:
        return None
    if not exists and fallback_tail is not None:
        return fallback_tail

    window_start, window_end = _window_chapter_range(semantic_window)
    chapter_start = _clamp_int(raw_tail.get("chapter_start", window_end), window_start, window_end, default=window_end)
    chapter_end = _clamp_int(raw_tail.get("chapter_end", chapter_start), chapter_start, window_end, default=chapter_start)
    text_excerpt = str(raw_tail.get("text_excerpt", "") or "").strip()
    if not text_excerpt and fallback_tail is not None:
        text_excerpt = fallback_tail.text
    if not text_excerpt:
        text_excerpt = _tail_excerpt_from_text(semantic_window.text)
    summary = str(raw_tail.get("summary", "") or "").strip() or (fallback_tail.summary if fallback_tail else "")
    reason = str(raw_tail.get("reason", "") or "").strip() or (fallback_tail.reason if fallback_tail else "")
    missing_parts = _coerce_list(raw_tail.get("missing_parts", [])) or (fallback_tail.missing_parts if fallback_tail else [])

    return IncompleteTail(
        tail_id=f"{semantic_window.semantic_window_id}_tail",
        novel_source=semantic_window.novel_source,
        arc_name=semantic_window.arc_name,
        from_semantic_window_id=semantic_window.semantic_window_id,
        from_source_chunk_ids=semantic_window.source_chunk_ids[:],
        source_spans=_derive_source_spans_from_window(semantic_window, chapter_start, chapter_end, "tail_part"),
        text=text_excerpt[:1200],
        summary=summary[:240] or "尾部未闭合剧情",
        reason=reason or "尾部尚未形成明确 outcome/state_delta",
        missing_parts=missing_parts or ["outcome", "state_delta"],
        carry_to_next_chunk=bool(raw_tail.get("carry_to_next_chunk", True)),
        carried_to_semantic_window_id="",
        status="open",
    )


def _fallback_atomize_semantic_window(
    client,
    semantic_window: SemanticWindow,
    novel_name: str,
    alias_registry: NovelAliasRegistry,
    character_memory: Dict[str, str],
    failure_reason: str,
) -> Tuple[List[PlotAtom], Optional[IncompleteTail], List[str]]:
    text = semantic_window.text
    pass1: Dict[str, Any] = {}
    pass2: Dict[str, Any] = {}
    try:
        pass1 = _pass1_objective(client, semantic_window.arc_name, text, character_memory)
        pass2 = _pass2_subjective(client, semantic_window.arc_name, text, pass1)
    except Exception as exc:
        failure_reason = f"{failure_reason}|legacy_fallback_failed:{exc}"

    _update_character_memory(character_memory, pass1.get("characters", []))
    raw_characters = _coerce_list(pass1.get("characters", []))
    raw_location = str(pass1.get("location", "") or "").strip()
    raw_core_action = str(pass1.get("core_action", "") or "").strip()
    raw_summary = _build_raw_summary_from_passes(text, pass1, pass2)
    chapter_start, chapter_end = _window_chapter_range(semantic_window)

    atom = PlotAtom(
        atom_id=f"{semantic_window.semantic_window_id}_fallback_atom1",
        arc_name=semantic_window.arc_name,
        novel_source=novel_name,
        chapter_count=max(1, chapter_end - chapter_start + 1),
        chapter_start=chapter_start,
        chapter_end=chapter_end,
        raw_characters=raw_characters,
        character_keys=alias_registry.character_keys_for(raw_characters),
        raw_location=raw_location,
        raw_core_action=raw_core_action,
        raw_summary=raw_summary,
        characters=alias_registry.alias_characters(raw_characters),
        core_action=sanitize_text_entities(raw_core_action or raw_summary, raw_characters, raw_location, alias_registry),
        cultivation_elements=sanitize_text_list(pass1.get("cultivation_elements", []), raw_characters, raw_location, alias_registry),
        location=alias_registry.alias_location(raw_location),
        causality_precondition=sanitize_text_entities(pass2.get("causality_precondition", ""), raw_characters, raw_location, alias_registry),
        causality_consequence=sanitize_text_entities(pass2.get("causality_consequence", ""), raw_characters, raw_location, alias_registry),
        motivation=sanitize_text_entities(pass2.get("motivation", ""), raw_characters, raw_location, alias_registry),
        conflict_type=sanitize_text_entities(pass2.get("conflict_type", ""), raw_characters, raw_location, alias_registry),
        narrative_function=sanitize_text_entities(pass2.get("narrative_function", ""), raw_characters, raw_location, alias_registry),
        emotion=sanitize_text_entities(pass2.get("emotion", ""), raw_characters, raw_location, alias_registry),
        tension_level=_clamp_int(pass2.get("tension_level", 5), 1, 10, default=5),
        summary=sanitize_text_entities(raw_summary, raw_characters, raw_location, alias_registry)[:240],
        semantic_window_id=semantic_window.semantic_window_id,
        atom_index_in_window=1,
        atom_type="transition",
        actor=sanitize_text_entities(_first_non_empty(raw_characters), raw_characters, raw_location, alias_registry),
        goal=sanitize_text_entities(pass2.get("motivation", ""), raw_characters, raw_location, alias_registry),
        obstacle=sanitize_text_entities(pass2.get("conflict_type", ""), raw_characters, raw_location, alias_registry),
        action=sanitize_text_entities(raw_core_action or raw_summary, raw_characters, raw_location, alias_registry),
        outcome=sanitize_text_entities(pass2.get("causality_consequence", ""), raw_characters, raw_location, alias_registry)
        or "fallback 模式下基于现有信息生成的兼容结果",
        state_delta={},
        is_complete=True,
        source_spans=_derive_source_spans_from_window(semantic_window, chapter_start, chapter_end, "complete_part"),
        boundary_reason="fallback_due_to_ai_parse_failure",
        warnings=[failure_reason],
    )
    atom.state_delta = _infer_state_delta_from_atom(atom)
    if not atom.summary:
        atom.summary = _fallback_summary_from_text(text)
    if not atom.raw_summary:
        atom.raw_summary = atom.summary

    tail = None
    if _looks_like_incomplete_tail(text):
        tail = IncompleteTail(
            tail_id=f"{semantic_window.semantic_window_id}_tail",
            novel_source=semantic_window.novel_source,
            arc_name=semantic_window.arc_name,
            from_semantic_window_id=semantic_window.semantic_window_id,
            from_source_chunk_ids=semantic_window.source_chunk_ids[:],
            source_spans=_derive_source_spans_from_window(semantic_window, chapter_end, chapter_end, "tail_part"),
            text=_tail_excerpt_from_text(text),
            summary="fallback 检测到尾部剧情未闭合",
            reason="fallback 模式下检测到明显悬念或新事件开头",
            missing_parts=["outcome", "state_delta"],
            carry_to_next_chunk=True,
            carried_to_semantic_window_id="",
            status="open",
        )

    return [atom], tail, [f"窗口 {semantic_window.semantic_window_id} 使用 fallback 抽取: {failure_reason}"]


def _save_plot_atoms_output(all_atoms: Dict[str, List[PlotAtom]]) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _STEP2_PLOT_ATOMS_FILENAME
    payload = {novel_name: [asdict(atom) for atom in atoms] for novel_name, atoms in all_atoms.items()}
    write_json_file(out_path, payload)
    return out_path


def _save_semantic_windows_output(all_windows: Dict[str, List[SemanticWindow]]) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _STEP2_WINDOWS_FILENAME
    payload = {novel_name: [asdict(window) for window in windows] for novel_name, windows in all_windows.items()}
    write_json_file(out_path, payload)
    print(f"[Step 2] Semantic windows saved -> {out_path}", flush=True)
    return out_path


def _save_tail_carry_log_output(all_tails: Dict[str, List[IncompleteTail]]) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _STEP2_TAILS_FILENAME
    payload = {novel_name: [asdict(tail) for tail in tails] for novel_name, tails in all_tails.items()}
    write_json_file(out_path, payload)
    print(f"[Step 2] Tail carry log saved -> {out_path}", flush=True)
    return out_path


def _coerce_loaded_atom_record(item: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(item or {})
    record.setdefault("chapter_count", 1)
    record.setdefault("chapter_start", 0)
    record.setdefault("chapter_end", 0)
    record.setdefault("raw_characters", [])
    record.setdefault("character_keys", [])
    record.setdefault("raw_location", "")
    record.setdefault("raw_core_action", "")
    record.setdefault("raw_summary", "")
    record.setdefault("characters", [])
    record.setdefault("core_action", "")
    record.setdefault("cultivation_elements", [])
    record.setdefault("location", "")
    record.setdefault("causality_precondition", "")
    record.setdefault("causality_consequence", "")
    record.setdefault("motivation", "")
    record.setdefault("conflict_type", "")
    record.setdefault("narrative_function", "")
    record.setdefault("emotion", "")
    record.setdefault("tension_level", 5)
    record.setdefault("summary", "")
    record.setdefault("semantic_window_id", "")
    record.setdefault("atom_index_in_window", 0)
    record.setdefault("atom_type", "")
    record.setdefault("actor", "")
    record.setdefault("goal", "")
    record.setdefault("obstacle", "")
    record.setdefault("action", "")
    record.setdefault("outcome", "")
    record.setdefault("state_delta", {})
    record.setdefault("is_complete", True)
    record.setdefault("source_spans", [])
    record.setdefault("boundary_reason", "")
    record.setdefault("warnings", [])
    return record


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
    atom.semantic_window_id = str(getattr(atom, "semantic_window_id", "") or "")
    atom.atom_index_in_window = int(getattr(atom, "atom_index_in_window", 0) or 0)
    atom.atom_type = str(getattr(atom, "atom_type", "") or "")
    atom.actor = sanitize_text_entities(getattr(atom, "actor", ""), atom.raw_characters, atom.raw_location, alias_registry)
    atom.goal = sanitize_text_entities(getattr(atom, "goal", ""), atom.raw_characters, atom.raw_location, alias_registry)
    atom.obstacle = sanitize_text_entities(getattr(atom, "obstacle", ""), atom.raw_characters, atom.raw_location, alias_registry)
    atom.action = sanitize_text_entities(getattr(atom, "action", ""), atom.raw_characters, atom.raw_location, alias_registry)
    atom.outcome = sanitize_text_entities(getattr(atom, "outcome", ""), atom.raw_characters, atom.raw_location, alias_registry)
    atom.state_delta = _coerce_state_delta(getattr(atom, "state_delta", {}))
    atom.is_complete = bool(getattr(atom, "is_complete", True))
    atom.source_spans = _coerce_source_spans(getattr(atom, "source_spans", []))
    atom.boundary_reason = str(getattr(atom, "boundary_reason", "") or "")
    atom.warnings = _coerce_list(getattr(atom, "warnings", []))
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


def _build_raw_summary_from_passes(text: str, pass1_result: Dict[str, Any], pass2_result: Dict[str, Any]) -> str:
    core_action = str(pass1_result.get("core_action", "") or "").strip()
    motivation = str(pass2_result.get("motivation", "") or "").strip()
    conflict_type = str(pass2_result.get("conflict_type", "") or "").strip()
    consequence = str(pass2_result.get("causality_consequence", "") or "").strip()
    emotion = str(pass2_result.get("emotion", "") or "").strip()

    fragments: List[str] = []
    if core_action:
        fragments.append(f"事件围绕{core_action}展开")
    if motivation:
        fragments.append(f"主要动机是{motivation}")
    if conflict_type:
        fragments.append(f"冲突落在{conflict_type}")
    if consequence:
        fragments.append(f"后续将引出{consequence}")
    if emotion:
        fragments.append(f"整体氛围偏向{emotion}")
    summary = "；".join(_dedupe_text_fragments(fragments))
    return (summary or _fallback_summary_from_text(text))[:240]


def _build_atom_summary_from_fields(atom: PlotAtom) -> str:
    fragments = [atom.actor, atom.action or atom.core_action, atom.outcome, atom.goal]
    clean = [str(item or "").strip() for item in fragments if str(item or "").strip()]
    if clean:
        return "；".join(clean[:3])[:240]
    return (atom.raw_summary or atom.causality_consequence or atom.core_action or "事件围绕连续冲突推进")[:240]


def _fallback_summary_from_text(text: str) -> str:
    preview_lines: List[str] = []
    for line in str(text or "").splitlines():
        clean = line.strip()
        if clean:
            preview_lines.append(clean)
        if len(preview_lines) >= 2:
            break
    if preview_lines:
        return " / ".join(dict.fromkeys(preview_lines))[:240]
    merged_preview = " ".join(str(text or "").strip().replace("\n", " ").split())
    return merged_preview[:240] if merged_preview else "事件围绕连续冲突推进"


def _validate_atom_completeness(atom: PlotAtom) -> List[str]:
    issues: List[str] = []
    if not str(atom.summary or "").strip():
        issues.append("summary")
    if not str(atom.outcome or "").strip():
        issues.append("outcome")
    if not isinstance(atom.state_delta, dict) or not atom.state_delta:
        issues.append("state_delta")
    return issues


def _repair_incomplete_middle_atom(atom: PlotAtom, issues: List[str]) -> PlotAtom:
    atom.warnings.append(f"incomplete_middle_atom:{','.join(issues)}")
    if not atom.summary:
        atom.summary = _build_atom_summary_from_fields(atom)
        atom.raw_summary = atom.raw_summary or atom.summary
    if not atom.outcome:
        atom.outcome = atom.causality_consequence or atom.summary or atom.core_action
    if not atom.causality_consequence:
        atom.causality_consequence = atom.outcome
    if not atom.state_delta:
        atom.state_delta = _infer_state_delta_from_atom(atom)
    atom.boundary_reason = _append_note(atom.boundary_reason, f"warning: {'/'.join(issues)} 由程序级兼容补全")
    return atom


def _tail_from_incomplete_atom(atom: PlotAtom, semantic_window: SemanticWindow, issues: List[str]) -> IncompleteTail:
    return IncompleteTail(
        tail_id=f"{semantic_window.semantic_window_id}_tail",
        novel_source=semantic_window.novel_source,
        arc_name=semantic_window.arc_name,
        from_semantic_window_id=semantic_window.semantic_window_id,
        from_source_chunk_ids=semantic_window.source_chunk_ids[:],
        source_spans=atom.source_spans[:],
        text=(atom.raw_summary or atom.summary or atom.core_action)[:1200],
        summary=(atom.summary or atom.raw_summary or "尾部未闭合剧情")[:240],
        reason=f"最后一个 atom 缺失 {','.join(issues)}，转为尾部顺延",
        missing_parts=issues[:],
        carry_to_next_chunk=True,
        carried_to_semantic_window_id="",
        status="open",
    )


def _derive_source_spans_from_window(
    semantic_window: SemanticWindow,
    chapter_start: int,
    chapter_end: int,
    span_role: str,
) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    for span in semantic_window.source_spans:
        span_start = int(span.get("chapter_start", 0) or 0)
        span_end = int(span.get("chapter_end", 0) or span_start or 0)
        overlap_start = max(chapter_start, span_start)
        overlap_end = min(chapter_end, span_end)
        if overlap_start <= overlap_end:
            spans.append(
                {
                    "chunk_id": span.get("chunk_id", ""),
                    "chapter_start": overlap_start,
                    "chapter_end": overlap_end,
                    "span_role": span_role,
                }
            )
    if spans:
        return spans
    if semantic_window.source_chunk_ids:
        return [
            {
                "chunk_id": semantic_window.source_chunk_ids[-1],
                "chapter_start": chapter_start,
                "chapter_end": chapter_end,
                "span_role": span_role,
            }
        ]
    return []


def _window_chapter_range(semantic_window: SemanticWindow) -> Tuple[int, int]:
    starts = [
        int(span.get("chapter_start", 0) or 0)
        for span in semantic_window.source_spans
        if int(span.get("chapter_start", 0) or 0) > 0
    ]
    ends = [
        int(span.get("chapter_end", 0) or 0)
        for span in semantic_window.source_spans
        if int(span.get("chapter_end", 0) or 0) > 0
    ]
    if starts and ends:
        return min(starts), max(ends)
    return 1, 1


def _infer_atom_type(summary: Any, narrative_function: Any, conflict_type: Any, action: Any, outcome: Any) -> str:
    text = " ".join(_flatten_texts([summary, narrative_function, conflict_type, action, outcome]))
    if any(keyword in text for keyword in _SETUP_KEYWORDS):
        return "setup"
    if any(keyword in text for keyword in _TRIGGER_KEYWORDS):
        return "trigger"
    if any(keyword in text for keyword in _CONFLICT_KEYWORDS):
        return "conflict"
    if any(keyword in text for keyword in _CHOICE_KEYWORDS):
        return "choice"
    if any(keyword in text for keyword in _ATTEMPT_KEYWORDS):
        return "attempt"
    if any(keyword in text for keyword in _REVERSAL_KEYWORDS):
        return "reversal"
    if any(keyword in text for keyword in _REVEAL_KEYWORDS):
        return "reveal"
    if any(keyword in text for keyword in _GAIN_KEYWORDS):
        return "gain"
    if any(keyword in text for keyword in _LOSS_KEYWORDS):
        return "loss"
    if any(keyword in text for keyword in _RELATIONSHIP_KEYWORDS):
        return "relationship"
    if any(keyword in text for keyword in _HOOK_OPEN_KEYWORDS):
        return "hook_open"
    if any(keyword in text for keyword in _HOOK_CLOSE_KEYWORDS):
        return "hook_close"
    if any(keyword in text for keyword in _CONSEQUENCE_KEYWORDS):
        return "consequence"
    return "transition"


def _infer_state_delta_from_atom(atom: PlotAtom) -> Dict[str, Any]:
    text = " ".join(_flatten_texts([atom.summary, atom.outcome, atom.causality_consequence, atom.conflict_type, atom.narrative_function]))
    state_delta: Dict[str, Any] = {}
    if any(keyword in text for keyword in ("证明", "打脸", "扬名", "认可", "震慑", "名声")):
        state_delta["reputation"] = "+"
    if any(keyword in text for keyword in ("失势", "丢脸", "质疑", "羞辱")):
        state_delta.setdefault("reputation", "-")
    if any(keyword in text for keyword in ("获得", "夺得", "得到", "收获", "拜入", "突破")):
        state_delta["resource"] = "+"
    if any(keyword in text for keyword in ("失去", "损失", "消耗", "付出", "代价")):
        state_delta.setdefault("resource", "-")
    if any(keyword in text for keyword in ("结盟", "和解", "靠近", "认可", "救下")):
        state_delta["relationship"] = "+"
    if any(keyword in text for keyword in ("决裂", "误会", "背叛", "仇视")):
        state_delta.setdefault("relationship", "-")
    if any(keyword in text for keyword in ("受伤", "重伤", "吐血", "反噬")):
        state_delta["injury"] = "+"
    if any(keyword in text for keyword in ("暴露", "揭露", "识破", "察觉")):
        state_delta["secret_exposure"] = "+"
    if any(keyword in text for keyword in ("盯上", "关注", "追杀", "通缉", "悬赏")):
        state_delta["enemy_attention"] = "+"
    if any(keyword in text for keyword in ("伏笔", "线索", "后续", "隐患", "下一", "更大")):
        state_delta["open_hook"] = "+"
    if not state_delta:
        state_delta["open_hook"] = "optional"
    return state_delta


def _coerce_state_delta(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if str(key).strip()}
    return {}


def _coerce_source_spans(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    spans: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        spans.append(
            {
                "chunk_id": str(item.get("chunk_id", "") or ""),
                "chapter_start": int(item.get("chapter_start", 0) or 0),
                "chapter_end": int(item.get("chapter_end", 0) or 0),
                "span_role": str(item.get("span_role", "") or ""),
            }
        )
    return spans


def _event_text(event: NarrativeEvent) -> str:
    summary = str(getattr(event, "summary", "") or "").strip()
    if summary:
        return summary
    return "\n\n".join(
        str(chapter or "").strip()
        for chapter in getattr(event, "chapters", [])
        if str(chapter or "").strip()
    ).strip()


def _looks_like_incomplete_tail(text: str) -> bool:
    tail_text = _tail_excerpt_from_text(text, limit=500)
    if not tail_text:
        return False
    if any(keyword in tail_text for keyword in _FALLBACK_TAIL_KEYWORDS):
        return True
    return tail_text.endswith(("？", "?", "……", "...", "…"))


def _tail_excerpt_from_text(text: str, limit: int = 1000) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    excerpt = clean[-limit:]
    if len(excerpt) > 1200:
        excerpt = excerpt[-1200:]
    return excerpt.strip()


def _append_note(base: str, note: str) -> str:
    base_text = str(base or "").strip()
    note_text = str(note or "").strip()
    if not base_text:
        return note_text
    if not note_text:
        return base_text
    return f"{base_text} | {note_text}"


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


def _dedupe_list(values: List[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _first_non_empty(values: List[str]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _flatten_texts(values: List[Any]) -> List[str]:
    flattened: List[str] = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(_flatten_texts(value))
            continue
        text = str(value or "").strip()
        if text:
            flattened.append(text)
    return flattened


def _novel_prefix(novel_name: str) -> str:
    base = str(novel_name or "novel").strip().rsplit(".", 1)[0]
    base = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", base)
    return base.strip("_") or "novel"


def _clamp_int(value: Any, low: int, high: int, default: int = 0) -> int:
    try:
        num = int(value)
    except Exception:
        num = default
    if high < low:
        high = low
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
        temperature=0.2,
    )
    data = safe_json_load(raw)
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
        temperature=0.2,
    )
    data = safe_json_load(raw)
    return data if isinstance(data, dict) else {}
