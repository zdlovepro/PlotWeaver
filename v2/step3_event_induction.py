from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import config
from pipeline.core.common_json import read_json_file, write_json_file
from pipeline.core.common_text import dedupe_texts, flatten_text_values
from pipeline.step2_extraction import PlotAtom


_STEP3_FILENAME = "step3_induced_events.json"


@dataclass
class InducedEvent:
    event_id: str
    novel_source: str
    arc_name: str
    summary: str
    raw_summary: str = ""
    conflict_hint: str = ""
    function_hint: str = ""
    source_atom_ids: List[str] = field(default_factory=list)
    chapter_start: int = 0
    chapter_end: int = 0
    chapter_count: int = 0
    characters: List[str] = field(default_factory=list)
    raw_characters: List[str] = field(default_factory=list)
    character_keys: List[str] = field(default_factory=list)
    tension_peak: int = 0
    state_inputs: List[str] = field(default_factory=list)
    state_outputs: List[str] = field(default_factory=list)
    relationship_delta: List[str] = field(default_factory=list)
    resource_delta: List[str] = field(default_factory=list)
    hook_open: List[str] = field(default_factory=list)
    hook_close: List[str] = field(default_factory=list)
    power_state: str = ""
    stage_index: int = -1
    identity_state: str = ""


def induce_all(all_atoms: Dict[str, List[PlotAtom]]) -> Dict[str, List[InducedEvent]]:
    result: Dict[str, List[InducedEvent]] = {}
    for novel_name, atoms in all_atoms.items():
        grouped: Dict[str, List[PlotAtom]] = defaultdict(list)
        for atom in atoms:
            if not bool(getattr(atom, "is_complete", True)):
                print(
                    f"[Step 3] Warning: skipping incomplete PlotAtom {atom.atom_id} "
                    f"(semantic_window_id={getattr(atom, 'semantic_window_id', '')})"
                )
                continue
            grouped[atom.arc_name or "未命名卷"].append(atom)

        induced_events: List[InducedEvent] = []
        for arc_name, bucket in grouped.items():
            ordered = sorted(bucket, key=lambda atom: (atom.chapter_start, atom.chapter_end, atom.atom_id))
            induced_events.extend(_induce_arc_events(novel_name, arc_name, ordered))
        result[novel_name] = induced_events

    save_step3_output(result)
    return result


def save_step3_output(all_events: Dict[str, List[InducedEvent]]) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _STEP3_FILENAME
    payload = {novel_name: [asdict(event) for event in events] for novel_name, events in all_events.items()}
    write_json_file(out_path, payload)
    print(f"[Step 3] Intermediate output saved -> {out_path}")
    return out_path


def load_step3_output(intermediate_dir: str | Path | None = None) -> Dict[str, List[InducedEvent]]:
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)
    in_path = inter_dir / _STEP3_FILENAME
    if not in_path.exists():
        raise FileNotFoundError(f"Step 3 intermediate file not found: {in_path}")
    raw = read_json_file(in_path)
    result: Dict[str, List[InducedEvent]] = {}
    for novel_name, events_data in raw.items():
        result[novel_name] = [InducedEvent(**item) for item in events_data]
    print(f"[Step 3] Loaded intermediate output from {in_path}")
    return result


def _induce_arc_events(novel_name: str, arc_name: str, atoms: List[PlotAtom]) -> List[InducedEvent]:
    events: List[InducedEvent] = []
    start = 0
    event_index = 0
    while start < len(atoms):
        cluster: List[PlotAtom] = []
        idx = start
        while idx < len(atoms):
            atom = atoms[idx]
            cluster.append(atom)
            chapter_span = _cluster_chapter_span(cluster)
            if _should_close_cluster(cluster, atoms, idx, chapter_span):
                break
            idx += 1
        events.append(_cluster_to_event(novel_name, arc_name, cluster, event_index))
        event_index += 1
        start += max(1, len(cluster))
    return events


def _should_close_cluster(cluster: List[PlotAtom], atoms: List[PlotAtom], current_index: int, chapter_span: int) -> bool:
    if current_index + 1 >= len(atoms):
        return True
    if chapter_span >= 6:
        return True

    latest = cluster[-1]
    next_atom = atoms[current_index + 1]
    phase_shift = _phase_shift_score(cluster, next_atom)
    same_chapter = (
        _atom_chapter_start(latest) == _atom_chapter_end(latest)
        and _atom_chapter_start(next_atom) == _atom_chapter_end(next_atom)
        and _atom_chapter_start(latest) == _atom_chapter_start(next_atom)
    )
    if same_chapter:
        if _cluster_raw_text_length(cluster) >= max(4200, int(config.MAX_TEXT_CHUNK_LENGTH * 1.8)):
            return True
        return len(cluster) >= 3 and phase_shift >= 2
    if chapter_span < 2:
        return False
    if chapter_span >= 2 and phase_shift >= 3:
        return True
    if chapter_span >= 3 and (
        _atom_has_boundary(latest)
        or _next_atom_starts_new_phase(cluster, atoms, current_index)
        or phase_shift >= 2
    ):
        return True
    if chapter_span >= 4 and (
        phase_shift >= 1
        or _cluster_raw_text_length(cluster) >= max(2200, int(config.MAX_TEXT_CHUNK_LENGTH * 0.8))
    ):
        return True
    return False


def _cluster_to_event(novel_name: str, arc_name: str, cluster: List[PlotAtom], event_index: int) -> InducedEvent:
    return InducedEvent(
        event_id=f"{arc_name}_induced_event{event_index}",
        novel_source=novel_name,
        arc_name=arc_name,
        summary=_build_cluster_synopsis(cluster, use_raw=False),
        raw_summary=_build_cluster_synopsis(cluster, use_raw=True),
        conflict_hint=_top_value(atom.conflict_type for atom in cluster),
        function_hint=_top_value(atom.narrative_function for atom in cluster),
        source_atom_ids=[atom.atom_id for atom in cluster],
        chapter_start=min(_atom_chapter_start(atom) for atom in cluster),
        chapter_end=max(_atom_chapter_end(atom) for atom in cluster),
        chapter_count=_cluster_chapter_span(cluster),
        characters=_dedupe_texts(alias for atom in cluster for alias in getattr(atom, "characters", [])),
        raw_characters=_dedupe_texts(alias for atom in cluster for alias in getattr(atom, "raw_characters", [])),
        character_keys=_dedupe_texts(key for atom in cluster for key in getattr(atom, "character_keys", [])),
        tension_peak=max(int(getattr(atom, "tension_level", 0) or 0) for atom in cluster),
        state_inputs=_derive_state_inputs(cluster),
        state_outputs=_derive_state_outputs(cluster),
        relationship_delta=_derive_relationship_delta(cluster),
        resource_delta=_derive_resource_delta(cluster),
        hook_open=_derive_hook_terms(cluster, kind="open"),
        hook_close=_derive_hook_terms(cluster, kind="close"),
        power_state=_derive_power_state(cluster, arc_name),
        identity_state=_derive_identity_state(cluster),
    )


def _build_cluster_summary(cluster: List[PlotAtom], use_raw: bool) -> str:
    fragments = []
    for atom in cluster:
        text = atom.raw_summary if use_raw else atom.summary
        clean = str(text or "").strip()
        if clean:
            fragments.append(clean)
    fragments = _dedupe_texts(fragments)
    if not fragments:
        return "主线围绕连续冲突与关系变化展开。"
    if len(fragments) == 1:
        combined = fragments[0]
    elif len(fragments) == 2:
        combined = f"{fragments[0]}；并牵出{fragments[1]}"
    else:
        combined = f"{fragments[0]}；随后转入{fragments[1]}；并在后段引出{fragments[2]}"
    return _clip_summary_text(combined, limit=420)


def _clip_summary_text(text: str, limit: int = 280) -> str:
    clean = str(text or "").strip()
    if len(clean) <= limit:
        return clean
    window = clean[: limit + 1]
    punctuation = "。！？；;.!?"
    cut = max(window.rfind(mark) for mark in punctuation)
    if cut >= max(40, int(limit * 0.45)):
        return window[: cut + 1].strip()
    return window[:limit].rstrip("锛屻€傦紱;,.!? ")


def _build_cluster_synopsis(cluster: List[PlotAtom], use_raw: bool) -> str:
    lead_atom = _pick_focus_atom(cluster)
    lead_action = _clean_summary_fragment(lead_atom.raw_core_action if use_raw else lead_atom.core_action)
    lead_summary = _clean_summary_fragment(lead_atom.raw_summary if use_raw else lead_atom.summary)
    outcome = _clean_summary_fragment(cluster[-1].causality_consequence or cluster[-1].core_action)
    location = _clean_summary_fragment(lead_atom.raw_location if use_raw else lead_atom.location)
    conflict = _clean_summary_fragment(_top_value(atom.conflict_type for atom in cluster))
    function_hint = _clean_summary_fragment(_top_value(atom.narrative_function for atom in cluster))

    parts: List[str] = []
    if lead_action:
        parts.append(lead_action)
    elif lead_summary:
        parts.append(lead_summary)
    if conflict:
        parts.append(f"冲突集中在{conflict}")
    if outcome and outcome not in "".join(parts):
        parts.append(f"结果是{outcome}")
    if function_hint and function_hint not in "".join(parts):
        parts.append(f"事件功能偏向{function_hint}")
    if location and location not in "".join(parts):
        parts.append(f"主要场景在{location}")

    if not parts:
        parts.append("这一事件围绕连续冲突与状态变化展开")
    combined = "；".join(parts[:4])
    return _clip_summary_text(combined, limit=180 if use_raw else 140)


def _atom_has_boundary(atom: PlotAtom) -> bool:
    text = " ".join(_flatten_text_values([atom.summary, atom.raw_summary, atom.narrative_function, atom.conflict_type, atom.causality_consequence]))
    keywords = ("转折", "翻盘", "余波", "收束", "后果", "伏笔", "钩子", "高潮", "对决", "逆袭")
    return any(keyword in text for keyword in keywords)


def _next_atom_resets(atoms: List[PlotAtom], current_index: int) -> bool:
    if current_index + 1 >= len(atoms):
        return True
    next_atom = atoms[current_index + 1]
    text = " ".join(_flatten_text_values([next_atom.summary, next_atom.raw_summary, next_atom.narrative_function]))
    return any(keyword in text for keyword in ("开篇", "铺垫", "试探", "入场"))


def _next_atom_starts_new_phase(cluster: List[PlotAtom], atoms: List[PlotAtom], current_index: int) -> bool:
    if current_index + 1 >= len(atoms):
        return True
    return _phase_shift_score(cluster, atoms[current_index + 1]) >= 2


def _phase_shift_score(cluster: List[PlotAtom], next_atom: PlotAtom) -> int:
    recent_atoms = cluster[-2:]
    score = 0
    if _field_shift((atom.narrative_function for atom in recent_atoms), next_atom.narrative_function):
        score += 1
    if _field_shift((atom.conflict_type for atom in recent_atoms), next_atom.conflict_type):
        score += 1
    if _location_shift(recent_atoms, next_atom):
        score += 1
    if _character_continuity_score(recent_atoms, next_atom) < 0.34:
        score += 1
    if _status_quo_shift(next_atom):
        score += 1
    return score


def _field_shift(values, next_value: Any) -> bool:
    current = _normalize_logic_text(_top_value(values))
    upcoming = _normalize_logic_text(next_value)
    if not current or not upcoming:
        return False
    if current == upcoming:
        return False
    if current in upcoming or upcoming in current:
        return False
    return True


def _location_shift(recent_atoms: List[PlotAtom], next_atom: PlotAtom) -> bool:
    current = _normalize_logic_text(_top_value(atom.raw_location or atom.location for atom in recent_atoms))
    upcoming = _normalize_logic_text(next_atom.raw_location or next_atom.location)
    if not current or not upcoming:
        return False
    if current == upcoming:
        return False
    if current in upcoming or upcoming in current:
        return False
    return True


def _character_continuity_score(recent_atoms: List[PlotAtom], next_atom: PlotAtom) -> float:
    recent_keys = {
        key
        for atom in recent_atoms
        for key in getattr(atom, "character_keys", [])
        if str(key or "").strip()
    }
    next_keys = {key for key in getattr(next_atom, "character_keys", []) if str(key or "").strip()}
    if not recent_keys or not next_keys:
        return 1.0
    return len(recent_keys & next_keys) / max(1, min(len(recent_keys), len(next_keys)))


def _status_quo_shift(atom: PlotAtom) -> bool:
    text = " ".join(
        _flatten_text_values(
            [
                atom.summary,
                atom.raw_summary,
                atom.core_action,
                atom.raw_core_action,
                atom.causality_consequence,
                atom.narrative_function,
            ]
        )
    )
    keywords = ("加入", "进入", "成为", "转入", "调入", "拜入", "外出", "归来", "晋升", "受命", "接取任务", "开始学习")
    return any(keyword in text for keyword in keywords)


def _pick_focus_atom(cluster: List[PlotAtom]) -> PlotAtom:
    def sort_key(atom: PlotAtom) -> tuple[int, int, int]:
        weight = 0
        if getattr(atom, "core_action", "") or getattr(atom, "raw_core_action", ""):
            weight += 2
        if getattr(atom, "causality_consequence", ""):
            weight += 2
        if getattr(atom, "conflict_type", ""):
            weight += 1
        return (weight, int(getattr(atom, "tension_level", 0) or 0), _atom_chapter_end(atom))

    return max(cluster, key=sort_key)


def _clean_summary_fragment(text: Any) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    clean = clean.replace("\n", " ").replace("\r", " ")
    clean = " ".join(clean.split())
    return clean[:56]


def _normalize_logic_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    filtered = [ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"]
    return "".join(filtered)[:48]


def _cluster_chapter_span(cluster: List[PlotAtom]) -> int:
    covered: set[int] = set()
    for atom in cluster:
        covered.update(range(_atom_chapter_start(atom), _atom_chapter_end(atom) + 1))
    return max(1, len(covered))


def _cluster_raw_text_length(cluster: List[PlotAtom]) -> int:
    return sum(len(str(atom.raw_summary or atom.summary or "")) + len(str(atom.raw_core_action or atom.core_action or "")) for atom in cluster)


def _atom_chapter_start(atom: PlotAtom) -> int:
    return int(getattr(atom, "chapter_start", 0) or 0) or 1


def _atom_chapter_end(atom: PlotAtom) -> int:
    start = _atom_chapter_start(atom)
    return int(getattr(atom, "chapter_end", 0) or 0) or start


def _top_value(values) -> str:
    counter = Counter(item for item in _flatten_text_values(list(values)) if item)
    return counter.most_common(1)[0][0] if counter else ""


def _derive_state_inputs(cluster: List[PlotAtom]) -> List[str]:
    values: List[Any] = []
    for atom in cluster:
        values.extend([atom.causality_precondition, atom.motivation])
    return _limit_event_terms(values, limit=8)


def _derive_state_outputs(cluster: List[PlotAtom]) -> List[str]:
    values: List[Any] = []
    for atom in cluster:
        values.extend([atom.causality_consequence, atom.core_action])
    outputs = _limit_event_terms(values, limit=8)
    if outputs:
        return outputs
    fallback = [atom.summary for atom in cluster if atom.summary]
    return _limit_event_terms(fallback, limit=4)


def _derive_relationship_delta(cluster: List[PlotAtom]) -> List[str]:
    deltas: List[str] = []
    for atom in cluster:
        if len(getattr(atom, "character_keys", [])) >= 2 and atom.emotion:
            deltas.append(f"相关角色关系转向{atom.emotion}")
    return _dedupe_texts(deltas)[:4]


def _derive_resource_delta(cluster: List[PlotAtom]) -> List[str]:
    resource_keywords = ("法宝", "丹", "药", "灵石", "功法", "资源", "血脉", "传承", "令牌", "阵盘")
    values: List[str] = []
    for atom in cluster:
        for item in getattr(atom, "cultivation_elements", []):
            text = str(item or "").strip()
            if text and any(keyword in text for keyword in resource_keywords):
                values.append(text)
    return _dedupe_texts(values)[:6]


def _derive_hook_terms(cluster: List[PlotAtom], kind: str) -> List[str]:
    if kind == "open":
        keywords = ("伏笔", "线索", "后续", "隐患", "更大", "下一", "秘密", "真相")
        values = [atom.causality_consequence for atom in cluster[-2:]] + [atom.summary for atom in cluster[-1:]]
    else:
        keywords = ("破局", "揭露", "翻盘", "和解", "救出", "斩杀", "夺得", "突破", "解决", "脱身")
        values = [atom.core_action for atom in cluster] + [atom.summary for atom in cluster[-2:]]
    matched = []
    for text in _flatten_text_values(values):
        if any(keyword in text for keyword in keywords):
            matched.append(text)
    return _limit_event_terms(matched, limit=4)


def _derive_power_state(cluster: List[PlotAtom], arc_name: str) -> str:
    values: List[str] = []
    for atom in cluster:
        for item in getattr(atom, "cultivation_elements", []):
            text = str(item or "").strip()
            if text and any(keyword in text for keyword in ("境", "期", "层", "重", "阶", "劫", "台")):
                values.append(text)
    if values:
        return _dedupe_texts(values)[0]
    # Arc labels describe source organization, not the protagonist's progression.
    return ""


def _derive_identity_state(cluster: List[PlotAtom]) -> str:
    identity_keywords = ("弟子", "长老", "亲传", "外门", "内门", "掌柜", "少主", "客卿", "散修", "逃亡", "潜伏", "卧底", "供奉")
    values: List[str] = []
    for atom in cluster:
        text = " ".join(_flatten_text_values([atom.summary, atom.raw_summary, atom.core_action, atom.location]))
        for fragment in _split_logic_fragments(text):
            if any(keyword in fragment for keyword in identity_keywords):
                values.append(fragment)
    return _dedupe_texts(values)[0] if values else ""


def _limit_event_terms(values: List[Any], limit: int) -> List[str]:
    deduped: List[str] = []
    for value in values:
        for fragment in _flatten_text_values([value]):
            for piece in _split_logic_fragments(fragment):
                if piece and piece not in deduped:
                    deduped.append(piece)
                if len(deduped) >= limit:
                    return deduped
    return deduped


def _split_logic_fragments(text: str) -> List[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    fragments = []
    for piece in cleaned.replace("；", "，").replace("。", "，").split("，"):
        item = piece.strip("、/ \t\r\n")
        if len(item) >= 2:
            fragments.append(item[:32])
    return fragments


_flatten_text_values = flatten_text_values
_dedupe_texts = dedupe_texts
