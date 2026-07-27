from __future__ import annotations

import difflib
import json
import re
import hashlib
from pathlib import Path
from typing import Any

from .contracts import ChapterRecord
from .extract import extract_record
from .jsonio import read_json, read_jsonl, write_json
from .model import ModelSettings, complete_json
from .paths import corpus_dir, runs_dir


def _tokens(value: Any) -> set[str]:
    def values(item: Any) -> list[str]:
        if isinstance(item, dict):
            return [piece for value in item.values() for piece in values(value)]
        if isinstance(item, list):
            return [piece for value in item for piece in values(value)]
        return [str(item or "")]
    text = " ".join(values(value))
    words = set(re.findall(r"[A-Za-z0-9_]{2,}", text))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    grams = {chinese[index:index + 2] for index in range(max(0, len(chinese) - 1))}
    return words | grams


def _score(left: set[str], right: set[str]) -> float | None:
    if not left and not right:
        return None
    return round(len(left & right) / max(len(left | right), 1), 4)


def _field_score(source: Any, generated: Any) -> float | None:
    return _score(_tokens(source), _tokens(generated))


def _diagnose(scores: dict[str, float | None], offline: bool) -> list[str]:
    messages: list[str] = []
    labels = {
        "event_structure": "事件链", "scene_beats": "场景节拍", "character_deltas": "人物状态变化", "relation_deltas": "人物关系变化",
        "temporal_spatial": "时间空间状态", "cultivation": "修炼与资源约束",
        "information": "信息差与伏笔", "pacing": "节奏曲线",
    }
    for key, label in labels.items():
        score = scores.get(key)
        if score is None:
            messages.append(f"{label}未被当前样本观测到，不能视为通过；请选择包含该要素的章节继续评测。")
        elif score < 0.5:
            messages.append(f"{label}覆盖不足：补强对应 ChapterState 字段或生成提示词约束。")
    if offline:
        messages.append("本次为离线诊断，生成文本是结构预览；配置模型后重跑以获得正文级保真指标。")
    return messages or ["所有已测字段达到当前阈值；使用留出章节继续验证泛化能力。"]


def _semantic_judgment(source_state: dict[str, Any], generated_state: dict[str, Any]) -> dict[str, Any]:
    schema = {
        "scores": {"events": 0, "scenes": 0, "characters": 0, "relations": 0, "time_space": 0, "information": 0, "pacing": 0, "logic": 0},
        "hallucinations": ["string"],
        "missing_facts": ["string"],
        "logic_violations": [{"scene_no": 0, "rule": "", "description": ""}],
        "priority_repairs": ["string"],
    }
    prompt = f"""任务：评估“生成章节状态”对“原章节状态”的语义保真度。允许措辞、句序和叙述细节不同；只判断事件、场景、人物、关系、时空、信息和节奏是否由同一组结构化事实支持。

只输出 JSON object，必须符合此 schema：
{json.dumps(schema, ensure_ascii=False, indent=2)}

打分范围是 0 到 100 的整数。`hallucinations` 只列生成状态中原状态不支持的具体事实；`missing_facts` 只列原状态中生成状态遗漏的关键事实；`logic_violations` 检查实体身份、行动前置条件、时间单调性、地点连续性、资源变化、关系变化和知识传播；`priority_repairs` 给出可执行的抽取或复现提示词改进项。

原章节状态：
{json.dumps(source_state, ensure_ascii=False)}

生成章节状态：
{json.dumps(generated_state, ensure_ascii=False)}"""
    return complete_json("你是小说章节保真评估器。所有字段值必须使用中文，不能因改写措辞不同而扣分。", prompt, ModelSettings.from_environment(), max_tokens=3_000)


def evaluate(author_id: str, work_id: str, chapter_no: int, run_id: str, offline: bool = False, state_filename: str = "chapter_states.jsonl") -> Path:
    folder = corpus_dir(author_id, work_id)
    records = read_jsonl(folder / "chapters.jsonl")
    if Path(state_filename).name != state_filename:
        raise ValueError("state filename must not contain a directory")
    states = read_jsonl(folder / state_filename)
    suffix = f"/{chapter_no:04d}"
    source = next((row for row in records if row["chapter_id"].endswith(suffix)), None)
    source_state = next((row for row in states if row["chapter_id"].endswith(suffix)), None)
    draft_path = runs_dir(author_id, run_id) / work_id / "reconstructed_chapter.md"
    if not source or not source_state or not draft_path.exists():
        raise ValueError("Source chapter, extracted state, or reconstruction draft is missing")
    draft = draft_path.read_text(encoding="utf-8")
    generated_record = ChapterRecord(
        chapter_id=f"{source['chapter_id']}:generated", author_id=author_id, work_id=work_id,
        chapter_no=chapter_no, source_chapter_no=chapter_no, title="generated", text=draft,
        char_count=len(draft), source_hash=hashlib.sha256(draft.encode("utf-8")).hexdigest(),
    )
    generated_state = extract_record(generated_record, offline=offline).to_dict()
    text_ratio = difflib.SequenceMatcher(None, source["text"], draft).ratio()
    structural = {
        "event_structure": _field_score(source_state.get("event_chain"), generated_state.get("event_chain")),
        "scene_beats": _field_score(source_state.get("scene_beats"), generated_state.get("scene_beats")),
        "character_deltas": _field_score(source_state.get("character_deltas"), generated_state.get("character_deltas")),
        "relation_deltas": _field_score(source_state.get("relation_deltas"), generated_state.get("relation_deltas")),
        "temporal_spatial": _field_score(source_state.get("temporal_spatial_state"), generated_state.get("temporal_spatial_state")),
        "cultivation": _field_score(source_state.get("cultivation_state"), generated_state.get("cultivation_state")),
        "information": _field_score(source_state.get("information_state"), generated_state.get("information_state")),
        "pacing": _field_score(source_state.get("pacing_curve"), generated_state.get("pacing_curve")),
    }
    semantic_judgment = {} if offline else _semantic_judgment(source_state, generated_state)
    semantic_scores = semantic_judgment.get("scores", {}) if isinstance(semantic_judgment, dict) else {}
    numeric_semantic = [value for value in semantic_scores.values() if isinstance(value, int) and 0 <= value <= 100]
    report = {
        "chapter_id": source["chapter_id"],
        "mode": "fidelity",
        "scores": {
            **structural,
            "text_sequence_ratio": round(text_ratio, 4),
            "target_char_count": len(source["text"]),
            "generated_char_count": len(draft),
            "model_semantic_average": round(sum(numeric_semantic) / len(numeric_semantic), 2) if numeric_semantic else None,
        },
        "gap_diagnosis": _diagnose(structural, offline),
        "generated_state": generated_state,
        "semantic_judgment": semantic_judgment,
    }
    target = draft_path.parent / "fidelity_report.json"
    write_json(target, report)
    return target
