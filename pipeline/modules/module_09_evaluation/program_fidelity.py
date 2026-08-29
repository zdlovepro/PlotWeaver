"""Independent fidelity evaluation for V3 program-generation sequences.

The generator receives only the chapter program.  This module runs afterwards:
it compares a published draft with anonymised annotation structure and locally
measured style features, never by supplying source prose to the generator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from .contracts import ChapterAnnotation, ChapterDocument, build_chapter_document
from .jsonio import read_json, read_jsonl, write_json
from .model import ModelSettings, complete_json
from .paths import corpus_dir, runs_dir, validate_author_id
from .style_distillation import build_chapter_style_card


def _annotation_path(folder: Path, limit: int) -> Path:
    exact = folder / f"chapter_annotations.sample-{limit}.jsonl"
    if exact.is_file():
        return exact
    candidates = sorted(folder.glob("chapter_annotations.sample-*.jsonl"))
    if not candidates:
        raise FileNotFoundError("缺少高密度章节标注，不能进行保真评测")
    return candidates[-1]


def _fact_text(annotation: ChapterAnnotation, fact_id: str) -> str:
    facts = {item.fact_id: item for item in annotation.facts}
    entities = {item.entity_id: item.canonical_name for item in annotation.entities}
    fact = facts.get(fact_id)
    if fact is None:
        return fact_id
    value = fact.value or entities.get(fact.object_id, "")
    return " ".join(piece for piece in (entities.get(fact.subject_id, "未知主体"), fact.predicate, value) if piece)


def annotation_structure(annotation: ChapterAnnotation) -> dict[str, Any]:
    """Return only causal structure; source spans and quotations are excluded."""

    names = {item.entity_id: item.canonical_name for item in annotation.entities}
    event_by_id = {item.event_id: item for item in annotation.events}
    anchors = {item.anchor_id: item.label for item in annotation.time_anchors}
    return {
        "章节ID": annotation.chapter_id,
        "实体": [{"名称": item.canonical_name, "类型": item.kind} for item in annotation.entities],
        "事实": [{"类型": item.kind, "陈述": _fact_text(annotation, item.fact_id), "确定性": item.certainty} for item in annotation.facts],
        "事件": [{
            "顺序": item.order,
            "概述": item.summary,
            "参与者": [names.get(value, value) for value in item.participant_ids],
            "行动": item.action,
            "阻力": item.obstacle,
            "决断": item.decision,
            "前置": [_fact_text(annotation, value) for value in item.precondition_fact_ids],
            "结果": [_fact_text(annotation, value) for value in item.outcome_fact_ids],
            "代价": [_fact_text(annotation, value) for value in item.cost_fact_ids],
        } for item in annotation.events],
        "场景": [{
            "顺序": item.order,
            "目标": item.objective,
            "参与者": [names.get(value, value) for value in item.participant_ids],
            "事件": [event_by_id[value].summary for value in item.event_ids if value in event_by_id],
            "入场": [_fact_text(annotation, value) for value in item.entry_fact_ids],
            "出场": [_fact_text(annotation, value) for value in item.exit_fact_ids],
            "张力": item.tension,
        } for item in annotation.scenes],
        "状态变化": [{
            "操作": item.operation,
            "事件": event_by_id[item.event_id].summary if item.event_id in event_by_id else item.event_id,
            "之前": _fact_text(annotation, item.before_fact_id) if item.before_fact_id else "",
            "之后": _fact_text(annotation, item.after_fact_id) if item.after_fact_id else "",
        } for item in annotation.state_changes],
        "时间关系": [{
            "前事件": event_by_id[item.before_event_id].summary if item.before_event_id in event_by_id else item.before_event_id,
            "后事件": event_by_id[item.after_event_id].summary if item.after_event_id in event_by_id else item.after_event_id,
            "关系": item.kind,
            "时间锚点": anchors.get(item.anchor_id, ""),
        } for item in annotation.temporal_relations],
        "空间关系": [{
            "主体": names.get(item.subject_id, item.subject_id),
            "地点": names.get(item.location_id, item.location_id),
            "关系": item.kind,
        } for item in annotation.spatial_relations],
    }


def _source_text(document: ChapterDocument) -> str:
    return "\n".join(unit.text for unit in document.annotation_units)


def style_comparison(reference: ChapterDocument, generated: str) -> dict[str, Any]:
    reference_metrics = {item.metric_id: item for item in build_chapter_style_card(reference).metrics}
    generated_document = build_chapter_document(
        f"{reference.chapter_id}:generated",
        hashlib.sha256(generated.encode("utf-8")).hexdigest(),
        generated,
        "Generated",
    )
    generated_metrics = {item.metric_id: item for item in build_chapter_style_card(generated_document).metrics}
    rows: list[dict[str, Any]] = []
    for metric_id, source in reference_metrics.items():
        actual = generated_metrics.get(metric_id, source).value
        floor = 0.05 if source.unit == "ratio" else 1.0
        relative_difference = abs(actual - source.value) / max(abs(source.value), floor)
        rows.append({
            "metric_id": metric_id,
            "target": source.value,
            "actual": actual,
            "relative_difference": round(relative_difference, 6),
        })
    mean_difference = fmean(min(item["relative_difference"], 1.0) for item in rows) if rows else 1.0
    return {
        "style_similarity": round(max(0.0, 1.0 - mean_difference), 4),
        "reference_char_count": len(_source_text(reference)),
        "generated_char_count": len(generated),
        "char_count_ratio": round(len(generated) / max(len(_source_text(reference)), 1), 4),
        "metrics": rows,
    }


_SCORE_KEYS = ("事件因果", "状态覆盖", "时空一致", "场景推进", "逻辑自洽")


def _judge_structure(expected: dict[str, Any], program: dict[str, Any], draft: str, generation: dict[str, Any]) -> dict[str, Any]:
    example = {
        "评分": {key: 4 for key in _SCORE_KEYS},
        "遗漏": ["缺少某一明确结果"],
        "矛盾": [],
        "优点": ["事件因果顺序清楚"],
        "下一步修复": ["补写遗漏的结果及其人物反应"],
    }
    prompt = f"""你是中文小说结构保真评测器。评测对象由章节程序生成，生成时没有得到原章正文。现在只根据“期望匿名结构”、章节程序、生成校验记录和生成正文判断：事件、状态、时间、空间、场景推进和逻辑是否保留。不得评价词句是否相同，不得根据常识补充输入之外的剧情。

只输出一个合法 JSON object，顶层字段必须且只能是 `评分`、`遗漏`、`矛盾`、`优点`、`下一步修复`。`评分` 必须且只能含 `事件因果`、`状态覆盖`、`时空一致`、`场景推进`、`逻辑自洽` 五项 1-5 整数。只要 `遗漏` 或 `矛盾` 非空，任何评分不得为 5；缺少关键事件、结果、状态变化或时空顺序必须写入 `遗漏` 或 `矛盾`，不能只降分。JSON 示例：
{json.dumps(example, ensure_ascii=False, indent=2)}

<期望匿名结构>
{json.dumps(expected, ensure_ascii=False)}
</期望匿名结构>
<章节程序>
{json.dumps(program, ensure_ascii=False)}
</章节程序>
<生成校验记录>
{json.dumps(generation, ensure_ascii=False)}
</生成校验记录>
<生成正文>
{draft}
</生成正文>
"""
    raw = complete_json(
        "你是严格的中文小说结构保真评测器。只输出指定 JSON。",
        prompt,
        ModelSettings.from_environment(),
        attempts=4,
        max_tokens=3_500,
    )
    if not isinstance(raw, dict) or set(raw) != {"评分", "遗漏", "矛盾", "优点", "下一步修复"}:
        raise ValueError("结构评测返回的顶层 JSON 契约不正确")
    scores = raw.get("评分")
    if not isinstance(scores, dict) or set(scores) != set(_SCORE_KEYS):
        raise ValueError("结构评测评分字段不完整")
    if any(not isinstance(scores[key], int) or not 1 <= scores[key] <= 5 for key in _SCORE_KEYS):
        raise ValueError("结构评测评分必须是 1-5 整数")
    for key in ("遗漏", "矛盾", "优点", "下一步修复"):
        if not isinstance(raw[key], list) or any(not isinstance(item, str) for item in raw[key]):
            raise ValueError(f"结构评测.{key} 必须是字符串数组")
    return raw


def hard_structure_pass(judgment: dict[str, Any]) -> bool:
    if not isinstance(judgment, dict) or judgment.get("遗漏") or judgment.get("矛盾"):
        return False
    scores = judgment.get("评分")
    return isinstance(scores, dict) and all(isinstance(scores.get(key), int) and scores[key] >= 4 for key in _SCORE_KEYS)


def evaluate_program_sequence(
    author_id: str,
    work_id: str,
    index_path: Path,
    sequence_path: Path,
    *,
    annotation_limit: int = 20,
    run_id: str = "program-fidelity",
) -> Path:
    """Evaluate a sequential V3 run without exposing source prose to drafting."""

    author_id = validate_author_id(author_id)
    index = read_json(index_path)
    sequence = read_json(sequence_path)
    if not isinstance(index, dict) or not isinstance(sequence, dict):
        raise ValueError("规划索引和顺序生成摘要必须是 JSON object")
    if index.get("author_id") != author_id or sequence.get("author_id") != author_id:
        raise ValueError("评测输入的 author_id 不一致")
    folder = corpus_dir(author_id, work_id)
    annotations = [ChapterAnnotation.from_dict(item) for item in read_jsonl(_annotation_path(folder, annotation_limit))]
    documents = {
        str(item.get("chapter_id", "")): ChapterDocument.from_dict(item)
        for item in read_jsonl(folder / "chapter_documents.jsonl")
    }
    indexed = {int(item.get("chapter_no", 0)): item for item in index.get("chapters", []) if isinstance(item, dict)}
    expected = {position: annotation for position, annotation in enumerate(annotations, start=1)}
    destination = runs_dir(author_id, run_id) / "program_fidelity"
    destination.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for row in sequence.get("chapters", []):
        if not isinstance(row, dict):
            continue
        chapter_no = int(row.get("chapter_no", 0))
        annotation = expected.get(chapter_no)
        program_row = indexed.get(chapter_no)
        result: dict[str, Any] = {"chapter_no": chapter_no, "chapter_id": str(row.get("chapter_id", "")), "sequence_status": row.get("status", "")}
        if annotation is None or program_row is None:
            result.update({"status": "failed", "reason": "缺少对应的标注或章节程序"})
        elif row.get("status") != "passed":
            result.update({"status": "not_evaluable", "reason": "章节没有通过生成硬门槛"})
        else:
            output_dir = Path(str(row.get("output_dir", "")))
            draft_path = output_dir / "generated_draft.txt"
            program_path = Path(str(program_row.get("program", "")))
            reference = documents.get(annotation.chapter_id)
            if not draft_path.is_file() or not program_path.is_file() or reference is None:
                result.update({"status": "failed", "reason": "缺少已发布正文、章节程序或原始文档"})
            else:
                draft = draft_path.read_text(encoding="utf-8")
                generation = read_json(output_dir / "generation_manifest.json", {})
                try:
                    judgment = _judge_structure(annotation_structure(annotation), read_json(program_path, {}), draft, generation if isinstance(generation, dict) else {})
                    result.update({
                        "status": "passed" if hard_structure_pass(judgment) else "failed",
                        "structure_hard_passed": hard_structure_pass(judgment),
                        "structure_judgment": judgment,
                        "style_comparison": style_comparison(reference, draft),
                    })
                except Exception as exc:
                    result.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "structure_hard_passed": False})
        results.append(result)
        write_json(destination / f"chapter-{chapter_no:04d}.result.json", result)

    evaluated = [item for item in results if item.get("sequence_status") == "passed"]
    passed = [item for item in evaluated if item.get("status") == "passed"]
    report = destination / "fidelity_summary.json"
    write_json(report, {
        "schema_version": "1.0",
        "author_id": author_id,
        "work_id": work_id,
        "index_path": str(index_path),
        "sequence_path": str(sequence_path),
        "planned_chapters": len(indexed),
        "generated_chapters": len(evaluated),
        "structure_hard_pass_rate": round(len(passed) / len(evaluated), 4) if evaluated else 0.0,
        "mean_style_similarity": round(fmean(item["style_comparison"]["style_similarity"] for item in passed), 4) if passed else None,
        "chapters": results,
    })
    return report
