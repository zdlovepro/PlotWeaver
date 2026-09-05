"""Run repeatable, input-bounded reconstruction regression tests for a compiled skill."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from .contracts import ChapterDocument, build_chapter_document
from .generation import DEFAULT_MAX_REPAIRS, generate_chapter
from .jsonio import read_jsonl, write_json
from .model import ModelSettings, complete_json
from .paths import corpus_dir, runs_dir
from .style_distillation import build_chapter_style_card
from .style_execution import budget_from_metrics


DEFAULT_FIDELITY_CHAPTERS = 5


def _annotation_path(folder: Path, limit: int) -> Path:
    exact = folder / f"chapter_annotations.sample-{limit}.jsonl"
    if exact.exists():
        return exact
    candidates = sorted(folder.glob("chapter_annotations.sample-*.jsonl"))
    if not candidates:
        raise FileNotFoundError("需要章节标注产物：请先运行第 2 阶段抽取")
    return candidates[-1]


def _fact_text(fact: dict[str, Any], names: dict[str, str]) -> str:
    subject = names.get(str(fact.get("subject_id", "")), "未知主体")
    object_name = names.get(str(fact.get("object_id", "")), "")
    value = str(fact.get("value", "")).strip() or object_name
    predicate = str(fact.get("predicate", "")).strip()
    return " ".join(piece for piece in (subject, predicate, value) if piece)


def _annotation_structure(annotation: dict[str, Any]) -> dict[str, Any]:
    """Drop all evidence/offset fields and retain only generation-relevant structure."""

    entities = [
        {"name": str(item.get("canonical_name", "")), "kind": str(item.get("kind", "")), "aliases": list(item.get("aliases", []))}
        for item in annotation.get("entities", []) if isinstance(item, dict) and str(item.get("canonical_name", "")).strip()
    ]
    names = {str(item.get("entity_id", "")): str(item.get("canonical_name", "")) for item in annotation.get("entities", []) if isinstance(item, dict)}
    facts_by_id = {str(item.get("fact_id", "")): _fact_text(item, names) for item in annotation.get("facts", []) if isinstance(item, dict)}
    events_by_id = {str(item.get("event_id", "")): str(item.get("summary", "")) for item in annotation.get("events", []) if isinstance(item, dict)}
    anchors_by_id = {str(item.get("anchor_id", "")): str(item.get("label", "")) for item in annotation.get("time_anchors", []) if isinstance(item, dict)}
    facts = [{"kind": str(item.get("kind", "")), "statement": facts_by_id.get(str(item.get("fact_id", "")), ""), "certainty": item.get("certainty", 1.0)} for item in annotation.get("facts", []) if isinstance(item, dict)]
    events = [{
        "order": item.get("order", 0), "summary": str(item.get("summary", "")), "participants": [names.get(str(value), str(value)) for value in item.get("participant_ids", [])],
        "action": str(item.get("action", "")), "obstacle": str(item.get("obstacle", "")), "decision": str(item.get("decision", "")),
        "preconditions": [facts_by_id.get(str(value), str(value)) for value in item.get("precondition_fact_ids", [])],
        "outcomes": [facts_by_id.get(str(value), str(value)) for value in item.get("outcome_fact_ids", [])],
        "costs": [facts_by_id.get(str(value), str(value)) for value in item.get("cost_fact_ids", [])],
    } for item in annotation.get("events", []) if isinstance(item, dict)]
    raw_events = [item for item in annotation.get("events", []) if isinstance(item, dict)]
    for index, item in enumerate(events, start=1):
        item["contract_id"] = f"target-event-{index:03d}"
    event_views_by_id = {str(raw.get("event_id", "")): event for raw, event in zip(raw_events, events)}
    scenes = [{
        "order": item.get("order", 0), "participants": [names.get(str(value), str(value)) for value in item.get("participant_ids", [])],
        "events": [dict(event_views_by_id.get(str(value), {"summary": events_by_id.get(str(value), str(value))})) for value in item.get("event_ids", [])], "objective": str(item.get("objective", "")),
        "entry_facts": [facts_by_id.get(str(value), str(value)) for value in item.get("entry_fact_ids", [])],
        "exit_facts": [facts_by_id.get(str(value), str(value)) for value in item.get("exit_fact_ids", [])], "tension": item.get("tension", 0),
        "locations": [names.get(str(value), str(value)) for value in item.get("location_ids", [])],
        "time_anchors": [anchors_by_id.get(str(value), str(value)) for value in item.get("time_anchor_ids", [])],
    } for item in annotation.get("scenes", []) if isinstance(item, dict)]
    for index, item in enumerate(scenes, start=1):
        item["contract_id"] = f"target-scene-{index:03d}"
    state_changes = [{
        "operation": str(item.get("operation", "")), "event": events_by_id.get(str(item.get("event_id", "")), ""),
        "before": facts_by_id.get(str(item.get("before_fact_id", "")), ""), "after": facts_by_id.get(str(item.get("after_fact_id", "")), ""),
        "contract_id": f"target-change-{index:03d}",
    } for index, item in enumerate(annotation.get("state_changes", []), start=1) if isinstance(item, dict)]
    return {
        "chapter_id": str(annotation.get("chapter_id", "")), "entities": entities, "facts": facts, "events": events, "scenes": scenes,
        "state_changes": state_changes,
        "time_anchors": [{"label": str(item.get("label", "")), "kind": str(item.get("kind", ""))} for item in annotation.get("time_anchors", []) if isinstance(item, dict)],
        "temporal_relations": [{
            "before": events_by_id.get(str(item.get("before_event_id", "")), ""), "after": events_by_id.get(str(item.get("after_event_id", "")), ""),
            "kind": str(item.get("kind", "")), "anchor": anchors_by_id.get(str(item.get("anchor_id", "")), ""),
        } for item in annotation.get("temporal_relations", []) if isinstance(item, dict)],
        "spatial_relations": [{
            "subject": names.get(str(item.get("subject_id", "")), ""), "location": names.get(str(item.get("location_id", "")), ""), "kind": str(item.get("kind", "")),
        } for item in annotation.get("spatial_relations", []) if isinstance(item, dict)],
    }


def _source_text(document: ChapterDocument) -> str:
    return "\n".join(unit.text for unit in document.annotation_units)


def _style_comparison(reference: ChapterDocument, generated: str) -> dict[str, Any]:
    reference_metrics = {item.metric_id: item for item in build_chapter_style_card(reference).metrics}
    generated_document = build_chapter_document(
        f"{reference.chapter_id}:generated", hashlib.sha256(generated.encode("utf-8")).hexdigest(), generated, "Generated",
    )
    generated_metrics = {item.metric_id: item for item in build_chapter_style_card(generated_document).metrics}
    rows = []
    for metric_id, source in reference_metrics.items():
        target = source.value
        actual = generated_metrics.get(metric_id, source).value
        floor = 0.05 if source.unit == "ratio" else 1.0
        relative_difference = abs(actual - target) / max(abs(target), floor)
        rows.append({"metric_id": metric_id, "target": target, "actual": actual, "relative_difference": round(relative_difference, 6)})
    mean_difference = fmean(min(item["relative_difference"], 1.0) for item in rows) if rows else 1.0
    return {
        "style_similarity": round(max(0.0, 1.0 - mean_difference), 4),
        "target_char_count": len(_source_text(reference)), "generated_char_count": len(generated),
        "char_count_ratio": round(len(generated) / max(len(_source_text(reference)), 1), 4), "metrics": rows,
    }


def _judge_structure(expected: dict[str, Any], plan: dict[str, Any], draft: str, validation: dict[str, Any]) -> dict[str, Any]:
    schema = {
        "scores": {"event_causality": 1, "state_coverage": 1, "temporal_spatial": 1, "scene_pacing": 1, "logic": 1},
        "missing_requirements": ["string"], "contradictions": ["string"], "strengths": ["string"], "next_repairs": ["string"],
    }
    prompt = f"""任务：评估一份“结构保真重建”的质量。输入中的期望结构由独立抽取的实体、事实、事件、场景和时空关系构成；生成器从未得到原章正文。请判断生成正文是否保留了期望结构的关键内容与因果，不要比较措辞，更不要补充输入之外的事实。

只输出一个合法 JSON object，必须符合此 schema：
{json.dumps(schema, ensure_ascii=False, indent=2)}

`scores` 的五项均为 1-5 整数：1 为严重缺失或冲突，3 为主体可辨但有重要遗漏，5 为关键结构完整且逻辑自洽。`missing_requirements` 和 `contradictions` 必须指向输入中可见的具体结构；`next_repairs` 必须可执行。只要 `missing_requirements` 或 `contradictions` 非空，任何分项都不得给 5，且总评不得视为通过。

期望结构：
{json.dumps(expected, ensure_ascii=False)}

生成计划：
{json.dumps(plan, ensure_ascii=False)}

连续性校验：
{json.dumps(validation, ensure_ascii=False)}

生成正文：
{draft}
"""
    return complete_json("你是中文小说结构保真评测器。只输出中文 JSON。", prompt, ModelSettings.from_environment(), attempts=6, max_tokens=3_500)


def _hard_structure_pass(judgment: dict[str, Any]) -> bool:
    """Do not let a high model score override an explicit missing requirement."""

    if not isinstance(judgment, dict) or judgment.get("error"):
        return False
    if judgment.get("missing_requirements") or judgment.get("contradictions"):
        return False
    scores = judgment.get("scores")
    if not isinstance(scores, dict):
        return False
    required = ("event_causality", "state_coverage", "temporal_spatial", "scene_pacing", "logic")
    return all(isinstance(scores.get(key), int) and scores[key] >= 4 for key in required)


def run_fidelity_regression(
    author_id: str,
    work_id: str,
    *,
    chapter_limit: int = DEFAULT_FIDELITY_CHAPTERS,
    run_id: str = "fidelity-regression",
    max_repairs: int = DEFAULT_MAX_REPAIRS,
) -> Path:
    """Generate and assess the first N annotated chapters without passing source prose to the generator."""

    if chapter_limit < 1:
        raise ValueError("chapter_limit must be positive")
    folder = corpus_dir(author_id, work_id)
    annotations = read_jsonl(_annotation_path(folder, chapter_limit))[:chapter_limit]
    documents = {item["chapter_id"]: ChapterDocument.from_dict(item) for item in read_jsonl(folder / "chapter_documents.jsonl")}
    if len(annotations) < chapter_limit:
        raise ValueError(f"只有 {len(annotations)} 章可用于保真回归，少于请求的 {chapter_limit} 章")
    target = runs_dir(author_id, run_id) / "fidelity"
    target.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    prior_context: list[dict[str, Any]] = []
    for number, annotation in enumerate(annotations, start=1):
        expected = _annotation_structure(annotation)
        reference = documents.get(expected["chapter_id"])
        if reference is None:
            raise ValueError(f"缺少章节原始文档：{expected['chapter_id']}")
        chapter_dir = target / f"chapter-{number:02d}"
        inputs = chapter_dir / "inputs"
        inputs.mkdir(parents=True, exist_ok=True)
        reference_budget = budget_from_metrics(build_chapter_style_card(reference).metrics, chapter_char_count=len(_source_text(reference)))
        state = {
            "reconstruction_mode": "fidelity_test_only", "chapter_id": expected["chapter_id"],
            "known_prior_chapters": prior_context, "known_entities": expected["entities"], "known_facts": expected["facts"],
            "current_chapter_target": {"time_anchors": expected["time_anchors"], "temporal_relations": expected["temporal_relations"], "spatial_relations": expected["spatial_relations"]},
        }
        brief = {
            "reconstruction_mode": "fidelity_test_only", "chapter_goal": "完整重建已提供的目标事件、场景、状态变化与结局，不新增输入外情节。",
            "required_events": expected["events"], "required_scenes": expected["scenes"], "required_state_changes": expected["state_changes"],
            "style_budget_override": reference_budget,
        }
        write_json(chapter_dir / "expected_structure.json", expected)
        write_json(chapter_dir / "reference_style_budget.json", reference_budget)
        write_json(inputs / "story_state.json", state)
        write_json(inputs / "chapter_brief.json", brief)
        scoped_run = f"{run_id}-chapter-{number:02d}"
        try:
            generation = generate_chapter(author_id, inputs / "story_state.json", inputs / "chapter_brief.json", run_id=scoped_run, max_repairs=max_repairs)
            generated_dir = Path(str(generation["output_dir"]))
            draft = (generated_dir / "generated_draft.txt").read_text(encoding="utf-8")
            plan = json.loads((generated_dir / "generated_plan.json").read_text(encoding="utf-8"))
            validation = json.loads((generated_dir / "validation.json").read_text(encoding="utf-8"))
            assessment = json.loads((generated_dir / "style_assessment.json").read_text(encoding="utf-8"))
            comparison = _style_comparison(reference, draft)
            try:
                judgment = _judge_structure(expected, plan, draft, validation)
            except RuntimeError as exc:
                judgment = {"error": f"结构评测模型调用失败：{exc}"}
            scores = judgment.get("scores", {}) if isinstance(judgment, dict) else {}
            numeric_scores = [value for value in scores.values() if isinstance(value, int) and 1 <= value <= 5]
            result = {
                "chapter_no": number, "chapter_id": expected["chapter_id"], "generation": generation, "generated_dir": str(generated_dir),
                "style_comparison": comparison, "style_assessment": assessment, "structure_judgment": judgment,
                "structure_score_mean": round(fmean(numeric_scores), 3) if numeric_scores else None,
                "structure_hard_passed": _hard_structure_pass(judgment),
            }
        except Exception as exc:  # Preserve the failed chapter and continue the regression set.
            result = {"chapter_no": number, "chapter_id": expected["chapter_id"], "error": f"{type(exc).__name__}: {exc}"}
        write_json(chapter_dir / "result.json", result)
        results.append(result)
        prior_context.append({"chapter_id": expected["chapter_id"], "events": expected["events"], "state_changes": expected["state_changes"]})
        successful = [item for item in results if "error" not in item]
        write_json(target / "summary.partial.json", {"chapter_limit": chapter_limit, "completed": len(results), "successful": len(successful), "chapters": results})

    successful = [item for item in results if "error" not in item]
    summary = {
        "schema_version": "1.0", "mode": "fidelity_regression", "author_id": author_id, "work_id": work_id,
        "chapter_limit": chapter_limit, "successful_chapters": len(successful), "failed_chapters": chapter_limit - len(successful),
        "mean_style_similarity": round(fmean(item["style_comparison"]["style_similarity"] for item in successful), 4) if successful else None,
        "mean_char_count_ratio": round(fmean(item["style_comparison"]["char_count_ratio"] for item in successful), 4) if successful else None,
        "style_budget_pass_rate": round(sum(bool(item["style_assessment"].get("passed")) for item in successful) / len(successful), 4) if successful else 0.0,
        "continuity_pass_rate": round(sum(bool(item["generation"].get("continuity_passed")) for item in successful) / len(successful), 4) if successful else 0.0,
        "structure_hard_pass_rate": round(sum(bool(item.get("structure_hard_passed")) for item in successful) / len(successful), 4) if successful else 0.0,
        "mean_structure_score": round(fmean(item["structure_score_mean"] for item in successful if item["structure_score_mean"] is not None), 3) if successful else None,
        "chapters": results,
    }
    report = target / "fidelity_summary.json"
    write_json(report, summary)
    return report
