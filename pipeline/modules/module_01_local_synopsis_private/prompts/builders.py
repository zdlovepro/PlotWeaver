from __future__ import annotations

import json
from typing import Any

from ....contracts import ChapterDocument
from ..windowing import LocalRequestBatch, SynopsisWindow, chapter_paragraph_alias_maps
from .registry import render_template

def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _rows(window: SynopsisWindow) -> list[dict[str, str]]:
    aliases = dict(window.unit_aliases)
    return [{"段落ID": aliases[item.unit_id], "原文": item.text} for item in window.target_units]


def _batch_source(batch: LocalRequestBatch) -> list[dict[str, Any]]:
    return [{"块ID": window.block_id, "原文段落": _rows(window)} for window in batch.windows]


def _chapter_source(document: ChapterDocument) -> list[dict[str, str]]:
    aliases, _ = chapter_paragraph_alias_maps(document)
    return [{"段落ID": aliases[unit.unit_id], "原文": unit.text} for unit in document.annotation_units]


def chapter_segmentation_prompt(document: ChapterDocument) -> str:
    return render_template("chapter_segmentation", CHAPTER_SOURCE_JSON=_json(_chapter_source(document)))


def chapter_segmentation_review_prompt(document: ChapterDocument, plan: dict[str, Any]) -> str:
    return render_template("chapter_segmentation_review", CHAPTER_SOURCE_JSON=_json(_chapter_source(document)),
                           BLOCK_IDS_JSON=_json([row["块ID"] for row in plan["章节分块"]]),
                           PLAN_JSON=_json(plan))


def chapter_segmentation_recheck_prompt(document: ChapterDocument, plan: dict[str, Any]) -> str:
    return render_template("chapter_segmentation_recheck",
                           CHAPTER_SOURCE_JSON=_json(_chapter_source(document)),
                           BLOCK_IDS_JSON=_json([row["块ID"] for row in plan["章节分块"]]),
                           PLAN_JSON=_json(plan))


def chapter_segmentation_repair_prompt(document: ChapterDocument, plan: dict[str, Any],
                                       issues: list[dict[str, Any]]) -> str:
    return render_template("chapter_segmentation_repair", CHAPTER_SOURCE_JSON=_json(_chapter_source(document)),
                           PLAN_JSON=_json(plan), ISSUES_JSON=_json(issues))


def local_summary_prompt(batch: LocalRequestBatch, *,
                         prior_context: list[dict[str, Any]] | None = None) -> str:
    return render_template("local_summary", BLOCK_IDS_JSON=_json([w.block_id for w in batch.windows]),
                           PRIOR_CONTEXT_JSON=_json(prior_context or []),
                           BLOCK_SOURCE_JSON=_json(_batch_source(batch)))


def _local_review_cases(batch: LocalRequestBatch, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_block = {row["块ID"]: row for row in _batch_source(batch)}
    cases = []
    for target in targets:
        block_id = str(target["块ID"])
        cases.append({"目标ID": target["目标ID"], "块ID": block_id,
                      "候选梗概": target["梗概"],
                      "可用证据": by_block[block_id]["原文段落"]})
    return cases


def local_risk_prompt(batch: LocalRequestBatch, targets: list[dict[str, Any]]) -> str:
    return render_template("local_risk_review",
                           TARGET_IDS_JSON=_json([row["目标ID"] for row in targets]),
                           REVIEW_CASES_JSON=_json(_local_review_cases(batch, targets)))


def local_major_adjudication_prompt(batch: LocalRequestBatch, targets: list[dict[str, Any]],
                                     findings: list[dict[str, Any]]) -> str:
    target_ids = {row["目标ID"] for row in findings}
    selected_targets = [row for row in targets if row["目标ID"] in target_ids]
    selected_blocks = {row["块ID"] for row in selected_targets}
    selected_source = [row for row in _batch_source(batch) if row["块ID"] in selected_blocks]
    return render_template("local_major_adjudication",
                           ISSUE_IDS_JSON=_json([row["问题ID"] for row in findings]),
                           BLOCK_SOURCE_JSON=_json(selected_source),
                           TARGETS_JSON=_json(selected_targets), FINDINGS_JSON=_json(findings))


def local_summary_repair_prompt(batch: LocalRequestBatch, targets: list[dict[str, Any]],
                                issues: list[dict[str, Any]]) -> str:
    failed_ids = list(dict.fromkeys(str(issue["块ID"]) for issue in issues))
    failed_source = [row for row in _batch_source(batch) if row["块ID"] in set(failed_ids)]
    failed_targets = [row for row in targets if row["块ID"] in set(failed_ids)]
    return render_template("local_repair",
                           FAILED_BLOCK_IDS_JSON=_json(failed_ids),
                           BLOCK_SOURCE_JSON=_json(failed_source), TARGETS_JSON=_json(failed_targets),
                           ISSUES_JSON=_json(issues))


def local_repair_verification_prompt(batch: LocalRequestBatch, cases: list[dict[str, Any]]) -> str:
    block_ids = list(dict.fromkeys(str(case["块ID"]) for case in cases))
    selected = set(block_ids)
    source = [row for row in _batch_source(batch) if row["块ID"] in selected]
    return render_template(
        "local_repair_verification",
        ISSUE_IDS_JSON=_json([case["问题ID"] for case in cases]),
        BLOCK_SOURCE_JSON=_json(source),
        VERIFICATION_CASES_JSON=_json(cases),
    )


def boundary_extract_prompt(window: SynopsisWindow, *, kind: str) -> str:
    name = "章首状态" if kind == "opening" else "章末状态"
    rows = _rows(window)
    anchor_id = rows[0]["段落ID"] if kind == "opening" else rows[-1]["段落ID"]
    rule = (
        "章首状态描述正文第一段展开时已经成立或正在发生的局面。第一段是内容锚点；"
        "窗口中的后续段落只可用于消解第一段的指代，不能把后来才发生的发展写入章首。"
        if kind == "opening" else
        "章末状态描述整个章末收束片段在章节结束时共同形成的最终局面。最后一段标记章节终点，"
        "但不是唯一可用证据；同一收束场景前几段中的离开、留下、反应或未决状态，只要共同构成"
        "最终局面，就可以合并表达，不要求每个动作在最后一句中再次明说仍在持续。不要写入收束"
        "片段之外已经结束的早期事件。"
    )
    return render_template("boundary_extract", BOUNDARY_NAME=name, BOUNDARY_RULE=rule,
                           BOUNDARY_ANCHOR_ID=anchor_id, BOUNDARY_SOURCE_JSON=_json(rows))


def boundary_review_prompt(window: SynopsisWindow, *, kind: str, value: str) -> str:
    name = "章首状态" if kind == "opening" else "章末状态"
    rows = _rows(window)
    anchor_id = rows[0]["段落ID"] if kind == "opening" else rows[-1]["段落ID"]
    rule = (
        "候选应描述正文第一段展开时的局面。后续段落只能帮助消解第一段指代；"
        "若候选加入后来才发生的发展，属于边界错位。"
        if kind == "opening" else
        "候选应表达章末收束片段共同形成的最终局面，而不是逐字复写最后一段。"
        "同一收束场景中相邻段落的连续动作、反应和状态可以被合并；只有候选把主体、结果或"
        "完成状态写错，拼入收束片段之外的事件，或组成原文不存在的关系时，才构成实质问题。"
    )
    return render_template("boundary_review", BOUNDARY_NAME=name, BOUNDARY_ANCHOR_ID=anchor_id,
                           BOUNDARY_RULE=rule,
                           BOUNDARY_SOURCE_JSON=_json(_rows(window)), CANDIDATE_JSON=_json(value))


def chapter_summary_prompt(local_rows: list[dict[str, Any]]) -> str:
    return render_template("chapter_summary",
                           LOCAL_IDS_JSON=_json([row["局部梗概ID"] for row in local_rows]),
                           LOCAL_ROWS_JSON=_json(local_rows))


def chapter_summary_review_prompt(local_rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    return render_template("chapter_summary_review", LOCAL_ROWS_JSON=_json(local_rows),
                           SUMMARY_JSON=_json(summary))


def chapter_summary_coverage_review_prompt(local_rows: list[dict[str, Any]],
                                           summary: dict[str, Any]) -> str:
    return render_template("chapter_summary_coverage_review",
                           LOCAL_IDS_JSON=_json([row["局部梗概ID"] for row in local_rows]),
                           LOCAL_ROWS_JSON=_json(local_rows), SUMMARY_JSON=_json(summary))


def chapter_summary_repair_prompt(local_rows: list[dict[str, Any]], summary: dict[str, Any],
                                  fidelity_issues: list[dict[str, str]],
                                  missing_points: list[dict[str, str]]) -> str:
    return render_template("chapter_summary_repair",
                           LOCAL_IDS_JSON=_json([row["局部梗概ID"] for row in local_rows]),
                           FIDELITY_ISSUES_JSON=_json(fidelity_issues),
                           MISSING_POINTS_JSON=_json(missing_points),
                           SUMMARY_JSON=_json(summary), LOCAL_ROWS_JSON=_json(local_rows))
