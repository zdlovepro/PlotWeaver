from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from ...common.jsonio import read_json, write_json
from ...common.model import ModelOutputError, ModelSettings, complete_json, json_system_message
from ...contracts import (BoundaryFrame, ChapterDocument, ChapterSynopsisBundle, LocalSynopsis,
    SynopsisIssue)
from .parser import (SynopsisContractError, apply_local_summary_patch, build_chapter_synopsis,
    build_local_synopsis, chapter_local_rows, parse_boundary_value,
    parse_chapter_segmentation, parse_chapter_summary, parse_local_summary_batch, paragraph_alias_maps)
from .prompts import (BOUNDARY_PROMPT_REVISION,
    CHAPTER_SUMMARY_PROMPT_REVISION, LOCAL_MAJOR_ADJUDICATION_PROMPT_REVISION,
    LOCAL_REPAIR_PROMPT_REVISION, LOCAL_REPAIR_VERIFICATION_PROMPT_REVISION,
    LOCAL_RISK_PROMPT_REVISION,
    LOCAL_SUMMARY_PROMPT_REVISION, LOCAL_SYSTEM_PROMPT,
    SEGMENTATION_PROMPT_REVISION, SEGMENTATION_RECHECK_PROMPT_REVISION,
    SEGMENTATION_REPAIR_PROMPT_REVISION, SEGMENTATION_REVIEW_PROMPT_REVISION,
    boundary_extract_prompt, boundary_review_prompt,
    chapter_segmentation_prompt, chapter_segmentation_recheck_prompt, chapter_segmentation_repair_prompt,
    chapter_segmentation_review_prompt,
    chapter_summary_coverage_review_prompt, chapter_summary_prompt,
    chapter_summary_repair_prompt, chapter_summary_review_prompt,
    local_major_adjudication_prompt, local_repair_verification_prompt,
    local_risk_prompt, local_summary_prompt, local_summary_repair_prompt)
from .quality import assess_synopsis_quality
from .request_budget import (CHAPTER_REQUEST_CHAR_LIMIT, LOCAL_REQUEST_CHAR_LIMIT,
    SEGMENTATION_REQUEST_CHAR_LIMIT, RequestLedger)
from .review import (GroundedSemanticIssue, ReviewContractError, parse_boundary_review,
    parse_chapter_coverage, parse_chapter_fidelity, parse_issue_adjudication,
    parse_local_risk, parse_repair_verification, parse_segmentation_recheck,
    parse_segmentation_review, validate_chapter_coverage_issues,
    validate_grounded_issues)
from .verification import ChapterExecutionResult
from .windowing import (ChapterNavigationPlan, LocalRequestBatch, SynopsisWindow,
    build_boundary_window, build_local_request_batches, build_synopsis_windows_from_plan,
    chapter_paragraph_alias_maps, split_oversized_navigation_blocks)

MODULE_VERSION = "21.2.0"
SEGMENTATION_MAX_SOURCE_CHARS = 120_000
LOCAL_PRIOR_CONTEXT_LIMIT = 3
SEGMENTATION_OUTPUT_TOKENS = 5000
LOCAL_OUTPUT_TOKENS = 3200
LOCAL_REVIEW_OUTPUT_TOKENS = 4200
CHAPTER_OUTPUT_TOKENS = 4200
CHAPTER_REVIEW_OUTPUT_TOKENS = 3000
JsonCompletion = Callable[..., dict[str, Any]]


class SynopsisExtractionError(RuntimeError):
    def __init__(self, message: str, *, code: str = "SYNOPSIS_EXTRACTION_FAILED"):
        self.code = code
        super().__init__(message)


class LocalBatchReviewError(SynopsisExtractionError):
    def __init__(self, batch_id: str, *, candidates: list[dict[str, Any]],
                 decisions: list[dict[str, Any]], failed_blocks: list[dict[str, Any]]):
        super().__init__(f"{batch_id}局部梗概修订后仍存在已确认的重大错误",
                         code="LOCAL_MAJOR_ERROR_UNRESOLVED")
        self.failed_stage = f"local-batch:{batch_id}"
        self.local_candidates = tuple(dict(row) for row in candidates)
        self.decisions = tuple(dict(row) for row in decisions)
        self.details = {"batch_id": batch_id, "failed_blocks": failed_blocks}


class LocalBatchContractError(SynopsisExtractionError):
    def __init__(self, batch_id: str, *, failed_stage: str,
                 candidates: list[dict[str, Any]], decisions: list[dict[str, Any]],
                 contract_errors: list[str], raw_response: dict[str, Any] | None = None):
        super().__init__(f"{batch_id}在{failed_stage}返回了不符合契约的模型结果",
                         code="LOCAL_BATCH_CONTRACT_FAILED")
        self.failed_stage = failed_stage
        self.local_candidates = tuple(dict(row) for row in candidates)
        self.decisions = tuple(dict(row) for row in decisions)
        self.details = {
            "batch_id": batch_id,
            "failed_stage": failed_stage,
            "contract_errors": list(contract_errors),
        }
        if raw_response is not None:
            self.details["raw_response"] = raw_response
            self.raw_response = raw_response


class ChapterSummaryReviewError(SynopsisExtractionError):
    def __init__(self, *, summary: dict[str, str], decisions: list[dict[str, Any]],
                 material_issues: list[dict[str, str]],
                 missing_points: list[dict[str, str]], coverage_reason: str):
        reasons = []
        if material_issues:
            reasons.append(f"仍有{len(material_issues)}项实质错误")
        if missing_points:
            reasons.append(coverage_reason or f"仍有{len(missing_points)}项必要遗漏")
        super().__init__(f"章节梗概单次修订后仍未通过：{'；'.join(reasons)}",
                         code="CHAPTER_SUMMARY_MAJOR_ERROR_UNRESOLVED")
        self.failed_stage = "chapter-summary"
        self.summary = dict(summary)
        self.decisions = tuple(dict(row) for row in decisions)
        self.details = {"material_issues": material_issues,
                        "missing_points": missing_points}


class ChapterSummaryContractError(SynopsisExtractionError):
    def __init__(self, *, failed_stage: str, summary: dict[str, str] | None,
                 decisions: list[dict[str, Any]], contract_errors: list[str],
                 raw_response: dict[str, Any] | None = None):
        super().__init__(f"章节梗概在{failed_stage}返回了不符合契约的模型结果",
                         code="CHAPTER_SUMMARY_CONTRACT_FAILED")
        self.failed_stage = failed_stage
        self.summary = dict(summary) if summary is not None else None
        self.decisions = tuple(dict(row) for row in decisions)
        self.details = {
            "failed_stage": failed_stage,
            "contract_errors": list(contract_errors),
        }
        if raw_response is not None:
            self.details["raw_response"] = raw_response
            self.raw_response = raw_response


CONTRACT_OUTPUT_ERRORS = (ModelOutputError, SynopsisContractError, ReviewContractError)


def _contract_error_messages(exc: Exception) -> list[str]:
    values = getattr(exc, "errors", None)
    if isinstance(values, (list, tuple)):
        messages = [str(value) for value in values if str(value).strip()]
        if messages:
            return messages
    return [str(exc)]


def _fingerprint(payload: Any) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _checkpoint_path(checkpoint_dir: Path, document: ChapterDocument, stage: str,
                     identity: str, revision: str) -> Path:
    digest = hashlib.sha256(f"{document.source_hash}|{MODULE_VERSION}|{revision}|{stage}|{identity}".encode()).hexdigest()
    return checkpoint_dir / f"{digest}.{stage}.json"


def _request_record(system: str, prompt: str, settings: ModelSettings, *, max_tokens: int,
                    thinking: bool) -> dict[str, Any]:
    effective_system = json_system_message(system)
    messages = [{"role": "system", "content": effective_system}, {"role": "user", "content": prompt}]
    return {"base_system": system, "system": effective_system, "user": prompt, "messages": messages,
            "model": settings.model, "base_url": settings.base_url, "thinking": thinking,
            "max_tokens": max_tokens, "system_character_count": len(effective_system),
            "user_character_count": len(prompt), "input_character_count": len(effective_system) + len(prompt),
            "request_fingerprint": _fingerprint({"messages": messages, "model": settings.model,
                                                  "thinking": thinking, "max_tokens": max_tokens})}


def _complete_checkpointed(completion: JsonCompletion, *, checkpoint_dir: Path,
    document: ChapterDocument, stage: str, identity: str, revision: str, system: str,
    prompt: str, settings: ModelSettings, max_tokens: int, resume: bool, ledger: RequestLedger,
    per_request_limit: int, thinking: bool = False) -> dict[str, Any]:
    request = _request_record(system, prompt, settings, max_tokens=max_tokens, thinking=thinking)
    input_chars = ledger.reserve_logical(stage=stage, system=str(request["system"]), prompt=prompt,
                                         per_request_limit=per_request_limit)
    request_identity = f"{identity}|{request['request_fingerprint']}"
    path = _checkpoint_path(checkpoint_dir, document, stage, request_identity, revision)
    if resume:
        cached = read_json(path)
        if isinstance(cached, dict) and cached.get("stage") == stage and isinstance(cached.get("response"), dict):
            ledger.mark_cache_hit()
            ledger.mark_response(stage=stage, identity=identity, cached=True)
            return cached["response"]
    write_json(path.with_suffix(".request.json"), {"stage": stage, "module_version": MODULE_VERSION,
        "prompt_revision": revision, "source_hash": document.source_hash, "identity": request_identity,
        "request": request})
    path.with_suffix(".request.txt").write_text(f"[SYSTEM]\n{request['system']}\n\n[USER]\n{prompt}\n", encoding="utf-8")

    def observe(attempt: int, content: str, parse_error: str | None, metadata: dict[str, Any]) -> None:
        path.with_suffix(f".attempt-{attempt:02d}.raw.txt").write_text(content, encoding="utf-8")
        write_json(path.with_suffix(f".attempt-{attempt:02d}.meta.json"), {"stage": stage, "attempt": attempt,
            "model": settings.model, "input_character_count": input_chars, "output_character_count": len(content),
            "json_parse_succeeded": parse_error is None, "parse_error": parse_error,
            "finish_reason": metadata.get("finish_reason"), "usage": metadata.get("usage", {})})

    ledger.mark_actual_call(stage=stage, input_chars=input_chars)
    kwargs: dict[str, Any] = {"attempts": 1, "max_tokens": max_tokens, "thinking": thinking,
                              "allow_thinking_fallback": False}
    if completion is complete_json: kwargs["attempt_observer"] = observe
    try:
        response = completion(system, prompt, settings, **kwargs)
    except Exception as exc:
        setattr(exc, "failed_stage", stage); raise
    write_json(path, {"stage": stage, "module_version": MODULE_VERSION, "prompt_revision": revision,
        "source_hash": document.source_hash, "identity": request_identity, "response": response,
        "request": request})
    ledger.mark_response(stage=stage, identity=identity, cached=False)
    return response


def _call(completion: JsonCompletion, *, document: ChapterDocument, checkpoint_dir: Path,
          ledger: RequestLedger, stage: str, identity: str, revision: str, prompt: str,
          settings: ModelSettings, max_tokens: int, resume: bool, kind: str,
          thinking: bool = False) -> dict[str, Any]:
    limits = {"segmentation": SEGMENTATION_REQUEST_CHAR_LIMIT, "local": LOCAL_REQUEST_CHAR_LIMIT,
              "chapter": CHAPTER_REQUEST_CHAR_LIMIT}
    try:
        return _complete_checkpointed(completion, checkpoint_dir=checkpoint_dir, document=document,
            stage=stage, identity=identity, revision=revision, system=LOCAL_SYSTEM_PROMPT, prompt=prompt,
            settings=settings, max_tokens=max_tokens, resume=resume, ledger=ledger,
            per_request_limit=limits[kind], thinking=thinking)
    except Exception as exc:
        setattr(exc, "failed_stage", stage); raise


def _parse_stage(stage: str, payload: dict[str, Any], parser: Callable[[], Any],
                 *, ledger: RequestLedger) -> Any:
    """保留模型已返回这一事实，并把结构错误定位到精确阶段。"""
    try:
        value = parser()
        ledger.mark_parse(stage=stage, succeeded=True)
        return value
    except (ReviewContractError, SynopsisContractError) as exc:
        ledger.mark_parse(stage=stage, succeeded=False)
        setattr(exc, "failed_stage", stage)
        setattr(exc, "raw_response", payload)
        raise


def _navigation_payload(document: ChapterDocument, plan: ChapterNavigationPlan) -> dict[str, Any]:
    aliases, _ = chapter_paragraph_alias_maps(document)
    return {"章节分块": [{"块ID": block.block_id, "起始段落ID": aliases[block.start_unit_id],
        "结束段落ID": aliases[block.end_unit_id], "标签": block.label} for block in plan.blocks]}


def _segmentation_allowed_split_ids(
    document: ChapterDocument, plan: ChapterNavigationPlan,
) -> dict[str, tuple[str, ...]]:
    """列出每个分区内部可作为后续独立推进起点的段落ID。"""

    aliases, _ = chapter_paragraph_alias_maps(document)
    units = document.annotation_units
    return {
        block.block_id: tuple(
            aliases[units[index].unit_id]
            for index in range(block.start_index + 1, block.end_index + 1)
        )
        for block in plan.blocks
    }


def extract_chapter_segmentation(document: ChapterDocument, *, settings: ModelSettings,
    checkpoint_dir: Path, resume: bool, completion: JsonCompletion, ledger: RequestLedger,
    chapter_thinking: bool) -> ChapterNavigationPlan:
    if sum(len(u.text) for u in document.annotation_units) > SEGMENTATION_MAX_SOURCE_CHARS:
        raise SynopsisExtractionError("整章原文超过导航请求允许长度", code="SEGMENTATION_SOURCE_TOO_LONG")
    payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
        stage="chapter-segmentation", identity=document.chapter_id, revision=SEGMENTATION_PROMPT_REVISION,
        prompt=chapter_segmentation_prompt(document),
        settings=settings.for_quality(), max_tokens=SEGMENTATION_OUTPUT_TOKENS, resume=resume,
        kind="segmentation", thinking=chapter_thinking)
    plan = _parse_stage("chapter-segmentation", payload,
                        lambda: parse_chapter_segmentation(document, payload), ledger=ledger)
    review_payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
        stage="chapter-segmentation-review", identity=document.chapter_id,
        revision=SEGMENTATION_REVIEW_PROMPT_REVISION,
        prompt=chapter_segmentation_review_prompt(document, _navigation_payload(document, plan)),
        settings=settings.for_review(), max_tokens=SEGMENTATION_OUTPUT_TOKENS, resume=resume,
        kind="segmentation")
    review = _parse_stage("chapter-segmentation-review", review_payload, lambda: parse_segmentation_review(
        review_payload, expected_ids=[block.block_id for block in plan.blocks]), ledger=ledger)
    if review.passed:
        return plan
    repair_payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
        stage="chapter-segmentation-repair", identity=document.chapter_id,
        revision=SEGMENTATION_REPAIR_PROMPT_REVISION,
        prompt=chapter_segmentation_repair_prompt(document, _navigation_payload(document, plan), list(review.issues)),
        settings=settings.for_quality(), max_tokens=SEGMENTATION_OUTPUT_TOKENS, resume=resume,
        kind="segmentation")
    plan = _parse_stage("chapter-segmentation-repair", repair_payload,
                        lambda: parse_chapter_segmentation(document, repair_payload), ledger=ledger)
    review_payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
        stage="chapter-segmentation-recheck", identity=document.chapter_id + "-recheck",
        revision=SEGMENTATION_RECHECK_PROMPT_REVISION,
        prompt=chapter_segmentation_recheck_prompt(document, _navigation_payload(document, plan)),
        settings=settings.for_review(), max_tokens=SEGMENTATION_OUTPUT_TOKENS, resume=resume,
        kind="segmentation")
    review = _parse_stage("chapter-segmentation-recheck", review_payload,
        lambda: parse_segmentation_recheck(
            review_payload, allowed_split_ids=_segmentation_allowed_split_ids(document, plan)), ledger=ledger)
    if review.passed:
        return plan
    split_points = {
        str(issue["块ID"]): tuple(str(value) for value in issue["拆分起点段落ID"])
        for issue in review.issues
    }
    return split_oversized_navigation_blocks(document, plan, split_points)


def _flat_candidates(candidates: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    return [dict(row) for rows in candidates.values() for row in rows]


def _recent_local_context(local_synopses: list[LocalSynopsis]) -> list[dict[str, Any]]:
    rows = chapter_local_rows(tuple(local_synopses))
    return [dict(row) for row in rows[-LOCAL_PRIOR_CONTEXT_LIMIT:]]


def _generate_local_batch(batch: LocalRequestBatch, *, completion: JsonCompletion,
    document: ChapterDocument, checkpoint_dir: Path, ledger: RequestLedger, settings: ModelSettings,
    resume: bool, prior_context: list[dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
        stage="local-summary-draft", identity=batch.batch_id, revision=LOCAL_SUMMARY_PROMPT_REVISION,
        prompt=local_summary_prompt(batch, prior_context=prior_context),
        settings=settings.for_quality(), max_tokens=LOCAL_OUTPUT_TOKENS,
        resume=resume, kind="local")
    return _parse_stage("local-summary-draft", payload,
                        lambda: parse_local_summary_batch(payload, batch), ledger=ledger)


def _local_sources(batch: LocalRequestBatch) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for window in batch.windows:
        aliases = dict(window.unit_aliases)
        target_id = f"{window.block_id}-S001"
        result[target_id] = {aliases[unit.unit_id]: unit.text for unit in window.target_units}
    return result


def _audit_local_batch(batch: LocalRequestBatch, candidates: dict[str, list[dict[str, str]]], *,
    completion: JsonCompletion, document: ChapterDocument, checkpoint_dir: Path,
    ledger: RequestLedger, settings: ModelSettings, resume: bool,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, Any]], list[SynopsisIssue]]:
    targets = _flat_candidates(candidates)
    decisions: list[dict[str, Any]] = []
    semantic_issues: list[SynopsisIssue] = []
    payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
        stage="local-summary-risk-review", identity=batch.batch_id, revision=LOCAL_RISK_PROMPT_REVISION,
        prompt=local_risk_prompt(batch, targets), settings=settings.for_review(),
        max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS, resume=resume, kind="local")
    candidate_map = {row["目标ID"]: row["梗概"] for row in targets}
    parsed_risk = _parse_stage("local-summary-risk-review", payload,
        lambda: parse_local_risk(payload, candidates=candidate_map), ledger=ledger)
    audits = parsed_risk.audits
    contract_normalizations = list(parsed_risk.contract_normalizations)
    raw_warnings = tuple(issue for audit in audits for issue in audit.warnings)
    raw_majors = tuple(issue for audit in audits for issue in audit.major_candidates)
    valid_warnings, rejected_warnings = validate_grounded_issues(
        raw_warnings, sources_by_target=_local_sources(batch),
        candidates_by_target=candidate_map)
    valid_majors, rejected_majors = validate_grounded_issues(
        raw_majors, sources_by_target=_local_sources(batch),
        candidates_by_target=candidate_map)
    rejected = rejected_warnings + rejected_majors
    semantic_issues.extend(SynopsisIssue("warning", "LOCAL_SEMANTIC_WARNING",
        f"{issue.target_id}：{issue.explanation}") for issue in valid_warnings)
    semantic_issues.extend(SynopsisIssue("warning", "REVIEW_EVIDENCE_INVALID",
        f"{row.get('目标ID', '')}：{row.get('驳回原因', '')}") for row in rejected)
    semantic_issues.extend(SynopsisIssue("warning", "MODEL_OUTPUT_NORMALIZED",
        f"{row.get('目标ID', '')} {row.get('字段', '')}：{row.get('说明', '')}".strip())
        for row in contract_normalizations)
    decisions.append({"stage": "local-risk", "batch_id": batch.batch_id,
        "passed": True,
        "has_warnings": bool(valid_warnings or rejected or contract_normalizations),
        "warnings": [issue.to_dict() for issue in valid_warnings],
        "blocking_issues": [],
        "major_candidates": [issue.to_dict() for issue in valid_majors],
        "rejected_evidence": list(rejected),
        "contract_normalizations": contract_normalizations})
    if not valid_majors:
        return candidates, decisions, semantic_issues

    issue_rows = []
    block_by_target = {row["目标ID"]: row["块ID"] for row in targets}
    for issue in valid_majors:
        issue_rows.append({**issue.to_dict(), "块ID": block_by_target[issue.target_id]})
    payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
        stage="local-summary-major-adjudication", identity=batch.batch_id,
        revision=LOCAL_MAJOR_ADJUDICATION_PROMPT_REVISION,
        prompt=local_major_adjudication_prompt(batch, targets, issue_rows),
        settings=settings.for_review(), max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
        resume=resume, kind="local")
    adjudications = _parse_stage("local-summary-major-adjudication", payload,
        lambda: parse_issue_adjudication(payload,
            expected_ids=[issue.issue_id for issue in valid_majors]), ledger=ledger)
    decision_by_id = {row.issue_id: row for row in adjudications}
    confirmed = tuple(issue for issue in valid_majors if decision_by_id[issue.issue_id].confirmed)
    decisions.append({"stage": "local-major-adjudication", "batch_id": batch.batch_id,
        "passed": True, "requires_repair": bool(confirmed),
        "has_warnings": False, "warnings": [],
        "blocking_issues": [],
        "confirmed_issues": [issue.to_dict() for issue in confirmed],
        "decisions": [{"issue_id": row.issue_id, "confirmed": row.confirmed, "reason": row.reason}
                      for row in adjudications]})
    if not confirmed:
        return candidates, decisions, semantic_issues

    confirmed_rows = [{**issue.to_dict(), "块ID": block_by_target[issue.target_id]}
                      for issue in confirmed]
    failed_blocks = {str(row["块ID"]) for row in confirmed_rows}
    original = {block_id: [dict(row) for row in rows] for block_id, rows in candidates.items()}
    payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
        stage="local-summary-repair", identity=batch.batch_id, revision=LOCAL_REPAIR_PROMPT_REVISION,
        prompt=local_summary_repair_prompt(batch, targets, confirmed_rows),
        settings=settings.for_quality(), max_tokens=LOCAL_OUTPUT_TOKENS,
        resume=resume, kind="local")
    repaired = _parse_stage("local-summary-repair", payload,
        lambda: apply_local_summary_patch(candidates, payload, failed_blocks), ledger=ledger)
    verify_cases = [{**row, "原梗概": original[row["块ID"]][0]["梗概"],
                     "修订后梗概": repaired[row["块ID"]][0]["梗概"]} for row in confirmed_rows]
    payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
        stage="local-summary-repair-verification", identity=batch.batch_id,
        revision=LOCAL_REPAIR_VERIFICATION_PROMPT_REVISION,
        prompt=local_repair_verification_prompt(batch, verify_cases),
        settings=settings.for_review(), max_tokens=LOCAL_REVIEW_OUTPUT_TOKENS,
        resume=resume, kind="local")
    verification = _parse_stage("local-summary-repair-verification", payload,
        lambda: parse_repair_verification(payload,
            expected_ids=[issue.issue_id for issue in confirmed]), ledger=ledger)
    decisions.append({"stage": "local-repair-verification", "batch_id": batch.batch_id,
        "passed": verification.passed, "resolved_issue_ids": list(verification.resolved_issue_ids),
        "unresolved_issue_ids": list(verification.unresolved_issue_ids),
        "warnings": [], "has_warnings": False,
        "blocking_issues": [row for row in confirmed_rows
                            if row["问题ID"] in verification.unresolved_issue_ids]})
    if not verification.passed:
        failed = [row for row in confirmed_rows if row["问题ID"] in verification.unresolved_issue_ids]
        raise LocalBatchReviewError(batch.batch_id, candidates=_flat_candidates(repaired),
                                    decisions=decisions, failed_blocks=failed)
    return repaired, decisions, semantic_issues


def _build_local_values(batch: LocalRequestBatch,
                        candidates: dict[str, list[dict[str, str]]]) -> tuple[LocalSynopsis, ...]:
    support: dict[str, tuple[str, ...]] = {}
    for window in batch.windows:
        source_ids = tuple(paragraph_alias_maps(window)[0].values())
        for row in candidates[window.block_id]:
            support[row["目标ID"]] = source_ids
    return tuple(build_local_synopsis(window, candidates[window.block_id], support)
                 for window in batch.windows)


def _extract_boundary(window: SynopsisWindow, *, kind: str, completion: JsonCompletion,
    document: ChapterDocument, checkpoint_dir: Path, ledger: RequestLedger, settings: ModelSettings,
    resume: bool) -> tuple[BoundaryFrame, dict[str, Any], list[SynopsisIssue]]:
    prefix = "chapter-opening" if kind == "opening" else "chapter-ending"
    payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
        stage=prefix + "-extract", identity=window.window_id, revision=BOUNDARY_PROMPT_REVISION,
        prompt=boundary_extract_prompt(window, kind=kind), settings=settings.for_quality(),
        max_tokens=1200, resume=resume, kind="chapter")
    value = _parse_stage(prefix + "-extract", payload,
                         lambda: parse_boundary_value(payload, kind=kind), ledger=ledger)
    payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
        stage=prefix + "-review", identity=window.window_id, revision=BOUNDARY_PROMPT_REVISION,
        prompt=boundary_review_prompt(window, kind=kind, value=value), settings=settings.for_review(),
        max_tokens=1200, resume=resume, kind="chapter")
    review = _parse_stage(prefix + "-review", payload,
                          lambda: parse_boundary_review(payload, candidate=value), ledger=ledger)
    aliases = dict(window.unit_aliases)
    sources = {"boundary": {aliases[unit.unit_id]: unit.text for unit in window.target_units}}
    candidates = {"boundary": value}
    valid_warnings, rejected_warnings = validate_grounded_issues(
        review.warnings, sources_by_target=sources, candidates_by_target=candidates)
    valid_majors, rejected_majors = validate_grounded_issues(
        review.major_issues, sources_by_target=sources, candidates_by_target=candidates)
    rejected = rejected_warnings + rejected_majors
    decision = {"stage": prefix, "passed": not valid_majors,
                "has_warnings": bool(valid_warnings or rejected),
                "warnings": [issue.to_dict() for issue in valid_warnings],
                "blocking_issues": [issue.to_dict() for issue in valid_majors],
                "rejected_evidence": list(rejected)}
    if valid_majors:
        exc = SynopsisExtractionError(f"{prefix}存在{len(valid_majors)}项有证据的重大错误",
                                      code="BOUNDARY_MAJOR_ERROR")
        exc.failed_stage = prefix + "-review"
        exc.details = {"blocking_issues": [issue.to_dict() for issue in valid_majors]}
        exc.partial_decisions = (decision,)
        raise exc
    source_ids = ((window.target_units[0].unit_id,)
                  if kind == "opening"
                  else tuple(unit.unit_id for unit in window.target_units))
    warnings = [SynopsisIssue("warning", "BOUNDARY_SEMANTIC_WARNING",
                              f"{prefix}：{issue.explanation}") for issue in valid_warnings]
    warnings.extend(SynopsisIssue("warning", "REVIEW_EVIDENCE_INVALID",
                                  f"{prefix}：{row.get('驳回原因', '')}") for row in rejected)
    return BoundaryFrame(value, source_ids), decision, warnings


def _review_chapter_summary(local_rows: list[dict[str, Any]], summary: dict[str, str], *,
    completion: JsonCompletion, document: ChapterDocument, checkpoint_dir: Path,
    ledger: RequestLedger, settings: ModelSettings, resume: bool,
    suffix: str = "") -> tuple[Any, Any, list[dict[str, Any]], list[SynopsisIssue]]:
    review_decisions: list[dict[str, Any]] = []
    semantic_issues: list[SynopsisIssue] = []
    phase = "recheck" if suffix else "initial"
    active_stage = "chapter-summary-fidelity-review"
    try:
        payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
            stage=active_stage, identity=document.chapter_id + suffix,
            revision=CHAPTER_SUMMARY_PROMPT_REVISION,
            prompt=chapter_summary_review_prompt(local_rows, summary), settings=settings.for_review(),
            max_tokens=CHAPTER_REVIEW_OUTPUT_TOKENS, resume=resume, kind="chapter")
        fidelity = _parse_stage(active_stage, payload,
            lambda: parse_chapter_fidelity(payload, summary=summary["内容"], local_rows=local_rows),
            ledger=ledger)
        chapter_sources = {"chapter-summary": {
            str(row["局部梗概ID"]): str(row["梗概"]) for row in local_rows}}
        chapter_candidates = {"chapter-summary": summary["内容"]}
        valid_fidelity_warnings, rejected_fidelity_warnings = validate_grounded_issues(
            fidelity.warnings, sources_by_target=chapter_sources,
            candidates_by_target=chapter_candidates)
        valid_fidelity_majors, rejected_fidelity_majors = validate_grounded_issues(
            fidelity.major_issues, sources_by_target=chapter_sources,
            candidates_by_target=chapter_candidates)
        fidelity = type(fidelity)(valid_fidelity_majors, valid_fidelity_warnings)
        fidelity_rejected = rejected_fidelity_warnings + rejected_fidelity_majors
        semantic_issues.extend(SynopsisIssue("warning", "CHAPTER_SEMANTIC_WARNING", issue.explanation)
                               for issue in valid_fidelity_warnings)
        semantic_issues.extend(SynopsisIssue("warning", "REVIEW_EVIDENCE_INVALID",
            f"章节事实审查：{row.get('驳回原因', '')}") for row in fidelity_rejected)
        review_decisions.append(
            {"stage": "chapter-summary-fidelity", "phase": phase, "passed": fidelity.passed,
             "material_error_count": len(fidelity.major_issues),
             "warning_count": len(fidelity.warnings),
             "has_warnings": bool(fidelity.warnings or fidelity_rejected),
             "blocking_issues": [issue.to_dict() for issue in fidelity.major_issues],
             "warnings": [issue.to_dict() for issue in fidelity.warnings],
             "rejected_evidence": list(fidelity_rejected)}
        )

        active_stage = "chapter-summary-coverage-review"
        payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
            stage=active_stage, identity=document.chapter_id + suffix,
            revision=CHAPTER_SUMMARY_PROMPT_REVISION,
            prompt=chapter_summary_coverage_review_prompt(local_rows, summary), settings=settings.for_review(),
            max_tokens=CHAPTER_REVIEW_OUTPUT_TOKENS, resume=resume, kind="chapter")
        local_ids = [str(row["局部梗概ID"]) for row in local_rows]
        coverage = _parse_stage(active_stage, payload,
            lambda: parse_chapter_coverage(payload, local_ids=local_ids), ledger=ledger)
        valid_coverage_majors, rejected_coverage_majors = validate_chapter_coverage_issues(
            coverage.major_omissions, local_ids=set(local_ids))
        valid_optional, rejected_optional = validate_chapter_coverage_issues(
            coverage.optional_additions, local_ids=set(local_ids))
        coverage = type(coverage)(valid_coverage_majors, valid_optional)
        coverage_rejected = rejected_coverage_majors + rejected_optional
        semantic_issues.extend(SynopsisIssue("warning", "CHAPTER_OPTIONAL_ADDITION",
            issue.explanation) for issue in coverage.optional_additions)
        semantic_issues.extend(SynopsisIssue("warning", "REVIEW_EVIDENCE_INVALID",
            f"章节完整性审查：{row.get('驳回原因', '')}") for row in coverage_rejected)
        review_decisions.append(
            {"stage": "chapter-summary-coverage", "phase": phase, "passed": coverage.passed,
             "has_warnings": bool(coverage.optional_additions or coverage_rejected),
             "blocking_issues": [issue.to_dict() for issue in coverage.major_omissions],
             "warnings": [issue.to_dict() for issue in coverage.optional_additions],
             "rejected_evidence": list(coverage_rejected)}
        )
        return fidelity, coverage, review_decisions, semantic_issues
    except CONTRACT_OUTPUT_ERRORS as exc:
        if not getattr(exc, "failed_stage", None):
            setattr(exc, "failed_stage", active_stage)
        setattr(exc, "partial_decisions", tuple(review_decisions))
        raise


def _chapter_summary(local_rows: list[dict[str, Any]], *, completion: JsonCompletion,
    document: ChapterDocument, checkpoint_dir: Path, ledger: RequestLedger, settings: ModelSettings,
    resume: bool) -> tuple[dict[str, Any], list[dict[str, Any]], list[SynopsisIssue]]:
    summary: dict[str, str] | None = None
    decisions: list[dict[str, Any]] = []
    semantic_issues: list[SynopsisIssue] = []
    active_stage = "chapter-summary"
    try:
        payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
            stage=active_stage, identity=document.chapter_id, revision=CHAPTER_SUMMARY_PROMPT_REVISION,
            prompt=chapter_summary_prompt(local_rows), settings=settings.for_quality(),
            max_tokens=CHAPTER_OUTPUT_TOKENS,
            resume=resume, kind="chapter")
        summary = _parse_stage(active_stage, payload, lambda: parse_chapter_summary(payload), ledger=ledger)
        fidelity, coverage, review_rows, review_warnings = _review_chapter_summary(
            local_rows, summary, completion=completion,
            document=document, checkpoint_dir=checkpoint_dir, ledger=ledger, settings=settings,
            resume=resume)
        decisions.extend(review_rows)
        semantic_issues.extend(review_warnings)
        if not fidelity.passed or not coverage.passed:
            active_stage = "chapter-summary-repair"
            payload = _call(completion, document=document, checkpoint_dir=checkpoint_dir, ledger=ledger,
                stage=active_stage, identity=document.chapter_id, revision=CHAPTER_SUMMARY_PROMPT_REVISION,
                prompt=chapter_summary_repair_prompt(local_rows, summary,
                    [] if fidelity.passed else [issue.to_dict() for issue in fidelity.major_issues],
                    [] if coverage.passed else [issue.to_dict() for issue in coverage.major_omissions]),
                settings=settings.for_quality(), max_tokens=CHAPTER_OUTPUT_TOKENS,
                resume=resume, kind="chapter")
            summary = _parse_stage(active_stage, payload, lambda: parse_chapter_summary(payload), ledger=ledger)
            fidelity, coverage, review_rows, review_warnings = _review_chapter_summary(
                local_rows, summary, completion=completion,
                document=document, checkpoint_dir=checkpoint_dir, ledger=ledger, settings=settings,
                resume=resume, suffix="-recheck")
            decisions.extend(review_rows)
            semantic_issues.extend(review_warnings)
        if not fidelity.passed or not coverage.passed:
            raise ChapterSummaryReviewError(
                summary=summary, decisions=decisions,
                material_issues=[issue.to_dict() for issue in fidelity.major_issues],
                missing_points=[issue.to_dict() for issue in coverage.major_omissions],
                coverage_reason="仍有章节级重大遗漏")
        decisions.append({"stage": "chapter-summary", "passed": True,
                          "local_segment_ids": [row["局部梗概ID"] for row in local_rows]})
        return summary, decisions, semantic_issues
    except CONTRACT_OUTPUT_ERRORS as exc:
        partial_decisions = getattr(exc, "partial_decisions", ())
        decisions.extend(dict(row) for row in partial_decisions)
        failed_stage = str(getattr(exc, "failed_stage", active_stage))
        raise ChapterSummaryContractError(
            failed_stage=failed_stage,
            summary=summary,
            decisions=decisions,
            contract_errors=_contract_error_messages(exc),
            raw_response=getattr(exc, "raw_response", None),
        ) from exc


def extract_document(document: ChapterDocument, *, settings: ModelSettings, checkpoint_dir: Path,
    resume: bool = True, completion: JsonCompletion = complete_json,
    chapter_thinking: bool = False) -> ChapterExecutionResult:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ledger = RequestLedger(sum(len(unit.text) for unit in document.annotation_units))
    local_candidates: list[dict[str, Any]] = []; decisions: list[dict[str, Any]] = []
    semantic_issues: list[SynopsisIssue] = []
    chapter_candidate: dict[str, Any] | None = None
    active_stage = "chapter-segmentation"
    try:
        plan = extract_chapter_segmentation(document, settings=settings, checkpoint_dir=checkpoint_dir,
            resume=resume, completion=completion, ledger=ledger, chapter_thinking=chapter_thinking)
        windows = build_synopsis_windows_from_plan(document, plan)
        batches = build_local_request_batches(windows)
        ledger.configure_fixed_flow(len(batches), len(windows))
        locals_list: list[LocalSynopsis] = []
        for batch in batches:
            active_stage = f"local-batch:{batch.batch_id}"
            try:
                prior_context = _recent_local_context(locals_list)
                candidates = _generate_local_batch(batch, completion=completion, document=document,
                    checkpoint_dir=checkpoint_dir, ledger=ledger, settings=settings,
                    resume=resume, prior_context=prior_context)
                initial_rows = _flat_candidates(candidates)
                local_candidates.extend({**row, "phase": "initial", "selected": False}
                                        for row in initial_rows)
                candidates, batch_decisions, batch_warnings = _audit_local_batch(
                    batch, candidates, completion=completion, document=document,
                    checkpoint_dir=checkpoint_dir, ledger=ledger, settings=settings, resume=resume)
                changed = {row["目标ID"]: row["梗概"] for row in initial_rows} != {
                    row["目标ID"]: row["梗概"] for row in _flat_candidates(candidates)}
                if changed:
                    local_candidates.extend({**row, "phase": "repaired", "selected": True}
                                            for row in _flat_candidates(candidates))
                else:
                    for row in local_candidates[-len(initial_rows):]:
                        row["selected"] = True
                values = _build_local_values(batch, candidates)
                locals_list.extend(values)
                decisions.extend(batch_decisions)
                semantic_issues.extend(batch_warnings)
                decisions.append({"stage": "local", "batch_id": batch.batch_id, "passed": True,
                                  "prior_context_ids": [row["局部梗概ID"] for row in prior_context],
                                  "segment_ids": [s.segment_id for local in values for s in local.segments]})
            except LocalBatchReviewError as exc:
                local_candidates.extend({**row, "phase": "repaired", "selected": False}
                                        for row in exc.local_candidates)
                decisions.extend(exc.decisions)
                raise
            except CONTRACT_OUTPUT_ERRORS as exc:
                wrapped = LocalBatchContractError(batch.batch_id,
                    failed_stage=str(getattr(exc, "failed_stage", active_stage)),
                    candidates=[], decisions=[], contract_errors=_contract_error_messages(exc),
                    raw_response=getattr(exc, "raw_response", None))
                raise wrapped from exc
        local_synopses = tuple(locals_list)
        if not any(local.segments for local in local_synopses):
            raise SynopsisExtractionError("整章没有提取到剧情推进", code="NO_LOCAL_PLOT")
        active_stage = "chapter-opening"
        opening_window = build_boundary_window(document, kind="opening")
        opening, opening_decision, opening_warnings = _extract_boundary(
            opening_window, kind="opening", completion=completion, document=document,
            checkpoint_dir=checkpoint_dir, ledger=ledger, settings=settings, resume=resume)
        decisions.append(opening_decision); semantic_issues.extend(opening_warnings)
        active_stage = "chapter-ending"
        ending_window = build_boundary_window(document, kind="ending")
        ending, ending_decision, ending_warnings = _extract_boundary(
            ending_window, kind="ending", completion=completion, document=document,
            checkpoint_dir=checkpoint_dir, ledger=ledger, settings=settings, resume=resume)
        decisions.append(ending_decision); semantic_issues.extend(ending_warnings)
        local_rows = chapter_local_rows(local_synopses)
        active_stage = "chapter-summary"
        try:
            summary, summary_decisions, summary_warnings = _chapter_summary(
                local_rows, completion=completion, document=document,
                checkpoint_dir=checkpoint_dir, ledger=ledger, settings=settings, resume=resume)
        except ChapterSummaryContractError as exc:
            decisions.extend(exc.decisions)
            decisions.append({
                "stage": "chapter-summary-contract",
                "passed": False,
                "failed_stage": exc.failed_stage,
                "contract_errors": list(exc.details["contract_errors"]),
            })
            if exc.summary is not None:
                chapter_candidate = build_chapter_synopsis(
                    document.chapter_id, opening, ending, local_rows, exc.summary).to_dict()
            raise
        except ChapterSummaryReviewError as exc:
            decisions.extend(exc.decisions)
            chapter_candidate = build_chapter_synopsis(
                document.chapter_id, opening, ending, local_rows, exc.summary).to_dict()
            raise
        decisions.extend(summary_decisions)
        semantic_issues.extend(summary_warnings)
        chapter = build_chapter_synopsis(document.chapter_id, opening, ending, local_rows, summary)
        chapter_candidate = chapter.to_dict()
        quality = assess_synopsis_quality(document, local_synopses, chapter,
                                          semantic_issues=tuple(semantic_issues))
        bundle = ChapterSynopsisBundle(document.chapter_id, document.source_hash, local_synopses, chapter, quality)
        bundle.validate()
        return ChapterExecutionResult(document.chapter_id, document.source_hash, bundle,
            tuple(local_candidates), chapter_candidate, tuple(decisions), None, ledger.to_dict(),
            tuple(dict(row) for row in ledger.stage_records))
    except Exception as exc:
        for row in getattr(exc, "partial_decisions", ()):
            value = dict(row)
            if value not in decisions:
                decisions.append(value)
        raw_response = getattr(exc, "raw_response", None)
        if isinstance(raw_response, dict):
            decisions.append({
                "stage": str(getattr(exc, "failed_stage", active_stage)),
                "passed": False,
                "has_warnings": False,
                "warnings": [],
                "blocking_issues": [],
                "contract_errors": _contract_error_messages(exc),
                "raw_response": raw_response,
            })
        failure = {
            "chapter_id": document.chapter_id, "code": str(getattr(exc, "code", type(exc).__name__)),
            "failed_stage": str(getattr(exc, "failed_stage", active_stage)), "message": str(exc),
        }
        details = getattr(exc, "details", None)
        if isinstance(details, dict) and details:
            failure["details"] = details
        return ChapterExecutionResult(document.chapter_id, document.source_hash, None,
            tuple(local_candidates), chapter_candidate, tuple(decisions), failure, ledger.to_dict(),
            tuple(dict(row) for row in ledger.stage_records))
