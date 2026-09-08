from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ...common.jsonio import write_json, write_jsonl
from .verification import ChapterExecutionResult


def _aggregate_metrics(results: tuple[ChapterExecutionResult, ...]) -> dict[str, Any]:
    numeric = ("source_character_count", "logical_request_count", "actual_model_call_count",
               "cache_hit_count", "logical_input_character_count", "actual_input_character_count")
    total = {key: sum(int(result.request_metrics.get(key, 0)) for result in results) for key in numeric}
    total["input_amplification_ratio"] = round(total["logical_input_character_count"] / max(1, total["source_character_count"]), 6)
    total["max_single_request_input_chars"] = max((int(r.request_metrics.get("max_single_request_input_chars", 0)) for r in results), default=0)
    total["planned_local_batch_count"] = sum(int(r.request_metrics.get("planned_local_batch_count", 0)) for r in results)
    total["planned_local_block_count"] = sum(int(r.request_metrics.get("planned_local_block_count", 0)) for r in results)
    for metric in ("stage_logical_request_count", "stage_actual_model_call_count", "stage_input_character_count"):
        values: dict[str, int] = {}
        for result in results:
            for stage, count in dict(result.request_metrics.get(metric, {})).items(): values[stage] = values.get(stage, 0) + int(count)
        total[metric] = dict(sorted(values.items()))
    total["warnings"] = [dict(warning, chapter_id=result.chapter_id) for result in results for warning in result.request_metrics.get("warnings", [])]
    return total


def write_synopsis_artifacts(stage_dir: Path, results: Iterable[ChapterExecutionResult],
                             manifest: dict[str, Any], *, source_integrity: dict[str, Any]) -> dict[str, Path]:
    results = tuple(results)
    if not results: raise ValueError("没有章节执行结果")
    bundles = tuple(result.bundle for result in results if result.accepted and result.bundle is not None)
    for bundle in bundles: bundle.validate()
    failures = tuple(result.failure for result in results if result.failure is not None)
    output_dir = stage_dir / "output"; chapter_dir = output_dir / "chapters"; chapter_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "bundles": output_dir / "chapter_synopsis_bundles.jsonl",
        "chapter_synopses": output_dir / "chapter_synopses.jsonl",
        "local_synopses": output_dir / "local_synopses.jsonl",
        "candidate_local_synopses": output_dir / "candidate_local_synopses.jsonl",
        "candidate_chapter_synopses": output_dir / "candidate_chapter_synopses.jsonl",
        "quality": stage_dir / "quality_report.json", "semantic_quality": stage_dir / "semantic_quality_report.json",
        "failures": stage_dir / "chapter_failures.json", "source_integrity": stage_dir / "source_integrity_report.json",
        "manifest": stage_dir / "manifest.json",
    }
    write_jsonl(paths["bundles"], [b.to_dict() for b in bundles])
    write_jsonl(paths["chapter_synopses"], [{"schema_version": b.schema_version, "source_hash": b.source_hash,
                                              **b.chapter_synopsis.to_dict()} for b in bundles])
    write_jsonl(paths["local_synopses"], [{"schema_version": b.schema_version, "source_hash": b.source_hash,
                                            **local.to_dict()} for b in bundles for local in b.local_synopses])
    write_jsonl(paths["candidate_local_synopses"], [{"chapter_id": r.chapter_id, "source_hash": r.source_hash,
        **candidate} for r in results for candidate in r.local_candidates])
    write_jsonl(paths["candidate_chapter_synopses"], [{"chapter_id": r.chapter_id, "source_hash": r.source_hash,
        "chapter_candidate": r.chapter_candidate} for r in results if r.chapter_candidate is not None])
    for i, bundle in enumerate(bundles, 1):
        write_json(chapter_dir / f"{i:04d}-{bundle.chapter_id.rsplit('/', 1)[-1]}.json", bundle.to_dict())
    metrics = _aggregate_metrics(results)
    full_flow_completed = not failures and len(bundles) == len(results)
    all_decisions = [d for r in results for d in r.decisions]
    semantic_warning_count = sum(
        len(decision.get("warnings", [])) + len(decision.get("rejected_evidence", []))
        for decision in all_decisions)
    contract_normalization_count = sum(
        len(decision.get("contract_normalizations", [])) for decision in all_decisions)
    total_warning_count = semantic_warning_count + contract_normalization_count
    final_decisions: list[dict[str, Any]] = []
    for result in results:
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for decision in result.decisions:
            key = (str(decision.get("stage", "")),
                   str(decision.get("batch_id", decision.get("target_or_batch_id", ""))))
            latest[key] = decision
        final_decisions.extend(latest.values())
    semantic_blocking_count = sum(len(decision.get("blocking_issues", []))
                                  for decision in final_decisions if not decision.get("passed", True))
    contract_failure_count = sum("CONTRACT" in str((failure or {}).get("code", "")).upper()
                                 for failure in failures)
    write_json(paths["semantic_quality"], {"passed": full_flow_completed,
        "run_completed": True, "full_flow_completed": full_flow_completed,
        "has_warnings": total_warning_count > 0,
        "warning_count": total_warning_count,
        "semantic_warning_count": semantic_warning_count,
        "contract_normalization_count": contract_normalization_count,
        "semantic_blocking_issue_count": semantic_blocking_count,
        "contract_failure_count": contract_failure_count,
        "pipeline_failure_count": len(failures),
        "module_version": manifest.get("module_version"), "formal_bundle_count": len(bundles),
        "candidate_local_count": sum(len(r.local_candidates) for r in results),
        "candidate_chapter_count": sum(r.chapter_candidate is not None for r in results),
        "request_metrics": metrics,
        "stage_records": [record for result in results for record in result.stage_records],
        "decisions": all_decisions})
    write_json(paths["failures"], {"failure_count": len(failures), "failures": list(failures)})
    write_json(paths["source_integrity"], source_integrity)
    chapters = []
    for result in results:
        unique_candidates = {str(row.get("块ID", "")) for row in result.local_candidates
                             if str(row.get("块ID", ""))}
        selected_candidates = {str(row.get("块ID", "")) for row in result.local_candidates
                               if row.get("selected") and str(row.get("块ID", ""))}
        planned_blocks = int(result.request_metrics.get("planned_local_block_count", 0))
        progress = round(len(unique_candidates) / max(1, planned_blocks), 6) if planned_blocks else 0.0
        if result.accepted and result.bundle:
            chapters.append({"chapter_id": result.chapter_id, "processing_status": "accepted",
                             "pipeline_completed": True, "formal_output_written": True,
                             "candidate_local_count": len(result.local_candidates),
                             "generated_candidate_block_count": len(unique_candidates),
                             "selected_local_block_count": len(selected_candidates),
                             "planned_local_block_count": planned_blocks,
                             "generated_candidate_progress": progress,
                             "review_decision_count": len(result.decisions),
                             **result.bundle.quality.to_dict()})
        else:
            failure = result.failure or {}
            failure_code = str(failure.get("code", "failed"))
            content_status = ("not_verified" if "CONTRACT" in failure_code.upper()
                              or failure_code in {"ModelOutputError", "ReviewContractError", "SynopsisContractError"}
                              else "failed")
            chapters.append({"chapter_id": result.chapter_id, "processing_status": failure.get("code", "failed"),
                             "pipeline_completed": False, "formal_output_written": False,
                             "content_status": content_status,
                             "failed_stage": failure.get("failed_stage", "pipeline"),
                             "candidate_local_count": len(result.local_candidates),
                             "generated_candidate_block_count": len(unique_candidates),
                             "selected_local_block_count": len(selected_candidates),
                             "planned_local_block_count": planned_blocks,
                             "generated_candidate_progress": progress,
                             "review_decision_count": len(result.decisions),
                             "last_recorded_stage": (failure.get("failed_stage")
                                                     or (result.decisions[-1].get("stage")
                                                         if result.decisions else None))})
    quality_passed = full_flow_completed and bool(source_integrity.get("passed"))
    quality = {"passed": quality_passed, "run_completed": True,
        "processing_completed": full_flow_completed, "full_flow_completed": full_flow_completed,
        "attempted_chapter_count": len(results), "chapter_count": len(bundles),
        "verified_bundle_count": len(bundles), "candidate_local_count": sum(len(r.local_candidates) for r in results),
        "candidate_chapter_count": sum(r.chapter_candidate is not None for r in results),
        "source_window_coverage": round(sum(b.quality.source_window_coverage for b in bundles) / len(bundles), 6) if bundles else 0.0,
        "local_segment_count": sum(b.quality.local_segment_count for b in bundles),
        "chapters": chapters, "request_metrics": metrics,
        "source_integrity_passed": bool(source_integrity.get("passed"))}
    write_json(paths["quality"], quality)
    final_manifest = dict(manifest)
    final_manifest["accepted"] = quality["passed"]
    final_manifest["outputs"] = {name: str(path.relative_to(stage_dir)) for name, path in paths.items() if name != "manifest"}
    write_json(paths["manifest"], final_manifest)
    return paths
