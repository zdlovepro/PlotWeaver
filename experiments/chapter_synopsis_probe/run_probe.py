from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Callable


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiments.chapter_synopsis_probe.parser import (  # noqa: E402
    COVERAGE_CATEGORIES,
    GRANULARITY_CATEGORIES,
    ProbeProtocolError,
    ParsedReview,
    parse_category_review,
    parse_extraction,
    parse_fidelity_review,
)
from experiments.chapter_synopsis_probe.prompts import (  # noqa: E402
    EXTRACTION_PROMPT_REVISION,
    REVIEW_PROMPT_REVISION,
    coverage_review_prompt,
    extraction_prompt,
    fidelity_review_prompt,
    granularity_review_prompt,
)
from experiments.chapter_synopsis_probe.report import build_markdown, build_summary  # noqa: E402
from pipeline.common.jsonio import read_jsonl, write_json  # noqa: E402
from pipeline.common.model import (  # noqa: E402
    ModelOutputError,
    ModelServiceError,
    ModelSettings,
    complete_json,
    json_system_message,
)
from pipeline.common.paths import ROOT, corpus_dir, safe_work_id, validate_author_id  # noqa: E402
from pipeline.contracts.source import ChapterDocument  # noqa: E402


EXPERIMENT_VERSION = "1.0.0"
STATUS_OK = {"passed", "unreviewed"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="独立、单轮测试模型的章节梗概提取能力，不运行第一模块。",
    )
    parser.add_argument("--author-id", required=True)
    parser.add_argument("--work-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=3, help="未指定--chapter时读取前N章")
    parser.add_argument("--chapter", help="按chapter_no选择，例如1或1,3,5")
    parser.add_argument("--audit", choices=("none", "full"), default="full")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model-tier", choices=("base", "quality"), default="base")
    parser.add_argument("--max-source-chars", type=int, default=30_000)
    parser.add_argument("--max-tokens", type=int, default=4_000)
    return parser


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _safe_component(value: str, label: str) -> str:
    normalized = safe_work_id(value)
    if normalized != value or not value:
        raise ValueError(f"{label}包含不安全字符：{value!r}")
    return normalized


def _parse_chapter_numbers(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    result: set[int] = set()
    for item in raw.split(","):
        value = item.strip()
        if not value.isdigit() or int(value) <= 0:
            raise ValueError(f"--chapter只接受正整数列表：{raw}")
        result.add(int(value))
    return result


def _load_chapters(author_id: str, work_id: str, *, chapter_numbers: set[int] | None, limit: int) -> list[dict[str, Any]]:
    source_dir = corpus_dir(author_id, work_id)
    document_rows = read_jsonl(source_dir / "chapter_documents.jsonl")
    record_rows = read_jsonl(source_dir / "chapters.jsonl")
    if not document_rows:
        raise FileNotFoundError(f"没有找到章节文档：{source_dir / 'chapter_documents.jsonl'}")
    record_by_id = {str(row.get("chapter_id", "")): row for row in record_rows}
    loaded: list[dict[str, Any]] = []
    for index, row in enumerate(document_rows, start=1):
        document = ChapterDocument.from_dict(row)
        document.validate()
        record = record_by_id.get(document.chapter_id, {})
        chapter_no = int(record.get("chapter_no", index))
        if chapter_numbers is not None and chapter_no not in chapter_numbers:
            continue
        loaded.append({
            "document": document,
            "chapter_no": chapter_no,
            "title": str(record.get("title", "")).strip() or f"第{chapter_no}章",
        })
        if chapter_numbers is None and len(loaded) >= limit:
            break
    if chapter_numbers is not None:
        found = {int(item["chapter_no"]) for item in loaded}
        missing = sorted(chapter_numbers - found)
        if missing:
            raise ValueError(f"作品中不存在指定章节：{','.join(map(str, missing))}")
    if not loaded:
        raise ValueError("没有选中可测试章节")
    return loaded


def _source_rows(document: ChapterDocument) -> list[dict[str, str]]:
    # The probe evaluates the complete chapter exactly as loaded.  Keep every
    # stable source unit available to reviewers instead of silently removing
    # editorial or separator units through the production annotation policy.
    return [{"段落ID": unit.unit_id, "原文": unit.text} for unit in document.units]


def _invoke_once(
    *, stage: str, chapter_dir: Path, system: str, prompt: str,
    settings: ModelSettings, max_tokens: int, thinking: bool,
) -> dict[str, Any]:
    _write_text(chapter_dir / f"{stage}.prompt.txt", prompt)
    write_json(chapter_dir / f"{stage}.request.json", {
        "stage": stage,
        "transport": "pipeline.common.model.complete_json",
        "model": settings.model,
        "thinking_requested": thinking,
        "thinking_effective": bool(thinking and settings.thinking_enabled),
        "reasoning_effort": settings.reasoning_effort,
        "max_tokens": max_tokens,
        "attempts": 1,
        "allow_thinking_fallback": False,
        "messages": [
            {"role": "system", "content": json_system_message(system)},
            {"role": "user", "content": prompt},
        ],
    })

    observed = False

    def observe(attempt: int, raw: str, parse_error: str | None, metadata: dict[str, Any]) -> None:
        nonlocal observed
        observed = True
        _write_text(chapter_dir / f"{stage}.raw.txt", raw)
        write_json(chapter_dir / f"{stage}.attempt.json", {
            "attempt": attempt,
            "parse_error": parse_error,
            "response_metadata": metadata,
            "raw_character_count": len(raw),
        })

    try:
        payload = complete_json(
            system,
            prompt,
            settings,
            attempts=1,
            max_tokens=max_tokens,
            thinking=thinking,
            allow_thinking_fallback=False,
            attempt_observer=observe,
        )
    except Exception as exc:
        if not observed:
            write_json(chapter_dir / f"{stage}.attempt.json", {
                "attempt": 1,
                "call_error": f"{type(exc).__name__}: {exc}",
                "response_metadata": {},
                "raw_character_count": 0,
            })
        raise
    write_json(chapter_dir / f"{stage}.result.json", payload)
    return payload


def _review_result(parsed: ParsedReview, payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": parsed.status,
        "issues": list(parsed.issues),
        "protocol_errors": [],
        "model_result": payload,
    }
    if parsed.checks is not None:
        result["checks"] = parsed.checks
    return result


def _run_review(
    *, name: str, chapter_dir: Path, system: str, prompt: str,
    settings: ModelSettings, max_tokens: int, thinking: bool,
    parse: Callable[[dict[str, Any]], ParsedReview],
) -> dict[str, Any]:
    try:
        payload = _invoke_once(
            stage=name,
            chapter_dir=chapter_dir,
            system=system,
            prompt=prompt,
            settings=settings,
            max_tokens=max_tokens,
            thinking=thinking,
        )
        return _review_result(parse(payload), payload)
    except ProbeProtocolError as exc:
        return {"status": "protocol_failed", "issues": [], "protocol_errors": list(exc.errors)}
    except ModelServiceError as exc:
        return {"status": "model_service_failed", "issues": [], "protocol_errors": [str(exc)]}
    except ModelOutputError as exc:
        return {"status": "protocol_failed", "issues": [], "protocol_errors": [str(exc)]}
    except Exception as exc:
        return {
            "status": "protocol_failed",
            "issues": [],
            "protocol_errors": [f"{type(exc).__name__}: {exc}"],
        }


def _final_review_status(reviews: dict[str, dict[str, Any]]) -> str:
    statuses = {str(item.get("status", "")) for item in reviews.values()}
    if "model_service_failed" in statuses:
        return "model_service_failed"
    if "protocol_failed" in statuses:
        return "review_protocol_failed"
    if "inconclusive" in statuses:
        return "inconclusive"
    if "needs_revision" in statuses:
        return "content_failed"
    return "passed" if statuses == {"passed"} else "inconclusive"


def _run_chapter(
    chapter: dict[str, Any], *, run_dir: Path, extraction_settings: ModelSettings,
    review_settings: ModelSettings, audit_mode: str, thinking: bool,
    max_source_chars: int, max_tokens: int,
) -> dict[str, Any]:
    document: ChapterDocument = chapter["document"]
    chapter_no = int(chapter["chapter_no"])
    title = str(chapter["title"])
    chapter_dir = run_dir / f"{chapter_no:04d}"
    chapter_dir.mkdir(parents=True, exist_ok=False)
    source_rows = _source_rows(document)
    write_json(chapter_dir / "source.json", {
        "chapter_id": document.chapter_id,
        "chapter_no": chapter_no,
        "title": title,
        "source_hash": document.source_hash,
        "source_chars": len(document.text),
        "text": document.text,
        "paragraphs": source_rows,
    })
    decision: dict[str, Any] = {
        "chapter_id": document.chapter_id,
        "chapter_no": chapter_no,
        "title": title,
        "source_hash": document.source_hash,
        "source_chars": len(document.text),
        "synopsis": "",
        "reviews": {},
        "model_call_count": 0,
        "status": "pending",
    }
    if len(document.text) > max_source_chars:
        decision["status"] = "source_too_large"
        decision["error"] = f"原文{len(document.text)}字符，超过上限{max_source_chars}，未截断也未调用模型"
        write_json(chapter_dir / "decision.json", decision)
        return decision

    prompt = extraction_prompt(chapter_id=document.chapter_id, title=title, source_text=document.text)
    decision["model_call_count"] += 1
    try:
        payload = _invoke_once(
            stage="extraction",
            chapter_dir=chapter_dir,
            system="你是中文小说章节梗概提取器。只根据当前章节正文生成章节级梗概。",
            prompt=prompt,
            settings=extraction_settings,
            max_tokens=max_tokens,
            thinking=thinking,
        )
        extraction = parse_extraction(payload, chapter_id=document.chapter_id)
    except ModelServiceError as exc:
        decision["status"] = "model_service_failed"
        decision["error"] = str(exc)
        write_json(chapter_dir / "decision.json", decision)
        return decision
    except (ModelOutputError, ProbeProtocolError) as exc:
        decision["status"] = "extraction_protocol_failed"
        decision["error"] = str(exc)
        if isinstance(exc, ProbeProtocolError):
            decision["protocol_errors"] = list(exc.errors)
        write_json(chapter_dir / "decision.json", decision)
        return decision
    except Exception as exc:
        decision["status"] = "extraction_protocol_failed"
        decision["error"] = f"{type(exc).__name__}: {exc}"
        write_json(chapter_dir / "decision.json", decision)
        return decision

    synopsis = extraction["章节梗概"]
    decision["synopsis"] = synopsis
    if audit_mode == "none":
        decision["status"] = "unreviewed"
        write_json(chapter_dir / "decision.json", decision)
        return decision

    allowed_source_ids = {row["段落ID"] for row in source_rows}
    review_specs: tuple[tuple[str, str, str, Callable[[dict[str, Any]], ParsedReview]], ...] = (
        (
            "fidelity",
            "你是章节梗概忠实性审校员。只检查忠实性，不检查覆盖和粒度。",
            fidelity_review_prompt(
                chapter_id=document.chapter_id, title=title, source_rows=source_rows, synopsis=synopsis,
            ),
            lambda value: parse_fidelity_review(value, allowed_source_ids=allowed_source_ids),
        ),
        (
            "coverage",
            "你是章节梗概主要推进覆盖审校员。只检查覆盖，不检查忠实性和粒度。",
            coverage_review_prompt(
                chapter_id=document.chapter_id, title=title, source_rows=source_rows, synopsis=synopsis,
            ),
            lambda value: parse_category_review(
                value,
                allowed_source_ids=allowed_source_ids,
                allowed_categories=COVERAGE_CATEGORIES,
                prefix="C",
                label="覆盖审校",
            ),
        ),
        (
            "granularity",
            "你是章节梗概粒度审校员。只检查章节级粒度，不检查忠实性和覆盖。",
            granularity_review_prompt(
                chapter_id=document.chapter_id, title=title, source_rows=source_rows, synopsis=synopsis,
            ),
            lambda value: parse_category_review(
                value,
                allowed_source_ids=allowed_source_ids,
                allowed_categories=GRANULARITY_CATEGORIES,
                prefix="G",
                label="粒度审校",
            ),
        ),
    )
    reviews: dict[str, dict[str, Any]] = {}
    for name, system, review_prompt, parse in review_specs:
        print(f"[章节梗概实验] {document.chapter_id}：{name}", flush=True)
        decision["model_call_count"] += 1
        reviews[name] = _run_review(
            name=name,
            chapter_dir=chapter_dir,
            system=system,
            prompt=review_prompt,
            settings=review_settings,
            max_tokens=max_tokens,
            thinking=thinking,
            parse=parse,
        )
    decision["reviews"] = reviews
    decision["status"] = _final_review_status(reviews)
    write_json(chapter_dir / "decision.json", decision)
    return decision


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    author_id = validate_author_id(args.author_id)
    work_id = _safe_component(args.work_id, "work_id")
    run_id = _safe_component(args.run_id, "run_id")
    if args.limit <= 0:
        raise ValueError("--limit必须大于0")
    if args.max_source_chars <= 0 or args.max_tokens <= 0:
        raise ValueError("--max-source-chars和--max-tokens必须大于0")
    chapter_numbers = _parse_chapter_numbers(args.chapter)
    chapters = _load_chapters(
        author_id,
        work_id,
        chapter_numbers=chapter_numbers,
        limit=args.limit,
    )
    run_dir = ROOT / "experiments" / "chapter_synopsis_probe" / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"实验目录已经存在，拒绝混入旧结果：{run_dir}")
    run_dir.mkdir(parents=True)

    base_settings = ModelSettings.from_environment()
    extraction_settings = base_settings.for_quality() if args.model_tier == "quality" else base_settings
    review_settings = base_settings.for_quality()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest: dict[str, Any] = {
        "experiment": "chapter_synopsis_probe",
        "experiment_version": EXPERIMENT_VERSION,
        "extraction_prompt_revision": EXTRACTION_PROMPT_REVISION,
        "review_prompt_revision": REVIEW_PROMPT_REVISION,
        "run_id": run_id,
        "author_id": author_id,
        "work_id": work_id,
        "audit_mode": args.audit,
        "thinking_requested": args.thinking,
        "model_tier": args.model_tier,
        "extraction_model": extraction_settings.model,
        "review_model": review_settings.model,
        "independent_review_model": extraction_settings.model != review_settings.model,
        "max_source_chars": args.max_source_chars,
        "max_tokens": args.max_tokens,
        "attempts_per_request": 1,
        "thinking_fallback": False,
        "repair_enabled": False,
        "selected_chapters": [item["document"].chapter_id for item in chapters],
        "started_at": started_at,
    }
    write_json(run_dir / "manifest.json", manifest)
    decisions: list[dict[str, Any]] = []
    for chapter in chapters:
        document: ChapterDocument = chapter["document"]
        print(f"[章节梗概实验] {document.chapter_id}：extraction", flush=True)
        decisions.append(_run_chapter(
            chapter,
            run_dir=run_dir,
            extraction_settings=extraction_settings,
            review_settings=review_settings,
            audit_mode=args.audit,
            thinking=args.thinking,
            max_source_chars=args.max_source_chars,
            max_tokens=args.max_tokens,
        ))
    summary = build_summary(decisions)
    manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest["status_counts"] = summary["status_counts"]
    manifest["model_call_count"] = summary["model_call_count"]
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "summary.json", summary)
    _write_text(run_dir / "report.md", build_markdown(manifest, decisions))
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(item.get("status") in STATUS_OK for item in decisions) else 2


if __name__ == "__main__":
    raise SystemExit(main())
