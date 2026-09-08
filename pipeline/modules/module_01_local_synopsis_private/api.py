from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ...common.jsonio import read_json, read_jsonl
from ...common.model import ModelSettings
from ...common.paths import corpus_dir, runs_dir, validate_author_id
from ...contracts import SYNOPSIS_SCHEMA_VERSION, ChapterDocument, build_chapter_document
from ...orchestrator import StageContext, StageResult
from .artifacts import write_synopsis_artifacts
from .prompts import PROMPT_REVISION
from .review import REVIEW_PROMPT_REVISION
from .service import MODULE_VERSION, SEGMENTATION_MAX_SOURCE_CHARS, extract_document


def _load_documents(
    author_id: str, work_id: str, limit: int
) -> tuple[tuple[ChapterDocument, ...], int]:
    work_dir = corpus_dir(author_id, work_id)
    documents = [ChapterDocument.from_dict(row) for row in read_jsonl(work_dir / "chapter_documents.jsonl") if isinstance(row, dict)]
    if not documents:
        documents = []
        for row in read_jsonl(work_dir / "chapters.jsonl"):
            text = str(row.get("text", ""))
            if not text.strip(): continue
            source_hash = str(row.get("source_hash", "")) or hashlib.sha256(text.encode("utf-8")).hexdigest()
            documents.append(build_chapter_document(str(row.get("chapter_id", "")), source_hash, text, str(row.get("title", ""))))
    narrative_documents = tuple(item for item in documents if item.is_narrative)
    selected = narrative_documents[:limit]
    if not selected: raise FileNotFoundError(f"没有找到可处理章节：{work_dir}")
    return selected, len(narrative_documents)


def extract_work(author_id: str, work_id: str, *, limit: int = 5, run_id: str = "module-01",
                 resume: bool = True, settings: ModelSettings | None = None,
                 chapter_thinking: bool = False) -> dict[str, Path]:
    author_id = validate_author_id(author_id)
    if limit < 1: raise ValueError("limit must be positive")
    settings = settings or ModelSettings.from_environment()
    documents, available_narrative_count = _load_documents(author_id, work_id, limit)
    stage_dir = runs_dir(author_id, run_id) / "module_01_local_synopsis"
    checkpoint_dir = stage_dir / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    integrity_rows = [{"chapter_id": d.chapter_id, "source_hash": d.source_hash,
                       "loaded_annotation_unit_count": len(d.annotation_units), "passed": bool(d.annotation_units)} for d in documents]
    source_integrity = {"passed": all(row["passed"] for row in integrity_rows), "chapter_count": len(integrity_rows),
                        "scope": "source_loaded_and_hashed", "chapters": integrity_rows}
    results = []
    for index, document in enumerate(documents, 1):
        print(f"[第一模块] 处理 {index}/{len(documents)}：{document.chapter_id}", flush=True)
        result = extract_document(document, settings=settings, checkpoint_dir=checkpoint_dir,
            resume=resume, chapter_thinking=chapter_thinking)
        results.append(result)
        print(f"[第一模块] {document.chapter_id}：{'完成' if result.accepted else '拒绝'}", flush=True)
    successful = [r.chapter_id for r in results if r.accepted]
    failed = [r.chapter_id for r in results if not r.accepted]
    manifest: dict[str, Any] = {
        "module": "module_01_local_synopsis", "module_version": MODULE_VERSION,
        "schema_version": SYNOPSIS_SCHEMA_VERSION, "prompt_revision": f"{PROMPT_REVISION}+{REVIEW_PROMPT_REVISION}",
        "author_id": author_id, "work_id": work_id, "run_id": run_id,
        "profile": "short_validation" if limit <= 20 else "full_book", "chapter_limit": limit,
        "scope": {
            "available_narrative_chapter_count": available_narrative_count,
            "selected_narrative_chapter_count": len(documents),
            "complete_work": (
                len(documents) == available_narrative_count and not failed
            ),
        },
        "chapter_ids": [d.chapter_id for d in documents], "successful_chapter_ids": successful,
        "failed_chapter_ids": failed, "input_hashes": {d.chapter_id: d.source_hash for d in documents},
        "model_policy": {
            "generation_model": settings.review_model or settings.model, "review_model": settings.review_model or settings.model,
            "chapter_thinking": chapter_thinking,
            "segmentation_max_source_chars": SEGMENTATION_MAX_SOURCE_CHARS,
            "flow": ["全章导航按主导推进划定语义范围，不设尺寸或总数限制", "首次独立审校分区是否过大或边界错误，过碎只诊断",
                     "最多完整修订一次；修订后终检只定位过大分区并由程序按段落起点拆分，过碎不判失败也不合并",
                     "相邻完整块组成传输批次；每批携带开始前最近3条已验收局部梗概作为只读前情",
                     "每个分区生成唯一一条局部梗概，前情只用于消解指代、别名和持续状态",
                     "每批一次局部事实风险扫描；证据必须由程序定位到所属分区，警告不阻断",
                     "局部审查按目标ID接纳；可无歧义恢复的空数组缺项、null、单对象和乱序由程序规范化并留痕",
                     "只有疑似重大错误进入一次独立裁决；成立后最小修正一次并只验证已确认问题",
                     "局部层不做覆盖度审查；提取并按三级结论核验章首章末状态",
                     "全部有序局部梗概直接生成完整章节梗概",
                     "章节梗概分别经过事实一致性和章节主线完整性审查，并最多最小修正一次"],
            "normal_logical_calls": "9 + 2 * local_batch_count",
            "maximum_logical_calls": "14 + 5 * local_batch_count",
            "forbidden_outputs": ["facts", "claims", "entities", "relations", "timeline", "knowledge_graph",
                                  "cross_chapter_threads", "narrative_function", "story_change", "style"],
        },
    }
    return write_synopsis_artifacts(stage_dir, results, manifest, source_integrity=source_integrity)


def run(context: StageContext) -> StageResult:
    paths = extract_work(context.author_id, context.work_id,
        limit=int(context.options.get("chapter_limit", 5)), run_id=context.run_id,
        resume=bool(context.options.get("resume", True)),
        chapter_thinking=bool(context.options.get("chapter_thinking", False)))
    quality = read_json(paths["quality"])
    return StageResult(module="module_01_local_synopsis", module_version=MODULE_VERSION,
        schema_version=SYNOPSIS_SCHEMA_VERSION, accepted=bool(isinstance(quality, dict) and quality.get("passed")),
        output_paths={key: value for key, value in paths.items() if key != "quality"}, quality_report=paths["quality"])
