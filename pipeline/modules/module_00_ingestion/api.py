from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from ...common.jsonio import write_json, write_jsonl
from ...common.paths import corpus_dir, safe_work_id, validate_author_id
from ...contracts import ChapterRecord, build_chapter_document, validate_document
from ...orchestrator import StageContext, StageResult


MODULE_VERSION = "1.0.0"
_CHAPTER_HEADING = re.compile(
    r"(?m)^\s*(?:正文\s*)?第\s*([0-9０-９一二三四五六七八九十百千万零〇两]+)\s*章\s*([^\r\n]*)$"
)
_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
           "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}


def read_source_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    candidates = ("utf-8-sig", "gb18030", "gbk", "utf-16")
    best: tuple[int, str, str] | None = None
    for encoding in candidates:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in text)
        replacement_penalty = text.count("�") * 500
        mojibake_penalty = sum(text.count(marker) for marker in ("锟", "閿", "鈥")) * 20
        score = cjk_count - replacement_penalty - mojibake_penalty
        if best is None or score > best[0]:
            best = (score, encoding, text)
    if best is None:
        raise ValueError(f"无法识别文本编码：{path}")
    return best[2].replace("\r\n", "\n").replace("\r", "\n"), best[1]


def chinese_number(raw: str, fallback: int) -> int:
    normalized = str(raw).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if normalized.isdigit():
        return int(normalized)
    if "万" in normalized:
        left, right = normalized.split("万", 1)
        return chinese_number(left, 1) * 10_000 + chinese_number(right, 0)
    total = 0
    current_digit = 0
    for char in normalized:
        if char in _DIGITS:
            current_digit = _DIGITS[char]
        elif char in _SMALL_UNITS:
            total += (current_digit or 1) * _SMALL_UNITS[char]
            current_digit = 0
        else:
            return fallback
    value = total + current_digit
    return value if value > 0 else fallback


def parse_chapters(text: str, author_id: str, work_id: str) -> list[ChapterRecord]:
    matches = list(_CHAPTER_HEADING.finditer(text))
    records: list[ChapterRecord] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        chapter_no = len(records) + 1
        source_no = chinese_number(match.group(1), chapter_no)
        title = match.group(2).strip(" \t-—:：") or f"第{source_no}章"
        chapter_id = f"{author_id}/{work_id}/{chapter_no:04d}"
        records.append(ChapterRecord(
            chapter_id=chapter_id,
            author_id=author_id,
            work_id=work_id,
            chapter_no=chapter_no,
            source_chapter_no=source_no,
            title=title,
            text=body,
            char_count=len(body),
            source_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        ))
    if not records:
        body = text.strip()
        if body:
            records.append(ChapterRecord(
                chapter_id=f"{author_id}/{work_id}/0001",
                author_id=author_id,
                work_id=work_id,
                chapter_no=1,
                source_chapter_no=1,
                title="第1章",
                text=body,
                char_count=len(body),
                source_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            ))
    for index, record in enumerate(records):
        record.previous_chapter_id = records[index - 1].chapter_id if index else ""
        record.next_chapter_id = records[index + 1].chapter_id if index + 1 < len(records) else ""
    return records


def ingest_file(author_id: str, source_path: Path, work_id: str | None = None) -> dict[str, Any]:
    author_id = validate_author_id(author_id)
    text, encoding = read_source_text(source_path)
    work_id = safe_work_id(work_id or f"work-{hashlib.sha256(source_path.name.encode('utf-8')).hexdigest()[:10]}")
    records = parse_chapters(text, author_id, work_id)
    documents = [
        build_chapter_document(record.chapter_id, record.source_hash, record.text, record.title)
        for record in records
    ]
    for document in documents:
        report = validate_document(document)
        if not report.passed:
            raise ValueError(f"章节文档校验失败：{document.chapter_id}: {report.to_dict()}")
    destination = corpus_dir(author_id, work_id)
    destination.mkdir(parents=True, exist_ok=True)
    write_jsonl(destination / "chapters.jsonl", [record.to_dict() for record in records])
    write_jsonl(destination / "chapter_documents.jsonl", [document.to_dict() for document in documents])
    metadata = {
        "author_id": author_id,
        "work_id": work_id,
        "source_file": source_path.name,
        "source_encoding": encoding,
        "source_title": source_path.stem,
        "chapter_count": len(records),
        "source_hash": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "document_schema_version": "2.1",
    }
    write_json(destination / "metadata.json", metadata)
    return metadata


def ingest_directory(author_id: str, source_dir: Path) -> list[dict[str, Any]]:
    files = sorted(source_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"目录中没有txt文件：{source_dir}")
    imported = [
        ingest_file(author_id, path, work_id=f"work-{index:03d}")
        for index, path in enumerate(files, start=1)
    ]
    write_json(corpus_dir(author_id) / "corpus_manifest.json", {
        "author_id": author_id,
        "source_dir": str(source_dir.resolve()),
        "work_ids": [item["work_id"] for item in imported],
        "source_files": [item["source_file"] for item in imported],
    })
    return imported


def run(context: StageContext) -> StageResult:
    source = context.input_paths.get("source")
    source_dir = context.input_paths.get("source_dir")
    if source:
        metadata = ingest_file(context.author_id, source, context.work_id or None)
        work_ids = [str(metadata["work_id"])]
    elif source_dir:
        imported = ingest_directory(context.author_id, source_dir)
        work_ids = [str(item["work_id"]) for item in imported]
    else:
        raise ValueError("语料导入需要 source 或 source_dir")
    return StageResult(
        module="module_00_ingestion",
        module_version=MODULE_VERSION,
        schema_version="2.1",
        accepted=True,
        output_paths={"corpus": corpus_dir(context.author_id)},
        diagnostics=tuple(f"已导入作品：{work_id}" for work_id in work_ids),
    )

