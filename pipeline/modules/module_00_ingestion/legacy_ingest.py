from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .contracts import ChapterRecord, build_chapter_document, validate_document
from .jsonio import write_json, write_jsonl
from .paths import corpus_dir, safe_work_id, validate_author_id


CHAPTER_RE = re.compile(r"(?m)^\s*第\s*([0-9０-９一二三四五六七八九十百千万零〇两]+)\s*章\s*([^\n\r]*)$")


def read_source_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    candidates = ("utf-8-sig", "gb18030", "gbk", "utf-16")
    best: tuple[int, str, str] | None = None
    for encoding in candidates:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        cjk = sum("\u4e00" <= char <= "\u9fff" for char in text)
        bad = text.count("�") * 200 + sum(token in text for token in ("銆", "锛", "鈥")) * 10
        score = cjk - bad
        if best is None or score > best[0]:
            best = (score, encoding, text)
    if best is None:
        raise ValueError(f"Unable to decode source file: {path}")
    return best[2].replace("\r\n", "\n").replace("\r", "\n"), best[1]


def _chapter_number(raw: str, fallback: int) -> int:
    clean = str(raw).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if clean.isdigit():
        return int(clean)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total = 0
    current = 0
    for char in clean:
        if char in digits:
            current = digits[char]
        elif char in units:
            total += (current or 1) * units[char]
            current = 0
        else:
            return fallback
    return total + current if total + current else fallback


def _clean_chapter_title(title: str, fallback: int) -> str:
    """Keep story titles while excluding trailing publication-facing notes."""

    clean = str(title or "").strip(" -—\t")
    clean = re.sub(
        r"\s*[（(][^（）()]{0,80}(?:月票|推荐票|订阅|收藏|打赏|第[一二三四五六七八九十0-9]+更|更新)[^（）()]{0,80}[）)]\s*$",
        "",
        clean,
    ).strip()
    return clean or f"Chapter {fallback}"


def _heading_quality(match: re.Match[str]) -> tuple[int, int]:
    """Rank duplicate headings; prefer plausible short chapter titles."""

    title = match.group(2).strip()
    length = len(title)
    sentence_marks = sum(char in "。！？!?；;" for char in title)
    punctuation = sum(char in "，、：:（）()“”\"'" for char in title)
    score = 0
    if 2 <= length <= 32:
        score += 100
    elif length == 1:
        score += 20
    else:
        score -= min(length, 100)
    score -= sentence_marks * 80 + punctuation * 3
    return score, -match.start()


def parse_chapters(text: str, author_id: str, work_id: str) -> list[ChapterRecord]:
    raw_matches: list[re.Match[str]] = []
    for match in CHAPTER_RE.finditer(text):
        source_number = _chapter_number(match.group(1), 0)
        if source_number <= 0 or not match.group(2).strip():
            continue
        raw_matches.append(match)
    groups: dict[int, list[re.Match[str]]] = {}
    for match in raw_matches:
        source_number = _chapter_number(match.group(1), 0)
        groups.setdefault(source_number, []).append(match)
    canonical_matches = [
        max(group, key=_heading_quality)
        for _, group in sorted(groups.items())
    ]
    canonical_matches.sort(key=lambda match: match.start())
    canonical_starts = {match.start() for match in canonical_matches}
    ignored_matches = [match for match in raw_matches if match.start() not in canonical_starts]
    records: list[ChapterRecord] = []
    for index, match in enumerate(canonical_matches):
        start = match.end()
        end = canonical_matches[index + 1].start() if index + 1 < len(canonical_matches) else len(text)
        body = text[start:end]
        for duplicate in sorted((item for item in ignored_matches if start <= item.start() < end), key=lambda item: item.start(), reverse=True):
            duplicate_start = duplicate.start() - start
            duplicate_end = duplicate.end() - start
            body = f"{body[:duplicate_start]}{body[duplicate_end:]}"
        body = body.strip()
        if not body:
            continue
        source_number = _chapter_number(match.group(1), index + 1)
        number = index + 1
        chapter_id = f"{author_id}/{work_id}/{number:04d}"
        records.append(ChapterRecord(
            chapter_id=chapter_id,
            author_id=author_id,
            work_id=work_id,
            chapter_no=number,
            source_chapter_no=source_number,
            title=_clean_chapter_title(match.group(2), number),
            text=body,
            char_count=len(body),
            source_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        ))
    if not records:
        body = text.strip()
        if body:
            records = [ChapterRecord(
                chapter_id=f"{author_id}/{work_id}/0001", author_id=author_id, work_id=work_id,
                chapter_no=1, source_chapter_no=1, title="Chapter 1", text=body, char_count=len(body),
                source_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            )]
    for index, record in enumerate(records):
        record.previous_chapter_id = records[index - 1].chapter_id if index else ""
        record.next_chapter_id = records[index + 1].chapter_id if index + 1 < len(records) else ""
    return records


def ingest_file(author_id: str, source_path: Path, work_id: str | None = None) -> dict:
    author_id = validate_author_id(author_id)
    text, encoding = read_source_text(source_path)
    work_id = safe_work_id(work_id or f"work-{hashlib.sha256(source_path.name.encode('utf-8')).hexdigest()[:10]}")
    records = parse_chapters(text, author_id, work_id)
    destination = corpus_dir(author_id, work_id)
    destination.mkdir(parents=True, exist_ok=True)
    documents = [
        build_chapter_document(record.chapter_id, record.source_hash, record.text, record.title)
        for record in records
    ]
    invalid_documents = [
        (document.chapter_id, validate_document(document).to_dict())
        for document in documents
        if not validate_document(document).passed
    ]
    if invalid_documents:
        raise ValueError(f"source document validation failed: {invalid_documents[0]}")
    write_jsonl(destination / "chapters.jsonl", [record.to_dict() for record in records])
    write_jsonl(destination / "chapter_documents.jsonl", [document.to_dict() for document in documents])
    metadata = {
        "author_id": author_id, "work_id": work_id, "source_file": source_path.name,
        "source_encoding": encoding, "source_title": source_path.stem, "chapter_count": len(records),
        "source_hash": hashlib.sha256(source_path.read_bytes()).hexdigest(), "document_schema_version": "2.0",
    }
    write_json(destination / "metadata.json", metadata)
    return metadata


def ingest_directory(author_id: str, source_dir: Path) -> list[dict]:
    files = sorted(source_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {source_dir}")
    imported = [ingest_file(author_id, path, work_id=f"work-{index:03d}") for index, path in enumerate(files, start=1)]
    write_json(corpus_dir(author_id) / "corpus_manifest.json", {
        "author_id": author_id,
        "source_dir": str(source_dir.resolve()),
        "work_ids": [item["work_id"] for item in imported],
        "source_files": [item["source_file"] for item in imported],
    })
    return imported
