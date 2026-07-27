from __future__ import annotations

import re
from pathlib import Path


# pipeline/<module>.py -> workspace root is one directory above ``pipeline``.
ROOT = Path(__file__).resolve().parents[1]
AUTHOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_author_id(author_id: str) -> str:
    value = str(author_id or "").strip()
    if not AUTHOR_ID_PATTERN.fullmatch(value):
        raise ValueError("author_id may contain only letters, numbers, underscores, and hyphens")
    return value


def corpus_dir(author_id: str, work_id: str | None = None) -> Path:
    base = ROOT / "corpus" / validate_author_id(author_id)
    return base / work_id if work_id else base


def active_work_ids(author_id: str) -> list[str]:
    """Return the work set selected by the latest corpus import.

    Older work folders are intentionally preserved as caches, but are not part of
    a new pipeline run unless the current corpus manifest names them.
    """
    base = corpus_dir(author_id)
    manifest = base / "corpus_manifest.json"
    if manifest.exists():
        import json
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        ids = payload.get("work_ids", []) if isinstance(payload, dict) else []
        return [str(work_id) for work_id in ids if (base / str(work_id) / "chapters.jsonl").exists()]
    return [path.name for path in sorted(base.iterdir()) if path.is_dir() and (path / "chapters.jsonl").exists()] if base.exists() else []


def runs_dir(author_id: str, run_id: str) -> Path:
    return ROOT / "runs" / validate_author_id(author_id) / run_id


def output_dir(author_id: str) -> Path:
    return ROOT / "output" / f"{validate_author_id(author_id)}_skill"


def safe_work_id(raw: str) -> str:
    value = re.sub(r"[<>:\\/?*\x00-\x1f]", "-", str(raw or "").strip())
    value = re.sub(r"\s+", "-", value).strip(".-")
    return value[:96] or "work"
