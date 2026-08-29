from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .jsonio import read_json, write_json
from .paths import runs_dir


SCHEMA_VERSION = "1.0"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path(author_id: str, run_id: str) -> Path:
    return runs_dir(author_id, run_id) / "pipeline_manifest.json"


def initialize_manifest(author_id: str, run_id: str, settings: dict[str, Any]) -> dict[str, Any]:
    path = manifest_path(author_id, run_id)
    existing = read_json(path, {})
    if existing:
        if existing.get("author_id") != author_id:
            raise ValueError("run_id already belongs to another author")
        return existing
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "author_id": author_id,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "stages": {},
    }
    write_json(path, manifest)
    return manifest


def record_stage(author_id: str, run_id: str, stage: int, name: str, artifacts: list[Path], details: dict[str, Any] | None = None) -> Path:
    path = manifest_path(author_id, run_id)
    manifest = read_json(path, {})
    if not manifest:
        raise ValueError("pipeline manifest has not been initialized")
    rows = []
    for artifact in artifacts:
        if artifact.exists() and artifact.is_file():
            rows.append({"path": str(artifact.resolve()), "sha256": _file_hash(artifact), "bytes": artifact.stat().st_size})
    manifest.setdefault("stages", {})[str(stage)] = {
        "name": name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": rows,
        "details": details or {},
    }
    write_json(path, manifest)
    return path
