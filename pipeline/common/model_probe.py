"""Small, source-free availability probe for the configured JSON model."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from .jsonio import write_json
from .model import ModelSettings, complete_json
from .paths import runs_dir, validate_author_id


def probe_model(author_id: str, *, run_id: str = "model-probe") -> Path:
    """Verify a non-empty strict-JSON completion before costly chapter work."""

    author_id = validate_author_id(author_id)
    target = runs_dir(author_id, run_id) / "model_probe.json"
    settings = ModelSettings.from_environment()
    started = time.monotonic()
    try:
        payload = complete_json(
            "你是中文 JSON 连通性探针。只输出指定 JSON。",
            "只输出一个合法 JSON object，字段必须且只能是 `状态`、`协议`，值分别为 `可用`、`json_object`。不得输出空对象、Markdown 或解释。",
            settings,
            attempts=1,
            max_tokens=64,
        )
        if payload != {"状态": "可用", "协议": "json_object"}:
            raise RuntimeError("模型探针返回了非预期 JSON 契约")
        write_json(target, {
            "schema_version": "1.0",
            "author_id": author_id,
            "status": "passed",
            "duration_ms": round((time.monotonic() - started) * 1000),
            "model": settings.model,
            "base_url": settings.base_url,
            "response_keys": sorted(payload),
        })
    except Exception as exc:
        write_json(target, {
            "schema_version": "1.0",
            "author_id": author_id,
            "status": "failed",
            "duration_ms": round((time.monotonic() - started) * 1000),
            "model": settings.model,
            "base_url": settings.base_url,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        })
        raise
    return target
