from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def safe_json_load(raw: str) -> Any:
    text = (raw or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        return json.loads(text.strip())
    except Exception:
        return {}


def write_json_file(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def read_json_file(path: str | Path) -> Any:
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8"))
