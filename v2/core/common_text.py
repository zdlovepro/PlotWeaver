from __future__ import annotations

from typing import Any, List


def flatten_text_values(values: List[Any]) -> List[str]:
    flattened: List[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            clean = value.strip()
            if clean:
                flattened.append(clean)
            continue
        if isinstance(value, (list, tuple, set)):
            flattened.extend(flatten_text_values(list(value)))
            continue
        if isinstance(value, dict):
            flattened.extend(flatten_text_values(list(value.values())))
            continue
        clean = str(value).strip()
        if clean:
            flattened.append(clean)
    return flattened


def dedupe_text_values(values: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        clean = str(value or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def dedupe_texts(values) -> List[str]:
    return dedupe_text_values([str(value or "") for value in values])


def coerce_text(value: Any) -> str:
    return " / ".join(flatten_text_values([value]))


def normalize_text(value: Any) -> str:
    return coerce_text(value).lower()
