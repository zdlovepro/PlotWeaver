from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Sequence


_STAGE_FIELD_CANDIDATES: Sequence[str] = (
    "progression_stages",
    "power_stages",
    "cultivation_realms",
    "realms",
    "levels",
)

_DIRECT_STAGE_FIELDS: Sequence[str] = (
    "stage",
    "power_stage",
    "power_state",
    "realm",
)

_NUMERIC_STAGE_FIELDS: Sequence[str] = (
    "stage_index",
    "progression_stage_index",
)

_LEGACY_NUMERIC_STAGE_FIELDS: Sequence[str] = (
    "realm_level",
    "level",
    "rank",
)

_TEXT_STAGE_FIELDS: Sequence[str] = (
    "logic_card.power_state",
    "original_summary",
    "adapted_summary",
    "summary",
    "event_plan",
    "state_updates",
)

_FALLBACK_STAGES: Sequence[tuple[str, int]] = (
    ("初始阶段", 1),
    ("低阶阶段", 10),
    ("中阶阶段", 20),
    ("高阶阶段", 30),
    ("顶阶阶段", 40),
    ("超凡阶段", 50),
)


@dataclass
class ProgressionStage:
    name: str
    stage_index: int
    aliases: List[str] = field(default_factory=list)
    category: str = ""
    description: str = ""


@dataclass
class StoryState:
    protagonist_stage: str = ""
    protagonist_stage_index: int = -1
    protagonist_identity: str = ""
    location: str = ""
    resources: set[str] = field(default_factory=set)
    open_hooks: set[str] = field(default_factory=set)
    closed_hooks: set[str] = field(default_factory=set)
    known_characters: set[str] = field(default_factory=set)


@dataclass
class StateIssue:
    severity: str
    issue_type: str
    message: str
    node_id: str = ""
    suggested_action: str = ""


def get_obj_field(obj: Any, field_name: str, default: Any = None) -> Any:
    if obj is None or not field_name:
        return default

    current = obj
    for part in field_name.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
            continue
        if isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if index < 0 or index >= len(current):
                return default
            current = current[index]
            continue
        if hasattr(current, part):
            current = getattr(current, part)
            continue
        return default
    return current


def build_progression_stages(world: dict[str, Any] | object | None = None) -> List[ProgressionStage]:
    payloads = _extract_stage_payloads(world)
    if not payloads:
        return _build_fallback_stages()

    stages: List[ProgressionStage] = []
    for index, payload in enumerate(payloads, start=1):
        stage = _coerce_progression_stage(payload, fallback_index=index)
        if stage:
            stages.append(stage)

    if not stages:
        return _build_fallback_stages()

    stages.sort(key=lambda item: (item.stage_index, item.name))
    return stages


def make_stage_lookup(stages: List[ProgressionStage]) -> Dict[str, ProgressionStage]:
    lookup: Dict[str, ProgressionStage] = {}
    for stage in stages:
        for alias in _dedupe_texts([stage.name] + list(stage.aliases)):
            for key in _lookup_keys_for_alias(alias):
                if key and key not in lookup:
                    lookup[key] = stage
    return lookup


def normalize_stage_name(text: str, stages: List[ProgressionStage]) -> str:
    matches = _match_stages_in_text(text, stages)
    if not matches:
        return ""
    matches.sort(key=lambda item: (len(item[1]), item[0].stage_index, len(item[0].name)), reverse=True)
    return matches[0][0].name


def stage_value(stage_name: str, stages: List[ProgressionStage]) -> int:
    if not stage_name:
        return -1

    lookup = make_stage_lookup(stages)
    for key in _lookup_keys_for_alias(stage_name):
        stage = lookup.get(key)
        if stage:
            return stage.stage_index
    return -1


def extract_max_stage_from_text(text: str, stages: List[ProgressionStage]) -> str:
    matches = _match_stages_in_text(text, stages)
    if not matches:
        return ""
    matches.sort(key=lambda item: (item[0].stage_index, len(item[1]), len(item[0].name)), reverse=True)
    return matches[0][0].name


def infer_stage_from_node(node: Any, stages: List[ProgressionStage]) -> str:
    if not stages or node is None:
        return ""

    for field_name in _DIRECT_STAGE_FIELDS:
        value = get_obj_field(node, field_name)
        inferred = _stage_from_value(value, stages, prefer_max=False)
        if inferred:
            return inferred

    for field_name in _NUMERIC_STAGE_FIELDS:
        numeric_value = _parse_int(get_obj_field(node, field_name))
        inferred = _stage_from_numeric_hint(numeric_value, stages)
        if inferred:
            return inferred

    for field_name in _TEXT_STAGE_FIELDS:
        value = get_obj_field(node, field_name)
        inferred = _stage_from_value(value, stages, prefer_max=True)
        if inferred:
            return inferred

    # A legacy realm_level is often a volume or display rank. Only map it when
    # the producer explicitly declares that it equals the world's stage_index.
    if _legacy_numeric_fields_are_stage_indexes(node):
        for field_name in _LEGACY_NUMERIC_STAGE_FIELDS:
            numeric_value = _parse_int(get_obj_field(node, field_name))
            inferred = _stage_from_numeric_hint(numeric_value, stages)
            if inferred:
                return inferred

    return ""


def _legacy_numeric_fields_are_stage_indexes(node: Any) -> bool:
    return bool(
        get_obj_field(node, "realm_level_is_stage_index", False)
        or get_obj_field(node, "metadata.realm_level_is_stage_index", False)
        or get_obj_field(node, "metadata.numeric_stage_fields_confirmed", False)
    )


def _extract_stage_payloads(world: dict[str, Any] | object | None) -> List[Any]:
    if world is None:
        return []

    if isinstance(world, list):
        return list(world)

    for field_name in _STAGE_FIELD_CANDIDATES:
        value = get_obj_field(world, field_name)
        if isinstance(value, list) and value:
            return list(value)
    return []


def _coerce_progression_stage(payload: Any, fallback_index: int) -> ProgressionStage | None:
    if isinstance(payload, str):
        name = payload.strip()
        if not name:
            return None
        return ProgressionStage(name=name, stage_index=fallback_index, aliases=[name])

    name = _first_non_empty(
        get_obj_field(payload, "name"),
        get_obj_field(payload, "title"),
        get_obj_field(payload, "label"),
        get_obj_field(payload, "stage"),
    )
    if not name:
        return None

    index = (
        _parse_int(get_obj_field(payload, "stage_index"))
        or _parse_int(get_obj_field(payload, "level"))
        or _parse_int(get_obj_field(payload, "rank"))
        or fallback_index
    )
    category = _first_non_empty(
        get_obj_field(payload, "category"),
        get_obj_field(payload, "type"),
        get_obj_field(payload, "system"),
    )
    aliases = _dedupe_texts(
        [name]
        + _flatten_texts(get_obj_field(payload, "aliases", []))
        + _flatten_texts(get_obj_field(payload, "keywords", []))
    )
    description_parts = _flatten_texts(
        [
            get_obj_field(payload, "description"),
            get_obj_field(payload, "breakthrough_condition"),
            get_obj_field(payload, "special_abilities", []),
            get_obj_field(payload, "abilities", []),
        ]
    )
    description = "；".join(_dedupe_texts(description_parts))
    return ProgressionStage(
        name=name,
        stage_index=index,
        aliases=aliases or [name],
        category=category,
        description=description,
    )


def _build_fallback_stages() -> List[ProgressionStage]:
    return [
        ProgressionStage(name=name, stage_index=stage_index, aliases=[name], description="Generic fallback stage.")
        for name, stage_index in _FALLBACK_STAGES
    ]


def _stage_from_value(value: Any, stages: List[ProgressionStage], prefer_max: bool) -> str:
    if value is None:
        return ""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _stage_from_numeric_hint(int(value), stages)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        numeric = _parse_int(text)
        if numeric is not None and text.isdigit():
            numeric_stage = _stage_from_numeric_hint(numeric, stages)
            if numeric_stage:
                return numeric_stage
        if prefer_max:
            return extract_max_stage_from_text(text, stages)
        return normalize_stage_name(text, stages)

    text = " ".join(_flatten_texts(value))
    if not text:
        return ""
    if prefer_max:
        return extract_max_stage_from_text(text, stages)
    return normalize_stage_name(text, stages)


def _stage_from_numeric_hint(stage_index: int | None, stages: List[ProgressionStage]) -> str:
    if stage_index is None:
        return ""
    for stage in stages:
        if stage.stage_index == stage_index:
            return stage.name
    return ""


def _match_stages_in_text(text: str, stages: List[ProgressionStage]) -> List[tuple[ProgressionStage, str]]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return []

    normalized_text = _normalize_lookup_key(raw_text)
    matches: List[tuple[ProgressionStage, str]] = []
    seen_names: set[str] = set()
    for stage in stages:
        best_alias = ""
        for alias in _dedupe_texts([stage.name] + list(stage.aliases)):
            if _alias_in_text(alias, raw_text, normalized_text) and len(alias) > len(best_alias):
                best_alias = alias
        if best_alias and stage.name not in seen_names:
            seen_names.add(stage.name)
            matches.append((stage, best_alias))
    return matches


def _alias_in_text(alias: str, raw_text: str, normalized_text: str) -> bool:
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        return False

    if _contains_cjk(clean_alias):
        return clean_alias in raw_text or _normalize_lookup_key(clean_alias) in normalized_text

    if any(not ch.isalnum() for ch in clean_alias) and clean_alias.casefold() in raw_text.casefold():
        return True

    if re.search(rf"(?<![A-Za-z0-9_]){re.escape(clean_alias)}(?![A-Za-z0-9_])", raw_text, flags=re.IGNORECASE):
        return True
    return _normalize_lookup_key(clean_alias) in normalized_text


def _lookup_keys_for_alias(alias: str) -> List[str]:
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        return []
    keys = [clean_alias]
    normalized = _normalize_lookup_key(clean_alias)
    if normalized and normalized not in keys:
        keys.append(normalized)
    return keys


def _normalize_lookup_key(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip()).casefold()


def _flatten_texts(value: Any) -> List[str]:
    flattened: List[str] = []
    if value is None:
        return flattened
    if isinstance(value, str):
        clean = value.strip()
        if clean:
            flattened.append(clean)
        return flattened
    if isinstance(value, dict):
        for item in value.values():
            flattened.extend(_flatten_texts(item))
        return flattened
    if isinstance(value, (list, tuple, set)):
        for item in value:
            flattened.extend(_flatten_texts(item))
        return flattened
    clean = str(value).strip()
    if clean:
        flattened.append(clean)
    return flattened


def _dedupe_texts(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        clean = str(value or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _parse_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    return None


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


__all__ = [
    "ProgressionStage",
    "StoryState",
    "StateIssue",
    "build_progression_stages",
    "make_stage_lookup",
    "normalize_stage_name",
    "stage_value",
    "extract_max_stage_from_text",
    "infer_stage_from_node",
    "get_obj_field",
]
