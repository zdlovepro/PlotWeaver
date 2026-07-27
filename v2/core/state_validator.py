from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from pipeline.core.story_state import (
    ProgressionStage,
    StateIssue,
    build_progression_stages,
    normalize_stage_name,
    stage_value,
    extract_max_stage_from_text,
    infer_stage_from_node,
    get_obj_field,
)


_EXPLANATION_KEYWORDS: Sequence[str] = (
    "突破",
    "晋升",
    "传承",
    "仪式",
    "觉醒",
    "进化",
    "晋级材料",
    "试炼奖励",
    "顿悟",
    "血脉觉醒",
)

_DIRECT_WIN_KEYWORDS: Sequence[str] = (
    "击杀",
    "反杀",
    "斩杀",
    "正面击败",
    "碾压",
    "单独击败",
)

_LIMITED_OUTCOME_KEYWORDS: Sequence[str] = (
    "逃脱",
    "拖住",
    "借禁制困住",
    "借外力逼退",
    "被救下",
    "周旋",
    "险些被杀",
)

_IDENTITY_BRIDGE_KEYWORDS: Sequence[str] = (
    "误入",
    "传送",
    "被俘",
    "潜入",
    "任务",
    "追查",
    "考核通过",
    "正式册封",
    "继承仪式",
    "身份暴露",
    "卧底",
    "伪装",
    "冒充",
    "受邀",
    "被任命",
)

_LOW_IDENTITY_KEYWORDS: Sequence[str] = (
    "普通成员",
    "底层成员",
    "学徒",
    "杂役",
    "新人",
    "见习",
    "候补",
    "实习",
    "外门",
    "外围成员",
    "平民",
)

_HIGH_IDENTITY_KEYWORDS: Sequence[str] = (
    "核心成员",
    "传承者",
    "掌权者",
    "圣子",
    "圣女",
    "首领",
    "长老",
    "亲传",
    "继承人",
    "负责人",
    "队长",
    "执事",
    "主祭",
    "统领",
    "议员",
    "高层",
)

_HOSTILE_IDENTITY_KEYWORDS: Sequence[str] = (
    "敌方势力成员",
    "敌对阵营成员",
    "敌方成员",
    "敌营成员",
)

_SAFE_LOCATION_KEYWORDS: Sequence[str] = (
    "宗门",
    "学院",
    "基地",
    "城市",
    "村镇",
    "安全区",
    "驻地",
    "营地",
    "学宫",
    "城池",
    "住处",
    "宅邸",
    "据点",
)

_HIGH_RISK_LOCATION_KEYWORDS: Sequence[str] = (
    "秘境",
    "异界",
    "敌方总部",
    "古遗迹",
    "星域",
    "深渊",
    "禁区",
    "遗迹",
    "地宫",
    "边荒",
    "虚空",
    "禁地",
    "敌营",
    "前线",
    "牢狱",
)

_LOCATION_BRIDGE_KEYWORDS: Sequence[str] = (
    "传送",
    "任务",
    "追杀",
    "误入",
    "开启入口",
    "远征",
    "被卷入",
    "护送",
    "调查",
    "进入",
    "抵达",
    "跃迁",
    "护航",
)


def stage_jump_threshold(stages: List[ProgressionStage]) -> int:
    ordered = _sorted_stages(stages)
    return 2 if len(ordered) <= 6 else 3


def validate_stage_transition(
    prev_stage: str,
    next_stage: str,
    stages: List[ProgressionStage],
    node_id: str = "",
    context_text: str = "",
) -> List[StateIssue]:
    issues: List[StateIssue] = []
    next_canonical = normalize_stage_name(next_stage, stages)
    if not next_canonical:
        issues.append(
            StateIssue(
                severity="minor",
                issue_type="unknown_stage",
                message=f"Unable to recognize next stage from value: {next_stage or '<empty>'}.",
                node_id=node_id,
                suggested_action="补充明确的成长阶段名称或别名。",
            )
        )
        return issues

    prev_canonical = normalize_stage_name(prev_stage, stages)
    if not prev_canonical:
        return issues

    prev_position = _stage_position(prev_canonical, stages)
    next_position = _stage_position(next_canonical, stages)
    if prev_position < 0 or next_position < 0:
        return issues

    if next_position < prev_position:
        severity = _lower_severity("fatal") if _has_any_keyword(context_text, _EXPLANATION_KEYWORDS) else "fatal"
        issues.append(
            StateIssue(
                severity=severity,
                issue_type="stage_regression",
                message=f"Stage regressed from {prev_canonical} to {next_canonical}.",
                node_id=node_id,
                suggested_action="补充阶段回落原因，或修正前后节点的成长状态。",
            )
        )
        return issues

    jump_distance = next_position - prev_position
    if jump_distance > stage_jump_threshold(stages):
        severity = _lower_severity("major") if _has_any_keyword(context_text, _EXPLANATION_KEYWORDS) else "major"
        issues.append(
            StateIssue(
                severity=severity,
                issue_type="stage_jump",
                message=f"Stage jumped too far from {prev_canonical} to {next_canonical}.",
                node_id=node_id,
                suggested_action="增加突破、觉醒、试炼奖励或传承桥接。",
            )
        )
    return issues


def validate_enemy_power(
    summary: str,
    protagonist_stage: str,
    stages: List[ProgressionStage],
    node_id: str = "",
) -> List[StateIssue]:
    text = str(summary or "").strip()
    if not text:
        return []

    protagonist_canonical = normalize_stage_name(protagonist_stage, stages)
    external_max_stage = extract_max_stage_from_text(text, stages)
    if not protagonist_canonical or not external_max_stage:
        return []

    protagonist_position = _stage_position(protagonist_canonical, stages)
    external_position = _stage_position(external_max_stage, stages)
    if protagonist_position < 0 or external_position < 0:
        return []

    if external_position - protagonist_position <= stage_jump_threshold(stages):
        return []

    if _has_any_keyword(text, _LIMITED_OUTCOME_KEYWORDS):
        return [
            StateIssue(
                severity="major",
                issue_type="power_gap_pressure",
                message=(
                    f"Encountered an external stage {external_max_stage} far above protagonist stage "
                    f"{protagonist_canonical}; outcome relies on escape, stalling, or outside help."
                ),
                node_id=node_id,
                suggested_action="补充借力、禁制、救援或拖延机制，让结果更清晰。",
            )
        ]

    if _has_any_keyword(text, _DIRECT_WIN_KEYWORDS):
        return [
            StateIssue(
                severity="fatal",
                issue_type="power_gap_win",
                message=(
                    f"Direct victory language appears against {external_max_stage}, which is far above "
                    f"protagonist stage {protagonist_canonical}."
                ),
                node_id=node_id,
                suggested_action="降低对手阶段，或补充强解释的借力/阵法/规则漏洞机制。",
            )
        ]

    return []


def validate_identity_transition(
    prev_identity: str,
    next_identity: str,
    summary: str,
    node_id: str = "",
) -> List[StateIssue]:
    prev_text = str(prev_identity or "").strip()
    next_text = str(next_identity or "").strip()
    if not prev_text or not next_text or prev_text == next_text:
        return []

    prev_level = _identity_level(prev_text)
    next_level = _identity_level(next_text)
    hostile_jump = _has_any_keyword(next_text, _HOSTILE_IDENTITY_KEYWORDS) and not _has_any_keyword(prev_text, _HOSTILE_IDENTITY_KEYWORDS)
    if (prev_level <= 1 and next_level >= 3) or hostile_jump:
        if not _has_any_keyword(summary, _IDENTITY_BRIDGE_KEYWORDS):
            return [
                StateIssue(
                    severity="major",
                    issue_type="identity_jump",
                    message=f"Identity jumped from {prev_text} to {next_text} without clear bridge.",
                    node_id=node_id,
                    suggested_action="补充潜入、册封、考核、继承或身份暴露等桥接事件。",
                )
            ]
    return []


def validate_location_transition(
    prev_location: str,
    next_location: str,
    summary: str,
    node_id: str = "",
) -> List[StateIssue]:
    prev_text = str(prev_location or "").strip()
    next_text = str(next_location or "").strip()
    if not prev_text or not next_text or prev_text == next_text:
        return []

    prev_level = _location_level(prev_text)
    next_level = _location_level(next_text)
    if prev_level <= 1 and next_level >= 3 and not _has_any_keyword(summary, _LOCATION_BRIDGE_KEYWORDS):
        return [
            StateIssue(
                severity="major",
                issue_type="location_jump",
                message=f"Location jumped from {prev_text} to {next_text} without travel or mission bridge.",
                node_id=node_id,
                suggested_action="补充传送、远征、任务、追杀、护送或入口开启等衔接。",
            )
        ]
    return []


def validate_skeleton_sequence(
    nodes: list,
    world: dict | object | None = None,
) -> List[StateIssue]:
    stages = build_progression_stages(world)
    issues: List[StateIssue] = []
    prev_stage = ""
    prev_identity = ""
    prev_location = ""

    for node in nodes:
        node_id = _node_id(node)
        context_text = _join_texts(
            [
                get_obj_field(node, "original_summary"),
                get_obj_field(node, "adapted_summary"),
                get_obj_field(node, "summary"),
                get_obj_field(node, "conflict_hint"),
                get_obj_field(node, "function_hint"),
                get_obj_field(node, "identity_state"),
                get_obj_field(node, "logic_card"),
            ]
        )
        current_stage = infer_stage_from_node(node, stages)
        protagonist_stage = current_stage or prev_stage
        current_identity = _infer_identity_from_obj(node)
        current_location = _infer_location_from_obj(node)

        if protagonist_stage:
            issues.extend(validate_stage_transition(prev_stage, protagonist_stage, stages, node_id=node_id, context_text=context_text))
            issues.extend(validate_enemy_power(context_text, protagonist_stage, stages, node_id=node_id))
        if current_identity:
            issues.extend(validate_identity_transition(prev_identity, current_identity, context_text, node_id=node_id))
        if current_location:
            issues.extend(validate_location_transition(prev_location, current_location, context_text, node_id=node_id))

        if current_stage:
            prev_stage = current_stage
        if current_identity:
            prev_identity = current_identity
        if current_location:
            prev_location = current_location

    return issues


def validate_reassembled_events(
    events: list,
    world: dict | object | None = None,
) -> List[StateIssue]:
    stages = build_progression_stages(world)
    issues: List[StateIssue] = []
    prev_stage = ""

    for event in events:
        node_id = _node_id(event)
        context_text = _join_texts(
            [
                get_obj_field(event, "adapted_summary"),
                get_obj_field(event, "summary"),
                get_obj_field(event, "event_plan"),
                get_obj_field(event, "state_updates"),
                get_obj_field(event, "active_characters"),
                get_obj_field(event, "logic_notes"),
            ]
        )
        current_stage = infer_stage_from_node(event, stages)
        protagonist_stage = current_stage or prev_stage
        if protagonist_stage:
            issues.extend(validate_stage_transition(prev_stage, protagonist_stage, stages, node_id=node_id, context_text=context_text))
            issues.extend(validate_enemy_power(context_text, protagonist_stage, stages, node_id=node_id))
        if current_stage:
            prev_stage = current_stage

    return issues


def build_forbidden_stage_terms(
    max_allowed_stage: str,
    stages: List[ProgressionStage],
    margin: int = 1,
) -> List[str]:
    canonical = normalize_stage_name(max_allowed_stage, stages)
    if not canonical:
        return []

    ordered = _sorted_stages(stages)
    max_position = _stage_position(canonical, ordered)
    if max_position < 0:
        return []

    cutoff = max_position + max(0, int(margin))
    forbidden: List[str] = []
    for position, stage in enumerate(ordered):
        if position <= cutoff:
            continue
        forbidden.extend([stage.name] + list(stage.aliases))
    return _dedupe_texts(forbidden)


def validate_volume_outline(
    volume: dict,
    world: dict | object | None = None,
) -> List[StateIssue]:
    stages = build_progression_stages(world)
    issues: List[StateIssue] = []
    volume_id = _node_id(volume)
    _min_stage, max_stage = _infer_volume_stage_bounds(volume, stages)
    if not max_stage:
        return issues

    max_position = _stage_position(max_stage, stages)
    forbidden_terms = build_forbidden_stage_terms(max_stage, stages, margin=1)

    chapter_summaries = get_obj_field(volume, "chapter_summaries", [])
    chapter_texts = _listify_text_blocks(chapter_summaries)
    ending_state = _join_texts([get_obj_field(volume, "ending_state")])
    raw_text = _join_texts([get_obj_field(volume, "raw_text")])

    for index, summary in enumerate(chapter_texts, start=1):
        chapter_id = f"{volume_id or 'volume'}_chapter_{index}"
        issues.extend(validate_enemy_power(summary, max_stage, stages, node_id=chapter_id))
        detected_max = extract_max_stage_from_text(summary, stages)
        if detected_max:
            detected_position = _stage_position(detected_max, stages)
            if detected_position > max_position + 1:
                severity = "fatal" if detected_position - max_position > stage_jump_threshold(stages) else "major"
                issues.append(
                    StateIssue(
                        severity=severity,
                        issue_type="volume_stage_overflow",
                        message=f"Chapter summary exceeds volume stage ceiling: {detected_max} > {max_stage}.",
                        node_id=chapter_id,
                        suggested_action="下调该章节出现的阶段，或上调本卷阶段范围并补足铺垫。",
                    )
                )
        if forbidden_terms and any(term and term in summary for term in forbidden_terms):
            issues.append(
                StateIssue(
                    severity="major",
                    issue_type="forbidden_stage_term",
                    message=f"Chapter summary contains stage terms beyond allowed range for this volume.",
                    node_id=chapter_id,
                    suggested_action="检查本卷阶段上限与章节摘要是否一致。",
                )
            )

    if ending_state:
        issues.extend(validate_enemy_power(ending_state, max_stage, stages, node_id=volume_id))
        detected_max = extract_max_stage_from_text(ending_state, stages)
        if detected_max:
            detected_position = _stage_position(detected_max, stages)
            if detected_position > max_position + 1:
                severity = "fatal" if detected_position - max_position > stage_jump_threshold(stages) else "major"
                issues.append(
                    StateIssue(
                        severity=severity,
                        issue_type="ending_stage_overflow",
                        message=f"Ending state exceeds volume stage ceiling: {detected_max} > {max_stage}.",
                        node_id=volume_id,
                        suggested_action="调整本卷结尾状态，或明确本卷阶段范围的上升过程。",
                    )
                )
        if forbidden_terms and any(term and term in ending_state for term in forbidden_terms):
            issues.append(
                StateIssue(
                    severity="major",
                    issue_type="forbidden_stage_term",
                    message="Ending state contains stage terms beyond allowed range for this volume.",
                    node_id=volume_id,
                    suggested_action="检查卷末状态是否越过本卷预设上限。",
                )
            )

    if raw_text:
        detected_max = extract_max_stage_from_text(raw_text, stages)
        if detected_max:
            detected_position = _stage_position(detected_max, stages)
            if detected_position > max_position + stage_jump_threshold(stages):
                issues.append(
                    StateIssue(
                        severity="major",
                        issue_type="volume_raw_stage_overflow",
                        message=f"Raw volume text references stage {detected_max}, which is well above allowed ceiling {max_stage}.",
                        node_id=volume_id,
                        suggested_action="检查原始卷设定文本是否混入更高阶段信息。",
                    )
                )

    return issues


def _sorted_stages(stages: List[ProgressionStage]) -> List[ProgressionStage]:
    return sorted(stages, key=lambda item: (item.stage_index, item.name))


def _stage_position(stage_name: str, stages: List[ProgressionStage]) -> int:
    canonical = normalize_stage_name(stage_name, stages)
    if not canonical:
        return -1
    for index, stage in enumerate(_sorted_stages(stages)):
        if stage.name == canonical:
            return index
    return -1


def _has_any_keyword(text: str, keywords: Sequence[str]) -> bool:
    haystack = str(text or "")
    return any(keyword in haystack for keyword in keywords)


def _lower_severity(severity: str) -> str:
    order = {"fatal": "major", "major": "minor", "minor": "minor"}
    return order.get(severity, severity)


def _identity_level(identity: str) -> int:
    text = str(identity or "")
    if _has_any_keyword(text, _HOSTILE_IDENTITY_KEYWORDS):
        return 3
    if _has_any_keyword(text, _HIGH_IDENTITY_KEYWORDS):
        return 3
    if _has_any_keyword(text, _LOW_IDENTITY_KEYWORDS):
        return 1
    return 2 if text else 0


def _location_level(location: str) -> int:
    text = str(location or "")
    if _has_any_keyword(text, _HIGH_RISK_LOCATION_KEYWORDS):
        return 3
    if _has_any_keyword(text, _SAFE_LOCATION_KEYWORDS):
        return 1
    return 2 if text else 0


def _infer_identity_from_obj(obj: Any) -> str:
    return _first_non_empty(
        get_obj_field(obj, "identity_state"),
        get_obj_field(obj, "protagonist_identity"),
        get_obj_field(obj, "identity"),
        get_obj_field(obj, "logic_card.identity_state"),
        get_obj_field(obj, "event_plan.identity_state"),
    )


def _infer_location_from_obj(obj: Any) -> str:
    return _first_non_empty(
        get_obj_field(obj, "location"),
        get_obj_field(obj, "raw_location"),
        get_obj_field(obj, "scene"),
        get_obj_field(obj, "setting"),
        get_obj_field(obj, "logic_card.location"),
        get_obj_field(obj, "logic_card.scene"),
        get_obj_field(obj, "event_plan.location"),
    )


def _node_id(obj: Any) -> str:
    return _first_non_empty(
        get_obj_field(obj, "node_id"),
        get_obj_field(obj, "event_id"),
        get_obj_field(obj, "id"),
        get_obj_field(obj, "arc_name"),
        "unknown_node",
    )


def _join_texts(values: Iterable[Any]) -> str:
    parts: List[str] = []
    for value in values:
        parts.extend(_flatten_texts(value))
    return " ".join(_dedupe_texts(parts))


def _listify_text_blocks(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        return [text for text in (_join_texts([item]) for item in value.values()) if text]
    if isinstance(value, (list, tuple, set)):
        return [text for text in (_join_texts([item]) for item in value) if text]
    text = str(value).strip()
    return [text] if text else []


def _flatten_texts(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        clean = value.strip()
        return [clean] if clean else []
    if isinstance(value, dict):
        parts: List[str] = []
        for item in value.values():
            parts.extend(_flatten_texts(item))
        return parts
    if isinstance(value, (list, tuple, set)):
        parts: List[str] = []
        for item in value:
            parts.extend(_flatten_texts(item))
        return parts
    clean = str(value).strip()
    return [clean] if clean else []


def _dedupe_texts(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        clean = str(value or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _infer_volume_stage_bounds(volume: Any, stages: List[ProgressionStage]) -> Tuple[str, str]:
    for field_name in ("realm_range", "stage_range", "power_range"):
        lower, upper = _parse_stage_range(get_obj_field(volume, field_name), stages)
        if lower or upper:
            return lower, upper

    ending_stage = extract_max_stage_from_text(_join_texts([get_obj_field(volume, "ending_state")]), stages)
    if ending_stage:
        return "", ending_stage

    chapter_stage = extract_max_stage_from_text(_join_texts([get_obj_field(volume, "chapter_summaries")]), stages)
    if chapter_stage:
        return "", chapter_stage

    raw_stage = extract_max_stage_from_text(_join_texts([get_obj_field(volume, "raw_text")]), stages)
    return "", raw_stage


def _parse_stage_range(value: Any, stages: List[ProgressionStage]) -> Tuple[str, str]:
    if value is None:
        return "", ""

    if isinstance(value, dict):
        lower = _coerce_stage_from_any(
            get_obj_field(value, "min") or get_obj_field(value, "start") or get_obj_field(value, "lower"),
            stages,
        )
        upper = _coerce_stage_from_any(
            get_obj_field(value, "max") or get_obj_field(value, "end") or get_obj_field(value, "upper"),
            stages,
        )
        if lower or upper:
            return lower, upper

    if isinstance(value, (list, tuple)):
        matched = [_coerce_stage_from_any(item, stages) for item in value]
        matched = [item for item in matched if item]
        if matched:
            ordered = sorted(matched, key=lambda item: _stage_position(item, stages))
            return ordered[0], ordered[-1]

    text = _join_texts([value])
    if not text:
        return "", ""
    stage_names = _extract_stage_names_from_text(text, stages)
    if not stage_names:
        return "", ""
    ordered = sorted(stage_names, key=lambda item: _stage_position(item, stages))
    return ordered[0], ordered[-1]


def _coerce_stage_from_any(value: Any, stages: List[ProgressionStage]) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for stage in stages:
            if stage.stage_index == int(value):
                return stage.name
        return ""
    text = _join_texts([value])
    return normalize_stage_name(text, stages) or extract_max_stage_from_text(text, stages)


def _extract_stage_names_from_text(text: str, stages: List[ProgressionStage]) -> List[str]:
    matches: List[str] = []
    raw_text = str(text or "")
    normalized_text = _normalize_text(raw_text)
    for stage in stages:
        aliases = [stage.name] + list(stage.aliases)
        for alias in _dedupe_texts(aliases):
            if _alias_in_text(alias, raw_text, normalized_text):
                matches.append(stage.name)
                break
    return _dedupe_texts(matches)


def _alias_in_text(alias: str, raw_text: str, normalized_text: str) -> bool:
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        return False
    if _contains_cjk(clean_alias):
        return clean_alias in raw_text or _normalize_text(clean_alias) in normalized_text
    if any(not ch.isalnum() for ch in clean_alias) and clean_alias.casefold() in raw_text.casefold():
        return True
    if re.search(rf"(?<![A-Za-z0-9_]){re.escape(clean_alias)}(?![A-Za-z0-9_])", raw_text, flags=re.IGNORECASE):
        return True
    return _normalize_text(clean_alias) in normalized_text


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip()).casefold()


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)
