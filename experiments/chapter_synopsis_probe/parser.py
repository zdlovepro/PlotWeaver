from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .prompts import FIDELITY_DIMENSIONS


STATUS_MAP = {"通过": "passed", "需修订": "needs_revision", "无法判断": "inconclusive"}
COVERAGE_CATEGORIES = {"重大遗漏", "错误合并", "顺序错误"}
GRANULARITY_CATEGORIES = {"过度细节", "过度拆分", "错误合并"}


class ProbeProtocolError(ValueError):
    def __init__(self, errors: list[str] | tuple[str, ...]):
        self.errors = tuple(str(item) for item in errors if str(item).strip())
        super().__init__("；".join(self.errors))


@dataclass(frozen=True)
class ParsedReview:
    status: str
    issues: tuple[dict[str, Any], ...]
    checks: dict[str, str] | None = None


def _exact_keys(payload: dict[str, Any], expected: set[str], label: str) -> list[str]:
    actual = set(payload)
    errors: list[str] = []
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"{label}缺少字段：{','.join(missing)}")
    if extra:
        errors.append(f"{label}包含额外字段：{','.join(extra)}")
    return errors


def parse_extraction(payload: dict[str, Any], *, chapter_id: str) -> dict[str, str]:
    errors = _exact_keys(payload, {"章节ID", "章节梗概"}, "提取返回")
    if payload.get("章节ID") != chapter_id:
        errors.append(f"章节ID必须为{chapter_id}")
    synopsis = str(payload.get("章节梗概", "")).strip()
    if not synopsis:
        errors.append("章节梗概不能为空")
    if errors:
        raise ProbeProtocolError(errors)
    return {"章节ID": chapter_id, "章节梗概": synopsis}


def _parse_issue_rows(
    payload: dict[str, Any], *, allowed_source_ids: set[str], category_field: str,
    allowed_categories: set[str], prefix: str,
) -> tuple[tuple[dict[str, Any], ...], list[str]]:
    errors: list[str] = []
    raw_issues = payload.get("问题")
    if not isinstance(raw_issues, list):
        return (), ["问题必须是数组"]
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected = {"问题ID", category_field, "说明", "相关原文段落ID"}
    for index, raw in enumerate(raw_issues, start=1):
        label = f"问题[{index}]"
        if not isinstance(raw, dict):
            errors.append(f"{label}必须是对象")
            continue
        errors.extend(_exact_keys(raw, expected, label))
        issue_id = str(raw.get("问题ID", "")).strip()
        category = str(raw.get(category_field, "")).strip()
        description = str(raw.get("说明", "")).strip()
        source_ids = raw.get("相关原文段落ID")
        if not issue_id or not issue_id.startswith(prefix):
            errors.append(f"{label}.问题ID必须以{prefix}开头")
        elif issue_id in seen:
            errors.append(f"{label}.问题ID重复：{issue_id}")
        if category not in allowed_categories:
            errors.append(f"{label}.{category_field}不受支持：{category}")
        if not description:
            errors.append(f"{label}.说明不能为空")
        normalized_ids: list[str] = []
        if not isinstance(source_ids, list) or not source_ids:
            errors.append(f"{label}.相关原文段落ID必须是非空数组")
        else:
            for source_id in source_ids:
                value = str(source_id).strip()
                if value not in allowed_source_ids:
                    errors.append(f"{label}引用不存在的原文段落ID：{value}")
                elif value not in normalized_ids:
                    normalized_ids.append(value)
        seen.add(issue_id)
        issues.append({
            "问题ID": issue_id,
            category_field: category,
            "说明": description,
            "相关原文段落ID": normalized_ids,
        })
    return tuple(issues), errors


def _parse_conclusion(payload: dict[str, Any], issues: tuple[dict[str, Any], ...]) -> tuple[str, list[str]]:
    raw = str(payload.get("结论", "")).strip()
    status = STATUS_MAP.get(raw, "")
    errors: list[str] = []
    if not status:
        errors.append(f"结论不受支持：{raw}")
    elif status == "passed" and issues:
        errors.append("结论为通过时问题必须为空")
    elif status == "needs_revision" and not issues:
        errors.append("结论为需修订时问题不能为空")
    return status, errors


def parse_fidelity_review(
    payload: dict[str, Any], *, allowed_source_ids: set[str],
) -> ParsedReview:
    errors = _exact_keys(payload, {"目标ID", "核对", "结论", "问题"}, "忠实性审校返回")
    if payload.get("目标ID") != "CHAPTER_SYNOPSIS":
        errors.append("目标ID必须为CHAPTER_SYNOPSIS")
    raw_checks = payload.get("核对")
    checks: dict[str, str] = {}
    if not isinstance(raw_checks, dict):
        errors.append("核对必须是对象")
    else:
        errors.extend(_exact_keys(raw_checks, set(FIDELITY_DIMENSIONS), "核对"))
        for dimension in FIDELITY_DIMENSIONS:
            value = str(raw_checks.get(dimension, "")).strip()
            if value not in {"一致", "不一致", "无法判断"}:
                errors.append(f"核对.{dimension}不受支持：{value}")
            checks[dimension] = value
    issues, issue_errors = _parse_issue_rows(
        payload,
        allowed_source_ids=allowed_source_ids,
        category_field="维度",
        allowed_categories=set(FIDELITY_DIMENSIONS),
        prefix="F",
    )
    errors.extend(issue_errors)
    issue_dimensions = {str(item.get("维度", "")) for item in issues}
    for dimension, value in checks.items():
        if value == "不一致" and dimension not in issue_dimensions:
            errors.append(f"核对.{dimension}为不一致但没有对应问题")
    for dimension in issue_dimensions:
        if checks.get(dimension) != "不一致":
            errors.append(f"问题维度{dimension}必须在核对中标为不一致")
    status, conclusion_errors = _parse_conclusion(payload, issues)
    errors.extend(conclusion_errors)
    if status == "passed" and any(value != "一致" for value in checks.values()):
        errors.append("结论为通过时六个核对维度必须全部一致")
    if status == "inconclusive" and "无法判断" not in checks.values():
        errors.append("结论为无法判断时至少一个核对维度必须为无法判断")
    if errors:
        raise ProbeProtocolError(errors)
    return ParsedReview(status=status, issues=issues, checks=checks)


def parse_category_review(
    payload: dict[str, Any], *, allowed_source_ids: set[str],
    allowed_categories: set[str], prefix: str, label: str,
) -> ParsedReview:
    errors = _exact_keys(payload, {"目标ID", "结论", "问题"}, f"{label}返回")
    if payload.get("目标ID") != "CHAPTER_SYNOPSIS":
        errors.append("目标ID必须为CHAPTER_SYNOPSIS")
    issues, issue_errors = _parse_issue_rows(
        payload,
        allowed_source_ids=allowed_source_ids,
        category_field="类别",
        allowed_categories=allowed_categories,
        prefix=prefix,
    )
    errors.extend(issue_errors)
    status, conclusion_errors = _parse_conclusion(payload, issues)
    errors.extend(conclusion_errors)
    if errors:
        raise ProbeProtocolError(errors)
    return ParsedReview(status=status, issues=issues)
