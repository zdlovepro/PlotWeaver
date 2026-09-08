from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

REVIEW_PROMPT_REVISION = "11.2-local-risk-contract-normalization"


class ReviewContractError(ValueError):
    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(dict.fromkeys(str(item) for item in errors if str(item).strip()))
        super().__init__("；".join(self.errors))


@dataclass(frozen=True)
class GroundedSemanticIssue:
    issue_id: str
    target_id: str
    severity: Literal["warning", "major_candidate"]
    issue_type: str
    candidate_fragment: str
    source_id: str
    source_quote: str
    explanation: str

    def to_dict(self) -> dict[str, str]:
        return {"问题ID": self.issue_id, "目标ID": self.target_id,
                "严重性": self.severity, "类型": self.issue_type,
                "候选片段": self.candidate_fragment, "证据ID": self.source_id,
                "证据片段": self.source_quote, "说明": self.explanation}


@dataclass(frozen=True)
class LocalRiskAudit:
    target_id: str
    warnings: tuple[GroundedSemanticIssue, ...]
    major_candidates: tuple[GroundedSemanticIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.major_candidates

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


@dataclass(frozen=True)
class LocalRiskParseResult:
    audits: tuple[LocalRiskAudit, ...]
    contract_normalizations: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class IssueAdjudication:
    issue_id: str
    confirmed: bool
    reason: str


@dataclass(frozen=True)
class RepairVerification:
    resolved_issue_ids: tuple[str, ...]
    unresolved_issue_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.unresolved_issue_ids


@dataclass(frozen=True)
class SegmentationReviewResult:
    passed: bool
    issues: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class BoundaryReviewResult:
    major_issues: tuple[GroundedSemanticIssue, ...]
    warnings: tuple[GroundedSemanticIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.major_issues

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


@dataclass(frozen=True)
class ChapterFidelityResult:
    major_issues: tuple[GroundedSemanticIssue, ...]
    warnings: tuple[GroundedSemanticIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.major_issues

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


@dataclass(frozen=True)
class ChapterCoverageIssue:
    issue_id: str
    local_ids: tuple[str, ...]
    missing_content: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {"问题ID": self.issue_id, "涉及局部梗概ID": list(self.local_ids),
                "缺失内容": self.missing_content, "说明": self.explanation}


@dataclass(frozen=True)
class ChapterCoverageResult:
    major_omissions: tuple[ChapterCoverageIssue, ...]
    optional_additions: tuple[ChapterCoverageIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.major_omissions

    @property
    def has_warnings(self) -> bool:
        return bool(self.optional_additions)


def _obj(value: Any, path: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{path}必须是对象")
        return {}
    return value


def _arr(value: Any, path: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(f"{path}必须是数组")
        return []
    return value


def _exact(value: dict[str, Any], keys: set[str], path: str, errors: list[str]) -> None:
    if set(value) != keys:
        errors.append(f"{path}字段必须恰好为{sorted(keys)}")


def _fields(value: dict[str, Any], required: set[str], path: str, errors: list[str],
            optional: set[str] | None = None) -> None:
    optional = optional or set()
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing:
        errors.append(f"{path}缺少字段{sorted(missing)}")
    if extra:
        errors.append(f"{path}包含未知字段{sorted(extra)}")


def _text(value: Any, path: str, errors: list[str], required: bool = True) -> str:
    if not isinstance(value, str):
        errors.append(f"{path}必须是字符串")
        return ""
    result = value.strip()
    if required and not result:
        errors.append(f"{path}不得为空")
    return result


def parse_segmentation_review(payload: dict[str, Any], *, expected_ids: list[str]) -> SegmentationReviewResult:
    errors: list[str] = []
    root = _obj(payload, "顶层", errors)
    _fields(root, {"分区审校"}, "顶层", errors, {"结论"})
    rows = _arr(root.get("分区审校"), "分区审校", errors)
    if len(rows) != len(expected_ids):
        errors.append("分区审校数量必须与分区数量一致")
    issues: list[dict[str, str]] = []
    for index, raw in enumerate(rows):
        item = _obj(raw, f"分区审校[{index}]", errors)
        _exact(item, {"块ID", "粒度", "边界", "说明"}, f"分区审校[{index}]", errors)
        block_id = _text(item.get("块ID"), "块ID", errors)
        if index >= len(expected_ids) or block_id != expected_ids[index]:
            errors.append(f"分区审校[{index}]块ID或顺序错误")
        granularity = _text(item.get("粒度"), "粒度", errors)
        boundary = _text(item.get("边界"), "边界", errors)
        if granularity not in {"合适", "过大", "过碎"}:
            errors.append(f"{block_id}粒度取值无效")
        if boundary not in {"正确", "错误"}:
            errors.append(f"{block_id}边界取值无效")
        explanation = _text(item.get("说明"), "说明", errors)
        if granularity == "过大" or boundary == "错误":
            issues.append({"块ID": block_id, "粒度": granularity, "边界": boundary, "说明": explanation})
    if errors:
        raise ReviewContractError(errors)
    return SegmentationReviewResult(not issues, tuple(issues))


def parse_segmentation_recheck(payload: dict[str, Any], *,
                               allowed_split_ids: dict[str, tuple[str, ...]]) -> SegmentationReviewResult:
    errors: list[str] = []
    root = _obj(payload, "顶层", errors)
    _fields(root, {"过大分区"}, "顶层", errors, {"结论"})
    rows = _arr(root.get("过大分区"), "过大分区", errors)
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        item = _obj(raw, f"过大分区[{index}]", errors)
        _exact(item, {"块ID", "拆分起点段落ID", "说明"}, f"过大分区[{index}]", errors)
        block_id = _text(item.get("块ID"), f"过大分区[{index}].块ID", errors)
        if block_id not in allowed_split_ids or block_id in seen:
            errors.append(f"过大分区块ID未知或重复：{block_id}")
        seen.add(block_id)
        split_ids = tuple(_text(value, f"{block_id}.拆分起点", errors)
                          for value in _arr(item.get("拆分起点段落ID"), f"{block_id}.拆分起点段落ID", errors))
        allowed = allowed_split_ids.get(block_id, ())
        if not split_ids or len(split_ids) != len(set(split_ids)):
            errors.append(f"{block_id}.拆分起点必须非空且不重复")
        if any(value not in allowed for value in split_ids):
            errors.append(f"{block_id}.拆分起点不在分区内部")
        known = [value for value in split_ids if value in allowed]
        if known != sorted(known, key=allowed.index):
            errors.append(f"{block_id}.拆分起点必须按原文顺序")
        issues.append({"块ID": block_id, "拆分起点段落ID": list(split_ids),
                       "说明": _text(item.get("说明"), f"{block_id}.说明", errors)})
    if errors:
        raise ReviewContractError(errors)
    return SegmentationReviewResult(not issues, tuple(issues))


def _parse_grounded_issue(raw: Any, path: str, *, issue_id: str,
                          target_id: str, severity: str, errors: list[str]) -> GroundedSemanticIssue:
    item = _obj(raw, path, errors)
    _fields(item, {"类型", "候选片段", "证据段落ID", "原文片段", "说明"},
            path, errors, {"问题ID"})
    fragment = _text(item.get("候选片段"), f"{path}.候选片段", errors)
    return GroundedSemanticIssue(
        issue_id, target_id,
        "warning" if severity == "warning" else "major_candidate",
        _text(item.get("类型"), f"{path}.类型", errors), fragment,
        _text(item.get("证据段落ID"), f"{path}.证据段落ID", errors),
        _text(item.get("原文片段"), f"{path}.原文片段", errors),
        _text(item.get("说明"), f"{path}.说明", errors))


def _local_risk_issue_rows(item: dict[str, Any], key: str, *, target_id: str,
                           normalizations: list[dict[str, str]],
                           errors: list[str]) -> list[Any]:
    """Normalize only representations whose review meaning is unambiguous."""
    if key not in item or item.get(key) is None:
        normalizations.append({
            "目标ID": target_id,
            "字段": key,
            "处理": "补为空数组",
            "说明": f"{key}缺失或为null；该目标未返回此类问题，程序按空数组接纳",
        })
        return []
    value = item[key]
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        normalizations.append({
            "目标ID": target_id,
            "字段": key,
            "处理": "单个对象包装为数组",
            "说明": f"{key}返回单个问题对象；程序保留该问题并包装为单元素数组",
        })
        return [value]
    errors.append(f"{target_id}.{key}必须是数组、单个问题对象、null或缺失")
    return []


def parse_local_risk(payload: dict[str, Any], *, candidates: dict[str, str]) -> LocalRiskParseResult:
    errors: list[str] = []
    normalizations: list[dict[str, str]] = []
    root = _obj(payload, "顶层", errors)
    _fields(root, {"审查结果"}, "顶层", errors, {"结论"})
    if "结论" in root:
        normalizations.append({
            "目标ID": "",
            "字段": "结论",
            "处理": "忽略派生字段",
            "说明": "总判断由程序根据问题数组计算",
        })
    rows = _arr(root.get("审查结果"), "审查结果", errors)
    expected = list(candidates)
    if len(rows) != len(expected):
        errors.append("审查结果数量必须与目标数量一致")

    rows_by_target: dict[str, tuple[int, dict[str, Any]]] = {}
    actual_order: list[str] = []
    for index, raw in enumerate(rows):
        item = _obj(raw, f"审查结果[{index}]", errors)
        _fields(item, {"目标ID"}, f"审查结果[{index}]", errors,
                {"警告", "重大疑似错误", "结论"})
        target_id = _text(item.get("目标ID"), f"审查结果[{index}].目标ID", errors)
        if not target_id:
            continue
        actual_order.append(target_id)
        if target_id not in candidates:
            errors.append(f"审查结果[{index}]目标ID未知：{target_id}")
            continue
        if target_id in rows_by_target:
            errors.append(f"审查结果目标ID重复：{target_id}")
            continue
        rows_by_target[target_id] = (index, item)

    missing = [target_id for target_id in expected if target_id not in rows_by_target]
    if missing:
        errors.append(f"审查结果缺少目标ID：{missing}")
    if not missing and not errors and actual_order != expected:
        normalizations.append({
            "目标ID": "",
            "字段": "审查结果",
            "处理": "按目标ID恢复顺序",
            "说明": "模型返回顺序与请求不同；程序按明确的目标ID重新排序",
        })

    results: list[LocalRiskAudit] = []
    for target_id in expected:
        located = rows_by_target.get(target_id)
        if located is None:
            continue
        index, item = located
        if "结论" in item:
            normalizations.append({
                "目标ID": target_id,
                "字段": "结论",
                "处理": "忽略派生字段",
                "说明": "该目标是否通过由程序根据两个问题数组计算",
            })
        warning_rows = _local_risk_issue_rows(
            item, "警告", target_id=target_id,
            normalizations=normalizations, errors=errors)
        major_rows = _local_risk_issue_rows(
            item, "重大疑似错误", target_id=target_id,
            normalizations=normalizations, errors=errors)
        warnings = tuple(_parse_grounded_issue(value, f"{target_id}.警告[{i}]",
            issue_id=f"{target_id}-W{i + 1:03d}", target_id=target_id,
            severity="warning", errors=errors)
            for i, value in enumerate(warning_rows))
        majors = tuple(_parse_grounded_issue(value, f"{target_id}.重大疑似错误[{i}]",
            issue_id=f"{target_id}-M{i + 1:03d}", target_id=target_id,
            severity="major", errors=errors)
            for i, value in enumerate(major_rows))
        results.append(LocalRiskAudit(target_id, warnings, majors))
    if errors:
        raise ReviewContractError(errors)
    return LocalRiskParseResult(tuple(results), tuple(normalizations))


def validate_grounded_issues(issues: Iterable[GroundedSemanticIssue], *,
                             sources_by_target: dict[str, dict[str, str]],
                             candidates_by_target: dict[str, str]) -> tuple[tuple[GroundedSemanticIssue, ...], tuple[dict[str, str], ...]]:
    accepted: list[GroundedSemanticIssue] = []
    rejected: list[dict[str, str]] = []
    for issue in issues:
        rows = sources_by_target.get(issue.target_id, {})
        reason = ""
        candidate = candidates_by_target.get(issue.target_id, "")
        if not issue.candidate_fragment or issue.candidate_fragment not in candidate:
            reason = "候选片段不能逐字定位到目标候选"
        elif issue.source_id not in rows:
            reason = "证据ID不属于目标分区"
        elif issue.source_quote not in rows[issue.source_id]:
            reason = "证据片段不能逐字定位到指定段落"
        if reason:
            rejected.append({**issue.to_dict(), "code": "REVIEW_EVIDENCE_INVALID", "驳回原因": reason})
        else:
            accepted.append(issue)
    return tuple(accepted), tuple(rejected)


def parse_issue_adjudication(payload: dict[str, Any], *, expected_ids: list[str]) -> tuple[IssueAdjudication, ...]:
    errors: list[str] = []
    root = _obj(payload, "顶层", errors)
    _exact(root, {"问题裁决"}, "顶层", errors)
    rows = _arr(root.get("问题裁决"), "问题裁决", errors)
    if len(rows) != len(expected_ids):
        errors.append("问题裁决数量不匹配")
    results = []
    for index, raw in enumerate(rows):
        item = _obj(raw, f"问题裁决[{index}]", errors)
        _exact(item, {"问题ID", "结论", "理由"}, f"问题裁决[{index}]", errors)
        issue_id = _text(item.get("问题ID"), f"问题裁决[{index}].问题ID", errors)
        if index >= len(expected_ids) or issue_id != expected_ids[index]:
            errors.append(f"问题裁决[{index}]问题ID或顺序错误")
        verdict = _text(item.get("结论"), f"{issue_id}.结论", errors)
        if verdict not in {"成立", "不成立"}:
            errors.append(f"{issue_id}.结论无效")
        results.append(IssueAdjudication(issue_id, verdict == "成立",
                                         _text(item.get("理由"), f"{issue_id}.理由", errors)))
    if errors:
        raise ReviewContractError(errors)
    return tuple(results)


def parse_repair_verification(payload: dict[str, Any], *, expected_ids: list[str]) -> RepairVerification:
    errors: list[str] = []
    root = _obj(payload, "顶层", errors)
    _exact(root, {"修复验证"}, "顶层", errors)
    rows = _arr(root.get("修复验证"), "修复验证", errors)
    if len(rows) != len(expected_ids):
        errors.append("修复验证数量不匹配")
    resolved: list[str] = []
    unresolved: list[str] = []
    for index, raw in enumerate(rows):
        item = _obj(raw, f"修复验证[{index}]", errors)
        _exact(item, {"问题ID", "结论", "理由"}, f"修复验证[{index}]", errors)
        issue_id = _text(item.get("问题ID"), f"修复验证[{index}].问题ID", errors)
        if index >= len(expected_ids) or issue_id != expected_ids[index]:
            errors.append(f"修复验证[{index}]问题ID或顺序错误")
        verdict = _text(item.get("结论"), f"{issue_id}.结论", errors)
        if verdict not in {"已解决", "未解决"}:
            errors.append(f"{issue_id}.结论无效")
        _text(item.get("理由"), f"{issue_id}.理由", errors)
        (resolved if verdict == "已解决" else unresolved).append(issue_id)
    if errors:
        raise ReviewContractError(errors)
    return RepairVerification(tuple(resolved), tuple(unresolved))


def parse_boundary_review(payload: dict[str, Any], *, candidate: str) -> BoundaryReviewResult:
    errors: list[str] = []
    root = _obj(payload, "顶层", errors)
    _fields(root, {"重大错误", "警告"}, "顶层", errors, {"结论", "说明"})
    warnings = tuple(_parse_grounded_issue(value, f"警告[{index}]",
        issue_id=f"BOUNDARY-W{index + 1:03d}", target_id="boundary",
        severity="warning", errors=errors)
        for index, value in enumerate(_arr(root.get("警告"), "警告", errors)))
    majors = tuple(_parse_grounded_issue(value, f"重大错误[{index}]",
        issue_id=f"BOUNDARY-M{index + 1:03d}", target_id="boundary",
        severity="major", errors=errors)
        for index, value in enumerate(_arr(root.get("重大错误"), "重大错误", errors)))
    if errors:
        raise ReviewContractError(errors)
    return BoundaryReviewResult(majors, warnings)


def parse_chapter_fidelity(payload: dict[str, Any], *, summary: str,
                           local_rows: list[dict[str, Any]]) -> ChapterFidelityResult:
    errors: list[str] = []
    root = _obj(payload, "顶层", errors)
    _fields(root, {"重大错误", "警告"}, "顶层", errors, {"结论"})

    def parse_rows(key: str, severity: str) -> tuple[GroundedSemanticIssue, ...]:
        results = []
        for index, raw in enumerate(_arr(root.get(key), key, errors)):
            item = _obj(raw, f"{key}[{index}]", errors)
            _fields(item, {"候选片段", "局部梗概ID", "输入片段", "说明"},
                    f"{key}[{index}]", errors, {"问题ID"})
            fragment = _text(item.get("候选片段"), f"{key}[{index}].候选片段", errors)
            source_id = _text(item.get("局部梗概ID"), f"{key}[{index}].局部梗概ID", errors)
            source_quote = _text(item.get("输入片段"), f"{key}[{index}].输入片段", errors)
            results.append(GroundedSemanticIssue(
                f"CS-F-{'W' if severity == 'warning' else 'M'}{index + 1:03d}", "chapter-summary",
                "warning" if severity == "warning" else "major_candidate", "章节事实差异",
                fragment, source_id, source_quote,
                _text(item.get("说明"), f"{key}[{index}].说明", errors)))
        return tuple(results)

    majors = parse_rows("重大错误", "major")
    warnings = parse_rows("警告", "warning")
    if errors:
        raise ReviewContractError(errors)
    return ChapterFidelityResult(majors, warnings)


def parse_chapter_coverage(payload: dict[str, Any], *, local_ids: list[str]) -> ChapterCoverageResult:
    errors: list[str] = []
    root = _obj(payload, "顶层", errors)
    _fields(root, {"重大遗漏", "可选补充"}, "顶层", errors, {"结论"})

    def parse_rows(key: str) -> tuple[ChapterCoverageIssue, ...]:
        results = []
        for index, raw in enumerate(_arr(root.get(key), key, errors)):
            item = _obj(raw, f"{key}[{index}]", errors)
            _fields(item, {"涉及局部梗概ID", "缺失内容", "说明"},
                    f"{key}[{index}]", errors, {"问题ID"})
            ids = tuple(_text(value, f"{key}[{index}].涉及局部梗概ID", errors)
                        for value in _arr(item.get("涉及局部梗概ID"), f"{key}[{index}].涉及局部梗概ID", errors))
            results.append(ChapterCoverageIssue(
                f"CS-C-{'W' if key == '可选补充' else 'M'}{index + 1:03d}", ids,
                _text(item.get("缺失内容"), f"{key}[{index}].缺失内容", errors),
                _text(item.get("说明"), f"{key}[{index}].说明", errors)))
        return tuple(results)

    majors = parse_rows("重大遗漏")
    optional = parse_rows("可选补充")
    if errors:
        raise ReviewContractError(errors)
    return ChapterCoverageResult(majors, optional)


def validate_chapter_coverage_issues(
    issues: Iterable[ChapterCoverageIssue], *, local_ids: set[str],
) -> tuple[tuple[ChapterCoverageIssue, ...], tuple[dict[str, Any], ...]]:
    accepted: list[ChapterCoverageIssue] = []
    rejected: list[dict[str, Any]] = []
    for issue in issues:
        unknown = [value for value in issue.local_ids if value not in local_ids]
        if not issue.local_ids or unknown:
            rejected.append({**issue.to_dict(), "code": "REVIEW_EVIDENCE_INVALID",
                             "驳回原因": f"局部梗概ID为空或未知：{unknown}"})
        else:
            accepted.append(issue)
    return tuple(accepted), tuple(rejected)
