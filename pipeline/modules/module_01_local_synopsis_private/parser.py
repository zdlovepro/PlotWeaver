from __future__ import annotations

from typing import Any, Iterable

from ...contracts import (BoundaryFrame, ChapterDocument, ChapterSummary, ChapterSynopsis,
    LocalPlotSegment, LocalSynopsis)
from .windowing import (ChapterNavigationPlan, LocalRequestBatch, NavigationBlock, SynopsisWindow,
    chapter_paragraph_alias_maps, validate_navigation_plan)


class SynopsisContractError(ValueError):
    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(dict.fromkeys(str(item) for item in errors if str(item).strip()))
        super().__init__("；".join(self.errors))


def _object(value: Any, path: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{path}必须是JSON对象")
        return {}
    return value


def _array(value: Any, path: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(f"{path}必须是数组")
        return []
    return value


def _exact(value: dict[str, Any], keys: set[str], path: str, errors: list[str]) -> None:
    if set(value) != keys:
        errors.append(f"{path}字段必须恰好为{sorted(keys)}，实际为{sorted(value)}")


def _text(value: Any, path: str, errors: list[str], *, required: bool = True) -> str:
    if not isinstance(value, str):
        errors.append(f"{path}必须是字符串")
        return ""
    result = value.strip()
    if required and not result:
        errors.append(f"{path}不得为空")
    return result


def paragraph_alias_maps(window: SynopsisWindow) -> tuple[dict[str, str], dict[str, str]]:
    aliases = dict(window.unit_aliases)
    stable_to_short = {unit_id: aliases[unit_id] for unit_id in window.target_unit_ids}
    return stable_to_short, {short: stable for stable, short in stable_to_short.items()}


def parse_chapter_segmentation(document: ChapterDocument, payload: dict[str, Any]) -> ChapterNavigationPlan:
    errors: list[str] = []
    root = _object(payload, "顶层", errors)
    _exact(root, {"章节分块"}, "顶层", errors)
    rows = _array(root.get("章节分块"), "章节分块", errors)
    _, short_to_stable = chapter_paragraph_alias_maps(document)
    index_by_id = {unit.unit_id: i for i, unit in enumerate(document.annotation_units)}
    blocks: list[NavigationBlock] = []
    for i, raw in enumerate(rows):
        item = _object(raw, f"章节分块[{i}]", errors)
        _exact(item, {"块ID", "起始段落ID", "结束段落ID", "标签"}, f"章节分块[{i}]", errors)
        block_id = _text(item.get("块ID"), f"章节分块[{i}].块ID", errors)
        if block_id != f"B{i + 1:03d}": errors.append(f"章节分块[{i}].块ID必须是B{i + 1:03d}")
        start_short = _text(item.get("起始段落ID"), f"章节分块[{i}].起始段落ID", errors)
        end_short = _text(item.get("结束段落ID"), f"章节分块[{i}].结束段落ID", errors)
        start, end = short_to_stable.get(start_short, ""), short_to_stable.get(end_short, "")
        if start_short and not start: errors.append(f"未知起始段落ID：{start_short}")
        if end_short and not end: errors.append(f"未知结束段落ID：{end_short}")
        blocks.append(NavigationBlock(block_id, start, end, _text(item.get("标签"), f"章节分块[{i}].标签", errors),
                                      index_by_id.get(start, -1), index_by_id.get(end, -1)))
    plan = ChapterNavigationPlan(document.chapter_id, tuple(blocks))
    errors.extend(v.message for v in validate_navigation_plan(document, plan))
    if errors: raise SynopsisContractError(errors)
    return plan


def parse_local_summary_batch(payload: dict[str, Any], batch: LocalRequestBatch) -> dict[str, list[dict[str, str]]]:
    errors: list[str] = []
    root = _object(payload, "顶层", errors); _exact(root, {"分块梗概"}, "顶层", errors)
    rows = _array(root.get("分块梗概"), "分块梗概", errors)
    expected = [w.block_id for w in batch.windows]
    if len(rows) != len(expected): errors.append("分块梗概数量必须与请求块数量一致")
    result: dict[str, list[dict[str, str]]] = {}
    for i, raw in enumerate(rows):
        item = _object(raw, f"分块梗概[{i}]", errors); _exact(item, {"块ID", "梗概"}, f"分块梗概[{i}]", errors)
        block_id = _text(item.get("块ID"), f"分块梗概[{i}].块ID", errors)
        if i >= len(expected) or block_id != expected[i]: errors.append(f"分块梗概[{i}].块ID或顺序错误")
        summary = _text(item.get("梗概"), f"分块梗概[{i}].梗概", errors)
        result[block_id] = [{"目标ID": f"{block_id}-S001", "块ID": block_id, "梗概": summary}]
    if errors: raise SynopsisContractError(errors)
    return result


def apply_local_summary_patch(candidates: dict[str, list[dict[str, str]]], payload: dict[str, Any],
                              failed_blocks: set[str]) -> dict[str, list[dict[str, str]]]:
    errors: list[str] = []; root = _object(payload, "顶层", errors)
    _exact(root, {"分块替换"}, "顶层", errors)
    replacements = _array(root.get("分块替换"), "分块替换", errors)
    seen: set[str] = set()
    replacement_values: dict[str, str] = {}
    for i, raw in enumerate(replacements):
        item = _object(raw, f"分块替换[{i}]", errors); _exact(item, {"块ID", "新梗概"}, f"分块替换[{i}]", errors)
        block_id = _text(item.get("块ID"), f"分块替换[{i}].块ID", errors)
        if block_id not in failed_blocks or block_id in seen or block_id not in candidates:
            errors.append(f"分块替换目标无效或重复：{block_id}")
        else:
            replacement_values[block_id] = _text(
                item.get("新梗概"), f"分块替换[{i}].新梗概", errors)
            seen.add(block_id)
    if seen != failed_blocks: errors.append("必须恰好替换全部未通过分区")
    if errors: raise SynopsisContractError(errors)
    result = {block_id: [dict(row) for row in rows] for block_id, rows in candidates.items()}
    for block_id, summary in replacement_values.items():
        result[block_id][0]["梗概"] = summary
    return result


def parse_boundary_value(payload: dict[str, Any], *, kind: str) -> str:
    errors: list[str] = []; key = "章首状态" if kind == "opening" else "章末状态"
    root = _object(payload, "顶层", errors); _exact(root, {key}, "顶层", errors)
    value = _text(root.get(key), key, errors)
    if errors: raise SynopsisContractError(errors)
    return value


def build_local_synopsis(window: SynopsisWindow, rows: list[dict[str, str]], support: dict[str, tuple[str, ...]]) -> LocalSynopsis:
    _, short_to_stable = paragraph_alias_maps(window)
    segments = tuple(LocalPlotSegment(row["目标ID"], i, row["梗概"],
        tuple(short_to_stable[x] for x in support[row["目标ID"]])) for i, row in enumerate(rows))
    value = LocalSynopsis(window.window_id, window.chapter_id, window.target_unit_ids, segments)
    value.validate(); return value


def chapter_local_rows(local_synopses: tuple[LocalSynopsis, ...]) -> list[dict[str, Any]]:
    return [{"局部梗概ID": s.segment_id, "梗概": s.summary} for local in local_synopses for s in local.segments]


def parse_chapter_summary(payload: dict[str, Any]) -> dict[str, str]:
    errors: list[str] = []; root = _object(payload, "顶层", errors)
    _exact(root, {"章节梗概"}, "顶层", errors)
    result = {"内容": _text(root.get("章节梗概"), "章节梗概", errors)}
    if errors: raise SynopsisContractError(errors)
    return result


def build_chapter_synopsis(chapter_id: str, opening: BoundaryFrame, ending: BoundaryFrame,
                           local_rows: list[dict[str, Any]], summary: dict[str, str]) -> ChapterSynopsis:
    return ChapterSynopsis(chapter_id, opening,
        ChapterSummary(summary["内容"], tuple(row["局部梗概ID"] for row in local_rows)), ending)
