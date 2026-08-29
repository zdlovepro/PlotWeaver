"""Turn an abstract author style profile into measurable chapter-level budgets."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from .contracts import AuthorStyleProfile, StyleMetric, build_chapter_document
from .style_distillation import build_chapter_style_card


STYLE_BUDGET_METRICS = (
    "chapter_char_count",
    "mean_sentence_chars",
    "mean_paragraph_chars",
    "short_paragraph_ratio",
    "dialogue_char_ratio",
    "perception_marker_ratio",
    "transition_marker_ratio",
    "question_sentence_ratio",
    "exclamation_sentence_ratio",
    "ellipsis_per_sentence",
)
HIGH_PRIORITY_METRICS = frozenset({
    "chapter_char_count",
    "mean_sentence_chars", "mean_paragraph_chars", "short_paragraph_ratio",
    "dialogue_char_ratio", "perception_marker_ratio", "transition_marker_ratio",
})
_METRIC_ACTIONS = {
    "chapter_char_count": ("在不新增计划外情节的前提下，补足已计划场景中的动作、对话、感知、阻力与结果，使篇幅覆盖完整事件链。", "删减重复解释和无结果的往复描写，保留每个场景的目标、阻力、转折与结果。"),
    "mean_sentence_chars": ("把动作、感知、判断或结果组合为完整复句，减少单个动作单独成句。", "拆分过度嵌套的复句，只保留一个主要因果链。"),
    "mean_paragraph_chars": ("让一个段落承载连续的动作—反应—结果，再在下一信息单元切段。", "把并列的信息单元拆开，保留关键落点作为短段。"),
    "short_paragraph_ratio": ("在关键判断、转折或情绪落点保留独立短段，其余信息保持连续段落。", "把非关键短段与相邻的因果或反应合并，避免阅读碎片化。"),
    "dialogue_char_ratio": ("在冲突、试探或信息交接处增加有结果的直接对话，并用动作承接每轮话语。", "压缩重复解释性的对话，把已知信息改为叙述或动作结果。"),
    "perception_marker_ratio": ("在关键动作或信息变化后补入角色看到、听到、感到或察觉到的具体反应。", "删去重复的感知提示，保留能改变判断或行动的感知。"),
    "transition_marker_ratio": ("在场景、时间或注意力变化处加入自然的承接词或结果性过渡，再继续推进。", "用动作结果、时间变化或感知承接替代生硬连接词，避免重复提示。"),
    "question_sentence_ratio": ("只在试探、疑问或需要推动回应时保留问句，其余信息用陈述句交代。", "把不需要回应的反问或连续提问改为陈述和动作反应。"),
    "exclamation_sentence_ratio": ("在真正的情绪落点保留感叹句，并让前后的陈述句提供蓄势。", "把普通强调改为动作、感知或语序变化，避免连续感叹。"),
    "ellipsis_per_sentence": ("在话语中断、犹豫或未尽之意处谨慎保留省略号。", "移除不承担停顿或信息缺口功能的省略号。"),
}


def _round(value: float) -> float:
    return round(float(value), 6)


def _band(target: float, minimum: float, maximum: float) -> tuple[float, float]:
    """Use the middle half of observed variation as a soft author target."""

    lower = minimum + (target - minimum) * 0.5
    upper = target + (maximum - target) * 0.5
    return _round(lower), _round(upper)


def style_execution_spec(profile: AuthorStyleProfile) -> dict[str, Any]:
    """Create an author-level default without turning style into hard quotas."""

    baseline_by_id = {item.metric_id: item for item in profile.baselines}
    metrics = []
    for metric_id in STYLE_BUDGET_METRICS:
        baseline = baseline_by_id.get(metric_id)
        if baseline is None:
            continue
        lower, upper = _band(baseline.mean, baseline.minimum, baseline.maximum)
        metrics.append({
            "metric_id": metric_id,
            "unit": baseline.unit,
            "target": _round(baseline.mean),
            "preferred_min": lower,
            "preferred_max": upper,
            "priority": "high" if metric_id in HIGH_PRIORITY_METRICS else "medium",
        })
    return {
        "schema_version": "1.0",
        "mode": "soft_budget",
        "instruction": "目标区间用于规划和修订，不能逐项机械凑数；剧情、可读性和连续性优先。",
        "metrics": metrics,
    }


def budget_from_metrics(metrics: Iterable[StyleMetric], *, chapter_char_count: int | None = None) -> dict[str, Any]:
    """Build a chapter-specific evaluation budget for fidelity experiments only."""

    metric_by_id = {item.metric_id: item for item in metrics}
    rows = []
    for metric_id in STYLE_BUDGET_METRICS:
        if metric_id == "chapter_char_count":
            if chapter_char_count is None or chapter_char_count <= 0:
                continue
            value = float(chapter_char_count)
            tolerance = max(500.0, value * 0.2)
            rows.append({
                "metric_id": metric_id,
                "unit": "count",
                "target": _round(value),
                "preferred_min": _round(max(120.0, value - tolerance)),
                "preferred_max": _round(value + tolerance),
                "priority": "high",
            })
            continue
        item = metric_by_id.get(metric_id)
        if item is None:
            continue
        if item.unit == "ratio":
            tolerance = max(0.02, item.value * 0.25)
        elif item.unit == "mean_chars":
            tolerance = max(4.0, item.value * 0.15)
        else:
            tolerance = max(3.0, item.value * 0.25)
        rows.append({
            "metric_id": metric_id,
            "unit": item.unit,
            "target": _round(item.value),
            "preferred_min": _round(max(0.0, item.value - tolerance)),
            "preferred_max": _round(item.value + tolerance),
            "priority": "high" if metric_id in HIGH_PRIORITY_METRICS else "medium",
        })
    return {
        "schema_version": "1.0",
        "mode": "fidelity_test_override",
        "instruction": "仅用于重建实验的量化验收；正常原创写作应使用作者级柔性预算。",
        "metrics": rows,
    }


def _budget_rows(budget: dict[str, Any]) -> list[dict[str, Any]]:
    rows = budget.get("metrics", []) if isinstance(budget, dict) else []
    result = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        metric_id = str(row.get("metric_id", "")).strip()
        if metric_id not in STYLE_BUDGET_METRICS:
            continue
        try:
            target = float(row["target"])
            lower = float(row["preferred_min"])
            upper = float(row["preferred_max"])
        except (KeyError, TypeError, ValueError):
            continue
        if target < 0 or lower < 0 or upper < lower:
            continue
        result.append({
            "metric_id": metric_id,
            "unit": str(row.get("unit", "")),
            "target": target,
            "preferred_min": lower,
            "preferred_max": upper,
            "priority": "high" if str(row.get("priority", "")) == "high" else "medium",
        })
    return result


def resolve_style_budget(
    default_spec: dict[str, Any],
    plan_budget: dict[str, Any] | None = None,
    override_budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Use a valid experiment override or constrain planner choices to defaults."""

    defaults = {item["metric_id"]: item for item in _budget_rows(default_spec)}
    override = _budget_rows(override_budget or {})
    if override:
        return {**(override_budget or {}), "metrics": override}
    planned = {item["metric_id"]: item for item in _budget_rows(plan_budget or {})}
    resolved = []
    for metric_id, base in defaults.items():
        candidate = planned.get(metric_id)
        if candidate is None:
            resolved.append(base)
            continue
        target = min(max(candidate["target"], base["preferred_min"]), base["preferred_max"])
        lower = min(max(candidate["preferred_min"], base["preferred_min"]), target)
        upper = max(min(candidate["preferred_max"], base["preferred_max"]), target)
        resolved.append({**base, "target": _round(target), "preferred_min": _round(lower), "preferred_max": _round(upper)})
    return {
        "schema_version": "1.0",
        "mode": "resolved_soft_budget",
        "instruction": str(default_spec.get("instruction", "")),
        "metrics": resolved,
    }


def assess_style_budget(text: str, budget: dict[str, Any]) -> dict[str, Any]:
    """Measure a draft and return machine-readable deviations for a repair pass."""

    document = build_chapter_document(
        "generated/style-check",
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        text,
        "Generated",
    )
    actual = {item.metric_id: item for item in build_chapter_style_card(document).metrics}
    actual["chapter_char_count"] = StyleMetric("chapter_char_count", float(len(text)), "count")
    rows = []
    weighted = []
    high_in_range = 0
    high_total = 0
    for target in _budget_rows(budget):
        metric = actual.get(target["metric_id"])
        if metric is None:
            continue
        value = metric.value
        in_range = target["preferred_min"] <= value <= target["preferred_max"]
        floor = 0.05 if metric.unit == "ratio" else 1.0
        relative_deviation = abs(value - target["target"]) / max(abs(target["target"]), floor)
        weight = 2.0 if target["priority"] == "high" else 1.0
        weighted.append(relative_deviation * weight)
        if target["priority"] == "high":
            high_total += 1
            high_in_range += int(in_range)
        rows.append({
            **target,
            "actual": _round(value),
            "in_preferred_range": in_range,
            "relative_deviation": _round(relative_deviation),
        })
    mean_deviation = _round(sum(weighted) / sum(2.0 if item["priority"] == "high" else 1.0 for item in rows)) if rows else 1.0
    required_high = max(1, int(high_total * 0.6 + 0.999))
    passed = bool(rows) and high_in_range >= required_high and mean_deviation <= 0.38
    priorities = [
        item["metric_id"] for item in sorted(rows, key=lambda value: value["relative_deviation"], reverse=True)
        if not item["in_preferred_range"]
    ][:4]
    return {
        "passed": passed,
        "mean_relative_deviation": mean_deviation,
        "high_priority_in_range": high_in_range,
        "high_priority_required": required_high,
        "metrics": rows,
        "repair_priorities": priorities,
    }


def style_execution_directives(
    budget: dict[str, Any],
    assessment: dict[str, Any] | None = None,
    *,
    limit: int = 4,
) -> list[dict[str, str]]:
    """Translate numeric style targets into a short, ordered writing checklist."""

    assessment_by_id = {
        str(item.get("metric_id", "")): item
        for item in (assessment or {}).get("metrics", [])
        if isinstance(item, dict)
    }
    rows = _budget_rows(budget)
    candidates = []
    for row in rows:
        observed = assessment_by_id.get(row["metric_id"])
        if observed is None:
            direction = "maintain"
            severity = 0.0
        else:
            actual = float(observed.get("actual", row["target"]))
            direction = "increase" if actual < row["preferred_min"] else "decrease" if actual > row["preferred_max"] else "maintain"
            severity = float(observed.get("relative_deviation", 0.0))
        candidates.append((severity, row, direction))
    candidates.sort(
        key=lambda item: (item[0], item[1]["metric_id"] == "chapter_char_count", item[1]["priority"] == "high"),
        reverse=True,
    )
    result = []
    for _, row, direction in candidates[:max(1, limit)]:
        increase, decrease = _METRIC_ACTIONS[row["metric_id"]]
        action = increase if direction == "increase" else decrease if direction == "decrease" else "保持当前做法，只在不破坏剧情时微调。"
        result.append({
            "metric_id": row["metric_id"],
            "direction": direction,
            "target_range": f"{row['preferred_min']}–{row['preferred_max']}",
            "action": action,
        })
    return result


def style_execution_blueprint(budget: dict[str, Any]) -> dict[str, Any]:
    """Translate a budget into countable, chapter-level drafting anchors."""

    rows = {item["metric_id"]: item for item in _budget_rows(budget)}
    length = rows.get("chapter_char_count")
    mean_paragraph = rows.get("mean_paragraph_chars")
    mean_sentence = rows.get("mean_sentence_chars")
    result: dict[str, Any] = {"instruction": "按此蓝图安排正文，但不得删改计划事件或以凑数破坏可读性。"}
    if length is None:
        return result
    target_chars = length["target"]
    result["chapter_char_count"] = {"target": round(target_chars), "minimum": round(length["preferred_min"]), "maximum": round(length["preferred_max"])}
    if mean_paragraph is not None:
        target_paragraphs = max(1, round(target_chars / max(mean_paragraph["target"], 1)))
        lower = max(1, round(length["preferred_min"] / max(mean_paragraph["preferred_max"], 1)))
        upper = max(lower, round(length["preferred_max"] / max(mean_paragraph["preferred_min"], 1)))
        result["paragraph_count"] = {"target": target_paragraphs, "minimum": lower, "maximum": upper}
        short_ratio = rows.get("short_paragraph_ratio")
        if short_ratio is not None:
            result["short_paragraph_count"] = {
                "target": round(target_paragraphs * short_ratio["target"]),
                "minimum": max(0, round(lower * short_ratio["preferred_min"])),
                "maximum": max(0, round(upper * short_ratio["preferred_max"])),
            }
    if mean_sentence is not None:
        result["sentence_count"] = {
            "target": max(1, round(target_chars / max(mean_sentence["target"], 1))),
            "minimum": max(1, round(length["preferred_min"] / max(mean_sentence["preferred_max"], 1))),
            "maximum": max(1, round(length["preferred_max"] / max(mean_sentence["preferred_min"], 1))),
        }
    sentence_target = result.get("sentence_count", {}).get("target", 0)
    dialogue = rows.get("dialogue_char_ratio")
    if dialogue is not None:
        result["dialogue_char_count"] = {
            "target": round(target_chars * dialogue["target"]),
            "minimum": round(length["preferred_min"] * dialogue["preferred_min"]),
            "maximum": round(length["preferred_max"] * dialogue["preferred_max"]),
        }
    for metric_id, key in (("perception_marker_ratio", "perception_sentence_count"), ("transition_marker_ratio", "transition_sentence_count")):
        row = rows.get(metric_id)
        if row is not None and sentence_target:
            result[key] = {
                "target": round(sentence_target * row["target"]),
                "minimum": max(0, round(sentence_target * row["preferred_min"])),
                "maximum": max(0, round(sentence_target * row["preferred_max"])),
            }
    return result


def style_repair_prompt(
    draft: str,
    chapter_plan: dict[str, Any],
    budget: dict[str, Any],
    assessment: dict[str, Any],
) -> str:
    """A Chinese repair prompt that preserves the plan while correcting drift."""

    directives = style_execution_directives(budget, assessment, limit=6)
    blueprint = style_execution_blueprint(budget)
    import json
    return f"""任务：修订一段中文小说正文，使其更接近给定的文风预算，同时不得改变章节计划中的事件顺序、人物、状态变化、时间、位置或结果。

只输出修订后的连续正文，不输出标题、解释、Markdown 或 JSON。不要删除计划中的关键行动、阻力、选择和后果；优先处理 `repair_priorities` 中的指标。预算是区间约束，不要为了逐项凑数而写成清单或破坏可读性。

先在心中按下列指令和篇幅蓝图安排句段、对话、感知与转场，再输出完整修订稿。若蓝图给出章节字符数下限，正文必须达到下限；扩写只能补足计划内场景的行动、阻力、对话、感知或结果。只处理列出的高优先事项，避免一次改动所有指标造成新的偏差：
{json.dumps(directives, ensure_ascii=False, indent=2)}

篇幅与句段蓝图：
{json.dumps(blueprint, ensure_ascii=False, indent=2)}

章节计划：
{json.dumps(chapter_plan, ensure_ascii=False)}

文风预算：
{json.dumps(budget, ensure_ascii=False)}

当前偏差：
{json.dumps(assessment, ensure_ascii=False)}

待修订正文：
{draft}
"""
