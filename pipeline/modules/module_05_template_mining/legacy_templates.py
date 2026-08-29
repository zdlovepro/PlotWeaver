from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .jsonio import read_jsonl, write_json, write_jsonl
from .model import ModelSettings, complete_json
from .paths import active_work_ids, corpus_dir, output_dir


DEFAULT_MAX_INPUT_CHARS = 48_000

TEMPLATE_SCHEMA = {
    "micro_templates": [{"name": "", "function": "", "role_slots": [], "interaction_beats": [], "selection_tags": [], "evidence_chapter_ids": []}],
    "event_templates": [{"name": "", "function": "", "preconditions": [], "role_slots": [], "beat_sequence": [], "state_delta": [], "variation_axes": [], "selection_tags": [], "forbidden": [], "evidence_chapter_ids": []}],
    "chapter_templates": [{"name": "", "chapter_function": "", "scene_sequence": [], "pacing_pattern": [], "ending_strategy": "", "selection_tags": [], "evidence_chapter_ids": []}],
}

MACRO_TEMPLATE_SCHEMA = {
    "macro_templates": [{"name": "", "arc_goal": "", "entry_state": [], "pressure_ladder": [], "turning_points": [], "payoff": [], "exit_state": [], "variation_axes": [], "selection_tags": [], "evidence_chapter_ids": []}],
}


def _compact_state(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chapter_id": row.get("chapter_id", ""),
        "chapter_function": row.get("chapter_function", ""),
        "entities": row.get("entities", []),
        "event_chain": row.get("event_chain", []),
        "scene_beats": row.get("scene_beats", []),
        "character_deltas": row.get("character_deltas", []),
        "relation_deltas": row.get("relation_deltas", []),
        "temporal_spatial_state": row.get("temporal_spatial_state", {}),
        "cultivation_state": row.get("cultivation_state", {}),
        "information_state": row.get("information_state", {}),
        "pacing_curve": row.get("pacing_curve", {}),
    }


def _chunks(rows: Iterable[dict[str, Any]], max_input_chars: int) -> list[list[dict[str, Any]]]:
    limit = max(4_000, int(max_input_chars))
    result: list[list[dict[str, Any]]] = []
    batch: list[dict[str, Any]] = []
    size = 2
    for row in rows:
        row_size = len(json.dumps(row, ensure_ascii=False)) + 1
        if batch and size + row_size > limit:
            result.append(batch)
            batch, size = [], 2
        batch.append(row)
        size += row_size
    if batch:
        result.append(batch)
    return result


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("chapter_id", "")) for row in rows if row.get("chapter_id")]


def _offline_observation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = _ids(rows)
    micro: list[dict[str, Any]] = []
    event: list[dict[str, Any]] = []
    chapter: list[dict[str, Any]] = []
    has_relations = any(row.get("relation_deltas") for row in rows)
    has_information = any(row.get("information_state", {}).get("foreshadows_planted") or row.get("information_state", {}).get("foreshadows_resolved") for row in rows)
    has_obstacle = any(item.get("obstacle") for row in rows for item in row.get("event_chain", []))
    has_turn = any(item.get("turn") for row in rows for item in row.get("event_chain", []))
    if has_relations:
        micro.append({"name": "关系变化互动", "function": "通过人物立场或关系变化推动局部冲突", "role_slots": ["行动者", "关系对象", "压力来源"], "interaction_beats": ["关系受压", "立场或信息变化", "关系结果"], "selection_tags": ["关系", "立场", "冲突"], "evidence_chapter_ids": ids[:20]})
    if has_information:
        micro.append({"name": "信息差推进", "function": "用线索、秘密或伏笔改变读者与人物的信息分布", "role_slots": ["知情者", "未知者", "信息载体"], "interaction_beats": ["信息出现", "误读或隐瞒", "后续压力"], "selection_tags": ["信息", "伏笔", "秘密"], "evidence_chapter_ids": ids[:20]})
    if has_obstacle:
        micro.append({"name": "目标受阻互动", "function": "让行动目标遭遇具体阻力，再逼出选择", "role_slots": ["行动者", "阻碍者", "代价承担者"], "interaction_beats": ["提出目标", "阻力加压", "选择或代价"], "selection_tags": ["目标", "阻力", "代价"], "evidence_chapter_ids": ids[:20]})
    beats = ["触发", "行动", "阻力"] + (["转折"] if has_turn else []) + ["结果"]
    event.append({"name": "受压推进事件", "function": "在约束下推进目标并改变章节末状态", "preconditions": ["存在明确行动目标", "存在可见阻力或代价"], "role_slots": ["行动者", "阻碍者", "协助者或见证者"], "beat_sequence": beats, "state_delta": ["人物目标、关系、资源、位置或信息至少一项发生变化"], "variation_axes": ["场景", "阻力来源", "破局方式", "代价", "结尾压力"], "selection_tags": ["推进", "冲突", "转折"], "forbidden": ["来源作品专名", "来源作品具体情节", "来源原句"], "evidence_chapter_ids": ids[:30]})
    hooks = any(row.get("pacing_curve", {}).get("ending_hook") for row in rows)
    chapter.append({"name": "推进—冲突—转折章节", "chapter_function": "以行动推进为主，在结尾保留未完全解决的压力", "scene_sequence": ["建立当前目标与场景", "阻力或对抗升级", "转折并形成状态变化", "以钩子或后果离场"], "pacing_pattern": ["铺垫", "冲突", "高潮", "短收束"], "ending_strategy": "留下下一步行动、代价或未解信息" if hooks else "以状态变化收束并连接下一章", "selection_tags": ["推进", "章节钩子" if hooks else "状态变化"], "evidence_chapter_ids": ids[:30]})
    return {"micro_templates": micro, "event_templates": event, "chapter_templates": chapter}


def _prompt(rows: list[dict[str, Any]]) -> str:
    return f"""任务：从同一作者的章节叙事状态中挖掘可复用的多层叙事模板。模板只能描述抽象功能、角色槽位、节拍和可替换变量；不得保留或复述来源作品的人名、地名、专名、具体情节、具体设定或原句。

只能输出以下顶层字段：{json.dumps(list(TEMPLATE_SCHEMA), ensure_ascii=False)}。
输出必须符合这个 JSON schema：
{json.dumps(TEMPLATE_SCHEMA, ensure_ascii=False, indent=2)}

规则：
1. `micro_templates` 是局部互动模式，最多 8 条。
2. `event_templates` 是可执行事件模板，必须提供前置条件、角色槽位、节拍、状态变化、变体轴和禁用项，最多 8 条。
3. `chapter_templates` 是章节级节奏结构，最多 5 条。
4. `evidence_chapter_ids` 只能来自输入 chapter_id；每个模板至少 2 个，证据不足则不要输出该模板。
5. 只输出跨章节重复出现或可由多项证据支持的模式，不得把单一情节包装成模板。

章节状态批次：
{json.dumps(rows, ensure_ascii=False)}"""


def _macro_windows(work_rows: list[list[dict[str, Any]]], window_size: int = 30) -> list[list[dict[str, Any]]]:
    windows: list[list[dict[str, Any]]] = []
    for rows in work_rows:
        for start in range(0, len(rows), window_size):
            window = rows[start:start + window_size]
            if len(window) >= 5:
                windows.append(window)
    return windows


def _offline_macro_observation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = _ids(rows)
    hooks = sum(1 for row in rows if row.get("pacing_curve", {}).get("ending_hook"))
    return {"macro_templates": [{
        "name": "阶段目标—压力升级—阶段结算", "arc_goal": "围绕阶段性目标持续推进，并让阶段末状态发生明确变化",
        "entry_state": ["给出阶段目标、已有约束和初始资源或关系格局"],
        "pressure_ladder": ["初始阻力出现", "阻力扩展到资源、关系、规则或信息层面", "阶段末形成必须处理的更高压力"],
        "turning_points": ["至少一次策略、关系、信息或资源层面的反转"],
        "payoff": ["结算阶段收获、代价或关系变化"],
        "exit_state": ["保留进入下一阶段的新目标、后果或悬念"],
        "variation_axes": ["阶段目标", "压力来源", "反转手段", "代价", "阶段回报"],
        "selection_tags": ["阶段推进", "压力升级", "阶段结算", "章节钩子" if hooks else "状态变化"],
        "evidence_chapter_ids": ids,
    }]}


def _macro_prompt(rows: list[dict[str, Any]]) -> str:
    return f"""任务：以下是同一部作品中连续章节窗口的叙事状态。归纳其中反复出现的阶段级（宏观）叙事模板。宏观模板描述一段连续剧情如何开始、逐级加压、发生关键转折、完成阶段回报并进入下一阶段。

只能输出以下顶层字段：{json.dumps(list(MACRO_TEMPLATE_SCHEMA), ensure_ascii=False)}。
输出必须符合这个 JSON schema：
{json.dumps(MACRO_TEMPLATE_SCHEMA, ensure_ascii=False, indent=2)}

规则：
1. 最多输出 4 个宏观模板；模板必须是抽象结构，不得写入来源作品的专名、具体情节、设定或原句。
2. `evidence_chapter_ids` 只能引用输入中的连续章节，至少 5 个。
3. 压力阶梯和关键转折必须可用于规划多章剧情，不能只是单章事件摘要。
4. 证据不足时返回 `{{"macro_templates": []}}`。

连续章节窗口：
{json.dumps(rows, ensure_ascii=False)}"""


def _list(value: Any) -> list[str]:
    return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []


def _normalize(items: Any, kind: str) -> list[dict[str, Any]]:
    fields = {
        "micro": ("name", "function", "role_slots", "interaction_beats", "selection_tags", "evidence_chapter_ids"),
        "event": ("name", "function", "preconditions", "role_slots", "beat_sequence", "state_delta", "variation_axes", "selection_tags", "forbidden", "evidence_chapter_ids"),
        "chapter": ("name", "chapter_function", "scene_sequence", "pacing_pattern", "ending_strategy", "selection_tags", "evidence_chapter_ids"),
        "macro": ("name", "arc_goal", "entry_state", "pressure_ladder", "turning_points", "payoff", "exit_state", "variation_axes", "selection_tags", "evidence_chapter_ids"),
    }[kind]
    results: list[dict[str, Any]] = []
    for raw in items if isinstance(items, list) else []:
        if not isinstance(raw, dict):
            continue
        row: dict[str, Any] = {}
        for field in fields:
            row[field] = _list(raw.get(field)) if field not in {"name", "function", "chapter_function", "ending_strategy", "arc_goal"} else str(raw.get(field, "")).strip()
        if row.get("name") and row.get("evidence_chapter_ids"):
            results.append(row)
    return results


def _merge(observations: list[dict[str, Any]], kind: str, limit: int) -> list[dict[str, Any]]:
    key_fields = {"micro": ("name", "function"), "event": ("name", "function", "beat_sequence"), "chapter": ("name", "chapter_function", "scene_sequence"), "macro": ("name", "arc_goal", "pressure_ladder")}[kind]
    buckets: dict[str, dict[str, Any]] = {}
    source_key = f"{kind}_templates"
    for observation in observations:
        for row in _normalize(observation.get(source_key), kind):
            key = json.dumps([row.get(field) for field in key_fields], ensure_ascii=False, sort_keys=True)
            existing = buckets.get(key)
            if existing is None:
                buckets[key] = row
                continue
            for field, value in row.items():
                if isinstance(value, list):
                    existing[field] = list(dict.fromkeys([*existing.get(field, []), *value]))
    ordered = sorted(buckets.values(), key=lambda row: len(row.get("evidence_chapter_ids", [])), reverse=True)[:limit]
    for index, row in enumerate(ordered, start=1):
        row["template_id"] = f"{kind}-{index:03d}"
        row["support_count"] = len(row.get("evidence_chapter_ids", []))
    return ordered


def mine_templates(author_id: str, offline: bool = False, max_input_chars: int = DEFAULT_MAX_INPUT_CHARS, state_filename: str = "chapter_states.jsonl", allow_macro: bool = True) -> Path:
    base = corpus_dir(author_id)
    if "/" in state_filename or "\\" in state_filename:
        raise ValueError("state filename must not contain a directory")
    rows: list[dict[str, Any]] = []
    work_rows: list[list[dict[str, Any]]] = []
    for work in [base / work_id for work_id in active_work_ids(author_id)]:
        if work.is_dir():
            state_rows = [_compact_state(row) for row in read_jsonl(work / state_filename)]
            if state_rows:
                work_rows.append(state_rows)
                rows.extend(state_rows)
    if not rows:
        raise ValueError("No full chapter_states.jsonl found; complete extraction before template mining")
    batches = _chunks(rows, max_input_chars)
    observations: list[dict[str, Any]] = []
    settings = ModelSettings.from_environment()
    for index, batch in enumerate(batches, start=1):
        observation = _offline_observation(batch) if offline else complete_json("你是小说叙事模板挖掘器。所有字段值必须使用中文，只归纳可复用的抽象模式。", _prompt(batch), settings)
        observations.append({"batch_index": index, "chapter_count": len(batch), "observation": observation})
        print(f"[templates] {index}/{len(batches)} 批完成", flush=True)
    macro_observations: list[dict[str, Any]] = []
    windows = _macro_windows(work_rows) if allow_macro else []
    for index, window in enumerate(windows, start=1):
        observation = _offline_macro_observation(window) if offline else complete_json("你是小说宏观叙事模板挖掘器。所有字段值必须使用中文，只归纳多章可复用的抽象结构。", _macro_prompt(window), settings)
        macro_observations.append({"window_index": index, "chapter_count": len(window), "observation": observation})
        print(f"[templates-macro] {index}/{len(windows)} 窗口完成", flush=True)
    destination = output_dir(author_id)
    write_jsonl(destination / "template_mining_batches.jsonl", [*observations, *[{"kind": "macro_window", **row} for row in macro_observations]])
    bodies = [row["observation"] for row in observations]
    library = {
        "library_version": "1.0",
        "author_id": author_id,
        "mining_mode": "offline" if offline else "model",
        "source_chapter_count": len(rows),
        "source_state_file": state_filename,
        "macro_templates_available": allow_macro,
        "batch_count": len(batches),
        "macro_window_count": len(windows),
        "micro_templates": _merge(bodies, "micro", 80),
        "event_templates": _merge(bodies, "event", 80),
        "chapter_templates": _merge(bodies, "chapter", 40),
        "macro_templates": _merge([row["observation"] for row in macro_observations], "macro", 20),
    }
    target = destination / "template_library.json"
    write_json(target, library)
    return target


def select_templates(library: dict[str, Any], brief: dict[str, Any], per_kind: int = 3) -> dict[str, list[dict[str, Any]]]:
    query = json.dumps(brief, ensure_ascii=False)
    selected: dict[str, list[dict[str, Any]]] = {}
    for key in ("micro_templates", "event_templates", "chapter_templates", "macro_templates"):
        scored = []
        for row in library.get(key, []):
            tags = row.get("selection_tags", [])
            score = sum(3 for tag in tags if tag and tag in query) + int(row.get("support_count", 0))
            scored.append((score, row))
        selected[key] = [row for _, row in sorted(scored, key=lambda item: item[0], reverse=True)[:per_kind]]
    return selected
