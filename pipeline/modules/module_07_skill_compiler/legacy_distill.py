from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .jsonio import read_json, read_jsonl, write_json, write_jsonl
from .model import ModelSettings, complete_json
from .paths import active_work_ids, corpus_dir, output_dir


DEFAULT_MAX_INPUT_CHARS = 48_000

BATCH_SCHEMA = {
    "narrative_patterns": ["string"],
    "scene_and_pacing_patterns": ["string"],
    "character_and_relation_patterns": ["string"],
    "world_and_constraint_patterns": ["string"],
    "information_and_hook_patterns": ["string"],
    "language_surface_patterns": ["string"],
    "prompt_rules": {"planning": ["string"], "drafting": ["string"], "validation": ["string"]},
    "evidence_chapter_ids": ["string"],
}

PROFILE_SCHEMA = {
    "narrative_architecture": ["string"],
    "scene_and_pacing": ["string"],
    "character_and_relation": ["string"],
    "world_and_constraints": ["string"],
    "information_and_hooks": ["string"],
    "language_surface": ["string"],
    "prompt_engineering": {"planning_rules": ["string"], "drafting_rules": ["string"], "validation_rules": ["string"]},
    "confidence_and_limits": ["string"],
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
        "prose_metrics": row.get("prose_metrics", {}),
        "evidence": row.get("evidence", [])[:3],
    }


def _chunk_rows(rows: Iterable[dict[str, Any]], max_input_chars: int) -> list[list[dict[str, Any]]]:
    max_input_chars = max(4_000, int(max_input_chars))
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 2
    for row in rows:
        size = len(json.dumps(row, ensure_ascii=False)) + 1
        if current and current_size + size > max_input_chars:
            groups.append(current)
            current, current_size = [], 2
        current.append(row)
        current_size += size
    if current:
        groups.append(current)
    return groups


def _offline_batch_observation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    functions = [str(row.get("chapter_function", "")).strip() for row in rows if row.get("chapter_function")]
    hooks = [str(row.get("pacing_curve", {}).get("ending_hook", "")).strip() for row in rows if row.get("pacing_curve", {}).get("ending_hook")]
    events = sum(len(row.get("event_chain", [])) for row in rows)
    characters = sum(len(row.get("character_deltas", [])) for row in rows)
    return {
        "narrative_patterns": [f"本批 {len(rows)} 章记录到 {events} 个事件节点；章节功能样本：{'；'.join(functions[:6])}"],
        "scene_and_pacing_patterns": [f"本批可观测章节末钩子 {len(hooks)} 个；需以模型抽取补强具体节奏规律。"],
        "character_and_relation_patterns": [f"本批记录到 {characters} 条人物状态变化；离线模式不推断人物关系风格。"],
        "world_and_constraint_patterns": ["离线模式只保留结构化世界与约束字段，不做风格归纳。"],
        "information_and_hook_patterns": [f"钩子样本：{'；'.join(hooks[:5])}" if hooks else "未从本批获得有效钩子样本。"],
        "language_surface_patterns": ["离线模式不能可靠判断句式、修辞和语气。"],
        "prompt_rules": {
            "planning": ["先落实事件、人物、时间空间、约束和信息差，再开始正文。"],
            "drafting": ["正文必须覆盖章节状态中的事件转折和结尾钩子。"],
            "validation": ["检查人物、关系、地点、资源和伏笔与前文状态是否矛盾。"],
        },
        "evidence_chapter_ids": [str(row.get("chapter_id", "")) for row in rows[:12]],
    }


def _batch_prompt(rows: list[dict[str, Any]]) -> str:
    return f"""任务：从同一小说作者的一批章节叙事状态中归纳可迁移的写作约束。归纳的是抽象叙事机制，不得复述、复制或命名来源作品的具体人物、地名、功法、道具、情节或措辞。

只能输出以下顶层字段：{json.dumps(list(BATCH_SCHEMA), ensure_ascii=False)}。
输出必须符合此 JSON schema：
{json.dumps(BATCH_SCHEMA, ensure_ascii=False, indent=2)}

要求：
1. 每条规律都要具体到可以转写为提示词的程度，避免“文笔好”“节奏快”等空泛评价。
2. `evidence_chapter_ids` 只能填写输入中实际存在的 chapter_id。
3. `language_surface_patterns` 只能依据 evidence 中的短引文、对话比例和段落指标归纳；证据不足时明确写“证据不足”。
4. `prompt_rules` 必须分别覆盖章节规划、正文扩写和生成后校验。
5. 不得输出 schema 之外的字段。

章节状态批次：
{json.dumps(rows, ensure_ascii=False)}"""


def _merge_prompt(observations: list[dict[str, Any]]) -> str:
    return f"""任务：把多个章节批次的作者叙事观察合并成一个稳定、可执行的作者叙事 profile。

只能输出以下顶层字段：{json.dumps(list(PROFILE_SCHEMA), ensure_ascii=False)}。
输出必须符合此 JSON schema：
{json.dumps(PROFILE_SCHEMA, ensure_ascii=False, indent=2)}

合并规则：
1. 只保留被多个批次支持或被明确证据支持的规律；冲突时写入 `confidence_and_limits`。
2. 不得包含任何来源作品的专名、人物名、地名、具体情节、原句或可识别的设定。
3. `prompt_engineering` 的每条规则必须能直接作为后续章节规划、正文生成或校验提示词的一部分。
4. 不得输出 schema 之外的字段。

批次观察：
{json.dumps(observations, ensure_ascii=False)}"""


def _offline_profile(observations: list[dict[str, Any]]) -> dict[str, Any]:
    def unique(field: str) -> list[str]:
        values: list[str] = []
        for observation in observations:
            for value in observation.get(field, []):
                if value and value not in values:
                    values.append(value)
        return values[:30]

    return {
        "narrative_architecture": unique("narrative_patterns"),
        "scene_and_pacing": unique("scene_and_pacing_patterns"),
        "character_and_relation": unique("character_and_relation_patterns"),
        "world_and_constraints": unique("world_and_constraint_patterns"),
        "information_and_hooks": unique("information_and_hook_patterns"),
        "language_surface": unique("language_surface_patterns"),
        "prompt_engineering": {
            "planning_rules": unique_prompt_rules(observations, "planning"),
            "drafting_rules": unique_prompt_rules(observations, "drafting"),
            "validation_rules": unique_prompt_rules(observations, "validation"),
        },
        "confidence_and_limits": ["这是离线结构基线；请使用模型模式重跑阶段 5，以获得基于章节证据的叙事与语言归纳。"],
    }


def unique_prompt_rules(observations: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for observation in observations:
        for value in observation.get("prompt_rules", {}).get(key, []):
            if value and value not in values:
                values.append(value)
    return values[:30]


def distill(author_id: str, offline: bool = False, max_input_chars: int = DEFAULT_MAX_INPUT_CHARS, state_filename: str = "chapter_states.jsonl") -> Path:
    base = corpus_dir(author_id)
    if "/" in state_filename or "\\" in state_filename:
        raise ValueError("state filename must not contain a directory")
    works = [base / work_id for work_id in active_work_ids(author_id)]
    all_states: list[dict[str, Any]] = []
    work_summaries: list[dict[str, Any]] = []
    for work in works:
        states = [_compact_state(row) for row in read_jsonl(work / state_filename)]
        if not states:
            continue
        all_states.extend(states)
        work_summaries.append({"work_id": work.name, "chapter_count": len(states), "metadata": read_json(work / "metadata.json", {})})
    if not all_states:
        raise ValueError("No full chapter_states.jsonl found; complete extraction before distillation")
    template_library = read_json(output_dir(author_id) / "template_library.json", {})
    if not template_library:
        raise ValueError("template library is missing; complete template mining before distillation")
    batches = _chunk_rows(all_states, max_input_chars)
    observations: list[dict[str, Any]] = []
    settings = ModelSettings.from_environment()
    for index, batch in enumerate(batches, start=1):
        observation = _offline_batch_observation(batch) if offline else complete_json(
            "你是小说作者叙事风格蒸馏器。所有字段值必须使用中文，并且只能根据输入证据归纳。", _batch_prompt(batch), settings,
        )
        observations.append({"batch_index": index, "chapter_count": len(batch), "observation": observation})
        print(f"[distill] {index}/{len(batches)} 批完成", flush=True)
    batch_target = output_dir(author_id) / "distillation_batches.jsonl"
    write_jsonl(batch_target, observations)
    raw_observations = [row["observation"] for row in observations]
    profile_body = _offline_profile(raw_observations) if offline else complete_json(
        "你是小说作者叙事 profile 合并器。所有字段值必须使用中文，且严格遵循输出 JSON 结构。", _merge_prompt(raw_observations), settings,
    )
    profile = {
        "author_id": author_id,
        "profile_version": "2.0",
        "distillation_mode": "offline" if offline else "model",
        "works": work_summaries,
        "source_chapter_count": len(all_states),
        "source_state_file": state_filename,
        "batch_count": len(batches),
        "template_library_summary": {
            "library_version": template_library.get("library_version", ""),
            "micro_template_count": len(template_library.get("micro_templates", [])),
            "event_template_count": len(template_library.get("event_templates", [])),
            "chapter_template_count": len(template_library.get("chapter_templates", [])),
            "macro_template_count": len(template_library.get("macro_templates", [])),
        },
        **profile_body,
    }
    target = output_dir(author_id) / "author_narrative_profile.json"
    write_json(target, profile)
    return target
