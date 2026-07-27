from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .contracts import ChapterRecord, ChapterState
from .jsonio import read_jsonl, write_jsonl
from .model import ModelSettings, complete_json
from .paths import corpus_dir


DEFAULT_MAX_INPUT_CHARS = 12_000
DEFAULT_OVERLAP_CHARS = 500
DEFAULT_BATCH_SIZE = 10
DEFAULT_MERGE_BATCH_SIZE = 3
SAMPLE_SELECTIONS = ("sequential", "stratified")

CHAPTER_STATE_SCHEMA = {
    "chapter_function": "string",
    "entities": [{"entity_id": "", "canonical_name": "", "aliases": [], "entity_type": "", "evidence": ""}],
    "event_chain": [{"trigger": "", "action": "", "obstacle": "", "turn": "", "result": ""}],
    "scene_beats": [{"scene_no": 1, "time": "", "location": "", "participants": [], "objective": "", "key_actions": [], "dialogue_intents": [], "conflict": "", "turn": "", "result": "", "preconditions": [], "before_state": {"time": "", "location": "", "resources": [], "relationships": [], "knowledge": []}, "after_state": {"time": "", "location": "", "resources": [], "relationships": [], "knowledge": []}, "knowledge_updates": [{"entity": "", "learns": "", "evidence": ""}], "fact_provenance": [{"fact": "", "quote": ""}]}],
    "character_deltas": [{"character": "", "goal": "", "emotion": "", "resources": "", "cultivation": "", "secret": ""}],
    "relation_deltas": [{"source": "", "target": "", "relation": "", "change": "", "evidence": ""}],
    "temporal_spatial_state": {"time_anchor": "", "duration": "", "location": "", "moves": []},
    "cultivation_state": {"stage": "", "techniques": [], "items": [], "resources": [], "costs": [], "constraints": []},
    "information_state": {"reader_knows": [], "character_knowledge": [], "foreshadows_planted": [], "foreshadows_resolved": []},
    "pacing_curve": {"setup": 0, "conflict": 0, "climax": 0, "resolution": 0, "ending_hook": ""},
    "evidence": [{"field": "", "quote": ""}],
}

CHAPTER_ANALYSIS_SCHEMA = {
    "chapter_function": CHAPTER_STATE_SCHEMA["chapter_function"],
    "event_chain": CHAPTER_STATE_SCHEMA["event_chain"],
    "character_deltas": CHAPTER_STATE_SCHEMA["character_deltas"],
    "relation_deltas": CHAPTER_STATE_SCHEMA["relation_deltas"],
    "temporal_spatial_state": CHAPTER_STATE_SCHEMA["temporal_spatial_state"],
    "cultivation_state": CHAPTER_STATE_SCHEMA["cultivation_state"],
    "information_state": CHAPTER_STATE_SCHEMA["information_state"],
    "pacing_curve": CHAPTER_STATE_SCHEMA["pacing_curve"],
    "evidence": CHAPTER_STATE_SCHEMA["evidence"],
}

CHAPTER_STATE_EXAMPLE = {
    "chapter_function": "主角在受限条件下获得新的行动机会，并留下下一章危机。",
    "entities": [{"entity_id": "person-a", "canonical_name": "角色甲", "aliases": ["角色甲的小名"], "entity_type": "人物", "evidence": "角色甲收到了试炼通知"}],
    "event_chain": [{"trigger": "角色甲收到试炼通知", "action": "角色甲前往地点甲调查", "obstacle": "资源不足且角色乙阻拦", "turn": "角色甲发现隐藏线索", "result": "角色甲暂时脱困并决定继续追查"}],
    "scene_beats": [{"scene_no": 1, "time": "当日傍晚", "location": "地点甲的外院", "participants": ["person-a", "person-b"], "objective": "确认试炼通知的真伪", "key_actions": ["角色甲前往外院", "角色乙阻拦"], "dialogue_intents": ["角色乙施压", "角色甲隐藏真实判断"], "conflict": "角色乙阻止调查", "turn": "角色甲发现隐藏线索", "result": "角色甲决定夜间继续追查", "preconditions": ["角色甲已经收到试炼通知", "角色甲能够进入外院"], "before_state": {"time": "当日傍晚", "location": "地点甲外", "resources": ["试炼通知"], "relationships": ["角色甲与角色乙存在竞争"], "knowledge": ["角色甲知道通知异常" ]}, "after_state": {"time": "当日傍晚", "location": "地点甲的外院", "resources": ["试炼通知", "隐藏线索"], "relationships": ["竞争加剧"], "knowledge": ["角色甲知道隐藏线索" ]}, "knowledge_updates": [{"entity": "person-a", "learns": "通知与旧案有关", "evidence": "角色甲发现隐藏线索"}], "fact_provenance": [{"fact": "角色甲发现隐藏线索", "quote": "角色甲收到了试炼通知"}]}],
    "character_deltas": [{"character": "角色甲", "goal": "查明试炼异常", "emotion": "警惕", "resources": "获得一枚令牌", "cultivation": "修为未变化", "secret": "知晓令牌与旧案有关"}],
    "relation_deltas": [{"source": "角色甲", "target": "角色乙", "relation": "同门竞争", "change": "信任下降", "evidence": "角色乙阻止角色甲查看线索"}],
    "temporal_spatial_state": {"time_anchor": "当日傍晚", "duration": "约半个时辰", "location": "地点甲的外院", "moves": ["住所→外院"]},
    "cultivation_state": {"stage": "当前境界", "techniques": ["功法甲"], "items": ["令牌"], "resources": ["少量灵石"], "costs": ["消耗部分灵力"], "constraints": ["不可在宗门内公开争斗"]},
    "information_state": {"reader_knows": ["试炼可能被人为操控"], "character_knowledge": ["角色甲知道令牌异常", "角色乙不知道角色甲已获线索"], "foreshadows_planted": ["令牌背面的残缺纹路"], "foreshadows_resolved": []},
    "pacing_curve": {"setup": 2, "conflict": 3, "climax": 4, "resolution": 2, "ending_hook": "角色甲决定在夜间前往禁区"},
    "evidence": [{"field": "event_chain[0].trigger", "quote": "角色甲收到试炼通知"}],
}

CHAPTER_ANALYSIS_EXAMPLE = {key: CHAPTER_STATE_EXAMPLE[key] for key in CHAPTER_ANALYSIS_SCHEMA}

SCENE_CONTRACT_SCHEMA = {
    "entities": CHAPTER_STATE_SCHEMA["entities"],
    "scene_beats": CHAPTER_STATE_SCHEMA["scene_beats"],
}
ENTITY_SCHEMA = {"entities": CHAPTER_STATE_SCHEMA["entities"]}
SCENE_BEATS_SCHEMA = {"scene_beats": CHAPTER_STATE_SCHEMA["scene_beats"]}

SYSTEM_PROMPT = """你是章节叙事状态抽取器。只记录原章正文或给定前文状态能够支持的事实，不得猜测、补写或评价。所有字段值必须使用中文。"""


def _metrics(text: str) -> dict[str, Any]:
    dialogue = sum(len(piece) for piece in re.findall(r"[“\"].*?[”\"]", text))
    paragraphs = [item for item in text.splitlines() if item.strip()]
    return {"char_count": len(text), "paragraph_count": len(paragraphs), "dialogue_ratio": round(dialogue / max(len(text), 1), 4)}


def _offline_state(record: ChapterRecord) -> ChapterState:
    sentences = [item.strip() for item in re.split(r"[。！？!?]", record.text) if item.strip()]
    event = "；".join(sentences[:3])[:500]
    return ChapterState(
        chapter_id=record.chapter_id,
        chapter_function="离线基础提取：需由模型补全叙事功能",
        event_chain=[{"trigger": sentences[0][:160] if sentences else "", "action": event, "obstacle": "", "turn": "", "result": sentences[-1][:160] if sentences else ""}],
        scene_beats=[{"scene_no": 1, "time": "", "location": "", "participants": [], "objective": "", "key_actions": [event], "conflict": "", "turn": "", "result": sentences[-1][:160] if sentences else "", "evidence": event[:240]}] if event else [],
        prose_metrics=_metrics(record.text),
        evidence=[{"field": "event_chain", "quote": event[:240]}] if event else [],
        extraction_mode="offline",
    )


def _prompt(record: ChapterRecord, prior_state: dict[str, Any] | None, fragment_note: str = "") -> str:
    return f"""任务：从下方单章正文中抽取章节叙事状态。{fragment_note}

必须且只能包含这些顶层字段：{json.dumps(list(CHAPTER_ANALYSIS_SCHEMA), ensure_ascii=False)}。
字段结构必须符合以下 JSON schema：
{json.dumps(CHAPTER_ANALYSIS_SCHEMA, ensure_ascii=False, indent=2)}

下面是格式示例。示例中的“角色甲”“地点甲”“功法甲”等仅用于展示字段类型和粒度，绝对不得照抄到当前章节的输出中：
{json.dumps(CHAPTER_ANALYSIS_EXAMPLE, ensure_ascii=False, indent=2)}

缺失信息的填写规则：字符串填 \"\"；数组填 []；对象填 {{}}。不得删除字段、增加顶层字段或把不确定事实写成确定事实。
`evidence` 中的 quote 必须是原文短引文；无法提供证据的字段不要编造。
`pacing_curve` 的 setup、conflict、climax、resolution 必须是 0 到 5 的整数。
`event_chain` 必须按因果顺序写出 3 到 10 个可区分的事件节点，不能只写整章总述。
不要输出 entities 或 scene_beats；它们由下一阶段专门校对。不得编造年龄、外貌、贫富、物品、动机或未出现的对白。

章节标识：{record.chapter_id}
章节标题：{record.title}
前文状态（可能为空）：
{json.dumps(prior_state or {}, ensure_ascii=False)}

章节正文：
{record.text}"""


def _entity_prompt(record: ChapterRecord, prior_state: dict[str, Any] | None, draft_state: ChapterState) -> str:
    return f"""任务：只核对下方章节的实体表。这是一次事实校对，不是写作、总结或补全。

只能输出顶层字段：{json.dumps(list(ENTITY_SCHEMA), ensure_ascii=False)}。
输出必须符合此 JSON schema：
{json.dumps(ENTITY_SCHEMA, ensure_ascii=False, indent=2)}

硬性规则：
1. 实体使用稳定 entity_id；同一人物的本名、小名、称谓必须出现在同一实体的 aliases 中。若前文状态已有对应实体，必须沿用其 entity_id。
2. 只登记本章对状态转移必要的人物、地点和物品，不得虚构实体。
3. evidence 只能是一句不超过 80 字的原文短引文；不得包含换行、英文双引号或未转义反斜杠。
4. 只输出 JSON，不得解释。

前文实体上下文：
{json.dumps((prior_state or {}).get('entities', []), ensure_ascii=False)}

第一阶段草稿（仅供检查，不是事实来源）：
{json.dumps({"event_chain": draft_state.event_chain, "character_deltas": draft_state.character_deltas, "relation_deltas": draft_state.relation_deltas}, ensure_ascii=False)}

章节正文：
{record.text}"""


def _scene_beats_prompt(record: ChapterRecord, prior_state: dict[str, Any] | None, draft_state: ChapterState) -> str:
    return f"""任务：只核对下方章节的场景状态转移。实体表已经确定，场景只能引用其中的 entity_id。这是一次事实校对，不是写作、总结或补全。

只能输出顶层字段：{json.dumps(list(SCENE_BEATS_SCHEMA), ensure_ascii=False)}。
输出必须符合此 JSON schema：
{json.dumps(SCENE_BEATS_SCHEMA, ensure_ascii=False, indent=2)}

硬性规则：
1. 场景按正文顺序列出 2 到 6 个；相邻且没有状态变化的动作合并为一个场景。participants 和 knowledge_updates.entity 只能使用给定实体表中的 entity_id。
2. 每个场景至多 3 个 key_actions、至多 2 个 dialogue_intents、至多 2 个 preconditions、至多 2 条 knowledge_updates、至多 3 条 fact_provenance；每条普通文本不超过 45 个汉字，quote 不超过 60 个汉字。证据只需覆盖该场景最关键的行动、转折和状态变化，不能为每个细枝末节重复引文。
3. before_state 与 after_state 只保留本场景发生变化的 time、location、resources、relationships、knowledge；没有变化时对应值填 "" 或 []。不得把后面场景的地点、知识、资源、关系或结果提前写入前面场景；不得补造年龄、外貌、家境、具体物品、动机或对白。
4. quote 不得包含换行、英文双引号或未转义反斜杠。所有数组与对象都必须闭合；只输出 JSON，不得解释。

上一章尾部状态（可能为空；它是连续性约束，不是本章事实来源）：
{json.dumps({"chapter_id": (prior_state or {}).get("chapter_id", ""), "location": (prior_state or {}).get("location", ""), "cultivation": (prior_state or {}).get("cultivation", {})}, ensure_ascii=False)}

实体表：
{json.dumps(draft_state.entities, ensure_ascii=False)}

章节级事件草稿：
{json.dumps({"event_chain": draft_state.event_chain, "character_deltas": draft_state.character_deltas, "relation_deltas": draft_state.relation_deltas}, ensure_ascii=False)}

章节正文：
{record.text}"""


def _location_matches(left: str, right: str) -> bool:
    """Treat a named place and its contained sub-place as continuous."""
    left, right = left.strip(), right.strip()
    return not left or not right or left == right or left in right or right in left


def _has_movement(actions: Any) -> bool:
    action_text = " ".join(str(item) for item in actions if item).lower() if isinstance(actions, list) else ""
    return any(token in action_text for token in ("到", "赴", "前往", "进入", "离开", "返回", "移动", "赶往", "抵达", "来到", "出发", "上马车", "下山"))


def _scene_contract_issues(entities: list[dict[str, Any]], scenes: list[dict[str, Any]], prior_locations: dict[str, str] | None = None) -> dict[str, list[str]]:
    entity_ids = {str(item.get("entity_id", "")).strip() for item in entities if isinstance(item, dict)}
    aliases = {str(alias).strip() for item in entities if isinstance(item, dict) for alias in [item.get("canonical_name", ""), *(item.get("aliases", []) if isinstance(item.get("aliases", []), list) else [])]}
    unknown: list[str] = []
    missing_preconditions: list[str] = []
    space_discontinuities: list[str] = []
    empty_scene_contract: list[str] = ["正文非空但 scene_beats 为空，必须重新抽取场景。"]
    locations = dict(prior_locations or {})
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        number = str(scene.get("scene_no", "?"))
        if not scene.get("preconditions"):
            missing_preconditions.append(number)
        participants = scene.get("participants", []) if isinstance(scene.get("participants", []), list) else []
        before = scene.get("before_state", {}) if isinstance(scene.get("before_state"), dict) else {}
        after = scene.get("after_state", {}) if isinstance(scene.get("after_state"), dict) else {}
        before_location = str(before.get("location", "")).strip()
        after_location = str(after.get("location", "") or scene.get("location", "")).strip()
        moves = _has_movement(scene.get("key_actions", []))
        for participant in participants:
            value = str(participant).strip()
            if value and value not in entity_ids and value not in aliases and value not in unknown:
                unknown.append(value)
            previous = locations.get(value, "")
            if previous and before_location and not _location_matches(previous, before_location) and not moves:
                space_discontinuities.append(f"场景 {number}：实体 {value} 从“{previous}”跳到“{before_location}”，但 key_actions 未记录移动或抵达。")
            if value and after_location:
                locations[value] = after_location
    if scenes:
        empty_scene_contract = []
    return {"unknown_participants": unknown, "missing_preconditions": missing_preconditions, "space_discontinuities": space_discontinuities, "empty_scene_contract": empty_scene_contract}


def _entity_repair_prompt(record: ChapterRecord, entities: list[dict[str, Any]], unknown_participants: list[str]) -> str:
    return f"""任务：补全章节实体表，使其覆盖场景中已经出现但尚未登记的参与者。只输出合法 JSON object，符合：
{json.dumps(ENTITY_SCHEMA, ensure_ascii=False, indent=2)}

现有实体表：
{json.dumps(entities, ensure_ascii=False)}

待核对参与者名称：
{json.dumps(unknown_participants, ensure_ascii=False)}

规则：保留现有 entity_id；对每个待核对名称，若原文能确认是独立人物或地点，则添加稳定 entity_id、canonical_name、aliases、entity_type 与不超过 80 字的 evidence；若只是既有实体别名，则补入 aliases。不得输出未在原文出现的实体。

章节正文：
{record.text}"""


def _scene_repair_prompt(record: ChapterRecord, entities: list[dict[str, Any]], scenes: list[dict[str, Any]], issues: dict[str, list[str]], prior_locations: dict[str, str] | None = None) -> str:
    return f"""任务：修复章节场景状态机中的逻辑合同缺口。只输出合法 JSON object，符合：
{json.dumps(SCENE_BEATS_SCHEMA, ensure_ascii=False, indent=2)}

实体表（participants 和 knowledge_updates.entity 必须只使用其中 entity_id）：
{json.dumps(entities, ensure_ascii=False)}

当前场景草稿：
{json.dumps(scenes, ensure_ascii=False)}

检测到的问题：
{json.dumps(issues, ensure_ascii=False)}

上一章末尾已知地点（可能为空；只用于检查转场）：
{json.dumps(prior_locations or {}, ensure_ascii=False)}

修复规则：
1. 每个场景必须有至少一个 preconditions，且每项都由原文支持。每个场景保持 2 到 6 个，普通文本不超过 45 个汉字，quote 不超过 60 个汉字。
2. 若地点从上一场变化，key_actions 和 fact_provenance 必须显式记录移动或抵达。
3. 每个 participants 只能写 entity_id；不得用别名、称谓或新名字。
4. 不得改变已被原文支持的事件顺序，不得补造事实。quote 不得含换行、英文双引号或未转义反斜杠。

章节正文：
{record.text}"""


def _merge_entities(existing: list[dict[str, Any]], repaired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for entity in [*existing, *repaired]:
        if isinstance(entity, dict) and str(entity.get("entity_id", "")).strip():
            merged[str(entity["entity_id"]).strip()] = entity
    return list(merged.values())


def _normalise_entity_ids(state: ChapterState, known_entities: dict[str, dict[str, Any]]) -> None:
    """Reuse an existing id when a model repeats an entity under a new id."""
    aliases: dict[str, str] = {}
    for entity_id, entity in known_entities.items():
        for name in [entity.get("canonical_name", ""), *(entity.get("aliases", []) if isinstance(entity.get("aliases", []), list) else [])]:
            if str(name).strip():
                aliases[str(name).strip()] = entity_id
    remap: dict[str, str] = {}
    merged: dict[str, dict[str, Any]] = {}
    for entity in state.entities:
        if not isinstance(entity, dict):
            continue
        raw_id = str(entity.get("entity_id", "")).strip()
        names = [entity.get("canonical_name", ""), *(entity.get("aliases", []) if isinstance(entity.get("aliases", []), list) else [])]
        stable_id = next((aliases[str(name).strip()] for name in names if str(name).strip() in aliases), raw_id)
        if not stable_id:
            continue
        remap[raw_id] = stable_id
        current = {**entity, "entity_id": stable_id}
        if stable_id in merged:
            prior = merged[stable_id]
            current["aliases"] = list(dict.fromkeys([*(prior.get("aliases", []) if isinstance(prior.get("aliases", []), list) else []), *(current.get("aliases", []) if isinstance(current.get("aliases", []), list) else [])]))
            current["canonical_name"] = prior.get("canonical_name") or current.get("canonical_name", "")
        merged[stable_id] = current

    def resolve(value: Any) -> Any:
        text = str(value).strip()
        return remap.get(text, aliases.get(text, value)) if text else value

    state.entities = list(merged.values())
    for scene in state.scene_beats:
        if isinstance(scene, dict):
            scene["participants"] = [resolve(item) for item in scene.get("participants", [])] if isinstance(scene.get("participants", []), list) else []
            for update in scene.get("knowledge_updates", []) if isinstance(scene.get("knowledge_updates", []), list) else []:
                if isinstance(update, dict):
                    update["entity"] = resolve(update.get("entity", ""))
    for delta in state.character_deltas:
        if isinstance(delta, dict) and delta.get("entity_id"):
            delta["entity_id"] = resolve(delta["entity_id"])
    for delta in state.relation_deltas:
        if isinstance(delta, dict):
            for field in ("source_id", "target_id"):
                if delta.get(field):
                    delta[field] = resolve(delta[field])


def extract_record(record: ChapterRecord, prior_state: dict[str, Any] | None = None, offline: bool = False, fragment_note: str = "") -> ChapterState:
    if offline:
        return _offline_state(record)
    payload = complete_json(SYSTEM_PROMPT, _prompt(record, prior_state, fragment_note), ModelSettings.from_environment(), max_tokens=4_000)
    state = ChapterState.from_dict(payload, record.chapter_id)
    entity_contract = complete_json(
        "你是小说实体表校对器。只记录原文明确支持的实体与别名；所有字段值必须使用中文。",
        _entity_prompt(record, prior_state, state),
        ModelSettings.from_environment(),
        max_tokens=2_500,
    )
    state.entities = entity_contract.get("entities", []) if isinstance(entity_contract.get("entities"), list) else []
    scene_contract = complete_json(
        "你是小说场景状态机校对器。只记录原文明确支持的场景前置条件和状态变化；所有字段值必须使用中文。",
        _scene_beats_prompt(record, prior_state, state),
        ModelSettings.from_environment(),
        max_tokens=5_000,
    )
    state.scene_beats = scene_contract.get("scene_beats", []) if isinstance(scene_contract.get("scene_beats"), list) else []
    prior_locations = (prior_state or {}).get("locations", {}) if isinstance((prior_state or {}).get("locations", {}), dict) else {}
    issues = _scene_contract_issues(state.entities, state.scene_beats, prior_locations)
    if issues["unknown_participants"]:
        repaired_entities = complete_json(
            "你是小说实体表修复器。所有字段值必须使用中文，只能依据原文补充实体或别名。",
            _entity_repair_prompt(record, state.entities, issues["unknown_participants"]),
            ModelSettings.from_environment(),
            max_tokens=2_500,
        )
        state.entities = _merge_entities(state.entities, repaired_entities.get("entities", []) if isinstance(repaired_entities.get("entities"), list) else [])
        issues = _scene_contract_issues(state.entities, state.scene_beats, prior_locations)
    if issues["unknown_participants"] or issues["missing_preconditions"] or issues["space_discontinuities"] or issues["empty_scene_contract"]:
        repaired_scenes = complete_json(
            "你是小说场景状态机修复器。所有字段值必须使用中文，只能依据原文修复场景前置条件、实体引用与状态连续性。",
            _scene_repair_prompt(record, state.entities, state.scene_beats, issues, prior_locations),
            ModelSettings.from_environment(),
            max_tokens=5_000,
        )
        if isinstance(repaired_scenes.get("scene_beats"), list) and repaired_scenes["scene_beats"]:
            state.scene_beats = repaired_scenes["scene_beats"]
    state.prose_metrics = {**_metrics(record.text), **state.prose_metrics}
    state.extraction_mode = "model"
    return state


def split_chapter_text(text: str, max_input_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_input_chars:
        return [text]
    max_input_chars = max(1_000, int(max_input_chars))
    overlap_chars = max(0, min(int(overlap_chars), max_input_chars // 3))
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    if not paragraphs:
        paragraphs = [text]
    fragments: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [paragraph[index:index + max_input_chars] for index in range(0, len(paragraph), max_input_chars)] or [paragraph]
        for piece in pieces:
            candidate = f"{current}\n{piece}".strip() if current else piece
            if current and len(candidate) > max_input_chars:
                fragments.append(current)
                prefix = current[-overlap_chars:] if overlap_chars else ""
                current = f"{prefix}\n{piece}".strip()
            else:
                current = candidate
    if current:
        fragments.append(current)
    return fragments


def select_chapter_records(records: list[ChapterRecord], limit: int | None, selection: str = "sequential") -> list[ChapterRecord]:
    if limit is None or limit >= len(records):
        return records
    if limit <= 0:
        raise ValueError("limit must be positive")
    if selection not in SAMPLE_SELECTIONS:
        raise ValueError(f"selection must be one of: {', '.join(SAMPLE_SELECTIONS)}")
    if selection == "sequential":
        return records[:limit]
    if limit == 1:
        return [records[0]]
    indexes = sorted({round(index * (len(records) - 1) / (limit - 1)) for index in range(limit)})
    return [records[index] for index in indexes]


def _dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _merge_offline_states(chapter_id: str, states: list[ChapterState], mode: str) -> ChapterState:
    def merge_dicts(name: str) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for state in states:
            value = getattr(state, name)
            if isinstance(value, dict):
                for key, item in value.items():
                    if item not in (None, "", [], {}):
                        merged[key] = item
        return merged

    functions: list[str] = []
    for state in states:
        value = state.chapter_function.strip()
        if value and value not in functions:
            functions.append(value)

    return ChapterState(
        chapter_id=chapter_id,
        chapter_function="；".join(functions)[:600],
        entities=_dedupe_rows([item for state in states for item in state.entities]),
        event_chain=_dedupe_rows([event for state in states for event in state.event_chain]),
        scene_beats=_dedupe_rows([item for state in states for item in state.scene_beats]),
        character_deltas=_dedupe_rows([item for state in states for item in state.character_deltas]),
        relation_deltas=_dedupe_rows([item for state in states for item in state.relation_deltas]),
        temporal_spatial_state=merge_dicts("temporal_spatial_state"),
        cultivation_state=merge_dicts("cultivation_state"),
        information_state=merge_dicts("information_state"),
        pacing_curve=merge_dicts("pacing_curve"),
        prose_metrics=merge_dicts("prose_metrics"),
        evidence=_dedupe_rows([item for state in states for item in state.evidence]),
        extraction_mode=mode,
    )


def _merge_state_group(chapter_id: str, states: list[ChapterState], offline: bool) -> ChapterState:
    if len(states) == 1:
        return states[0]
    if offline:
        return _merge_offline_states(chapter_id, states, "offline-chunked")
    partials = [state.to_dict() for state in states]
    prompt = f"""任务：合并同一章节相邻片段的状态抽取结果，输出该完整章节的唯一状态。

必须且只能包含这些顶层字段：{json.dumps(list(CHAPTER_STATE_SCHEMA), ensure_ascii=False)}。
输出结构必须符合：
{json.dumps(CHAPTER_STATE_SCHEMA, ensure_ascii=False, indent=2)}

去除因片段重叠造成的重复事件和证据；不得新增 partial states 中没有支持的信息。若片段信息相互矛盾，在 evidence 中保留冲突证据，并采用更具体的描述。

待合并的片段状态：
{json.dumps(partials, ensure_ascii=False)}"""
    payload = complete_json("你是章节状态合并器。所有字段值必须使用中文，只依据输入的片段状态合并事实。", prompt, ModelSettings.from_environment())
    state = ChapterState.from_dict(payload, chapter_id)
    state.extraction_mode = "model-chunked"
    return state


def extract_chapter(record: ChapterRecord, prior_state: dict[str, Any] | None, offline: bool, max_input_chars: int, overlap_chars: int) -> ChapterState:
    fragments = split_chapter_text(record.text, max_input_chars, overlap_chars)
    if len(fragments) == 1:
        return extract_record(record, prior_state, offline)
    partials: list[ChapterState] = []
    for index, fragment in enumerate(fragments, start=1):
        fragment_record = replace(record, chapter_id=f"{record.chapter_id}:fragment-{index}", text=fragment, char_count=len(fragment))
        note = f"这是完整章节的第 {index}/{len(fragments)} 个连续片段。只抽取此片段可证实的内容，不要假定后续片段发生的事件。"
        partials.append(extract_record(fragment_record, prior_state, offline, note))
    while len(partials) > 1:
        partials = [_merge_state_group(record.chapter_id, partials[index:index + DEFAULT_MERGE_BATCH_SIZE], offline) for index in range(0, len(partials), DEFAULT_MERGE_BATCH_SIZE)]
    state = partials[0]
    state.chapter_id = record.chapter_id
    state.prose_metrics = _metrics(record.text)
    return state


def extract_work(author_id: str, work_id: str, offline: bool = False, limit: int | None = None, batch_size: int = DEFAULT_BATCH_SIZE, max_input_chars: int = DEFAULT_MAX_INPUT_CHARS, overlap_chars: int = DEFAULT_OVERLAP_CHARS, resume: bool = True, selection: str = "sequential") -> Path:
    folder = corpus_dir(author_id, work_id)
    records = [ChapterRecord(**row) for row in read_jsonl(folder / "chapters.jsonl")]
    records = select_chapter_records(records, limit, selection)
    suffix = "" if selection == "sequential" else f".{selection}"
    target = folder / ("chapter_states.jsonl" if limit is None else f"chapter_states.sample-{limit}{suffix}.jsonl")
    if limit is not None and resume:
        full_states = {row.get("chapter_id"): row for row in read_jsonl(folder / "chapter_states.jsonl")}
        wanted_ids = [record.chapter_id for record in records]
        if all(chapter_id in full_states for chapter_id in wanted_ids):
            write_jsonl(target, [full_states[chapter_id] for chapter_id in wanted_ids])
            print(f"[extract] {work_id}: reused {len(wanted_ids)} completed chapter states for sample {limit}", flush=True)
            return target
    completed = {row.get("chapter_id"): row for row in read_jsonl(target)} if resume else {}
    states: dict[str, dict[str, Any]] = {chapter_id: row for chapter_id, row in completed.items() if chapter_id}
    batch_size = max(1, int(batch_size))
    prior: dict[str, Any] | None = None
    entity_context: dict[str, dict[str, Any]] = {}
    entity_locations: dict[str, str] = {}

    def update_entity_context(items: Any) -> None:
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict) and str(item.get("entity_id", "")).strip():
                entity_context[str(item["entity_id"]).strip()] = item

    def update_entity_locations(state: ChapterState) -> None:
        for scene in state.scene_beats:
            if not isinstance(scene, dict):
                continue
            after = scene.get("after_state", {}) if isinstance(scene.get("after_state"), dict) else {}
            location = str(after.get("location", "") or scene.get("location", "")).strip()
            if location:
                for participant in scene.get("participants", []) if isinstance(scene.get("participants", []), list) else []:
                    entity_id = str(participant).strip()
                    if entity_id:
                        entity_locations[entity_id] = location

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        for record in batch:
            existing = states.get(record.chapter_id)
            if existing:
                update_entity_context(existing.get("entities", []))
                existing_state = ChapterState.from_dict(existing, record.chapter_id)
                update_entity_locations(existing_state)
                prior = {"chapter_id": existing["chapter_id"], "chapter_function": existing.get("chapter_function", ""), "cultivation": existing.get("cultivation_state", {}), "location": existing.get("temporal_spatial_state", {}).get("location", ""), "locations": dict(entity_locations), "entities": list(entity_context.values())}
                continue
            state = extract_chapter(record, prior, offline, max_input_chars, overlap_chars)
            _normalise_entity_ids(state, entity_context)
            states[record.chapter_id] = state.to_dict()
            update_entity_context(state.entities)
            update_entity_locations(state)
            prior = {"chapter_id": state.chapter_id, "chapter_function": state.chapter_function, "cultivation": state.cultivation_state, "location": state.temporal_spatial_state.get("location", ""), "locations": dict(entity_locations), "entities": list(entity_context.values())}
        ordered = [states[record.chapter_id] for record in records if record.chapter_id in states]
        write_jsonl(target, ordered)
        print(f"[extract] {work_id}: {min(start + len(batch), len(records))}/{len(records)}，已完成 {len(states)} 章", flush=True)
    return target
