from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .contracts import StoryState
from .jsonio import read_jsonl, write_json, write_jsonl
from .paths import corpus_dir


def _relation_key(source: str, target: str) -> str:
    return f"{source} -> {target}"


def _story_filename(state_filename: str) -> str:
    return state_filename.replace("chapter_states", "story_states", 1)


def _scene_ledger_filename(state_filename: str) -> str:
    return state_filename.replace("chapter_states", "scene_transitions", 1)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _entity_id(entity: dict[str, Any]) -> str:
    return _clean(entity.get("entity_id"))


def _register_entities(current: StoryState, entities: Any) -> None:
    for raw in entities if isinstance(entities, list) else []:
        if not isinstance(raw, dict):
            continue
        entity_id = _entity_id(raw)
        canonical_name = _clean(raw.get("canonical_name"))
        if not entity_id or not canonical_name:
            continue
        aliases = list(dict.fromkeys([canonical_name, *[_clean(alias) for alias in raw.get("aliases", []) if _clean(alias)]]))
        current.entity_registry[entity_id] = {
            "entity_id": entity_id,
            "canonical_name": canonical_name,
            "aliases": aliases,
            "entity_type": _clean(raw.get("entity_type")),
            "evidence": _clean(raw.get("evidence")),
        }
        for alias in aliases:
            current.alias_index[alias] = entity_id


def _resolve_entity(reference: Any, current: StoryState) -> str:
    value = _clean(reference)
    if not value:
        return ""
    if value in current.entity_registry:
        return value
    return current.alias_index.get(value, value)


def _entity_label(entity_id: str, current: StoryState) -> str:
    return _clean(current.entity_registry.get(entity_id, {}).get("canonical_name")) or entity_id


def _scene_dict(scene: Any) -> dict[str, Any]:
    return scene if isinstance(scene, dict) else {}


def _scene_entry(chapter_id: str, scene: dict[str, Any], current: StoryState) -> dict[str, Any]:
    participants = [_resolve_entity(item, current) for item in scene.get("participants", [])]
    participants = [item for item in participants if item]
    before = scene.get("before_state", {}) if isinstance(scene.get("before_state"), dict) else {}
    after = scene.get("after_state", {}) if isinstance(scene.get("after_state"), dict) else {}
    carried_before = {
        "time": current.temporal_anchor,
        "locations": {entity_id: current.locations.get(entity_id, "") for entity_id in participants},
        "knowledge": {entity_id: current.knowledge.get(entity_id, []) for entity_id in participants},
    }
    return {
        "scene_id": f"{chapter_id}:scene-{int(scene.get('scene_no', 0) or 0):02d}",
        "chapter_id": chapter_id,
        "scene_no": int(scene.get("scene_no", 0) or 0),
        "time": _clean(scene.get("time")),
        "location": _clean(scene.get("location")),
        "participants": participants,
        "participant_labels": [_entity_label(entity_id, current) for entity_id in participants],
        "objective": _clean(scene.get("objective")),
        "key_actions": scene.get("key_actions", []) if isinstance(scene.get("key_actions"), list) else [],
        "dialogue_intents": scene.get("dialogue_intents", []) if isinstance(scene.get("dialogue_intents"), list) else [],
        "preconditions": scene.get("preconditions", []) if isinstance(scene.get("preconditions"), list) else [],
        "declared_before_state": before,
        "declared_after_state": after,
        "carried_before_state": carried_before,
        "knowledge_updates": scene.get("knowledge_updates", []) if isinstance(scene.get("knowledge_updates"), list) else [],
        "fact_provenance": scene.get("fact_provenance", []) if isinstance(scene.get("fact_provenance"), list) else [],
        "result": _clean(scene.get("result")),
    }


def _apply_scene_state(current: StoryState, entry: dict[str, Any]) -> None:
    after = entry.get("declared_after_state", {})
    after_time = _clean(after.get("time")) or _clean(entry.get("time"))
    after_location = _clean(after.get("location")) or _clean(entry.get("location"))
    if after_time:
        current.temporal_anchor = after_time
    if after_location:
        for entity_id in entry.get("participants", []):
            current.locations[entity_id] = after_location
    for update in entry.get("knowledge_updates", []):
        if not isinstance(update, dict):
            continue
        entity_id = _resolve_entity(update.get("entity"), current)
        learned = _clean(update.get("learns"))
        if entity_id and learned:
            current.knowledge[entity_id] = list(dict.fromkeys([*current.knowledge.get(entity_id, []), learned]))
    current.last_scene = {
        "scene_id": entry["scene_id"],
        "time": current.temporal_anchor,
        "location": after_location,
        "participants": entry.get("participants", []),
    }


def build_story_states(author_id: str, work_id: str, state_filename: str = "chapter_states.jsonl") -> Path:
    folder = corpus_dir(author_id, work_id)
    if Path(state_filename).name != state_filename:
        raise ValueError("state filename must not contain a directory")
    chapter_states = read_jsonl(folder / state_filename)
    current = StoryState(chapter_id="")
    output: list[dict[str, Any]] = []
    scene_ledger: list[dict[str, Any]] = []
    for chapter in chapter_states:
        current = deepcopy(current)
        current.chapter_id = chapter["chapter_id"]
        _register_entities(current, chapter.get("entities", []))
        for scene in sorted((_scene_dict(item) for item in chapter.get("scene_beats", [])), key=lambda item: int(item.get("scene_no", 0) or 0)):
            entry = _scene_entry(current.chapter_id, scene, current)
            scene_ledger.append(entry)
            _apply_scene_state(current, entry)
        for delta in chapter.get("character_deltas", []):
            entity_id = _resolve_entity(delta.get("entity_id") or delta.get("character"), current)
            if entity_id:
                current.characters[entity_id] = {**current.characters.get(entity_id, {}), **delta, "entity_id": entity_id, "canonical_name": _entity_label(entity_id, current)}
        for delta in chapter.get("relation_deltas", []):
            source = _resolve_entity(delta.get("source_id") or delta.get("source"), current)
            target = _resolve_entity(delta.get("target_id") or delta.get("target"), current)
            if source and target:
                current.relations[_relation_key(source, target)] = {**delta, "source_id": source, "target_id": target, "source": _entity_label(source, current), "target": _entity_label(target, current)}
        space = chapter.get("temporal_spatial_state", {})
        current.temporal_anchor = _clean(space.get("time_anchor")) or current.temporal_anchor
        location = _clean(space.get("location"))
        if location:
            for delta in chapter.get("character_deltas", []):
                entity_id = _resolve_entity(delta.get("entity_id") or delta.get("character"), current)
                if entity_id:
                    current.locations[entity_id] = location
        cultivation = chapter.get("cultivation_state", {})
        if cultivation:
            current.cultivation.update(cultivation)
        info = chapter.get("information_state", {})
        current.open_foreshadows.extend(info.get("foreshadows_planted", []))
        resolved = {str(item) for item in info.get("foreshadows_resolved", [])}
        if resolved:
            current.open_foreshadows = [item for item in current.open_foreshadows if str(item) not in resolved]
        output.append(current.to_dict())
    target = folder / _story_filename(state_filename)
    write_jsonl(target, output)
    write_jsonl(folder / _scene_ledger_filename(state_filename), scene_ledger)
    return target


def _issue(issues: list[dict[str, str]], chapter_id: str, severity: str, rule: str, message: str) -> None:
    issues.append({"chapter_id": chapter_id, "severity": severity, "rule": rule, "message": message})


def validate_story_states(author_id: str, work_id: str, state_filename: str = "chapter_states.jsonl") -> Path:
    """Validate state transitions without repairing or silently changing facts."""
    folder = corpus_dir(author_id, work_id)
    story_path = folder / _story_filename(state_filename)
    ledger_path = folder / _scene_ledger_filename(state_filename)
    chapter_states = read_jsonl(folder / state_filename)
    story_states = read_jsonl(story_path)
    ledger = read_jsonl(ledger_path)
    if len(chapter_states) != len(story_states):
        raise ValueError("chapter-state and story-state counts differ; rebuild story states first")
    issues: list[dict[str, str]] = []
    known_entities: dict[str, str] = {}
    aliases: dict[str, str] = {}
    last_locations: dict[str, str] = {}
    for chapter, story in zip(chapter_states, story_states):
        chapter_id = _clean(chapter.get("chapter_id"))
        if chapter_id != story.get("chapter_id"):
            _issue(issues, chapter_id, "error", "chapter_alignment", "章节状态与连续状态的 chapter_id 不一致。")
        if not chapter.get("scene_beats"):
            _issue(issues, chapter_id, "error", "missing_scenes", "非空章节没有可用于状态转移的场景。")
        for entity in chapter.get("entities", []):
            if not isinstance(entity, dict):
                continue
            entity_id, canonical = _entity_id(entity), _clean(entity.get("canonical_name"))
            if not entity_id or not canonical:
                _issue(issues, chapter_id, "error", "entity_identity", "实体缺少 entity_id 或 canonical_name。")
                continue
            known_entities[entity_id] = canonical
            for alias in [canonical, *[_clean(item) for item in entity.get("aliases", [])]]:
                if not alias:
                    continue
                previous = aliases.get(alias)
                if previous and previous != entity_id:
                    _issue(issues, chapter_id, "error", "alias_collision", f"别名“{alias}”同时指向 {previous} 与 {entity_id}。")
                aliases[alias] = entity_id
        for relation in chapter.get("relation_deltas", []):
            source = _clean(relation.get("source_id") or relation.get("source"))
            target = _clean(relation.get("target_id") or relation.get("target"))
            if source and source == target:
                _issue(issues, chapter_id, "warning", "self_relation", f"实体 {source} 出现自指关系。")
        pacing = chapter.get("pacing_curve", {})
        if isinstance(pacing, dict):
            for key in ("setup", "conflict", "climax", "resolution"):
                value = pacing.get(key)
                if value not in (None, "") and (not isinstance(value, int) or not 0 <= value <= 5):
                    _issue(issues, chapter_id, "warning", "pacing_range", f"{key} 必须是 0 到 5 的整数。")
    for scene in ledger:
        chapter_id = _clean(scene.get("chapter_id"))
        participants = [_clean(item) for item in scene.get("participants", [])]
        for entity_id in participants:
            if entity_id not in known_entities:
                _issue(issues, chapter_id, "error", "unknown_scene_entity", f"场景参与者 {entity_id} 未在实体表中声明。")
        if scene.get("key_actions") and not scene.get("preconditions"):
            _issue(issues, chapter_id, "warning", "missing_preconditions", "场景有行动但没有前置条件。")
        if (scene.get("key_actions") or scene.get("result")) and not scene.get("fact_provenance"):
            _issue(issues, chapter_id, "warning", "missing_provenance", "场景行动或结果没有原文证据。")
        before = scene.get("declared_before_state", {}) if isinstance(scene.get("declared_before_state"), dict) else {}
        after = scene.get("declared_after_state", {}) if isinstance(scene.get("declared_after_state"), dict) else {}
        before_location = _clean(before.get("location"))
        after_location = _clean(after.get("location")) or _clean(scene.get("location"))
        moves = " ".join(str(item) for item in scene.get("key_actions", [])).lower()
        for entity_id in participants:
            previous_location = last_locations.get(entity_id, "")
            if previous_location and before_location and previous_location != before_location and not any(token in moves for token in ("到", "赴", "前往", "进入", "离开", "返回", "移动")):
                _issue(issues, chapter_id, "warning", "space_discontinuity", f"实体 {entity_id} 的地点从“{previous_location}”跳到“{before_location}”，但没有移动动作。")
            if after_location:
                last_locations[entity_id] = after_location
        for update in scene.get("knowledge_updates", []):
            if not isinstance(update, dict):
                continue
            entity_id, learned = _clean(update.get("entity")), _clean(update.get("learns"))
            if entity_id and entity_id not in known_entities:
                _issue(issues, chapter_id, "error", "unknown_knowledge_entity", f"知识更新引用未声明实体 {entity_id}。")
            if learned and not _clean(update.get("evidence")):
                _issue(issues, chapter_id, "warning", "knowledge_without_evidence", f"实体 {entity_id} 获得知识但没有证据。")
    report = {
        "work_id": work_id,
        "state_file": state_filename,
        "chapter_count": len(chapter_states),
        "scene_count": len(ledger),
        "issue_count": len(issues),
        "issues": issues,
    }
    target = folder / _story_filename(state_filename).replace(".jsonl", ".continuity_report.json")
    write_json(target, report)
    return target
