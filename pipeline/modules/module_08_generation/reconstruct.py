from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

from .jsonio import read_jsonl, write_json
from .model import ModelSettings, complete_text
from .paths import runs_dir


PROMPT_VERSION = "reconstruction.v4-state-machine"


def _plan_payload(chapter_state: dict[str, Any], prior_story_state: dict[str, Any], scene_ledger: list[dict[str, Any]], target_char_count: int) -> dict[str, Any]:
    events = chapter_state.get("event_chain", [])
    beats = []
    for index, event in enumerate(events, start=1):
        beats.append({"beat": index, "purpose": event.get("action", ""), "conflict": event.get("obstacle", ""), "turn": event.get("turn", ""), "outcome": event.get("result", "")})
    return {
        "chapter_id": chapter_state["chapter_id"], "chapter_function": chapter_state.get("chapter_function", ""),
        "beats": beats,
        "entities": chapter_state.get("entities", []),
        "scenes": chapter_state.get("scene_beats", []),
        "scene_transition_ledger": scene_ledger,
        "target_char_count": target_char_count,
        "prompt_version": PROMPT_VERSION,
        "constraints": {"cultivation": chapter_state.get("cultivation_state", {}), "information": chapter_state.get("information_state", {}), "prior_state": prior_story_state},
    }


def _offline_draft(plan: dict[str, Any]) -> str:
    lines = [f"# Reconstruction preview: {plan['chapter_id']}", ""]
    for scene in plan.get("scenes", []):
        lines.append(f"- 场景 {scene.get('scene_no', '')}：{scene.get('location', '')}；目标：{scene.get('objective', '')}；动作：{scene.get('key_actions', [])}；结果：{scene.get('result', '')}")
    if not plan.get("scenes"):
        for beat in plan["beats"]:
            lines.append(f"- 场景 {beat['beat']}：{beat['purpose']}；阻碍：{beat['conflict']}；转折：{beat['turn']}；结果：{beat['outcome']}")
    return "\n".join(lines)


def reconstruct(author_id: str, work_id: str, chapter_no: int, run_id: str, offline: bool = False, state_filename: str = "chapter_states.jsonl") -> Path:
    from .paths import corpus_dir
    folder = corpus_dir(author_id, work_id)
    if Path(state_filename).name != state_filename:
        raise ValueError("state filename must not contain a directory")
    states = read_jsonl(folder / state_filename)
    story_states = read_jsonl(folder / state_filename.replace("chapter_states", "story_states", 1))
    scene_ledger = read_jsonl(folder / state_filename.replace("chapter_states", "scene_transitions", 1))
    chapter_index = next((index for index, row in enumerate(states) if row["chapter_id"].endswith(f"/{chapter_no:04d}")), -1)
    chapter_state = states[chapter_index] if chapter_index >= 0 else None
    if not chapter_state:
        raise ValueError(f"No extracted state for chapter {chapter_no}")
    prior_story_state = story_states[chapter_index - 1] if chapter_index > 0 and chapter_index - 1 < len(story_states) else {}
    records = read_jsonl(folder / "chapters.jsonl")
    source_record = next((row for row in records if row["chapter_id"] == chapter_state["chapter_id"]), {})
    chapter_ledger = [row for row in scene_ledger if row.get("chapter_id") == chapter_state["chapter_id"]]
    plan = _plan_payload(chapter_state, prior_story_state, chapter_ledger, int(source_record.get("char_count", 0)))
    destination = runs_dir(author_id, run_id) / work_id
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "reconstruction_plan.json", plan)
    if offline:
        draft = _offline_draft(plan)
    else:
        system = """你是网络小说章节复现写作器。根据给定的结构化计划写出连续的中文小说正文。不要写提纲、解释、标题、Markdown 或 JSON；只输出章节正文。必须按 `scene_transition_ledger` 和 `scenes` 的顺序覆盖所有场景节拍。每个场景必须先满足 preconditions，随后只产生 declared_after_state 中允许的变化；不得把后续场景的知识、资源、关系或地点提前写入前面场景。实体只可使用 entities 与 prior_state 中已有的 entity_id、canonical_name 或 aliases，不能新增人物或替换身份。场景证据之外的具体年龄、外貌、家境、道具、经历、动机和对白一律不得补造；信息不足时采用中性表述。不得添加与计划矛盾的事实。"""
        target = plan["target_char_count"]
        user = f"""结构化复现计划：
{json.dumps(plan, ensure_ascii=False, indent=2)}

目标篇幅：{target} 个汉字。正文长度应介于 {int(target * 0.75)} 到 {int(target * 1.15)} 个汉字之间。不要在第一个节拍后总结或提前结束；使用自然段、对白和动作推进完整章节。"""
        max_continuations = max(0, int(os.environ.get("DISTILL_MAX_CONTINUATIONS", "1")))
        write_json(destination / "reconstruction_request.json", {"prompt_version": PROMPT_VERSION, "system": system, "user": user, "max_continuations": max_continuations})
        prose_max_tokens = min(12_000, max(4_000, int(target * 1.3)))
        draft = complete_text(system, user, ModelSettings.from_environment(), max_tokens=prose_max_tokens)
        minimum = int(target * 0.75)
        for continuation_index in range(max_continuations):
            if not minimum or len(draft) >= minimum:
                break
            addition = complete_text(
                "你是网络小说续写器。直接续写给定正文，保持人物、时空、修炼规则和叙事语气连续。不要复述已有内容，不要总结，不要解释，不要输出标题、Markdown 或 JSON；只输出新增正文。",
                f"""原始章节计划：
{json.dumps(plan, ensure_ascii=False, indent=2)}

已有正文：
{draft}

请续写尚未覆盖的节拍，使合并后的正文至少达到 {minimum} 个汉字。""",
                ModelSettings.from_environment(),
                max_tokens=prose_max_tokens,
            )
            if not addition:
                break
            draft = f"{draft}\n\n{addition}"
    draft_path = destination / "reconstructed_chapter.md"
    draft_path.write_text(draft, encoding="utf-8")
    return draft_path
