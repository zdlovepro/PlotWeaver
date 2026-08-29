from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .jsonio import read_json, write_json
from .model import ModelSettings, complete_text
from .paths import output_dir, runs_dir
from .templates import select_templates


def _offline_preview(brief: dict[str, Any], profile: dict[str, Any]) -> str:
    goal = str(brief.get("chapter_goal", ""))
    world = brief.get("world", {})
    characters = brief.get("characters", [])
    return "\n".join([
        "# Original chapter preview", "", f"Goal: {goal}",
        f"World: {world}", f"Characters: {characters}",
        f"Profile version: {profile.get('profile_version', '')}",
    ])


def generate_original(author_id: str, brief_path: Path, run_id: str, offline: bool = False) -> Path:
    brief = read_json(brief_path, {})
    if not isinstance(brief, dict) or not str(brief.get("chapter_goal", "")).strip():
        raise ValueError("brief JSON must contain a non-empty chapter_goal")
    profile = read_json(output_dir(author_id) / "author_narrative_profile.json", {})
    if not profile:
        raise ValueError("author profile is missing; run distill first")
    template_library = read_json(output_dir(author_id) / "template_library.json", {})
    if not template_library:
        raise ValueError("template library is missing; run template mining first")
    selected_templates = select_templates(template_library, brief)
    destination = runs_dir(author_id, run_id) / "original"
    destination.mkdir(parents=True, exist_ok=True)
    request = {"author_profile": profile, "selected_templates": selected_templates, "original_brief": brief, "mode": "original"}
    write_json(destination / "original_request.json", request)
    if offline:
        draft = _offline_preview(brief, profile)
    else:
        system = """你是原创网络小说章节写作器。根据作者叙事 profile、选中的抽象模板与全新的章节 brief 写出连续中文正文。只输出正文，不要输出标题、提纲、解释、Markdown 或 JSON。选中模板只能提供角色槽位、事件节拍、压力阶梯和变体轴；所有人物、世界观、专名和具体事件必须来自新 brief，不能复用来源作品的具体实体、场景或情节。"""
        draft = complete_text(system, f"""作者叙事 profile：
{json.dumps(profile, ensure_ascii=False, indent=2)}

按当前原创 brief 选中的抽象模板：
{json.dumps(selected_templates, ensure_ascii=False, indent=2)}

新的原创章节 brief：
{json.dumps(brief, ensure_ascii=False, indent=2)}""", ModelSettings.from_environment())
    target = destination / "original_chapter.md"
    target.write_text(draft, encoding="utf-8")
    return target
