from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .jsonio import read_json, write_json
from .paths import output_dir


CHAPTER_PLAN_EXAMPLE = {
    "chapter_goal": "让主角在受限条件下取得行动进展，并在结尾制造下一章压力",
    "selected_template_ids": ["macro-001", "event-003", "micro-006"],
    "scenes": [{"location": "场景名", "time": "时间锚点", "purpose": "场景目的", "conflict": "阻力", "turn": "转折", "exit_hook": "离场钩子"}],
    "continuity_checks": ["人物目标与资源不矛盾", "时间、空间和关系状态连续"],
}


def _rules(profile: dict[str, Any], key: str) -> str:
    rules = profile.get("prompt_engineering", {}).get(key, [])
    return "\n".join(f"- {rule}" for rule in rules if str(rule).strip()) or "- 以当前叙事状态和章节目标为唯一事实来源。"


def _feedback_rules(author_id: str, limit: int = 24) -> list[str]:
    root = output_dir(author_id).parents[1] / "runs" / author_id
    rules: list[str] = []
    if not root.exists():
        return rules
    reports = sorted(root.rglob("fidelity_report.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for report_path in reports:
        report = read_json(report_path, {})
        judgment = report.get("semantic_judgment", {}) if isinstance(report, dict) else {}
        for rule in judgment.get("priority_repairs", []) if isinstance(judgment, dict) else []:
            text = str(rule).strip()
            if text and text not in rules:
                rules.append(text)
        for violation in judgment.get("logic_violations", []) if isinstance(judgment, dict) else []:
            if isinstance(violation, dict):
                text = str(violation.get("description", "")).strip()
                if text and text not in rules:
                    rules.append(text)
        if len(rules) >= limit:
            break
    return rules[:limit]


def _prompt_engineering(profile: dict[str, Any], template_library: dict[str, Any], feedback_rules: list[str]) -> dict[str, Any]:
    hard_logic_rules = [
        "实体必须复用已有 entity_id；小名、全名和称谓只能作为 aliases。",
        "每个场景先满足 preconditions，再产生 after_state 中的状态变化。",
        "时间、地点、资源、关系和知识只能按场景顺序推进，不得后果倒灌。",
        "所有关键事实必须有 fact_provenance；证据不足时用中性表述，不得补造。",
    ]
    return {
        "version": "1.0",
        "profile_version": profile.get("profile_version", ""),
        "template_library_version": template_library.get("library_version", ""),
        "feedback_rule_count": len(feedback_rules),
        "stages": {
            "extraction": {"output": "ChapterState", "hard_rules": hard_logic_rules},
            "planning": {"output": "ChapterPlan", "rules": profile.get("prompt_engineering", {}).get("planning_rules", []), "template_selection": "先选宏观、章节、事件、微互动模板，再绑定到场景状态机。"},
            "drafting": {"output": "正文", "rules": profile.get("prompt_engineering", {}).get("drafting_rules", []), "hard_rules": hard_logic_rules},
            "validation": {"output": "逻辑校验报告", "rules": profile.get("prompt_engineering", {}).get("validation_rules", []), "checks": ["实体身份", "前置条件", "时间单调", "空间连续", "资源变化", "关系变化", "知识传播", "来源证据"]},
        },
        "feedback_rules": feedback_rules,
    }


def _template(profile: dict[str, Any], kind: str, feedback_rules: list[str]) -> str:
    feedback = "\n".join(f"- {rule}" for rule in feedback_rules) or "- 当前没有可用的评测反馈；仍须严格遵守状态机。"
    if kind == "chapter_plan":
        return f"""# 章节规划提示词

你是中文网文的章节规划器。读取 `<AuthorProfile>` 的抽象叙事约束，不得带入 profile 来源作品中的任何专名、情节或句子。只使用 `<StoryState>` 和 `<ChapterGoal>` 提供的原创世界、人物与设定。

规划规则：
{_rules(profile, 'planning_rules')}

历史评测反馈：
{feedback}

先从 `<TemplateLibrary>` 选择 1 至 3 个适配模板，写入 `selected_template_ids`；模板只提供抽象结构，具体内容必须来自当前原创输入。

输出必须是合法 JSON object，字段为 `chapter_goal`、`selected_template_ids`、`scenes`、`continuity_checks`。格式示例：

```json
{json.dumps(CHAPTER_PLAN_EXAMPLE, ensure_ascii=False, indent=2)}
```

<AuthorProfile>
{{AUTHOR_PROFILE_JSON}}
</AuthorProfile>
<TemplateLibrary>
{{TEMPLATE_LIBRARY_JSON}}
</TemplateLibrary>
<StoryState>
{{STORY_STATE_JSON}}
</StoryState>
<ChapterGoal>
{{CHAPTER_GOAL_JSON}}
</ChapterGoal>
"""
    if kind == "original_chapter":
        return f"""# 原创章节正文提示词

你是中文网文写作者。根据 `<ChapterPlan>` 写连续中文正文。作者 profile 只能提供抽象叙事约束；所有具体人物、世界、术语、事件与对话都必须来自这次原创输入。只输出正文，不输出标题、解释、Markdown 或 JSON。

正文规则：
{_rules(profile, 'drafting_rules')}

历史评测反馈：
{feedback}

<AuthorProfile>
{{AUTHOR_PROFILE_JSON}}
</AuthorProfile>
<SelectedTemplates>
{{SELECTED_TEMPLATES_JSON}}
</SelectedTemplates>
<StoryState>
{{STORY_STATE_JSON}}
</StoryState>
<ChapterPlan>
{{CHAPTER_PLAN_JSON}}
</ChapterPlan>
"""
    return f"""# 章节一致性校验提示词

你是小说章节校验器。比较 `<StoryState>`、`<ChapterPlan>` 和 `<Draft>`，只输出合法 JSON object：`passed`（布尔值）、`issues`（数组）、`repairs`（数组）。

校验规则：
{_rules(profile, 'validation_rules')}

历史评测反馈：
{feedback}

<StoryState>
{{STORY_STATE_JSON}}
</StoryState>
<ChapterPlan>
{{CHAPTER_PLAN_JSON}}
</ChapterPlan>
<Draft>
{{DRAFT_TEXT}}
</Draft>
"""


def compile_skill(author_id: str) -> Path:
    destination = output_dir(author_id)
    profile = read_json(destination / "author_narrative_profile.json", {})
    if not profile:
        raise ValueError("author profile is missing; run distill before compile")
    template_library = read_json(destination / "template_library.json", {})
    if not template_library:
        raise ValueError("template library is missing; run template mining before compile")
    feedback_rules = _feedback_rules(author_id)
    destination.mkdir(parents=True, exist_ok=True)
    skill = """---
name: author-narrative-skill
description: 根据蒸馏得到的小说作者叙事 profile，规划、生成并校验原创中文网文章节。使用时必须提供 profile、故事状态与章节目标。
---

# 作者叙事 Skill

1. 先读取 `author_narrative_profile.json`，仅采用其中的抽象叙事规律和提示词规则。
2. 从 `template_library.json` 选择与当前目标、冲突和故事状态相匹配的微模板、事件模板、章节模板和宏观模板；只复用抽象结构与槽位。
3. 使用 `prompt_templates/chapter_plan.md` 生成章节计划，再使用 `prompt_templates/original_chapter.md` 扩写正文。
4. 使用 `prompt_templates/continuity_check.md` 校验人物、关系、时间、空间、资源、修炼约束、信息差与伏笔。
5. 禁止复用来源作品的人物、地名、专名、设定、情节或原句；原创世界的具体内容必须来自当前输入。
"""
    source_file = str(profile.get("source_state_file", "chapter_states.jsonl"))
    if source_file != "chapter_states.jsonl":
        skill += f"\n> 注意：当前 profile 基于 `{source_file}`，属于限定章节样本；扩大语料后应重新蒸馏。\n"
    (destination / "SKILL.md").write_text(skill, encoding="utf-8")
    write_json(destination / "chapter_state_schema.json", {
        "schema_version": "2.0",
        "required": ["chapter_id", "event_chain", "character_deltas", "relation_deltas", "temporal_spatial_state", "cultivation_state", "information_state", "pacing_curve"],
    })
    templates = destination / "prompt_templates"
    templates.mkdir(exist_ok=True)
    for kind in ("chapter_plan", "original_chapter", "continuity_check"):
        (templates / f"{kind}.md").write_text(_template(profile, kind, feedback_rules), encoding="utf-8")
    write_json(destination / "skill_context.json", {
        "profile_version": profile.get("profile_version", ""),
        "distillation_mode": profile.get("distillation_mode", ""),
        "template_library_summary": profile.get("template_library_summary", {}),
        "prompt_engineering": profile.get("prompt_engineering", {}),
        "confidence_and_limits": profile.get("confidence_and_limits", []),
    })
    write_json(destination / "prompt_engineering.json", _prompt_engineering(profile, template_library, feedback_rules))
    return destination
