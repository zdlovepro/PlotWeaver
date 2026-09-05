"""Build and validate the portable filesystem anatomy of an author Skill.

This module contains no source-work names or prose.  It turns already accepted
single- or multi-work abstractions into progressively disclosed references and
small deterministic utilities.  Release evidence is handled by typed contracts;
directory validity never upgrades a draft to a validated author Skill.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable

from .contracts import AuthorStyleProfile, DistilledAuthorTrait, NarrativeTemplateLibrary, WorkContinuity


REQUIRED_REFERENCE_FILES = (
    "references/author_core.json",
    "references/work_deltas.json",
    "references/genre_baseline.json",
    "references/narrative_policy.json",
    "references/prose_realization.json",
    "references/dialogue_patterns.json",
    "references/long_form_strategy.json",
    "references/negative_constraints.json",
    "references/quality_thresholds.json",
    "references/evidence_report.json",
)

_FORBIDDEN_SOURCE_KEYS = frozenset({
    "source_text", "chapter_text", "original_text", "raw_text",
    "source_quote", "evidence_quote", "original_prose",
})


def normalize_skill_name(author_id: str) -> str:
    """Return a stable Skill frontmatter name without changing output folders."""

    stem = re.sub(r"[^a-z0-9]+", "-", str(author_id).strip().lower()).strip("-") or "novel"
    suffix = "-novel-author"
    stem = stem[: 64 - len(suffix)].rstrip("-")
    return f"{stem}{suffix}"


def _trait_from_constraint(profile: AuthorStyleProfile, constraint: Any) -> DistilledAuthorTrait:
    trait = DistilledAuthorTrait(
        trait_id=constraint.constraint_id,
        family=constraint.family,
        rule=constraint.rule,
        application=constraint.application,
        avoid=constraint.avoid,
        source_work_ids=(profile.work_id,),
        evidence_chapter_ids=constraint.evidence_chapter_ids,
        support_count=constraint.support_count,
        confidence=constraint.confidence,
        validation_status="single_work_candidate",
        applicable_contexts=(constraint.family,),
    )
    trait.validate()
    return trait


def _template_policy(template: Any, work_id: str) -> dict[str, Any]:
    return {
        "policy_id": template.template_id,
        "level": template.level,
        "purpose": template.purpose,
        "when": {"selection_tags": list(template.selection_tags)},
        "role_slots": list(template.role_slots),
        "realization_order": list(template.beat_sequence),
        "state_effects": list(template.state_effects),
        "variation_axes": list(template.variation_axes),
        "source_work_ids": [work_id],
        "evidence_chapter_ids": list(template.evidence_chapter_ids),
        "support_count": template.support_count,
        "confidence": template.confidence,
        "validation_status": "single_work_candidate",
    }


def build_draft_references(
    profile: AuthorStyleProfile,
    library: NarrativeTemplateLibrary,
    continuity: WorkContinuity,
    execution_spec: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Project one work into honest draft references without inventing author core."""

    traits = tuple(_trait_from_constraint(profile, item) for item in profile.constraints)
    policies = tuple(_template_policy(item, profile.work_id) for item in library.templates)
    dialogue_metrics = [
        item.to_dict() for item in profile.baselines
        if item.metric_id.startswith("dialogue_")
    ]
    dialogue_traits = [item.to_dict() for item in traits if item.family == "dialogue"]
    long_form = [item for item in policies if item["level"] in {"chapter", "thread", "arc"}]
    return {
        "references/author_core.json": {
            "schema_version": "1.0",
            "release_status": "draft",
            "validated_traits": [],
            "candidate_trait_count": len(traits),
            "reason": "当前只有单作品证据；候选规律保存在 work_deltas.json，不能冒充跨作品作者核心。",
        },
        "references/work_deltas.json": {
            "schema_version": "1.0",
            "release_status": "draft",
            "works": [{
                "work_id": profile.work_id,
                "chapter_ids": list(profile.chapter_ids),
                "candidate_traits": [item.to_dict() for item in traits],
                "baselines": [item.to_dict() for item in profile.baselines],
            }],
        },
        "references/genre_baseline.json": {
            "schema_version": "1.0",
            "status": "unavailable",
            "contrast_work_count": 0,
            "metrics": [],
            "reason": "尚未导入同题材对照作品，不能计算作者区分度。",
        },
        "references/narrative_policy.json": {
            "schema_version": "1.0",
            "release_status": "draft",
            "validated_policies": [],
            "candidate_policies": list(policies),
        },
        "references/prose_realization.json": {
            "schema_version": "1.0",
            "release_status": "draft",
            "baselines": [item.to_dict() for item in profile.baselines],
            "candidate_traits": [item.to_dict() for item in traits],
            "execution_spec": execution_spec,
            "interpretation": "数值是单作品观测区间，不是作者核心，也不得机械凑数。",
        },
        "references/dialogue_patterns.json": {
            "schema_version": "1.0",
            "release_status": "draft",
            "metrics": dialogue_metrics,
            "candidate_traits": dialogue_traits,
        },
        "references/long_form_strategy.json": {
            "schema_version": "1.0",
            "release_status": "draft",
            "validated_strategies": [],
            "candidate_strategies": long_form,
            "limitation": "当前策略来自有限章节模板，尚未通过跨作品卷级验证。",
        },
        "references/negative_constraints.json": {
            "schema_version": "1.0",
            "release_status": "draft",
            "candidate_avoid_rules": [
                {"trait_id": item.trait_id, "family": item.family, "avoid": item.avoid}
                for item in traits
            ],
            "source_safety_guardrails": list(profile.guardrails),
        },
        "references/quality_thresholds.json": {
            "schema_version": "1.0",
            "release_status": "draft",
            "hard_generation_gates": [
                "未来信息泄漏数必须为零。",
                "已提交人物、关系、时间、地点、资源和知识状态的硬矛盾数必须为零。",
                "核心事件和核心状态变化必须全部实现。",
                "候选图谱补丁未通过时不得发布正文或开始下一章。",
                "超过来源复写风险阈值的候选必须拒绝。",
            ],
            "release_requirements": {
                "content_audit_chapter_count": 5,
                "minimum_source_work_count": 2,
                "requires_cross_work_distillation": True,
                "requires_held_out_evaluation": True,
                "requires_long_form_evaluation": True,
                "requires_style_discrimination": True,
                "requires_copy_risk_check": True,
            },
        },
        "references/evidence_report.json": {
            "schema_version": "1.0",
            "release_status": "draft",
            "source_work_ids": [profile.work_id],
            "chapter_ids": list(profile.chapter_ids),
            "chapter_count": len(profile.chapter_ids),
            "candidate_trait_count": len(traits),
            "validated_author_trait_count": 0,
            "candidate_policy_count": len(policies),
            "continuity_observation_counts": {
                "global_entity_count": len(continuity.global_entities),
                "timeline_event_count": len(continuity.timeline_events),
                "state_ledger_entry_count": len(continuity.state_ledger),
                "location_transition_count": len(continuity.location_transitions),
            },
            "missing_evidence_gates": [
                "五章内容级语义审计结果",
                "第二部及以上训练作品",
                "同题材对照基线",
                "跨作品作者核心",
                "留出作品或连续章节盲测",
                "原创长篇和防复写验收",
            ],
        },
    }


def build_skill_markdown(skill_name: str) -> str:
    """Render a concise progressively disclosed Skill entry point."""

    return f"""---
name: {skill_name}
description: 使用证据化的小说作者写作规律规划、生成和校验中文长篇小说；适用于章节规划、场景写作、连续性维护、候选比较与结构保真测试。先检查发布状态，draft 规则只能用于受控实验，不能宣称是完整作者核心。
---

# 小说作者 Skill

1. 先读 `skill_manifest.json` 和 `references/evidence_report.json`。若 `release_status` 不是 `validated`，明确说明证据缺口；不得把单作品候选规律称为完整作者风格。
2. 规划长篇时读 `references/long_form_strategy.json` 与 `references/narrative_policy.json`；只选择与当前阶段、事件和场景职责匹配的策略。
3. 写正文时读 `references/author_core.json`。只有其中 `validated_traits` 可作为作者核心；需要单作品实验时才额外读取 `references/work_deltas.json`，并标明它是作品偏移候选。
4. 需要语言、对话或负面约束时，分别读取 `references/prose_realization.json`、`references/dialogue_patterns.json` 和 `references/negative_constraints.json`。数值是柔性观测区间，不得机械凑数。
5. 调用者提供的新故事图谱是唯一跨段事实底座；卷/阶段大纲、事件链、章节合同和已提交状态决定当前边界。作者参考只决定怎样安排与表达，不能提供来源角色、设定、情节或未来答案。
6. 每个段落写作包至少包含阶段目标、当前事件链、章节合同、场景职责、当前段事实子图、已提交状态、上一段受限收束和本段作者策略；不得输入来源原文或未来事件。
7. 候选正文先通过事实、状态、时空、因果、线索和复写硬门控，再比较作者策略贴合与可读性。每章先写 `chapter_graph_patch.candidate.json`；补丁未提交时不得开始下一章。
8. 选择作者规则时可运行 `scripts/select_style_context.py`；交付或迁移前运行 `scripts/self_check.py`。失败时报告具体证据缺口或错误层，不得自动降低硬门槛。
"""


def build_openai_yaml(skill_name: str, author_id: str) -> str:
    display_name = f"{author_id} 小说作者 Skill"
    short_description = "使用证据化作者策略规划、生成并校验中文长篇小说章节"
    default_prompt = f"使用 ${skill_name} 根据当前故事状态规划并写作下一章，同时校验连续性。"

    def quote(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'

    return "\n".join((
        "interface:",
        f"  display_name: {quote(display_name)}",
        f"  short_description: {quote(short_description)}",
        f"  default_prompt: {quote(default_prompt)}",
        "policy:",
        "  allow_implicit_invocation: true",
        "",
    ))


def self_check_script() -> str:
    """Return a dependency-free package checker stored inside the Skill."""

    required = json.dumps(["SKILL.md", "agents/openai.yaml", *REQUIRED_REFERENCE_FILES], ensure_ascii=False, indent=2)
    forbidden = json.dumps(sorted(_FORBIDDEN_SOURCE_KEYS), ensure_ascii=False)
    return f'''#!/usr/bin/env python3
"""Validate this packaged novel-author Skill without reading source novels."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

REQUIRED = {required}
FORBIDDEN_KEYS = set({forbidden})


def walk(value, path="$"):
    issues = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                issues.append(f"{{path}}.{{key}}: 禁止保存来源正文键")
            issues.extend(walk(item, f"{{path}}.{{key}}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(walk(item, f"{{path}}[{{index}}]"))
    elif isinstance(value, str):
        if re.search(r"(?:[A-Za-z]:\\\\|/Users/|/home/)", value):
            issues.append(f"{{path}}: 包含编译机绝对路径")
    return issues


def main():
    root = Path(__file__).resolve().parents[1]
    issues = [f"缺少文件：{{name}}" for name in REQUIRED if not (root / name).is_file()]
    if (root / "README.md").exists():
        issues.append("Skill 根目录不得包含 README.md")
    skill = (root / "SKILL.md").read_text(encoding="utf-8") if (root / "SKILL.md").exists() else ""
    match = re.match(r"^---\\n(.*?)\\n---", skill, re.DOTALL)
    if not match or not re.search(r"^name:\\s*[a-z0-9]+(?:-[a-z0-9]+)*\\s*$", match.group(1), re.MULTILINE):
        issues.append("SKILL.md 缺少合法的小写连字符 name")
    for relative in REQUIRED:
        path = root / relative
        if path.suffix == ".json" and path.exists():
            try:
                issues.extend(walk(json.loads(path.read_text(encoding="utf-8")), relative))
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"{{relative}}: JSON 无法读取：{{exc}}")
    result = {{"passed": not issues, "root": str(root), "issues": issues}}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
'''


def select_style_context_script() -> str:
    """Return a small deterministic selector for progressive reference loading."""

    return '''#!/usr/bin/env python3
"""Select a bounded set of author/work rules for one scene task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def grams(text):
    compact = "".join(str(text).lower().split())
    return {compact[index:index + 2] for index in range(max(0, len(compact) - 1))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-role", default="")
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--max-rules", type=int, default=6)
    args = parser.parse_args()
    if args.max_rules <= 0:
        raise SystemExit("--max-rules 必须为正数")
    root = Path(__file__).resolve().parents[1]
    core = load(root / "references" / "author_core.json")
    deltas = load(root / "references" / "work_deltas.json")
    policies = load(root / "references" / "narrative_policy.json")
    candidates = []
    for item in core.get("validated_traits", []):
        candidates.append({"kind": "author_core", **item})
    for work in deltas.get("works", []):
        for item in work.get("candidate_traits", []):
            candidates.append({"kind": "work_delta_candidate", "work_id": work.get("work_id", ""), **item})
    for item in policies.get("validated_policies", []) + policies.get("candidate_policies", []):
        candidates.append({"kind": "narrative_policy", **item})
    wanted = set(args.family)
    role_grams = grams(args.scene_role)
    ranked = []
    for item in candidates:
        family = str(item.get("family", item.get("level", "")))
        body = " ".join(str(item.get(key, "")) for key in ("rule", "application", "purpose", "when"))
        score = (4 if wanted and family in wanted else 0) + len(role_grams & grams(body))
        if not wanted and not args.scene_role:
            score = 1
        ranked.append((score, str(item.get("trait_id", item.get("policy_id", ""))), item))
    selected = [item for score, _, item in sorted(ranked, key=lambda row: (-row[0], row[1])) if score > 0][:args.max_rules]
    print(json.dumps({"release_status": core.get("release_status", "draft"), "selected": selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''


def _walk_json(value: Any, path: str = "$") -> list[str]:
    issues: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_SOURCE_KEYS:
                issues.append(f"{path}.{key}: forbidden source-prose key")
            issues.extend(_walk_json(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(_walk_json(item, f"{path}[{index}]"))
    elif isinstance(value, str) and re.search(r"(?:[A-Za-z]:\\|/Users/|/home/)", value):
        issues.append(f"{path}: absolute machine path")
    return issues


def assess_skill_directory(destination: Path, expected_files: Iterable[str]) -> tuple[str, ...]:
    """Check package anatomy and source-safety without external dependencies."""

    expected = tuple(expected_files)
    issues = [f"missing generated file: {name}" for name in expected if not (destination / name).is_file()]
    if (destination / "README.md").exists():
        issues.append("Skill package must not contain README.md")
    skill_path = destination / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""
    frontmatter = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not frontmatter:
        issues.append("SKILL.md frontmatter is missing")
    elif not re.search(r"^name:\s*[a-z0-9]+(?:-[a-z0-9]+)*\s*$", frontmatter.group(1), re.MULTILINE):
        issues.append("SKILL.md name is not lowercase hyphen-case")
    for relative in expected:
        path = destination / relative
        if path.suffix != ".json" or not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(f"{relative}: invalid JSON: {exc}")
            continue
        issues.extend(f"{relative}: {item}" for item in _walk_json(payload))
    return tuple(issues)
