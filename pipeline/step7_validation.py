"""
step7_validation.py – Adversarial Plagiarism Check & Output

Responsibilities:
  - Run exact-match NER check against source texts (threshold < 5%).
  - Use DeepSeek as an "Anti-Plagiarism Editor" to find the top 3 most similar
    tropes. If it fails, flag specific nodes for Step 5 re-generation.
  - Output `novel_world_bible.md` and `volume_1_to_N_outline.md`.

Output:
  ValidationResult with pass/fail status and flagged node IDs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tenacity import retry, stop_after_attempt, wait_exponential

import config
from pipeline.step3_knowledge_base import FusedWorld
from pipeline.step4_role_casting import CharacterSheet, NarrativeSkeleton
from pipeline.step5_reassembly import ReassembledEvent
from pipeline.step6_generation import VolumeOutline
from pipeline.utils import get_deepseek_client, chat_completion_json


@dataclass
class ValidationResult:
    passed: bool
    ner_overlap_ratio: float
    flagged_event_ids: List[str] = field(default_factory=list)
    similar_tropes: List[str] = field(default_factory=list)
    notes: str = ""


# ── Public API ────────────────────────────────────────────────────────────────

def validate_and_output(
    volumes: List[VolumeOutline],
    reassembled_events: List[ReassembledEvent],
    skeleton: NarrativeSkeleton,
    fused_world: FusedWorld,
    source_texts: Dict[str, str],
    output_dir: str | Path,
) -> ValidationResult:
    """
    Run plagiarism checks, then write output files.

    Returns ValidationResult. If not passed, flagged_event_ids are set so
    the orchestrator can send them back to Step 5 for re-generation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_deepseek_client()

    # 1. Collect all text from the new outline
    outline_text = _compile_outline_text(volumes)

    # 2. NER exact-match check
    ner_ratio, flagged_ner = _ner_exact_check(outline_text, source_texts)
    print(f"[Step 7] NER overlap ratio: {ner_ratio:.2%}")

    # 3. Adversarial DeepSeek check
    similar_tropes, flagged_trope_events = _adversarial_check(
        client, outline_text, source_texts, reassembled_events
    )

    flagged = list(set(flagged_ner + flagged_trope_events))
    passed = (
        ner_ratio < config.PLAGIARISM_NER_THRESHOLD and len(flagged_trope_events) == 0
    )

    result = ValidationResult(
        passed=passed,
        ner_overlap_ratio=ner_ratio,
        flagged_event_ids=flagged,
        similar_tropes=similar_tropes,
        notes=(
            "通过验证。" if passed
            else f"发现 {len(flagged)} 个需要修改的节点，NER重合率 {ner_ratio:.2%}。"
        ),
    )

    # 4. Always write output files (even if not fully passed, for review)
    _write_world_bible(fused_world, skeleton.character_sheet, output_dir)
    _write_volume_outline(volumes, fused_world, output_dir)
    _write_validation_report(result, output_dir)

    print(f"[Step 7] Validation {'PASSED ✓' if passed else 'FAILED ✗'}: {result.notes}")
    return result


# ── NER exact-match check ─────────────────────────────────────────────────────

def _extract_named_entities(text: str) -> Set[str]:
    """
    Heuristic NER for Chinese text: extract 2-4 character sequences that look
    like named entities (person names, place names, technique names).
    This is intentionally lightweight – a production system would use a proper
    NER model.
    """
    # Match typical Chinese proper nouns: 2-6 Chinese characters
    entities: Set[str] = set()
    # Look for capitalised words in mixed text
    entities.update(re.findall(r"[A-Z][a-zA-Z]{2,}", text))
    # Chinese noun phrases: 2-6 chars followed by common suffixes
    suffixes = r"(?:功|诀|剑|刀|拳|掌|宗|门|宫|殿|界|境|峰|山|城|洞|府|道|法|经|典|丹|器|符|阵)"
    entities.update(re.findall(rf"[\u4e00-\u9fff]{{1,5}}{suffixes}", text))
    return entities


def _ner_exact_check(
    outline_text: str, source_texts: Dict[str, str]
) -> tuple[float, List[str]]:
    new_entities = _extract_named_entities(outline_text)
    if not new_entities:
        return 0.0, []

    source_entities: Set[str] = set()
    for src_text in source_texts.values():
        source_entities.update(_extract_named_entities(src_text))

    overlap = new_entities & source_entities
    ratio = len(overlap) / len(new_entities)
    flagged: List[str] = []  # NER check flags the whole outline, not specific events

    if ratio >= config.PLAGIARISM_NER_THRESHOLD:
        flagged = ["global_ner_overlap"]

    return ratio, flagged


# ── Adversarial DeepSeek check ────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _adversarial_check(
    client,
    outline_text: str,
    source_texts: Dict[str, str],
    events: List[ReassembledEvent],
) -> tuple[List[str], List[str]]:
    """
    Run an adversarial 'anti-plagiarism editor' DeepSeek session.
    Returns (similar_tropes_list, flagged_event_ids).
    """
    # Build a compact representation of source summaries
    source_summaries: List[str] = []
    for novel_name, src_text in source_texts.items():
        # Use configurable number of chars as a synopsis proxy
        source_summaries.append(
            f"原作《{novel_name}》摘要：{src_text[:config.PLAGIARISM_SOURCE_SYNOPSIS_LENGTH]}"
        )
    source_summary_text = "\n".join(source_summaries[:5])

    # Use configurable chars of new outline for the check
    new_outline_excerpt = outline_text[:config.PLAGIARISM_OUTLINE_EXCERPT_LENGTH]

    prompt = (
        "你是一位极其严苛的网文反抄袭鉴定专家。\n\n"
        "以下是多本原作的简要摘要：\n"
        f"{source_summary_text}\n\n"
        "以下是新生成大纲的节选：\n"
        f"{new_outline_excerpt}\n\n"
        "请找出新大纲中与原作在以下方面最相似的最多3处：\n"
        "- 核心金手指/外挂设定\n"
        "- 打脸/退婚/废材逆袭等套路\n"
        "- 经典标志性桥段\n\n"
        "如果某处相似度极高（可一眼看出改写痕迹），请标记其需要重写。\n"
        "以JSON输出：\n"
        '{"similar_tropes": ["相似点描述1", "相似点描述2"], '
        '"needs_rewrite": true或false, '
        '"rewrite_reason": "如需重写说明原因"}'
    )
    raw = chat_completion_json(
        client,
        system=(
            "你是专业的反抄袭鉴定专家，没有任何偏袒，只输出合法JSON。"
            "你与生成此大纲的AI无关，独立客观评估。"
        ),
        user=prompt,
        json_mode=True,
    )
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, AttributeError):
        data = {}

    similar_tropes = data.get("similar_tropes", [])
    needs_rewrite = data.get("needs_rewrite", False)

    flagged_ids: List[str] = []
    if needs_rewrite and events:
        # Flag the last N events for re-generation (heuristic: high-tension events)
        high_tension = sorted(events, key=lambda e: e.realm_level, reverse=True)[:3]
        flagged_ids = [e.event_id for e in high_tension]

    return similar_tropes, flagged_ids


# ── Output writers ────────────────────────────────────────────────────────────

def _compile_outline_text(volumes: List[VolumeOutline]) -> str:
    parts = []
    for vol in volumes:
        parts.append(f"## {vol.volume_title}\n")
        for ch in vol.chapter_summaries:
            parts.append(f"{ch}\n")
    return "\n".join(parts)


def _write_world_bible(
    fused_world: FusedWorld,
    character_sheet: CharacterSheet,
    output_dir: Path,
) -> None:
    lines = [
        f"# 世界圣经 – {fused_world.world_name}",
        "",
        f"## 全局主题",
        f"> {fused_world.global_theme}",
        "",
        "## 修炼体系",
        "",
        "| 境界 | 名称 | 突破条件 | 特殊能力 |",
        "|------|------|----------|----------|",
    ]
    for realm in fused_world.cultivation_realms:
        abilities = "、".join(realm.special_abilities) or "—"
        lines.append(
            f"| 第{realm.level}境 | {realm.name} | {realm.breakthrough_condition} | {abilities} |"
        )

    lines += [
        "",
        "## 主角设定",
        "",
        f"**姓名：** {character_sheet.protagonist.name}",
        f"**道心：** {character_sheet.protagonist.dao_heart}",
        f"**战斗风格：** {character_sheet.protagonist.combat_style}",
        f"**性格缺陷：** {character_sheet.protagonist.personality_flaw}",
        f"**背景：** {character_sheet.protagonist.background}",
        f"**特质：** {', '.join(character_sheet.protagonist.traits)}",
        "",
        "## 配角一览",
        "",
    ]
    for char in character_sheet.supporting:
        lines += [
            f"### {char.name}（{char.role}）",
            f"- 起始境界：{char.realm_start}",
            f"- 道心：{char.dao_heart}",
            f"- 战斗风格：{char.combat_style}",
            f"- 缺陷：{char.personality_flaw}",
            "",
        ]
    lines += [
        "## 角色关系",
        "",
        character_sheet.relationship_summary,
    ]

    output_path = output_dir / "novel_world_bible.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Step 7] World bible written to {output_path}")


def _write_volume_outline(
    volumes: List[VolumeOutline],
    fused_world: FusedWorld,
    output_dir: Path,
) -> None:
    lines = [
        f"# 《{fused_world.world_name}》完整分卷大纲",
        "",
        f"> 全局主题：{fused_world.global_theme}",
        "",
    ]
    for vol in volumes:
        lines += [
            f"---",
            f"## 第{vol.volume_number}卷：{vol.volume_title}",
            f"**境界跨度：** {vol.realm_range}",
            f"**张力曲线：** {vol.tension_curve}",
            "",
            "### 章节大纲",
            "",
        ]
        for ch_summary in vol.chapter_summaries:
            lines.append(f"{ch_summary}")
            lines.append("")
        lines += [
            f"**卷末状态：** {vol.ending_state}",
            "",
        ]

    n = len(volumes)
    output_path = output_dir / f"volume_1_to_{n}_outline.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Step 7] Outline written to {output_path}")


def _write_validation_report(result: ValidationResult, output_dir: Path) -> None:
    lines = [
        "# 验证报告",
        "",
        f"**验证结果：** {'✓ 通过' if result.passed else '✗ 未通过'}",
        f"**NER重合率：** {result.ner_overlap_ratio:.2%}（阈值 < {config.PLAGIARISM_NER_THRESHOLD:.0%}）",
        f"**说明：** {result.notes}",
        "",
        "## 发现的相似套路",
        "",
    ]
    if result.similar_tropes:
        for trope in result.similar_tropes:
            lines.append(f"- {trope}")
    else:
        lines.append("无明显相似套路。")

    if result.flagged_event_ids:
        lines += [
            "",
            "## 需要重写的情节节点",
            "",
        ]
        for eid in result.flagged_event_ids:
            lines.append(f"- `{eid}`")

    output_path = output_dir / "validation_report.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Step 7] Validation report written to {output_path}")
