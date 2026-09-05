from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Set

from tenacity import retry, stop_after_attempt, wait_exponential

import config
from pipeline.core.story_models import CharacterSheet, NarrativeSkeleton
from pipeline.core.common_json import write_json_file
from pipeline.core.state_validator import (
    validate_reassembled_events,
    validate_skeleton_sequence,
    validate_volume_outline,
)
from pipeline.core.utils import chat_completion_json, get_deepseek_client
from pipeline.core.world_building_core import FusedWorld
from pipeline.step11_reassembly import ReassembledEvent
from pipeline.step12_generation import VolumeOutline


def _quality_chat_completion_json(client, **kwargs) -> str:
    return chat_completion_json(client, model_name=config.DEEPSEEK_LAST_STEPS_MODEL, **kwargs)


@dataclass
class ValidationResult:
    passed: bool
    ner_overlap_ratio: float
    flagged_event_ids: List[str] = field(default_factory=list)
    similar_tropes: List[str] = field(default_factory=list)
    state_issues: List[Dict[str, str]] = field(default_factory=list)
    notes: str = ""


def validate_and_output(
    volumes: List[VolumeOutline],
    reassembled_events: List[ReassembledEvent],
    skeleton: NarrativeSkeleton,
    fused_world: FusedWorld,
    source_texts: Dict[str, str],
    output_dir: str | Path,
) -> ValidationResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outline_text = _compile_outline_text(volumes)
    ner_ratio, flagged_ner = _ner_exact_check(outline_text, source_texts)
    print(f"[Step 13] NER overlap ratio: {ner_ratio:.2%}")

    client = get_deepseek_client()
    similar_tropes, flagged_trope_events = _adversarial_check(client, outline_text, source_texts, reassembled_events)

    state_issues = _collect_state_issues(skeleton, reassembled_events, volumes, fused_world)
    blocking_state_issues = [item for item in state_issues if item["severity"] in {"major", "fatal"}]
    if blocking_state_issues:
        print(f"[Step 13] Blocking state issues: {len(blocking_state_issues)}", flush=True)

    flagged = list(set(flagged_ner + flagged_trope_events))
    passed = (
        ner_ratio < config.PLAGIARISM_NER_THRESHOLD
        and len(flagged_trope_events) == 0
        and not blocking_state_issues
    )
    result = ValidationResult(
        passed=passed,
        ner_overlap_ratio=ner_ratio,
        flagged_event_ids=flagged,
        similar_tropes=similar_tropes,
        state_issues=state_issues,
        notes=("通过验证。" if passed else f"发现 {len(flagged)} 个需要修改的节点，NER重合率 {ner_ratio:.2%}。"),
    )

    _write_world_bible(fused_world, skeleton.character_sheet, output_dir)
    _write_volume_outline(volumes, fused_world, output_dir)
    _write_validation_report(result, output_dir)
    _write_validation_result_json(result)
    print(f"[Step 13] Validation {'PASSED' if passed else 'FAILED'}: {result.notes}")
    return result


def _collect_state_issues(
    skeleton: NarrativeSkeleton,
    events: List[ReassembledEvent],
    volumes: List[VolumeOutline],
    fused_world: FusedWorld,
) -> List[Dict[str, str]]:
    issues = list(validate_skeleton_sequence(skeleton.nodes, fused_world))
    issues.extend(validate_reassembled_events(events, fused_world))
    for volume in volumes:
        issues.extend(validate_volume_outline(asdict(volume), fused_world))

    result: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for issue in issues:
        payload = {
            "severity": str(issue.severity),
            "issue_type": str(issue.issue_type),
            "message": str(issue.message),
            "node_id": str(issue.node_id),
            "suggested_action": str(issue.suggested_action),
        }
        key = (payload["severity"], payload["issue_type"], payload["node_id"], payload["message"])
        if key not in seen:
            seen.add(key)
            result.append(payload)
    return result


def _write_validation_result_json(result: ValidationResult) -> None:
    path = Path(config.INTERMEDIATE_DIR) / "step13_validation_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(path, asdict(result))
    print(f"[Step 13] Validation result saved -> {path.name}")


def _compile_outline_text(volumes: List[VolumeOutline]) -> str:
    parts = []
    for volume in volumes:
        parts.append(f"## {volume.volume_title}\n")
        for chapter in volume.chapter_summaries:
            parts.append(f"{chapter}\n")
    return "\n".join(parts)


def _extract_named_entities(text: str) -> Set[str]:
    entities = set(re.findall(r"[\u4e00-\u9fff]{2,6}", text))
    stopwords = {"主角", "宗门", "前辈", "灵气", "秘境", "功法", "丹药", "长老", "弟子", "修炼", "境界"}
    return {item for item in entities if item not in stopwords}


def _ner_exact_check(outline_text: str, source_texts: Dict[str, str]) -> tuple[float, List[str]]:
    new_entities = _extract_named_entities(outline_text)
    if not new_entities:
        return 0.0, []
    source_entities: Set[str] = set()
    for src_text in source_texts.values():
        source_entities.update(_extract_named_entities(src_text))
    overlap = new_entities & source_entities
    ratio = len(overlap) / len(new_entities)
    flagged = ["global_ner_overlap"] if ratio >= config.PLAGIARISM_NER_THRESHOLD else []
    return ratio, flagged


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _adversarial_check(client, outline_text: str, source_texts: Dict[str, str], events: List[ReassembledEvent]) -> tuple[List[str], List[str]]:
    source_summaries = [f"原作《{name}》摘要：{text[:config.PLAGIARISM_SOURCE_SYNOPSIS_LENGTH]}" for name, text in source_texts.items()]
    prompt = (
        "你是一位严苛的网文相似性检查员。\n"
        "请比较下面的新大纲与原作摘要，找出最多 3 个明显相似的套路。\n"
        "如果某处相似度高到需要重写，请标记 needs_rewrite 为 true。\n\n"
        f"原作摘要：\n{chr(10).join(source_summaries[:5])}\n\n"
        f"新大纲节选：\n{outline_text[:config.PLAGIARISM_OUTLINE_EXCERPT_LENGTH]}\n\n"
        '只返回 JSON，格式为 {"similar_tropes": ["..."], "needs_rewrite": true, "rewrite_reason": "..."}'
    )
    raw = _quality_chat_completion_json(client, system="只返回合法 JSON。", user=prompt, json_mode=True)
    try:
        data = json.loads(raw)
    except Exception:
        data = {}
    similar_tropes = data.get("similar_tropes", []) if isinstance(data, dict) else []
    needs_rewrite = bool(data.get("needs_rewrite", False)) if isinstance(data, dict) else False
    flagged_ids = []
    if needs_rewrite and events:
        high_tension = sorted(events, key=lambda item: item.realm_level, reverse=True)[:3]
        flagged_ids = [item.event_id for item in high_tension]
    return similar_tropes, flagged_ids


def _write_world_bible(fused_world: FusedWorld, character_sheet: CharacterSheet, output_dir: Path) -> None:
    lines = [
        f"# 世界圣经 - {fused_world.world_name}",
        "",
        "## 全局主题",
        f"> {fused_world.global_theme}",
        "",
        "## 修炼体系",
        "",
        "| 境界 | 名称 | 突破条件 | 特殊能力 |",
        "|------|------|----------|----------|",
    ]
    for realm in fused_world.cultivation_realms:
        abilities = "、".join(realm.special_abilities) or "-"
        lines.append(f"| 第{realm.level}境 | {realm.name} | {realm.breakthrough_condition} | {abilities} |")

    if character_sheet:
        lines += [
            "",
            "## 主角设定",
            "",
            f"**姓名：** {character_sheet.protagonist.name}",
            f"**道心：** {character_sheet.protagonist.dao_heart}",
            f"**战斗风格：** {character_sheet.protagonist.combat_style}",
            f"**性格缺陷：** {character_sheet.protagonist.personality_flaw}",
            f"**背景：** {character_sheet.protagonist.background}",
            "",
            "## 角色图鉴",
            "",
        ]
        for char in character_sheet.supporting:
            lines += [
                f"### {char.name}（{char.role}）",
                f"- **羁绊定位：** {char.bond_depth}",
                f"- **登场节点：** {char.entry_event}",
                f"- **退场节点：** {char.exit_event}",
                f"- **回场节点：** {char.return_event}",
                f"- **执念：** {char.dao_heart}",
                f"- **战斗风格：** {char.combat_style}",
                f"- **性格缺陷：** {char.personality_flaw}",
                f"- **背景：** {char.background}",
                "",
            ]

    output_path = output_dir / "novel_world_bible.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Step 13] World bible written to {output_path}")


def _write_volume_outline(volumes: List[VolumeOutline], fused_world: FusedWorld, output_dir: Path) -> None:
    lines = [
        f"# 《{fused_world.world_name}》完整分卷大纲",
        "",
        f"> 全局主题：{fused_world.global_theme}",
        "",
    ]
    for volume in volumes:
        lines += [
            "---",
            f"## 第{volume.volume_number}卷：{volume.volume_title}",
            f"**境界跨度：** {volume.realm_range}",
            f"**张力曲线：** {volume.tension_curve}",
            "",
            "### 章节大纲",
            "",
        ]
        lines.extend(volume.chapter_summaries)
        lines += ["", f"**卷末状态：** {volume.ending_state}", ""]

    output_path = output_dir / f"volume_1_to_{len(volumes)}_outline.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Step 13] Outline written to {output_path}")


def _write_validation_report(result: ValidationResult, output_dir: Path) -> None:
    lines = [
        "# 验证报告",
        "",
        f"**验证结果：** {'通过' if result.passed else '未通过'}",
        f"**NER重合率：** {result.ner_overlap_ratio:.2%}（阈值 < {config.PLAGIARISM_NER_THRESHOLD:.0%}）",
        f"**说明：** {result.notes}",
        "",
        "## 相似套路",
        "",
    ]
    if result.similar_tropes:
        lines.extend(f"- {trope}" for trope in result.similar_tropes)
    else:
        lines.append("无明显相似套路。")
    if result.flagged_event_ids:
        lines += ["", "## 需要重写的情节节点", ""]
        lines.extend(f"- `{eid}`" for eid in result.flagged_event_ids)
    if result.state_issues:
        lines += ["", "## 状态连续性问题", ""]
        lines.extend(
            f"- [{item['severity']}/{item['issue_type']}] {item['message']} (`{item['node_id']}`)"
            for item in result.state_issues
        )
    output_path = output_dir / "validation_report.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Step 13] Validation report written to {output_path}")
