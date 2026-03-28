"""
step6_generation.py – Sliding Window Volume Generation

Responsibilities:
  - Generate a detailed outline volume by volume using a sliding window approach.
  - System prompt is cached with: world bible, character sheet, global theme.
  - Per-volume user prompt includes: previous volume summary + current events.

Output:
  List[VolumeOutline] with the final detailed outline text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.step3_knowledge_base import FusedWorld
from pipeline.step4_role_casting import CharacterSheet
from pipeline.step5_reassembly import ReassembledEvent
from pipeline.utils import get_deepseek_client, chat_completion_json


@dataclass
class VolumeOutline:
    volume_number: int
    volume_title: str
    realm_range: str            # e.g. "第1境 → 第3境"
    chapter_summaries: List[str] = field(default_factory=list)
    tension_curve: str = ""
    ending_state: str = ""      # Last state – fed as context to next volume
    raw_text: str = ""


# ── Public API ────────────────────────────────────────────────────────────────

def generate_volumes(
    reassembled_events: List[ReassembledEvent],
    fused_world: FusedWorld,
    character_sheet: CharacterSheet,
) -> List[VolumeOutline]:
    """
    Group reassembled events into volumes (by arc) and generate detailed outlines.
    """
    client = get_deepseek_client()

    # Build the cached system context (world bible + character sheet)
    system_context = _build_system_context(fused_world, character_sheet)

    # Group events into volumes by arc_name
    volume_groups = _group_events_into_volumes(reassembled_events)
    print(f"[Step 6] Generating {len(volume_groups)} volumes...")

    volumes: List[VolumeOutline] = []
    previous_ending = ""

    for vol_idx, (arc_name, events) in enumerate(volume_groups.items(), start=1):
        print(f"[Step 6] Generating volume {vol_idx}: {arc_name}...")
        outline = _generate_single_volume(
            client=client,
            volume_number=vol_idx,
            arc_name=arc_name,
            events=events,
            system_context=system_context,
            previous_ending=previous_ending,
            fused_world=fused_world,
        )
        volumes.append(outline)
        previous_ending = outline.ending_state

    print(f"[Step 6] All {len(volumes)} volumes generated.")
    return volumes


# ── Internal helpers ──────────────────────────────────────────────────────────

def _group_events_into_volumes(
    events: List[ReassembledEvent],
) -> Dict[str, List[ReassembledEvent]]:
    groups: Dict[str, List[ReassembledEvent]] = {}
    for event in events:
        groups.setdefault(event.arc_name, []).append(event)
    # Sort volume groups by their minimum realm_level so volumes are always
    # generated in the correct cultivation-progression order regardless of the
    # order in which arc boundaries were first detected in the source text.
    return dict(
        sorted(
            groups.items(),
            key=lambda item: min((e.realm_level for e in item[1]), default=0),
        )
    )


def _build_system_context(
    fused_world: FusedWorld, character_sheet: CharacterSheet
) -> str:
    protagonist = character_sheet.protagonist
    realms_text = "\n".join(
        f"  第{r.level}境 {r.name}：{r.breakthrough_condition}"
        for r in fused_world.cultivation_realms
    )
    supporting_text = "\n".join(
        f"  - {c.name}（{c.role}）：{c.dao_heart}"
        for c in character_sheet.supporting
    )
    return (
        "===== 世界圣经 =====\n"
        f"世界名：{fused_world.world_name}\n"
        f"全局主题：{fused_world.global_theme}\n\n"
        f"修炼体系：\n{realms_text}\n\n"
        "===== 角色设定 =====\n"
        f"主角：{protagonist.name}\n"
        f"  道心：{protagonist.dao_heart}\n"
        f"  战斗风格：{protagonist.combat_style}\n"
        f"  性格缺陷：{protagonist.personality_flaw}\n"
        f"  背景：{protagonist.background}\n\n"
        f"配角：\n{supporting_text}\n\n"
        f"角色关系：{character_sheet.relationship_summary}\n"
        "===================="
    )


_VOLUME_SCHEMA = """\
{
  "volume_title": "本卷标题（富有诗意）",
  "realm_range": "本卷境界跨度（如：第1境 荒元境 → 第3境 凝煞境）",
  "chapter_summaries": [
    "第1章：...",
    "第2章：..."
  ],
  "tension_curve": "本卷张力曲线描述（开篇低谷→中期对抗→结尾高潮，50字）",
  "ending_state": "本卷结尾主角状态摘要（50字，用于下一卷衔接）"
}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _generate_single_volume(
    client,
    volume_number: int,
    arc_name: str,
    events: List[ReassembledEvent],
    system_context: str,
    previous_ending: str,
    fused_world: FusedWorld,
) -> VolumeOutline:
    events_text = "\n".join(
        f"{i+1}. [{e.pacing_role}] {e.adapted_summary}"
        for i, e in enumerate(events)
    )

    user_prompt = (
        f"你正在为修仙小说《{fused_world.world_name}》撰写第{volume_number}卷的细粒度大纲。\n"
        f"当前卷境界弧：{arc_name}\n\n"
        + (f"【上一卷结尾状态】\n{previous_ending}\n\n" if previous_ending else "")
        + f"【本卷情节序列】\n{events_text}\n\n"
        "请生成本卷的详细大纲，要求：\n"
        "1. 严格按照大境界弧分卷结构\n"
        "2. 每章摘要100-150字\n"
        "3. 伏笔与揭示节点明确标注\n"
        "4. 每章说明服务于全局主题的方式\n"
        "5. 张力曲线呈现起伏变化（不能一直高潮）\n"
        "6. 严禁使用原著人名、宗门名、功法名等专有名词——所有专有名词必须原创，与原著完全区分\n\n"
        f"请严格按照以下JSON Schema输出：\n{_VOLUME_SCHEMA}"
    )

    system_prompt = (
        "你是专业的修仙小说大纲生成专家。\n"
        "以下是本次写作的世界圣经和角色设定，在整个创作过程中保持一致：\n\n"
        f"{system_context}\n\n"
        "只输出合法JSON，不要有其他文字。"
    )

    raw = chat_completion_json(
        client,
        system=system_prompt,
        user=user_prompt,
        json_mode=True,
    )
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, AttributeError):
        data = {}

    return VolumeOutline(
        volume_number=volume_number,
        volume_title=data.get("volume_title", f"第{volume_number}卷·{arc_name}"),
        realm_range=data.get("realm_range", arc_name),
        chapter_summaries=data.get("chapter_summaries", []),
        tension_curve=data.get("tension_curve", ""),
        ending_state=data.get("ending_state", ""),
        raw_text=raw,
    )
