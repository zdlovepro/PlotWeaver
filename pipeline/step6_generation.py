"""
step6_generation.py - Sliding Window Volume Generation.

Expands reassembled plot events into per-volume chapter outlines while keeping
power scaling, continuity, and intra-volume variety under control.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from pipeline.step3_knowledge_base import FusedWorld
from pipeline.step4_role_casting import CharacterSheet
from pipeline.step5_reassembly import ReassembledEvent
from pipeline.utils import get_deepseek_client, chat_completion_json

_GENERIC_VOLUME_MOTIFS = (
    "神秘珠子",
    "暗中守护",
    "设局",
    "识破",
    "逆袭",
    "引爆",
    "试炼",
    "拍卖会",
    "秘境",
)


@dataclass
class VolumeOutline:
    volume_number: int
    volume_title: str
    realm_range: str
    chapter_summaries: List[str] = field(default_factory=list)
    tension_curve: str = ""
    ending_state: str = ""
    raw_text: str = ""


def generate_volumes(
    reassembled_events: List[ReassembledEvent],
    fused_world: FusedWorld,
    character_sheet: CharacterSheet,
) -> List[VolumeOutline]:
    client = get_deepseek_client()
    volume_groups = _group_events_into_volumes(reassembled_events)
    total_volumes = len(volume_groups)
    print(f"[Step 6] Generating {total_volumes} volumes. Expanding Events into Chapters...")

    volumes: List[VolumeOutline] = []
    previous_tail_chapters: List[str] = []
    previous_ending_state = ""

    for vol_idx, (arc_name, events) in enumerate(
        tqdm(volume_groups.items(), desc="[Step 6] Generating volumes", unit="vol"),
        start=1,
    ):
        tqdm.write(f"[Step 6] Generating volume {vol_idx}: {arc_name} (Events: {len(events)})...")

        current_stage_network = _get_stage_network_for_volume(
            character_sheet, vol_idx, total_volumes
        )
        system_context = _build_system_context(
            fused_world, character_sheet, current_stage_network
        )

        outline = _generate_single_volume(
            client=client,
            volume_number=vol_idx,
            arc_name=arc_name,
            events=events,
            system_context=system_context,
            previous_ending_state=previous_ending_state,
            previous_tail_chapters=previous_tail_chapters,
            fused_world=fused_world,
        )
        volumes.append(outline)

        previous_ending_state = outline.ending_state
        previous_tail_chapters = (
            outline.chapter_summaries[-5:]
            if len(outline.chapter_summaries) >= 5
            else outline.chapter_summaries
        )

    tqdm.write(f"[Step 6] All {len(volumes)} volumes generated.")
    return volumes


def _group_events_into_volumes(events: List[ReassembledEvent]) -> Dict[str, List[ReassembledEvent]]:
    groups: Dict[str, List[ReassembledEvent]] = {}
    for event in events:
        groups.setdefault(event.arc_name, []).append(event)
    return groups


def _get_stage_network_for_volume(
    char_sheet: CharacterSheet, vol_idx: int, total_vols: int
) -> str:
    if not char_sheet.relationship_networks:
        return "暂无关系网。"

    ratio = (vol_idx - 1) / total_vols if total_vols > 1 else 0
    stage_idx = int(ratio * len(char_sheet.relationship_networks))
    active_net = char_sheet.relationship_networks[
        min(stage_idx, len(char_sheet.relationship_networks) - 1)
    ]
    return (
        f"【本卷时段：{active_net.stage}】\n"
        f"活跃角色：{', '.join(active_net.active_characters)}\n"
        f"局势状态：{active_net.relationship_status}"
    )


def _build_system_context(
    fused_world: FusedWorld,
    character_sheet: CharacterSheet,
    current_stage_network: str,
) -> str:
    protagonist = character_sheet.protagonist
    realms_text = "\n".join(
        f"  {realm.name}（第{realm.level}境）：{realm.breakthrough_condition}"
        for realm in fused_world.cultivation_realms
    )
    supporting_text = "\n".join(
        f"  - [{char.role}] {char.name}：{char.dao_heart} | 背景：{char.background}"
        for char in character_sheet.supporting
    )
    return (
        "===== 世界圣经 =====\n"
        f"世界名：{fused_world.world_name}\n"
        f"全局主题：{fused_world.global_theme}\n"
        f"力量来源：{fused_world.power_source}\n"
        f"世界背景：{fused_world.world_background}\n\n"
        f"修炼体系：\n{realms_text}\n\n"
        "===== 角色图鉴 =====\n"
        f"主角：{protagonist.name}\n"
        f"  执念：{protagonist.dao_heart}\n"
        f"  战斗：{protagonist.combat_style}\n"
        f"  缺陷：{protagonist.personality_flaw}\n"
        f"配角：\n{supporting_text}\n\n"
        "===== 本卷动态关系 =====\n"
        f"{current_stage_network}\n"
    )


_VOLUME_SCHEMA = """\
{
  "volume_title": "本卷标题",
  "realm_range": "本卷境界跨度",
  "expanded_events": [
    {
      "event_name": "事件短标题",
      "chapters": [
        "第1章：80-120字中文剧情摘要",
        "第2章：80-120字中文剧情摘要",
        "第3章：80-120字中文剧情摘要"
      ]
    }
  ],
  "tension_curve": "张力曲线描述",
  "ending_state": "本卷结尾状态"
}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _generate_single_volume(
    client,
    volume_number: int,
    arc_name: str,
    events: List[ReassembledEvent],
    system_context: str,
    previous_ending_state: str,
    previous_tail_chapters: List[str],
    fused_world: FusedWorld,
) -> VolumeOutline:
    del fused_world  # kept in signature for compatibility with existing callers

    events_lines: List[str] = []
    max_realm_level = max([event.realm_level for event in events] + [1])

    for idx, event in enumerate(events, start=1):
        trope_tag = f" [微模板:{event.used_trope}]" if event.used_trope else ""
        events_lines.append(
            f"核心事件{idx}. [{event.pacing_role}]{trope_tag} {event.adapted_summary}"
        )
    events_text = "\n".join(events_lines)
    diversity_brief = _build_volume_diversity_brief(events)

    transition_parts: List[str] = []
    if previous_tail_chapters:
        transition_parts.append("【上卷末尾剧情】\n" + "\n".join(previous_tail_chapters))
    if previous_ending_state:
        transition_parts.append(f"【上卷结尾状态】\n{previous_ending_state}")
    transition_text = "\n\n".join(transition_parts)

    user_prompt = (
        f"请为第{volume_number}卷（{arc_name}）生成详细中文章节大纲。\n\n"
        f"{transition_text}\n\n"
        "【本卷必须推演的核心事件骨架】\n"
        f"{events_text}\n\n"
        "Rules:\n"
        "1. Expand each core event into 3 to 6 concrete chapters.\n"
        "2. Each event needs a distinct setup, escalation, reversal, and temporary resolution.\n"
        f"3. Keep combat scaling within roughly realm level {max_realm_level}.\n"
        "4. Write all chapter summaries in Chinese and keep each chapter around 80 to 120 Chinese characters.\n"
        "5. Show personality through tactics, choices, and dialogue instead of labels.\n"
        f"{diversity_brief}\n"
        "6. Volume-level anti-repetition: each expanded event must have a distinct trigger, obstacle, and payoff.\n"
        "7. Do not reuse the same combo of trap, secret help, clue spotting, and sudden reversal across multiple events in the same volume unless the skeleton explicitly requires it.\n\n"
        f"Return valid JSON only using this schema:\n{_VOLUME_SCHEMA}"
    )

    system_prompt = (
        "You are an expert Chinese web-novel outliner. "
        "Write concise but vivid Chinese chapter summaries. "
        "Keep chapter numbers continuous and avoid reusing names from unrelated novels.\n\n"
        f"{system_context}"
    )

    raw = chat_completion_json(client, system=system_prompt, user=user_prompt, json_mode=True)
    try:
        data = json.loads(raw)
    except Exception:
        data = {}

    expanded_events = data.get("expanded_events", [])
    if not _is_volume_payload_usable(expanded_events, events):
        expanded_events = _build_fallback_expanded_events(events)

    flattened_chapters: List[str] = []
    global_chapter_idx = 1

    for event_block in expanded_events:
        event_name = event_block.get("event_name", "主线推进")
        flattened_chapters.append(f"#### 剧情点：{event_name}")
        for chapter_text in event_block.get("chapters", []):
            clean_text = _strip_leading_chapter_label(str(chapter_text or "").strip())
            flattened_chapters.append(f"**第{global_chapter_idx}章**：{clean_text}")
            global_chapter_idx += 1
        flattened_chapters.append("")

    return VolumeOutline(
        volume_number=volume_number,
        volume_title=data.get("volume_title", f"第{volume_number}卷"),
        realm_range=data.get("realm_range", arc_name),
        chapter_summaries=flattened_chapters,
        tension_curve=data.get("tension_curve", "本卷张力持续上升。"),
        ending_state=data.get("ending_state", _build_fallback_ending_state(events)),
        raw_text=raw,
    )


def _build_volume_diversity_brief(events: List[ReassembledEvent]) -> str:
    motif_hits = []
    for motif in _GENERIC_VOLUME_MOTIFS:
        count = sum(1 for event in events if motif in event.adapted_summary)
        if count >= 2:
            motif_hits.append(f"{motif} x{count}")

    repeated_tropes = [event.used_trope for event in events if event.used_trope]
    duplicate_tropes = sorted(
        {trope for trope in repeated_tropes if repeated_tropes.count(trope) > 1}
    )

    lines = ["Volume anti-repetition briefing:"]
    lines.append("- Every event needs a visibly different dramatic engine.")
    if motif_hits:
        lines.append(
            "- These motifs are already repeated in the skeleton and should not become the default solution again: "
            + ", ".join(motif_hits)
        )
    if duplicate_tropes:
        lines.append(
            "- Reused micro templates detected in skeleton: "
            + ", ".join(duplicate_tropes)
        )
    if len(lines) == 2:
        lines.append("- No obvious repeated motif detected in the event skeleton.")
    return "\n".join(lines)


def _is_volume_payload_usable(
    expanded_events: Any,
    source_events: List[ReassembledEvent],
) -> bool:
    if not isinstance(expanded_events, list) or not expanded_events:
        return False

    valid_blocks = 0
    for block in expanded_events:
        if not isinstance(block, dict):
            continue
        chapters = block.get("chapters", [])
        if isinstance(chapters, list) and chapters:
            valid_blocks += 1

    return valid_blocks >= max(1, len(source_events) // 2)


def _build_fallback_expanded_events(events: List[ReassembledEvent]) -> List[Dict[str, Any]]:
    fallback_events: List[Dict[str, Any]] = []
    for event in events:
        event_name = _build_fallback_event_name(event)
        chapters = [
            f"铺垫当前目标与阻碍：{event.adapted_summary}",
            f"冲突升级并迫使主角调整策略：围绕[{event.pacing_role}]展开新的代价、选择与对抗。",
            "阶段性收束但留下后续压力：延续当前事件后果，并把卷内主线推向下一步。",
        ]
        fallback_events.append({"event_name": event_name, "chapters": chapters})
    return fallback_events


def _build_fallback_event_name(event: ReassembledEvent) -> str:
    summary = (event.adapted_summary or "").strip()
    if not summary:
        return "主线推进"
    compact = summary.replace("；", "，").replace("。", "，")
    return compact.split("，", 1)[0][:18] or "主线推进"


def _build_fallback_ending_state(events: List[ReassembledEvent]) -> str:
    if not events:
        return "本卷完成阶段性推进。"
    return f"本卷收束于：{events[-1].adapted_summary}"


def _strip_leading_chapter_label(text: str) -> str:
    if not text:
        return "主线推进。"

    if text.startswith("第") and "：" in text[:12]:
        return text.split("：", 1)[1].strip()
    if text.startswith("第") and ":" in text[:12]:
        return text.split(":", 1)[1].strip()
    if text.lower().startswith("chapter") and ":" in text[:16]:
        return text.split(":", 1)[1].strip()
    return text
