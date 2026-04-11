"""
step6_generation.py – Sliding Window Volume Generation (Event to Chapters Expansion)

Responsibilities:
  - Generate a detailed outline volume by volume.
  - EXPANDS each Plot Event into MULTIPLE chapters (1 Event -> 3~6 Chapters) to match true web novel pacing.
  - Enforces power scaling and "Show, don't tell" rules.
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


@dataclass
class VolumeOutline:
    volume_number: int
    volume_title: str
    realm_range: str
    chapter_summaries: List[str] = field(default_factory=list)  # Flattened list of chapter strings
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

    # 因为现在章节数变多了，我们只传上一卷最后的一个事件块（最后约3-5章）作为衔接
    previous_tail_chapters: List[str] = []
    previous_ending_state = ""

    for vol_idx, (arc_name, events) in enumerate(
        tqdm(volume_groups.items(), desc="[Step 6] Generating volumes", unit="vol"), start=1
    ):
        tqdm.write(f"[Step 6] Generating volume {vol_idx}: {arc_name} (Events: {len(events)})...")

        current_stage_network = _get_stage_network_for_volume(character_sheet, vol_idx, total_volumes)
        system_context = _build_system_context(fused_world, character_sheet, current_stage_network)

        outline = _generate_single_volume(
            client=client,
            volume_number=vol_idx,
            arc_name=arc_name,
            events=events,
            system_context=system_context,
            previous_ending_state=previous_ending_state,
            previous_tail_chapters=previous_tail_chapters,
            fused_world=fused_world
        )
        volumes.append(outline)

        previous_ending_state = outline.ending_state
        # 提取本卷最后生成的5章作为下一卷的绝对前置剧情
        if len(outline.chapter_summaries) >= 5:
            previous_tail_chapters = outline.chapter_summaries[-5:]
        else:
            previous_tail_chapters = outline.chapter_summaries

    tqdm.write(f"[Step 6] All {len(volumes)} volumes generated.")
    return volumes


def _group_events_into_volumes(events: List[ReassembledEvent]) -> Dict[str, List[ReassembledEvent]]:
    groups: Dict[str, List[ReassembledEvent]] = {}
    for event in events:
        groups.setdefault(event.arc_name, []).append(event)
    return groups

def _get_stage_network_for_volume(char_sheet: CharacterSheet, vol_idx: int, total_vols: int) -> str:
    if not char_sheet.relationship_networks: return "暂无关系网。"
    ratio = (vol_idx - 1) / total_vols if total_vols > 1 else 0
    stage_idx = int(ratio * len(char_sheet.relationship_networks))
    active_net = char_sheet.relationship_networks[min(stage_idx, len(char_sheet.relationship_networks) - 1)]
    return f"【本卷时期：{active_net.stage}】\n活跃角色：{', '.join(active_net.active_characters)}\n局势状态：{active_net.relationship_status}"

def _build_system_context(fused_world: FusedWorld, character_sheet: CharacterSheet, current_stage_network: str) -> str:
    protagonist = character_sheet.protagonist
    realms_text = "\n".join(f"  {r.name}（第{r.level}境）：{r.breakthrough_condition}" for r in fused_world.cultivation_realms)
    supporting_text = "\n".join(f"  - [{c.role}] {c.name}：{c.dao_heart} | 背景：{c.background}" for c in character_sheet.supporting)
    return (
        f"===== 世界圣经 =====\n世界名：{fused_world.world_name}\n全局主题：{fused_world.global_theme}\n"
        f"力量本源：{fused_world.power_source}\n世界背景：{fused_world.world_background}\n\n修炼体系：\n{realms_text}\n\n"
        f"===== 角色全图鉴 =====\n主角：{protagonist.name}\n  执念：{protagonist.dao_heart}\n  战斗：{protagonist.combat_style}\n  缺陷：{protagonist.personality_flaw}\n"
        f"配角：\n{supporting_text}\n\n===== 本卷动态人物关系 =====\n{current_stage_network}\n"
    )


_VOLUME_SCHEMA = """\
{
  "volume_title": "本卷标题",
  "realm_range": "本卷境界跨度",
  "expanded_events": [
    {
      "event_name": "事件的简短标题（如：坊市捡漏反杀）",
      "chapters": [
        "第X章：(80-100字，描写具体行为与起承转合)",
        "第X+1章：(80-100字，矛盾爆发)",
        "第X+2章：(80-100字，解决与收尾)"
      ]
    }
  ],
  "tension_curve": "张力曲线描述",
  "ending_state": "本卷结尾主角状态摘要（用于下卷绝对起点）"
}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _generate_single_volume(
    client, volume_number: int, arc_name: str, events: List[ReassembledEvent],
    system_context: str, previous_ending_state: str, previous_tail_chapters: List[str],
    fused_world: FusedWorld
) -> VolumeOutline:

    events_lines = []
    max_realm_level = max([e.realm_level for e in events] + [1])

    for i, e in enumerate(events):
        trope_tag = f" 【化用套路：{e.used_trope}】" if e.used_trope and e.used_trope != "无" else ""
        events_lines.append(f"核心事件{i+1}. [{e.pacing_role}]{trope_tag} {e.adapted_summary}")
    events_text = "\n".join(events_lines)

    transition_text = ""
    if previous_tail_chapters:
        transition_text += "【上一卷末尾剧情（必须顺滑承接）】\n" + "\n".join(previous_tail_chapters) + "\n"
    if previous_ending_state:
        transition_text += f"【上一卷结尾局势】\n{previous_ending_state}\n"

    user_prompt = (
        f"你正在为修仙大作撰写第 {volume_number} 卷（{arc_name}）的详细章回大纲。\n\n"
        f"{transition_text}\n"
        f"【本卷必须推演的核心事件骨架】\n{events_text}\n\n"
        "【大纲倍增扩写纪律】（极其重要）：\n"
        "1. 事件 ➔ 章节的扩写：上面提供的是“大事件骨架”。在网文中，1个事件绝不可能1章写完。你必须将【每一个核心事件】详细拆解为 3 到 6 个具体章节！\n"
        "   （例如：如果本卷有10个事件，你的 JSON 中的 chapters 总数必须达到 30 到 60 章！）\n"
        "2. 拆分逻辑：一个事件应拆分为“铺垫(发现端倪/遭遇挑衅) -> 发展(应对/遇险) -> 高潮(底牌尽出/反杀) -> 收尾(摸尸/境界突破)”。\n"
        f"3. 战力红线：本卷最高战力限制在【第{max_realm_level}境】上下，严格限制破坏力表现！\n"
        "4. 隐性人设 (Show, don't tell)：禁止直接复制人物的性格标签，用具体的行为、对话、阴招来体现他们的性格。\n"
        "5. 精炼���因为总章节数很多，每章摘要保持在 80-120 字即可，剔除废话，直击动作与剧情发展。\n\n"
        f"请严格按以下JSON Schema输出：\n{_VOLUME_SCHEMA}"
    )

    system_prompt = (
        "你是白金级修仙大纲总编剧，精通网文的“节奏注水”与“爽点拆解”技术。\n"
        "你绝不会把一个大高潮事件一笔带过，而是懂得将其拆解为连续数章的压迫与释放。\n"
        "CRITICAL: 严禁使用其他小说的原名。必须保持章节序号的连续递增（如 第1章, 第2章... 第40章）。\n\n"
        f"{system_context}\n\n只输出合法JSON。"
    )

    raw = chat_completion_json(client, system=system_prompt, user=user_prompt, json_mode=True)
    try:
        data = json.loads(raw)
    except Exception:
        data = {}

    # 将大模型生成的事件嵌套结构 (expanded_events) 扁平化，转换为 markdown 友好的列表
    flattened_chapters = []
    global_ch_idx = 1

    for evt in data.get("expanded_events", []):
        event_name = evt.get("event_name", "主线进展")
        flattened_chapters.append(f"#### 剧情点：{event_name}")
        for ch_text in evt.get("chapters", []):
            # 过滤掉模型可能自己带的“第X章：”，由我们统一编号保证连续性
            clean_text = ch_text.split("：", 1)[-1] if "：" in ch_text else ch_text
            flattened_chapters.append(f"**第{global_ch_idx}章**：{clean_text.strip()}")
            global_ch_idx += 1

        flattened_chapters.append("") # 加个空行分隔

    return VolumeOutline(
        volume_number=volume_number,
        volume_title=data.get("volume_title", f"第{volume_number}卷"),
        realm_range=data.get("realm_range", arc_name),
        chapter_summaries=flattened_chapters,
        tension_curve=data.get("tension_curve", ""),
        ending_state=data.get("ending_state", ""),
        raw_text=raw,
    )