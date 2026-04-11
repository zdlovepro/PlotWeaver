"""
step4_role_casting.py – Skeleton Extraction & Role Casting (Multi-Pass Architecture)

Responsibilities:
  - Extracts the pacing skeleton from the base novel.
  - Multi-pass LLM calls to prevent lazy generation:
      Pass 1: Protagonist & Core Cast.
      Pass 2: Event-specific Transient Cast (divided by story chunks).
      Pass 3: Evolving Relationship Networks based on the full cast.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

import config
from pipeline.step1_chunking import VolumeArc
from pipeline.step2_extraction import PlotAtom
from pipeline.step3_knowledge_base import KnowledgeBase, FusedWorld
from pipeline.utils import get_deepseek_client, chat_completion_json


@dataclass
class SkeletonNode:
    node_id: str
    arc_name: str
    realm_level: int
    pacing_role: str
    original_summary: str


@dataclass
class Character:
    name: str
    role: str
    bond_depth: str
    entry_event: str
    exit_event: str
    return_event: str
    dao_heart: str
    combat_style: str
    personality_flaw: str
    background: str


@dataclass
class StageNetwork:
    stage: str
    active_characters: List[str]
    relationship_status: str


@dataclass
class CharacterSheet:
    protagonist: Character
    supporting: List[Character] = field(default_factory=list)
    relationship_networks: List[StageNetwork] = field(default_factory=list)


@dataclass
class NarrativeSkeleton:
    base_novel: str
    nodes: List[SkeletonNode] = field(default_factory=list)
    character_sheet: Optional[CharacterSheet] = None


# ── Public API ──────────────────────────────────────────────────────────────

def build_skeleton(
    novel_arcs: dict[str, List[VolumeArc]],
    all_atoms: dict[str, List[PlotAtom]],
    kb: KnowledgeBase,
    fused_world: FusedWorld,
) -> NarrativeSkeleton:
    client = get_deepseek_client()

    base_novel = _select_base_novel(client, novel_arcs, fused_world)
    print(f"[Step 4] Selected base novel for pacing: {base_novel}")

    arcs = novel_arcs[base_novel]
    nodes = _extract_skeleton_nodes(arcs, all_atoms.get(base_novel, []))
    print(f"[Step 4] Skeleton generated with {len(nodes)} events.")

    character_sheet = _cast_characters_multipass(client, kb, fused_world, nodes)

    print(f"[Step 4] New protagonist: {character_sheet.protagonist.name}")
    print(f"[Step 4] Generated a high-quality cast of {len(character_sheet.supporting)} supporting characters via Multi-Pass!")

    return NarrativeSkeleton(
        base_novel=base_novel,
        nodes=nodes,
        character_sheet=character_sheet,
    )


# ── Internal Helpers (Skeleton) ─────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _select_base_novel(client, novel_arcs: dict[str, List[VolumeArc]], fused_world: FusedWorld) -> str:
    realm_names = [r.name for r in fused_world.cultivation_realms]
    candidates = [f"小说《{n}》：卷目分布 {[a.arc_name for a in arcs]}" for n, arcs in novel_arcs.items()]

    prompt = (
        "你是���仙小说结构分析师。\n"
        f"新统一修炼体系的境界序列为：{realm_names}\n\n"
        "以下是候选小说的卷目分布：\n" + "\n".join(candidates) + "\n\n"
        "请推荐最适合作为节奏骨架的小说：\n"
        '输出JSON格式：{"selected": "文件名.txt"}'
    )
    raw = chat_completion_json(client, system="只输出合法JSON。", user=prompt, json_mode=True)
    try:
        data = json.loads(raw)
        selected = data.get("selected", "")
        if selected in novel_arcs: return selected
    except Exception: pass
    return max(novel_arcs, key=lambda k: len(novel_arcs[k]))


def _extract_skeleton_nodes(arcs: List[VolumeArc], atoms: List[PlotAtom]) -> List[SkeletonNode]:
    atom_map: Dict[str, PlotAtom] = {a.atom_id: a for a in atoms}
    nodes: List[SkeletonNode] = []
    total_events = sum(len(arc.events) for arc in arcs)
    global_idx = 0
    for arc in arcs:
        realm_level = _realm_level_from_arc(arc.arc_name)
        for event in arc.events:
            atom = atom_map.get(event.event_id)
            pacing_role = _infer_pacing_role(global_idx, total_events, atom)
            nodes.append(
                SkeletonNode(
                    node_id=event.event_id, arc_name=arc.arc_name,
                    realm_level=realm_level, pacing_role=pacing_role,
                    original_summary=event.summary,
                )
            )
            global_idx += 1
    return nodes


def _infer_pacing_role(global_idx: int, total_events: int, atom: Optional[PlotAtom]) -> str:
    if atom and atom.narrative_function: return atom.narrative_function
    ratio = global_idx / total_events if total_events > 0 else 0
    if ratio < 0.05: return "开篇"
    elif ratio > 0.95: return "收尾"
    elif ratio < 0.3: return "铺垫与宗门/势力探索"
    elif ratio < 0.7: return "冲突升级与中坚成长"
    else: return "巅峰对决与大高潮"


def _realm_level_from_arc(arc_name: str) -> int:
    match = re.search(r"卷(\d+)", arc_name)
    if match: return int(match.group(1))
    return 0


# ── Multi-Pass Character Generation ──────────────────────────────────────────

_PASS1_SCHEMA = """{
  "protagonist": {
    "name": "原创姓名", "dao_heart": "执念", "combat_style": "战斗流派", "personality_flaw": "性格缺陷", "background": "背景故事"
  },
  "core_cast": [
    {
      "name": "极具修仙质感的名字或道号", "role": "女主/宿敌/师尊等", "bond_depth": "核心",
      "entry_event": "在【事件X】登场", "exit_event": "在【事件Y】退场", "return_event": "在【事件Z】返场",
      "dao_heart": "执念", "combat_style": "战斗流派", "personality_flaw": "缺陷", "background": "背景"
    }
  ]
}"""

_PASS2_SCHEMA = """{
  "event_specific_casts": [
    {
      "name": "极具修仙质感的配角名字（绝不能敷衍）", "role": "炮灰/坊市老板/暗杀者等", "bond_depth": "短暂过客 / 单次出场",
      "entry_event": "在【事件X】登场", "exit_event": "在【事件X】或【事件X+N】退场", "return_event": "无",
      "dao_heart": "个人欲望", "combat_style": "无或具体手段", "personality_flaw": "致命弱点", "background": "身份简述"
    }
  ]
}"""

_PASS3_SCHEMA = """{
  "relationship_networks": [
    {
      "stage": "事件1 至 事件N（阶段名）",
      "active_characters": ["人物A", "人物B"],
      "relationship_status": "极度具体的恩怨纠葛与局势说明"
    }
  ]
}"""


def _cast_characters_multipass(client, kb: KnowledgeBase, fused_world: FusedWorld, nodes: List[SkeletonNode]) -> CharacterSheet:
    events_text = "\n".join([f"事件{i+1} [{n.arc_name}]: {n.original_summary[:80]}..." for i, n in enumerate(nodes)])

    # === Pass 1: Protagonist & Core Cast ===
    print("[Step 4 - Pass 1] Generating Protagonist & Core Cast...")
    prompt1 = (
        f"世界观主题：{fused_world.global_theme}\n"
        f"【核心小说事件骨架（按先后顺序）】\n{events_text}\n\n"
        "任务：生成 1名主角 和 5-8名贯穿全书的核心宿敌/红颜/师尊。\n"
        "【严厉纪律】：\n"
        "1. 名字必须符合高端修仙文审美（如复姓慕容、百里，或带道号如‘血鸦老祖’）。\n"
        "2. 绝对禁止使用张三、李四、十三、十四这种敷衍名字！违者直接失败！\n"
        "3. 明确用【事件数字】填写 `entry_event` 等字段。\n\n"
        f"输出JSON Schema:\n{_PASS1_SCHEMA}"
    )
    raw1 = chat_completion_json(client, system="你是修仙群像大师，只输出JSON。", user=prompt1, json_mode=True)
    try: data1 = json.loads(raw1)
    except Exception: data1 = {}

    p_data = data1.get("protagonist", {})
    protagonist = Character(
        name=p_data.get("name", "苍玄"), role="主角", bond_depth="主角",
        entry_event="事件1", exit_event="大结局", return_event="",
        dao_heart=p_data.get("dao_heart", ""), combat_style=p_data.get("combat_style", ""),
        personality_flaw=p_data.get("personality_flaw", ""), background=p_data.get("background", "")
    )
    supporting = []
    for s in data1.get("core_cast", []):
        supporting.append(Character(
            name=s.get("name", ""), role=s.get("role", ""), bond_depth="核心",
            entry_event=s.get("entry_event", ""), exit_event=s.get("exit_event", ""), return_event=s.get("return_event", ""),
            dao_heart=s.get("dao_heart", ""), combat_style=s.get("combat_style", ""),
            personality_flaw=s.get("personality_flaw", ""), background=s.get("background", "")
        ))

    # === Pass 2: Transient / Event-Specific Characters ===
    print("[Step 4 - Pass 2] Generating Event-Specific/One-off Characters...")
    # 我们把事件分成前后两半，让它分批生成，避免脑力枯竭
    half = len(nodes) // 2
    events_p1 = "\n".join([f"事件{i+1}: {n.original_summary[:80]}" for i, n in enumerate(nodes[:half])])
    events_p2 = "\n".join([f"事件{i+1}: {n.original_summary[:80]}" for i, n in enumerate(nodes[half:], start=half)])

    for chunk_name, chunk_text in [("前半部", events_p1), ("后半部", events_p2)]:
        prompt2 = (
            f"以下是小说的【{chunk_name}】事件骨架：\n{chunk_text}\n\n"
            "任务：为这些事件生成 8-10 个【只活跃一两个事件的过客/龙套/单章反派】。\n"
            "【严厉��律】：\n"
            "1. 名字必须极其用心（如：枯骨散人、风雷阁执事、万宝楼掌柜、毒牙老叟）。\n"
            "2. 绝对禁止出现带有数字的敷衍名字（如李四、赵六、王十四）！这是死线！\n"
            "3. 明确他们在哪个事件出场，哪个事件退场（甚至可以是同一个事件出场并立刻被秒杀）。\n\n"
            f"输出JSON Schema:\n{_PASS2_SCHEMA}"
        )
        raw2 = chat_completion_json(client, system="你是专职设计精彩配角的编剧，只输出JSON。", user=prompt2, json_mode=True)
        try:
            data2 = json.loads(raw2)
            for s in data2.get("event_specific_casts", []):
                supporting.append(Character(
                    name=s.get("name", ""), role=s.get("role", ""), bond_depth="过客",
                    entry_event=s.get("entry_event", ""), exit_event=s.get("exit_event", ""), return_event=s.get("return_event", ""),
                    dao_heart=s.get("dao_heart", ""), combat_style=s.get("combat_style", ""),
                    personality_flaw=s.get("personality_flaw", ""), background=s.get("background", "")
                ))
        except Exception: pass

    # === Pass 3: Relationship Networks ===
    print("[Step 4 - Pass 3] Weaving the Relationship Networks...")
    all_names = [protagonist.name] + [s.name for s in supporting]
    prompt3 = (
        f"核心事件骨架：\n{events_text}\n\n"
        f"全角色名单（{len(all_names)}人）：\n{', '.join(all_names)}\n\n"
        "任务：根据事件发展，将全书划分为 3-4 个大阶段，推演每个阶段存活的活跃角色间的关系网。\n"
        "要求：不要记流水账！写出极具张力的势力博弈、背叛、救赎与死斗。\n\n"
        f"输出JSON Schema:\n{_PASS3_SCHEMA}"
    )
    raw3 = chat_completion_json(client, system="你是网文关系局势推演大师，只输出JSON。", user=prompt3, json_mode=True)
    try: data3 = json.loads(raw3)
    except Exception: data3 = {}

    stage_networks = [
        StageNetwork(
            stage=net.get("stage", f"阶段{i+1}"), active_characters=net.get("active_characters", []),
            relationship_status=net.get("relationship_status", "")
        )
        for i, net in enumerate(data3.get("relationship_networks", []))
    ]

    return CharacterSheet(protagonist=protagonist, supporting=supporting, relationship_networks=stage_networks)


# ── Intermediate I/O ────────────────────────────────────────────────────────

_STEP4_SKELETON_FILENAME = "step4_1_skeleton.json"
_STEP4_CHARS_FILENAME = "step4_2_characters.json"

def save_step4_output(skeleton: NarrativeSkeleton) -> None:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    skel_path = out_dir / _STEP4_SKELETON_FILENAME
    skel_data = {
        "base_novel": skeleton.base_novel,
        "total_events": len(skeleton.nodes),
        "nodes": [asdict(node) for node in skeleton.nodes],
    }
    with open(skel_path, "w", encoding="utf-8") as f:
        json.dump(skel_data, f, ensure_ascii=False, indent=2)

    chars_path = out_dir / _STEP4_CHARS_FILENAME
    if skeleton.character_sheet:
        char_data = {
            "protagonist": asdict(skeleton.character_sheet.protagonist),
            "supporting": [asdict(c) for c in skeleton.character_sheet.supporting],
            "relationship_networks": [asdict(n) for n in skeleton.character_sheet.relationship_networks],
        }
        with open(chars_path, "w", encoding="utf-8") as f:
            json.dump(char_data, f, ensure_ascii=False, indent=2)

    print(f"[Step 4] Intermediate outputs saved → {skel_path.name} & {chars_path.name}")


def load_step4_output(intermediate_dir: str | Path | None = None) -> NarrativeSkeleton:
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)

    skel_path = inter_dir / _STEP4_SKELETON_FILENAME
    with open(skel_path, "r", encoding="utf-8") as f:
        skel_data = json.load(f)
    nodes = [SkeletonNode(**n) for n in skel_data.get("nodes", [])]

    chars_path = inter_dir / _STEP4_CHARS_FILENAME
    character_sheet = None
    if chars_path.exists():
        with open(chars_path, "r", encoding="utf-8") as f:
            char_data = json.load(f)
        protagonist = Character(**char_data["protagonist"])
        supporting = [Character(**c) for c in char_data.get("supporting", [])]
        networks = [StageNetwork(**n) for n in char_data.get("relationship_networks", [])]
        character_sheet = CharacterSheet(protagonist=protagonist, supporting=supporting, relationship_networks=networks)

    return NarrativeSkeleton(
        base_novel=skel_data.get("base_novel", ""),
        nodes=nodes,
        character_sheet=character_sheet
    )