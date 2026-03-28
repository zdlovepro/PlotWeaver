"""
step4_role_casting.py – Skeleton Extraction & Role Casting

Responsibilities:
  - Select a base novel for narrative pacing (skeleton).
  - Fuse character traits from the RAG knowledge base to create:
      * A new protagonist with Dao heart, combat style, and flaws.
      * Supporting cast with new relationship graph.
  - The character sheet is created BEFORE plot reassembly to prevent OOC issues.

Output:
  NarrativeSkeleton with skeleton_nodes and CharacterSheet.
"""

from __future__ import annotations

import json
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
    pacing_role: str        # e.g. 开篇 / 冲突升级 / 大高潮 / 过渡 / 收尾
    original_summary: str


@dataclass
class Character:
    name: str
    role: str               # 主角 / 女主 / 反派 / 师尊 / 友人
    realm_start: str
    dao_heart: str          # Core Dao / philosophy
    combat_style: str
    personality_flaw: str
    background: str
    traits: List[str] = field(default_factory=list)


@dataclass
class CharacterSheet:
    protagonist: Character
    supporting: List[Character] = field(default_factory=list)
    relationship_summary: str = ""


@dataclass
class NarrativeSkeleton:
    base_novel: str
    nodes: List[SkeletonNode] = field(default_factory=list)
    character_sheet: Optional[CharacterSheet] = None


# ── Public API ────────────────────────────────────────────────────────────────

def build_skeleton(
    novel_arcs: dict[str, List[VolumeArc]],
    all_atoms: dict[str, List[PlotAtom]],
    kb: KnowledgeBase,
    fused_world: FusedWorld,
) -> NarrativeSkeleton:
    """
    Select the best base novel, extract the pacing skeleton, and cast characters.
    """
    client = get_deepseek_client()

    base_novel = _select_base_novel(client, novel_arcs, fused_world)
    print(f"[Step 4] Selected base novel for pacing: {base_novel}")

    arcs = novel_arcs[base_novel]
    nodes = _extract_skeleton_nodes(arcs, all_atoms.get(base_novel, []))
    print(f"[Step 4] Skeleton has {len(nodes)} nodes")

    character_sheet = _cast_characters(client, kb, fused_world)
    print(f"[Step 4] New protagonist: {character_sheet.protagonist.name}")

    return NarrativeSkeleton(
        base_novel=base_novel,
        nodes=nodes,
        character_sheet=character_sheet,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _select_base_novel(
    client,
    novel_arcs: dict[str, List[VolumeArc]],
    fused_world: FusedWorld,
) -> str:
    """Ask DeepSeek to recommend the best base novel for narrative pacing."""
    realm_names = [r.name for r in fused_world.cultivation_realms]
    candidates = []
    for novel_name, arcs in novel_arcs.items():
        arc_names = [a.arc_name for a in arcs]
        candidates.append(f"小说《{novel_name}》：境界弧 {arc_names}")

    prompt = (
        "你是修仙小说结构分析师。\n"
        f"新统一修炼体系的境界序列为：{realm_names}\n\n"
        "以下是各候选小说的境界弧分布：\n"
        + "\n".join(candidates)
        + "\n\n请根据以下标准推荐最适合作为叙事节奏骨架的小说：\n"
        "1. 修炼阶梯完整度（覆盖从凡人到巅峰）\n"
        "2. 弧线覆盖度（前期弱、中期宗门、后期仙界）\n"
        "3. 与新修炼体系的匹配度\n\n"
        '只输出被推荐小说的文件名（字符串），格式：{"selected": "文件名.txt"}'
    )
    raw = chat_completion_json(
        client,
        system="你是专业的叙事结构评估专家，只输出合法JSON。",
        user=prompt,
        json_mode=True,
    )
    try:
        data = json.loads(raw)
        selected = data.get("selected", "")
        if selected in novel_arcs:
            return selected
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: return the novel with the most arcs
    return max(novel_arcs, key=lambda k: len(novel_arcs[k]))


def _extract_skeleton_nodes(
    arcs: List[VolumeArc], atoms: List[PlotAtom]
) -> List[SkeletonNode]:
    """Convert the base novel's arc events into pacing skeleton nodes."""
    atom_map: Dict[str, PlotAtom] = {a.atom_id: a for a in atoms}
    nodes: List[SkeletonNode] = []
    for arc in arcs:
        for i, event in enumerate(arc.events):
            atom = atom_map.get(event.event_id)
            pacing_role = _infer_pacing_role(i, len(arc.events), atom)
            realm_level = _realm_level_from_arc(arc.arc_name)
            nodes.append(
                SkeletonNode(
                    node_id=event.event_id,
                    arc_name=arc.arc_name,
                    realm_level=realm_level,
                    pacing_role=pacing_role,
                    original_summary=event.summary,
                )
            )
    return nodes


def _infer_pacing_role(
    event_idx: int, total_events: int, atom: Optional[PlotAtom]
) -> str:
    if atom and atom.narrative_function:
        return atom.narrative_function
    if event_idx == 0:
        return "开篇"
    if event_idx == total_events - 1:
        return "收尾"
    if event_idx < total_events * 0.3:
        return "铺垫"
    if event_idx < total_events * 0.7:
        return "冲突升级"
    return "大高潮"


def _realm_level_from_arc(arc_name: str) -> int:
    order = [
        "炼气", "筑基", "金丹", "元婴", "化神",
        "炼虚", "合体", "大乘", "渡劫", "真仙",
    ]
    for i, realm in enumerate(order):
        if realm in arc_name:
            return i + 1
    return 0


_CHARACTER_SCHEMA = """\
{
  "protagonist": {
    "name": "原创姓名",
    "realm_start": "起始境界",
    "dao_heart": "道心/核心执念（20字以内）",
    "combat_style": "战斗风格（如：阵法苟流/刚猛体修/毒道奇袭）",
    "personality_flaw": "性格缺陷（如：过度多疑/执念太深/冷漠自闭）",
    "background": "背景故事简述（50字）",
    "traits": ["特质1", "特质2", "特质3"]
  },
  "supporting": [
    {
      "name": "原创姓名",
      "role": "角色定位（女主/反派/师尊/挚友）",
      "realm_start": "起始境界",
      "dao_heart": "其核心执念",
      "combat_style": "战斗风格",
      "personality_flaw": "缺陷",
      "background": "背景",
      "traits": ["特质1"]
    }
  ],
  "relationship_summary": "角色关系简述（100字以内）"
}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _cast_characters(
    client, kb: KnowledgeBase, fused_world: FusedWorld
) -> CharacterSheet:
    """Fuse character archetypes from RAG into a fully original cast."""
    archetype_results = kb.query_character_traits("修仙小说主角特质", n_results=8)
    archetypes_text = "\n".join(
        f"- {r['document']}" for r in archetype_results
    )
    realm_names = [r.name for r in fused_world.cultivation_realms[:3]]

    prompt = (
        "你是修仙小说角色设计大师。\n"
        f"当前世界名：{fused_world.world_name}\n"
        f"修炼体系起始境界：{realm_names}\n"
        f"全局主题：{fused_world.global_theme}\n\n"
        "以下是从多本原作中提炼的角色原型参考（仅作灵感参考，禁止直接复用）：\n"
        f"{archetypes_text}\n\n"
        "请设计一套全新的角色阵容（主角+3-4名配角），要求：\n"
        "1. 主角与现有修仙套路明显区分（避免普通废柴流/天才打脸流）\n"
        "2. 每个角色道心/执念各不相同，产生张力\n"
        "3. 关系网络不复刻原作师徒/情侣关系\n"
        "4. 所有名字需原创\n\n"
        f"请严格按照以下JSON Schema输出：\n{_CHARACTER_SCHEMA}"
    )
    raw = chat_completion_json(
        client,
        system="你是专业的修仙小说角色设计师，只输出合法JSON。",
        user=prompt,
        json_mode=True,
    )
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, AttributeError):
        data = {}

    proto = data.get("protagonist", {})
    protagonist = Character(
        name=proto.get("name", "苍玄"),
        role="主角",
        realm_start=proto.get("realm_start", fused_world.cultivation_realms[0].name
                               if fused_world.cultivation_realms else "凡人"),
        dao_heart=proto.get("dao_heart", "以我之力，逆天改命"),
        combat_style=proto.get("combat_style", ""),
        personality_flaw=proto.get("personality_flaw", ""),
        background=proto.get("background", ""),
        traits=proto.get("traits", []),
    )

    supporting = [
        Character(
            name=s.get("name", f"配角{i}"),
            role=s.get("role", "友人"),
            realm_start=s.get("realm_start", ""),
            dao_heart=s.get("dao_heart", ""),
            combat_style=s.get("combat_style", ""),
            personality_flaw=s.get("personality_flaw", ""),
            background=s.get("background", ""),
            traits=s.get("traits", []),
        )
        for i, s in enumerate(data.get("supporting", []))
    ]

    return CharacterSheet(
        protagonist=protagonist,
        supporting=supporting,
        relationship_summary=data.get("relationship_summary", ""),
    )


# ── Intermediate I/O ──────────────────────────────────────────────────────────

_STEP4_FILENAME = "step4_protagonist.json"


def save_step4_output(skeleton: NarrativeSkeleton) -> Path:
    """Serialise *skeleton* to ``intermediate_dir/step4_protagonist.json``."""
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _STEP4_FILENAME
    data = {
        "base_novel": skeleton.base_novel,
        "nodes": [asdict(node) for node in skeleton.nodes],
        "character_sheet": {
            "protagonist": asdict(skeleton.character_sheet.protagonist),
            "supporting": [asdict(c) for c in skeleton.character_sheet.supporting],
            "relationship_summary": skeleton.character_sheet.relationship_summary,
        } if skeleton.character_sheet else None,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[Step 4] Intermediate output saved → {out_path}")
    return out_path


def load_step4_output(intermediate_dir: str | Path | None = None) -> NarrativeSkeleton:
    """Load previously saved Step 4 output from ``intermediate_dir/step4_protagonist.json``."""
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)
    in_path = inter_dir / _STEP4_FILENAME
    if not in_path.exists():
        raise FileNotFoundError(
            f"Step 4 intermediate file not found: {in_path}\n"
            "Run the pipeline from Step 4 or earlier first to generate it."
        )
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = [SkeletonNode(**n) for n in data.get("nodes", [])]

    char_data = data.get("character_sheet")
    if char_data:
        protagonist = Character(**char_data["protagonist"])
        supporting = [Character(**c) for c in char_data.get("supporting", [])]
        character_sheet = CharacterSheet(
            protagonist=protagonist,
            supporting=supporting,
            relationship_summary=char_data.get("relationship_summary", ""),
        )
    else:
        character_sheet = None

    skeleton = NarrativeSkeleton(
        base_novel=data.get("base_novel", ""),
        nodes=nodes,
        character_sheet=character_sheet,
    )
    print(f"[Step 4] Loaded intermediate output from {in_path}")
    return skeleton
