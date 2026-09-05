"""
step5_world_fusion.py

Builds the shared world base, character traits, breakthroughs, and global theme.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from pipeline.core.artifacts import save_json_artifact, world_base_from_fused_world
from pipeline.core.utils import chat_completion_json, get_deepseek_client
from pipeline.core.world_building_core import (
    CultivationRealm,
    FusedWorld,
    KnowledgeBase,
    _character_mentions,
    _safe_json_load,
    _top_values,
    load_world_snapshot,
    save_world_snapshot,
)
from pipeline.step2_extraction import PlotAtom


_STEP5_FILENAME = "step5_world_fusion.json"
_STEP5_WORLD_BASE_FILENAME = "step5_world_base.json"
_CULTIVATION_SYSTEM_SCHEMA = {
    "world_name": "融合修真世界",
    "world_background": "一句话世界背景",
    "power_source": "灵气",
    "major_factions": ["宗门甲", "宗门乙"],
    "realms": [
        {
            "name": "炼气",
            "level": 1,
            "breakthrough_condition": "完成首轮稳定周天",
            "special_abilities": ["基础御气"],
        }
    ],
}


def build_world_base(all_atoms: Dict[str, List[PlotAtom]], kb: KnowledgeBase) -> FusedWorld:
    client = get_deepseek_client()
    all_atoms_flat = [atom for atoms in all_atoms.values() for atom in atoms]

    print("[Step 5] Extracting character traits & breakthrough opportunities...")
    traits = _extract_character_traits(all_atoms_flat)
    breakthroughs = _extract_breakthroughs(all_atoms_flat)
    kb.add_character_traits(traits)
    kb.add_breakthrough_opportunities(breakthroughs)

    print("[Step 5] Fusing cultivation system & extracting theme...")
    fused_world = _fuse_cultivation_system(client, all_atoms_flat)
    kb.add_cultivation_system(fused_world.cultivation_realms)
    fused_world.global_theme = _extract_global_theme(client, all_atoms_flat)
    return fused_world


def save_step5_output(fused_world: FusedWorld):
    path = save_world_snapshot(_STEP5_FILENAME, fused_world)
    artifact_path = save_json_artifact(Path(config.INTERMEDIATE_DIR) / _STEP5_WORLD_BASE_FILENAME, world_base_from_fused_world(fused_world))
    print(f"[Step 5] Intermediate output saved -> {path.name}; world base -> {artifact_path.name}")
    return path


def load_step5_output() -> FusedWorld:
    return load_world_snapshot(_STEP5_FILENAME)


def _extract_character_traits(atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    per_character: dict[str, Dict[str, Any]] = {}
    for atom in atoms:
        for mention in _character_mentions(atom):
            entry = per_character.setdefault(
                mention["key"],
                {"display": mention["display"], "raw": mention["raw"], "atoms": []},
            )
            entry["atoms"].append(atom)

    traits: List[Dict[str, Any]] = []
    ranked_characters = sorted(per_character.items(), key=lambda item: len(item[1]["atoms"]), reverse=True)[:30]
    for key, payload in ranked_characters:
        char_atoms = payload["atoms"]
        conflicts = _top_values([atom.conflict_type for atom in char_atoms], limit=2)
        motives = _top_values([atom.motivation for atom in char_atoms], limit=1)
        emotions = _top_values([atom.emotion for atom in char_atoms], limit=1)
        text = (
            f"{payload['display']}常卷入{('、'.join(conflicts) or '关键冲突')}，"
            f"主要受{('、'.join(motives) or '生存与成长')}驱动，"
            f"情绪底色偏向{('、'.join(emotions) or '压抑与克制')}。"
        )
        traits.append(
            {
                "id": f"trait_{uuid.uuid4().hex[:8]}",
                "text": text,
                "metadata": {
                    "character_key": key,
                    "character": payload["display"],
                    "raw_character": payload["raw"],
                    "conflict": conflicts,
                    "motivation": motives,
                },
            }
        )
    return traits


def _extract_breakthroughs(atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    grouped: dict[str, List[PlotAtom]] = defaultdict(list)
    for atom in atoms:
        if atom.arc_name:
            grouped[atom.arc_name].append(atom)

    items: List[Dict[str, Any]] = []
    for arc_name, arc_atoms in grouped.items():
        for atom in arc_atoms:
            if not atom.cultivation_elements and atom.tension_level < 7:
                continue
            elements = ", ".join(atom.cultivation_elements[:3]) or atom.location or "high-pressure node"
            text = (
                f"{arc_name}中，{elements}可作为突破契机，"
                f"尤其适合与{atom.conflict_type or '高压冲突'}叠加触发。"
            )
            items.append(
                {
                    "id": f"breakthrough_{uuid.uuid4().hex[:8]}",
                    "text": text,
                    "metadata": {
                        "arc_name": arc_name,
                        "atom_id": atom.atom_id,
                        "tension_level": atom.tension_level,
                    },
                }
            )
    return items[:40]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _fuse_cultivation_system(client, atoms: List[PlotAtom]) -> FusedWorld:
    all_elements: List[str] = []
    for atom in atoms:
        all_elements.extend(atom.cultivation_elements)
    unique_elements = list(dict.fromkeys(item for item in all_elements if item))[:120]
    if not unique_elements:
        unique_elements = ["灵气", "丹药", "宗门", "秘境", "传承"]

    prompt = (
        "请基于下面的来源元素，融合出一个统一的修真世界与修炼阶梯。"
        "所有字段内容都使用中文，只返回合法 JSON。\n\n"
        f"Schema: {json.dumps(_CULTIVATION_SYSTEM_SCHEMA, ensure_ascii=False)}\n\n"
        f"来源元素：\n{unique_elements}"
    )
    raw = chat_completion_json(
        client,
        system="只返回合法 JSON，所有字段内容都使用中文。",
        user=prompt,
        json_mode=True,
        temperature=0.5,
    )
    data = _safe_json_load(raw)
    if not isinstance(data, dict):
        data = {}

    realms: List[CultivationRealm] = []
    for index, item in enumerate(data.get("realms", []), start=1):
        realms.append(
            CultivationRealm(
                name=item.get("name", f"境界{index}"),
                level=int(item.get("level", index)),
                breakthrough_condition=item.get("breakthrough_condition", ""),
                special_abilities=item.get("special_abilities", []),
            )
        )

    if not realms:
        defaults = ["炼气", "筑基", "金丹", "元婴", "化神", "炼虚", "合体", "大乘"]
        realms = [
            CultivationRealm(name=name, level=index, breakthrough_condition="完成当前阶段并稳固根基。")
            for index, name in enumerate(defaults, start=1)
        ]

    realms.sort(key=lambda item: item.level)
    dag = nx.DiGraph()
    for realm in realms:
        dag.add_node(realm.name, level=realm.level)
    for index in range(len(realms) - 1):
        dag.add_edge(realms[index].name, realms[index + 1].name)

    return FusedWorld(
        cultivation_realms=realms,
        realm_dag=dag,
        world_name=data.get("world_name", "融合修真世界"),
        world_background=data.get("world_background", ""),
        power_source=data.get("power_source", "灵气"),
        major_factions=data.get("major_factions", []),
        raw_system_text=raw,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _extract_global_theme(client, atoms: List[PlotAtom]) -> str:
    motivations = _top_values([atom.motivation for atom in atoms], limit=20)
    prompt = (
        "请根据下面的动机列表，总结一句不超过20字的全局主题。"
        "只输出这一句中文主题，不要加解释。\n\n"
        f"动机列表：{motivations}"
    )
    result = chat_completion_json(
        client,
        system="只输出一句简短中文主题。",
        user=prompt,
        json_mode=False,
        temperature=0.3,
    )
    return result.strip().strip('"').strip("'")
