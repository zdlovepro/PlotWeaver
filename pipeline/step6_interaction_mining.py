"""
step6_interaction_mining.py

Extracts reusable interaction patterns and long relationship threads.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.core.utils import chat_completion_json, get_deepseek_client
from pipeline.core.world_building_core import (
    FusedWorld,
    KnowledgeBase,
    _character_mentions,
    _safe_json_load,
    load_world_snapshot,
    save_world_snapshot,
)
from pipeline.step2_extraction import PlotAtom


_STEP6_FILENAME = "step6_interaction_mining.json"


def mine_story_patterns(
    all_atoms: Dict[str, List[PlotAtom]],
    kb: KnowledgeBase,
    fused_world: FusedWorld,
) -> FusedWorld:
    client = get_deepseek_client()
    all_atoms_flat = [atom for atoms in all_atoms.values() for atom in atoms]

    print("[Step 6] Extracting micro interactions...")
    fused_world.micro_interactions = _extract_micro_interactions(client, all_atoms_flat)
    kb.add_micro_interactions(fused_world.micro_interactions)

    print("[Step 6] Extracting macro tropes...")
    fused_world.macro_tropes = _extract_macro_tropes(client, all_atoms_flat)

    print("[Step 6] Extracting plot threads...")
    fused_world.plot_threads = _extract_plot_threads(client, all_atoms_flat)
    return fused_world


def save_step6_output(fused_world: FusedWorld):
    path = save_world_snapshot(_STEP6_FILENAME, fused_world)
    print(f"[Step 6] Intermediate output saved -> {path.name}")
    return path


def load_step6_output() -> FusedWorld:
    return load_world_snapshot(_STEP6_FILENAME)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _call_llm_for_micro(client, summaries: str):
    schema = {
        "items": [
            {
                "interaction_name": "压制试探后的局面翻转",
                "applicable_scene": "交易 / 拍卖 / 资源争夺",
                "role_A": {"archetype": "压制者", "initial_psychology": "轻视"},
                "role_B": {"archetype": "主角", "initial_psychology": "克制观察"},
                "the_dance_of_interaction": {
                    "phase_1_probing": "试探底线",
                    "phase_2_escalation": "筹码与压力上桌",
                    "phase_3_reversal": "隐藏底牌被揭开",
                    "phase_4_resolution": "败方离场并埋下后续压力",
                },
                "bystander_effect": "围观者态度发生反转",
            }
        ]
    }
    prompt = (
        "请从下面的剧情摘要中提炼 3 到 5 个可复用的人物互动模板。"
        "重点描述试探、升级、翻盘和围观者态度变化。只返回合法 JSON，字段内容全部使用中文。\n\n"
        f"Schema: {json.dumps(schema, ensure_ascii=False)}\n\n"
        f"剧情摘要：\n{summaries}"
    )
    raw = chat_completion_json(client, system="只返回合法 JSON，字段内容全部使用中文。", user=prompt, json_mode=True, temperature=0.4)
    data = _safe_json_load(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    return []


def _extract_micro_interactions(client, atoms: List[PlotAtom]):
    valid_texts = [text for text in [atom.summary or atom.core_action for atom in atoms] if text and len(text) > 10]
    if not valid_texts:
        return []

    chunks = [valid_texts[i:i + 20] for i in range(0, len(valid_texts), 20)]
    if len(chunks) > 15:
        step = len(chunks) / 15
        chunks = [chunks[int(i * step)] for i in range(15)]

    extracted_items = []
    seen_names: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        print(f"  -> [Micro Extraction] Pass {index}/{len(chunks)}")
        summaries = "\n".join(f"- {text[:180]}" for text in chunk)
        for item in _call_llm_for_micro(client, summaries):
            name = str(item.get("interaction_name", "")).strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            extracted_items.append(item)
    return extracted_items


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _call_llm_for_macro(client, bird_eye_view: str):
    schema = {
        "items": [
            {
                "name": "低位积累后集中翻盘",
                "description": "主角先承受压力，再在关键节点掀翻局面。",
                "stages": ["低位受压", "资源积累", "局部翻转", "公开爆发"],
            }
        ]
    }
    prompt = (
        "请从下面的卷级概览中提炼 3 到 5 个宏观剧情模板。"
        "每个模板都要包含阶段推进，只返回合法 JSON，字段内容全部使用中文。\n\n"
        f"Schema: {json.dumps(schema, ensure_ascii=False)}\n\n"
        f"卷级概览：\n{bird_eye_view}"
    )
    raw = chat_completion_json(client, system="只返回合法 JSON，字段内容全部使用中文。", user=prompt, json_mode=True, temperature=0.4)
    data = _safe_json_load(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    return []


def _extract_macro_tropes(client, atoms: List[PlotAtom]):
    arc_summaries: dict[str, List[str]] = defaultdict(list)
    for atom in atoms:
        if atom.arc_name and atom.summary:
            arc_summaries[atom.arc_name].append(atom.summary)

    all_arcs = list(arc_summaries.items())
    if not all_arcs:
        return []

    chunks = [all_arcs[i:i + 10] for i in range(0, len(all_arcs), 10)]
    all_macro = []
    seen_names: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        print(f"  -> [Macro Extraction] Pass {index}/{len(chunks)}")
        bird_eye_view = "\n\n".join(
            f"[{arc}] " + " -> ".join(summary[:36] for summary in summaries[:6])
            for arc, summaries in chunk
        )
        for item in _call_llm_for_macro(client, bird_eye_view):
            name = str(item.get("name", "")).strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            all_macro.append(item)
    return all_macro


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _call_llm_for_threads(client, threads_data: str):
    schema = {
        "items": [
            {
                "name": "从互相利用走向共同护持",
                "thread_type": "关系长线",
                "stages": ["试探利用", "被迫合作", "共同负债", "关系重估"],
            }
        ]
    }
    prompt = (
        "请从下面的人物互动时间线中提炼 2 到 4 条长期关系线。"
        "只返回合法 JSON，字段内容全部使用中文。\n\n"
        f"Schema: {json.dumps(schema, ensure_ascii=False)}\n\n"
        f"互动时间线：\n{threads_data}"
    )
    raw = chat_completion_json(client, system="只返回合法 JSON，字段内容全部使用中文。", user=prompt, json_mode=True, temperature=0.4)
    data = _safe_json_load(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    return []


def _extract_plot_threads(client, atoms: List[PlotAtom]):
    pair_interactions: dict[tuple[str, str], Dict[str, List[str] | tuple[str, str]]] = {}
    for atom in atoms:
        mentions = _character_mentions(atom)
        if len(mentions) < 2:
            continue
        lead = mentions[:2]
        pair = tuple(sorted(item["key"] for item in lead))
        record = pair_interactions.setdefault(
            pair,
            {"display_pair": tuple(item["display"] for item in lead), "summaries": []},
        )
        record["summaries"].append(atom.summary or getattr(atom, "raw_summary", "") or atom.core_action)

    top_pairs = sorted(pair_interactions.items(), key=lambda item: len(item[1]["summaries"]), reverse=True)[:12]
    if not top_pairs:
        return []

    chunks = [top_pairs[i:i + 3] for i in range(0, len(top_pairs), 3)]
    all_threads = []
    seen_names: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        print(f"  -> [Thread Extraction] Pass {index}/{len(chunks)}")
        thread_text = "\n\n".join(
            f"[{record['display_pair'][0]} / {record['display_pair'][1]}] "
            + " -> ".join(summary[:60] for summary in record["summaries"][:8])
            for _pair, record in chunk
        )
        for item in _call_llm_for_threads(client, thread_text):
            name = str(item.get("name", "")).strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            all_threads.append(item)
    return all_threads
