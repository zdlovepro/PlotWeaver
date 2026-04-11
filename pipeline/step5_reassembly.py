"""
step5_reassembly.py – Character-driven Plot Reassembly (Event-level)

Key improvements:
1) REMOVED arc_filter: Events can now pull inspiration from ANY volume across ANY novel, enabling true plot fusion.
2) Enforces Power Scaling (战力匹配): Actions must match the current realm level.
3) Enforces "Show, don't tell": Forbids the LLM from verbatim copying personality traits.
4) Introduces Event-level Trope usage & Cooldowns.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

import config
from pipeline.step3_knowledge_base import KnowledgeBase, FusedWorld
from pipeline.step4_role_casting import NarrativeSkeleton, SkeletonNode, CharacterSheet
from pipeline.utils import get_deepseek_client, chat_completion_json


@dataclass
class ReassembledEvent:
    event_id: str
    arc_name: str
    realm_level: int
    pacing_role: str
    adapted_summary: str
    used_trope: str = ""
    source_atom_ids: List[str] = field(default_factory=list)
    is_bridge: bool = False


def reassemble_plot(
    skeleton: NarrativeSkeleton,
    kb: KnowledgeBase,
    fused_world: FusedWorld,
    only_event_ids: Optional[Set[str]] = None,
    previous_events: Optional[List[ReassembledEvent]] = None,
) -> List[ReassembledEvent]:
    client = get_deepseek_client()
    char_sheet = skeleton.character_sheet
    realm_system = _format_realm_system(fused_world)

    recent_context: List[str] = []
    prev_event_map = {e.event_id: e for e in (previous_events or [])}
    reassembled: List[ReassembledEvent] = []
    total_nodes = len(skeleton.nodes)

    trope_cooldowns: Dict[str, int] = {}
    EVENT_COOLDOWN_DURATION = 6

    for i, node in enumerate(tqdm(skeleton.nodes, desc="[Step 5] Reassembling plot", unit="node")):
        base_event_id = f"adapted_{node.node_id}"

        if only_event_ids and node.node_id not in only_event_ids:
            if base_event_id in prev_event_map:
                reused = prev_event_map[base_event_id]
                reassembled.append(reused)
                _update_rolling_context(recent_context, reused.adapted_summary)

                if reused.used_trope and reused.used_trope != "无":
                    trope_cooldowns[reused.used_trope] = EVENT_COOLDOWN_DURATION

                for t in list(trope_cooldowns.keys()):
                    if trope_cooldowns[t] > 0: trope_cooldowns[t] -= 1
                continue

        for t in list(trope_cooldowns.keys()):
            if trope_cooldowns[t] > 0: trope_cooldowns[t] -= 1

        available_tropes = [t for t in fused_world.classic_tropes if trope_cooldowns.get(t.get("name", ""), 0) == 0]
        suggested_tropes = random.sample(available_tropes, min(3, len(available_tropes))) if available_tropes else []

        current_stage_network = _get_current_stage_network(char_sheet, i, total_nodes)
        characters_desc = _format_characters_for_stage(char_sheet, current_stage_network)

        adapted = _adapt_node(
            client=client,
            kb=kb,
            node=node,
            characters_desc=characters_desc,
            realm_system=realm_system,
            recent_context=recent_context,
            suggested_tropes=suggested_tropes
        )
        reassembled.append(adapted)
        _update_rolling_context(recent_context, adapted.adapted_summary)

        if adapted.used_trope and adapted.used_trope != "无":
            trope_cooldowns[adapted.used_trope] = EVENT_COOLDOWN_DURATION
            tqdm.write(f"  -> 事件 {i+1} 化用套路：【{adapted.used_trope}】(冷却6个事件)")

    tqdm.write(f"[Step 5] Reassembled {len(reassembled)} coherent events.")
    return reassembled


def _get_current_stage_network(char_sheet: CharacterSheet, current_idx: int, total_nodes: int) -> str:
    if not char_sheet.relationship_networks: return "暂无阶段关系网。"
    ratio = current_idx / total_nodes if total_nodes > 0 else 0
    stage_idx = int(ratio * len(char_sheet.relationship_networks))
    active_net = char_sheet.relationship_networks[min(stage_idx, len(char_sheet.relationship_networks) - 1)]
    return f"【时期：{active_net.stage}】\n活跃角色：{', '.join(active_net.active_characters)}\n局势：{active_net.relationship_status}"

def _format_characters_for_stage(char_sheet: CharacterSheet, stage_desc: str) -> str:
    proto = char_sheet.protagonist
    proto_str = f"[主角] {proto.name} | 执念：{proto.dao_heart} | 缺陷：{proto.personality_flaw} | 战斗：{proto.combat_style}"
    supp_strs = [f"[{s.role}] {s.name} | 执念：{s.dao_heart}" for s in char_sheet.supporting]
    all_chars = proto_str + "\n" + "\n".join(supp_strs)
    return f"【角色图鉴】\n{all_chars}\n\n【本事件人物局势】\n{stage_desc}"

def _format_realm_system(fused_world: FusedWorld) -> str:
    return "\n".join([f"世界名：{fused_world.world_name}", "体系："] + [f"  {r.name}（第{r.level}境）：{r.breakthrough_condition}" for r in fused_world.cultivation_realms])

def _update_rolling_context(context_list: List[str], new_summary: str, max_history: int = 3):
    if new_summary:
        context_list.append(new_summary)
        if len(context_list) > max_history: context_list.pop(0)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _adapt_node(
    client, kb: KnowledgeBase, node: SkeletonNode, characters_desc: str,
    realm_system: str, recent_context: List[str], suggested_tropes: List[Dict[str, str]]
) -> ReassembledEvent:

    # 核心改动：移除了 arc_filter=node.arc_name，允许真正的跨卷、跨书灵金融合！
    similar_events = kb.query_events(
        query=node.original_summary or node.pacing_role,
        n_results=2
    )

    retrieved_text = "\n".join(f"- 灵感{i+1}：{e['document']}" for i, e in enumerate(similar_events))
    source_ids = [e["id"] for e in similar_events]
    context_str = "\n".join([f"前情{i+1}: {text}" for i, text in enumerate(recent_context)]) or "无前情。"
    tropes_text = "\n".join(f"- 【{t.get('name')}】：{t.get('description')}" for t in suggested_tropes) if suggested_tropes else "无可用套路。"

    user_prompt = (
        "你是顶级的修仙剧情编剧。任务：将骨架节点重组为一段150字左右的连贯剧情。\n\n"
        "【核心排雷要求】（违反将导致不合格）：\n"
        f"1) 战力边界：当前主角处于【第{node.realm_level}境】。主角的战斗和行事手段必须严格局限在该境界范围内，底层修士绝不可出现“毁天灭地、撕裂虚空”的夸张表现！\n"
        "2) 隐性人设 (Show, don't tell)：绝对禁止在剧情中直接复制粘贴角色的性格特征词（如'因为他腹黑/缺乏安全感...'）。你必须用具体的行为（如'他提前布置了三道雷符'）来隐晦地展现人设！\n"
        "3) 细节落地：禁止堆砌套话成语，要写出具体的手段、心机或利益冲突。\n"
        "4) 因果连贯：必须顺滑承接【近期前情回顾】。\n\n"
        "【套路化用（可选）】：以下是推荐套路，契合则用，不契合填'无'。\n"
        f"{tropes_text}\n\n"
        f"{characters_desc}\n\n"
        f"【原骨架走向】{node.original_summary}\n\n"
        f"【近期前情回顾】\n{context_str}\n\n"
        f"【灵感桥段（来自全库，可融合使用）】\n{retrieved_text}\n\n"
        "仅输出JSON：\n"
        "{"
        "\"adapted_summary\":\"具体剧情...\","
        "\"used_trope\":\"填入套路名，未用填'无'\""
        "}"
    )

    raw = chat_completion_json(client, system="你是编剧，只输出合法JSON。", user=user_prompt, json_mode=True)

    adapted_summary = node.original_summary or node.pacing_role
    used_trope = "无"
    try:
        data = json.loads(raw)
        adapted_summary = data.get("adapted_summary", adapted_summary) or adapted_summary
        used_trope = data.get("used_trope", "无")
    except (json.JSONDecodeError, AttributeError):
        pass

    return ReassembledEvent(
        event_id=f"adapted_{node.node_id}", arc_name=node.arc_name,
        realm_level=node.realm_level, pacing_role=node.pacing_role,
        adapted_summary=adapted_summary, used_trope=used_trope,
        source_atom_ids=source_ids, is_bridge=False
    )


# ── Intermediate I/O ────────────────────────────────────────────────────────

_STEP5_FILENAME = "step5_reassembled_plot.json"

def save_step5_output(events: List[ReassembledEvent]) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _STEP5_FILENAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in events], f, ensure_ascii=False, indent=2)
    print(f"[Step 5] Intermediate output saved → {out_path}", flush=True)
    return out_path

def load_step5_output(intermediate_dir: str | Path | None = None) -> List[ReassembledEvent]:
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)
    in_path = inter_dir / _STEP5_FILENAME
    if not in_path.exists(): raise FileNotFoundError(f"Step 5 file not found: {in_path}")
    with open(in_path, "r", encoding="utf-8") as f: data = json.load(f)
    return [ReassembledEvent(**e) for e in data]