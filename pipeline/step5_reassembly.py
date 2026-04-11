"""
step5_reassembly.py – Character-driven Plot Reassembly (With Macro & Micro Dynamics)

Key improvements:
1) Short-term Coherence: Rolling window of the last 3 events.
2) Long-term Coherence: Maps current progress (%) to the corresponding stage of Macro-Tropes and Plot Threads.
3) Micro Dynamics: Injects Director-level psychological push-and-pull templates.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

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

    for i, node in enumerate(tqdm(skeleton.nodes, desc="[Step 5] Reassembling plot", unit="node")):
        base_event_id = f"adapted_{node.node_id}"

        # 增量生成逻辑
        if only_event_ids and node.node_id not in only_event_ids:
            if base_event_id in prev_event_map:
                reused = prev_event_map[base_event_id]
                reassembled.append(reused)
                _update_rolling_context(recent_context, reused.adapted_summary)
                continue

        # 1. 中期连贯：当前活跃角色与恩怨局势
        current_stage_network = _get_current_stage_network(char_sheet, i, total_nodes)
        characters_desc = _format_characters_for_stage(char_sheet, current_stage_network)

        # 2. 长期连贯：计算当前进度，映射宏观套路与长线剧情的“当前阶段”
        progress_ratio = i / total_nodes if total_nodes > 1 else 0
        active_macro_stages = _get_active_macro_stages(fused_world.macro_tropes, fused_world.plot_threads, progress_ratio)

        # 3. 微观连贯：随��抽取一个极其细腻的微观博弈模板作为手法参考
        suggested_micro = random.choice(fused_world.micro_interactions) if getattr(fused_world, "micro_interactions", None) else None

        adapted = _adapt_node(
            client=client,
            kb=kb,
            node=node,
            characters_desc=characters_desc,
            realm_system=realm_system,
            recent_context=recent_context,
            active_macro_stages=active_macro_stages,
            suggested_micro=suggested_micro
        )
        reassembled.append(adapted)

        # 更新短期连贯窗口
        _update_rolling_context(recent_context, adapted.adapted_summary)

    tqdm.write(f"[Step 5] Reassembled {len(reassembled)} coherent events with macro/micro tracking.")
    return reassembled


# ── Long-term Coherence Tracking ─────────────────────────────────────────────

def _get_active_macro_stages(macro_tropes: List[Dict], plot_threads: List[Dict], progress_ratio: float) -> str:
    """根据全书进度百分比，精准定位长线剧情当前该写哪个阶段"""
    lines = []

    # 追踪套路主轴
    if macro_tropes:
        trope = macro_tropes[0] # 取第一主轴
        stages = trope.get("stages", [])
        if stages:
            idx = min(int(progress_ratio * len(stages)), len(stages) - 1)
            lines.append(f"【主线套路】：{trope.get('name')} -> 当前处于：{stages[idx]}")

    # 追踪感情/支线主轴
    if plot_threads:
        thread = plot_threads[0] # 取第一支线
        stages = thread.get("stages", [])
        if stages:
            idx = min(int(progress_ratio * len(stages)), len(stages) - 1)
            lines.append(f"【支线剧情（{thread.get('thread_type','')}）】：{thread.get('name')} -> 当前处于：{stages[idx]}")

    return "\n".join(lines) if lines else "无特殊长线约束。"


def _get_current_stage_network(char_sheet: CharacterSheet, current_idx: int, total_nodes: int) -> str:
    if not char_sheet.relationship_networks: return "暂无阶段关系网。"
    ratio = current_idx / total_nodes if total_nodes > 0 else 0
    stage_idx = int(ratio * len(char_sheet.relationship_networks))
    active_net = char_sheet.relationship_networks[min(stage_idx, len(char_sheet.relationship_networks) - 1)]
    return f"【近期群像局势（{active_net.stage}）】：\n活跃角色：{', '.join(active_net.active_characters)}\n局势与恩怨：{active_net.relationship_status}"

def _format_characters_for_stage(char_sheet: CharacterSheet, stage_desc: str) -> str:
    proto = char_sheet.protagonist
    proto_str = f"[主角] {proto.name} | 执念：{proto.dao_heart} | 缺陷：{proto.personality_flaw} | 战斗：{proto.combat_style}"
    supp_strs = [f"[{s.role}] {s.name} | 执念：{s.dao_heart}" for s in char_sheet.supporting]
    all_chars = proto_str + "\n" + "\n".join(supp_strs)
    return f"【角色图鉴】\n{all_chars}\n\n{stage_desc}"

def _format_realm_system(fused_world: FusedWorld) -> str:
    return "\n".join([f"世界名：{fused_world.world_name}"] + [f"  {r.name}（第{r.level}境）：{r.breakthrough_condition}" for r in fused_world.cultivation_realms])

def _update_rolling_context(context_list: List[str], new_summary: str, max_history: int = 3):
    if new_summary:
        context_list.append(new_summary)
        if len(context_list) > max_history: context_list.pop(0)


# ── Core Generation Helpers ──────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _adapt_node(
    client, kb: KnowledgeBase, node: SkeletonNode, characters_desc: str,
    realm_system: str, recent_context: List[str],
    active_macro_stages: str, suggested_micro: Optional[Dict[str, Any]]
) -> ReassembledEvent:

    # 跨卷无死角检索灵感
    similar_events = kb.query_events(query=node.original_summary or node.pacing_role, n_results=2)
    retrieved_text = "\n".join(f"- 灵感{i+1}：{e['document']}" for i, e in enumerate(similar_events))
    source_ids = [e["id"] for e in similar_events]

    context_str = "\n".join([f"前情{i+1}: {text}" for i, text in enumerate(recent_context)]) or "无前情。"

    micro_text = "无"
    if suggested_micro:
        micro_text = f"【{suggested_micro.get('interaction_name')}】\n角色A定位：{suggested_micro.get('role_A', {}).get('initial_psychology', '')}\n推拉过程：{json.dumps(suggested_micro.get('the_dance_of_interaction', {}), ensure_ascii=False)}"

    user_prompt = (
        "你是白金级修仙大纲总编剧，精通网文的草蛇灰线与情绪推拉。\n\n"
        "【当前任务】：将当前的骨架节点重组为一段连贯、有张力的剧情摘要（约150字）。\n"
        "你必须综合考虑【短期前情连贯】和【长线宏观进度】！\n\n"
        "=======================================\n"
        f"【全书长线进度追踪】（极度重要，你的剧情必须推动或符合这个阶段）：\n{active_macro_stages}\n\n"
        f"【本卷微观群像图鉴与局势】：\n{characters_desc}\n\n"
        f"【微观导演级推拉参考】（选用，用来刻画心理博弈）：\n{micro_text}\n\n"
        f"【过去三章短期前情】（必须顺滑承接！）：\n{context_str}\n\n"
        f"【当前骨架必须发生的核心事件】：{node.original_summary}\n"
        f"【可汲取的全书灵感碎片】：\n{retrieved_text}\n"
        "=======================================\n\n"
        "【排雷红线】：\n"
        f"1. 战力红线：主角目前仅在【第{node.realm_level}境】，严禁出现越阶太夸张的毁天灭地描写！\n"
        "2. 不要流水账：不要只写“主角去了哪里，得到了什么”，要写出人与人的利益冲突和阴谋拆招。\n\n"
        "仅输出JSON：\n"
        "{"
        "\"adapted_summary\":\"具体剧情摘要...\""
        "}"
    )

    raw = chat_completion_json(client, system="你是顶级编剧，只输出合法JSON。", user=user_prompt, json_mode=True)

    adapted_summary = node.original_summary or node.pacing_role
    try:
        data = json.loads(raw)
        adapted_summary = data.get("adapted_summary", adapted_summary) or adapted_summary
    except (json.JSONDecodeError, AttributeError):
        pass

    return ReassembledEvent(
        event_id=f"adapted_{node.node_id}", arc_name=node.arc_name,
        realm_level=node.realm_level, pacing_role=node.pacing_role,
        adapted_summary=adapted_summary,
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