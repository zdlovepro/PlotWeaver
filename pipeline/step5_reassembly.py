"""
step5_reassembly.py – Character-driven Plot Reassembly (With Macro & Micro Dynamics)

Key improvements:
1) Character Spillage Fixed: Only characters explicitly valid for the current event index are injected into the LLM prompt.
2) Long-term Coherence: Maps current progress (%) to the corresponding stage of Macro-Tropes and Plot Threads.
3) Micro Dynamics: Injects Director-level psychological push-and-pull templates.
"""

from __future__ import annotations

from collections import Counter
import json
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

import config
from pipeline.step3_knowledge_base import KnowledgeBase, FusedWorld
from pipeline.step4_role_casting import NarrativeSkeleton, SkeletonNode, CharacterSheet, Character
from pipeline.utils import get_deepseek_client, chat_completion_json

_GENERIC_REPEAT_MOTIFS = (
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
    recent_micro_names: List[str] = []
    prev_event_map = {e.event_id: e for e in (previous_events or [])}
    reassembled: List[ReassembledEvent] = []
    total_nodes = len(skeleton.nodes)

    for i, node in enumerate(tqdm(skeleton.nodes, desc="[Step 5] Reassembling plot", unit="node")):
        base_event_id = f"adapted_{node.node_id}"

        if only_event_ids and node.node_id not in only_event_ids:
            if base_event_id in prev_event_map:
                reused = prev_event_map[base_event_id]
                reassembled.append(reused)
                _update_rolling_context(recent_context, reused.adapted_summary)
                continue

        # 1. 精确的角色控制（过滤掉还没登场和已经退场的人）
        characters_desc = _format_characters_for_stage(char_sheet, i)

        # 2. 长期连贯
        progress_ratio = i / total_nodes if total_nodes > 1 else 0
        active_macro_stages = _get_active_macro_stages(fused_world.macro_tropes, fused_world.plot_threads, progress_ratio)

        # 3. 微观连贯
        suggested_micro = _pick_micro_interaction(
            getattr(fused_world, "micro_interactions", None),
            recent_micro_names,
        )
        anti_repeat_rules = _build_anti_repetition_rules(recent_context, recent_micro_names)

        adapted = _adapt_node(
            client=client,
            kb=kb,
            node=node,
            characters_desc=characters_desc,
            realm_system=realm_system,
            recent_context=recent_context,
            active_macro_stages=active_macro_stages,
            suggested_micro=suggested_micro,
            anti_repeat_rules=anti_repeat_rules,
        )
        reassembled.append(adapted)
        _update_rolling_context(recent_context, adapted.adapted_summary)
        _update_recent_micro_names(recent_micro_names, adapted.used_trope)

    tqdm.write(f"[Step 5] Reassembled {len(reassembled)} coherent events with strict character entry/exit limits.")
    return reassembled


def _is_char_active(char: Character, current_idx: int) -> bool:
    """根据 entry_event 和 exit_event 中的数字，判断角色在当前事件是否可见"""
    entry_match = re.search(r'\d+', char.entry_event)
    entry_idx = int(entry_match.group()) - 1 if entry_match else 0

    exit_match = re.search(r'\d+', char.exit_event)
    exit_idx = int(exit_match.group()) - 1 if exit_match else 9999

    return entry_idx <= current_idx <= exit_idx


def _format_characters_for_stage(char_sheet: CharacterSheet, current_idx: int) -> str:
    """只向 LLM 暴露本章有资格登场的角色，严格控制泄漏"""
    proto = char_sheet.protagonist
    active_chars = [f"[主角] {proto.name} | 执念：{proto.dao_heart} | 缺陷：{proto.personality_flaw} | 战斗：{proto.combat_style}"]

    for s in char_sheet.supporting:
        if _is_char_active(s, current_idx):
            active_chars.append(f"[{s.role}] {s.name} | 执念：{s.dao_heart}")

    return "【当前事件可调用的活跃角色池（严格限制，未列出的人物绝不应出场）】\n" + "\n".join(active_chars)


def _get_active_macro_stages(macro_tropes: List[Dict], plot_threads: List[Dict], progress_ratio: float) -> str:
    lines = []
    if macro_tropes:
        trope = macro_tropes[0]
        stages = trope.get("stages", [])
        if stages:
            idx = min(int(progress_ratio * len(stages)), len(stages) - 1)
            lines.append(f"【主线套路】：{trope.get('name')} -> 当前处于：{stages[idx]}")
    if plot_threads:
        thread = plot_threads[0]
        stages = thread.get("stages", [])
        if stages:
            idx = min(int(progress_ratio * len(stages)), len(stages) - 1)
            lines.append(f"【支线剧情】：{thread.get('name')} -> 当前处于：{stages[idx]}")
    return "\n".join(lines) if lines else "无特殊长线约束。"


def _format_realm_system(fused_world: FusedWorld) -> str:
    return "\n".join([f"世界名：{fused_world.world_name}"] + [f"  {r.name}（第{r.level}境）：{r.breakthrough_condition}" for r in fused_world.cultivation_realms])

def _update_rolling_context(context_list: List[str], new_summary: str, max_history: int = 3):
    if new_summary:
        context_list.append(new_summary)
        if len(context_list) > max_history: context_list.pop(0)


def _pick_micro_interaction(
    interactions: Optional[List[Dict[str, Any]]],
    recent_micro_names: List[str],
) -> Optional[Dict[str, Any]]:
    if not interactions:
        return None

    recent_name = recent_micro_names[-1] if recent_micro_names else ""
    candidates = [
        interaction
        for interaction in interactions
        if interaction.get("interaction_name", "") != recent_name
    ]
    pool = candidates or interactions
    return random.choice(pool)


def _update_recent_micro_names(recent_micro_names: List[str], used_trope: str, max_history: int = 3) -> None:
    trope = (used_trope or "").strip()
    if trope:
        recent_micro_names.append(trope)
        if len(recent_micro_names) > max_history:
            recent_micro_names.pop(0)


def _build_anti_repetition_rules(
    recent_context: List[str],
    recent_micro_names: List[str],
) -> str:
    if not recent_context and not recent_micro_names:
        return "No recent repetition pressure yet."

    motif_counter: Counter[str] = Counter()
    for summary in recent_context[-3:]:
        for motif in _GENERIC_REPEAT_MOTIFS:
            if motif in summary:
                motif_counter[motif] += 1

    repeated_motifs = [motif for motif, count in motif_counter.items() if count >= 2]
    recent_micro_text = ", ".join(recent_micro_names[-2:]) or "none"

    rules = [
        "Anti-repetition rules:",
        "- The next event must use a different obstacle, clue, reversal, and payoff from the last 3 events.",
        "- Do not repeat the exact combo of villain trap -> protagonist spots it -> ally secretly helps -> sudden reversal.",
        f"- Recently used micro templates: {recent_micro_text}. Pick a fresh interaction rhythm.",
    ]
    if repeated_motifs:
        rules.append(
            "- These motifs are already overused recently and should be avoided unless the current event summary explicitly demands them: "
            + ", ".join(repeated_motifs)
        )
    return "\n".join(rules)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _adapt_node(
    client, kb: KnowledgeBase, node: SkeletonNode, characters_desc: str,
    realm_system: str, recent_context: List[str],
    active_macro_stages: str, suggested_micro: Optional[Dict[str, Any]],
    anti_repeat_rules: str,
) -> ReassembledEvent:

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
        "=======================================\n"
        f"【全书长线进度追踪】（极度重要，你的剧情必须推动或符合这个阶段）：\n{active_macro_stages}\n\n"
        f"{characters_desc}\n\n"
        f"【微观导演级推拉参考】（选用，用来刻画心理博弈）：\n{micro_text}\n\n"
        f"【过去三章短期前情】（必须顺滑承接！）：\n{context_str}\n\n"
        f"【当前骨架必须发生的核心事件】：{node.original_summary}\n"
        f"【可汲取的全书灵感碎片】：\n{retrieved_text}\n"
        f"{anti_repeat_rules}\n"
        "=======================================\n\n"
        "【排雷红线】：\n"
        f"1. 战力红线：主角目前仅在【第{node.realm_level}境】。\n"
        "2. 不要流水账：要写出人与人的利益冲突和阴谋拆招。\n"
        "3. 严禁让未在【当前事件可调用的活跃角色池】中列出的人物出场！\n\n"
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
        used_trope=(suggested_micro or {}).get("interaction_name", ""),
        source_atom_ids=source_ids, is_bridge=False
    )


# ── Intermediate I/O ──

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
