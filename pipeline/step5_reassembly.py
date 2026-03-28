"""
step5_reassembly.py – Character-driven Plot Reassembly

Responsibilities:
  - Iterate through each skeleton node.
  - Retrieve equivalent events from the ChromaDB Events collection.
  - Prompt DeepSeek to adapt the retrieved event to the new protagonist's
    personality and the new cultivation system (alignment step).
  - Detect logical breaks using the realm DAG and retrieve Breakthrough
    Opportunities to bridge gaps.

Output:
  List[ReassembledEvent] – the new plot sequence ready for generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from pipeline.step3_knowledge_base import KnowledgeBase, FusedWorld
from pipeline.step4_role_casting import NarrativeSkeleton, SkeletonNode, Character
from pipeline.utils import get_deepseek_client, chat_completion_json


@dataclass
class ReassembledEvent:
    event_id: str
    arc_name: str
    realm_level: int
    pacing_role: str
    adapted_summary: str        # Alignment-adapted summary for new protagonist
    source_atom_ids: List[str] = field(default_factory=list)
    is_bridge: bool = False     # True = auto-generated breakthrough bridge


# ── Public API ────────────────────────────────────────────────────────────────

def reassemble_plot(
    skeleton: NarrativeSkeleton,
    kb: KnowledgeBase,
    fused_world: FusedWorld,
) -> List[ReassembledEvent]:
    """
    Build a new plot sequence by aligning skeleton nodes with retrieved events
    and the new protagonist's character.
    """
    client = get_deepseek_client()
    protagonist = skeleton.character_sheet.protagonist
    protagonist_desc = _format_protagonist(protagonist)
    realm_system = _format_realm_system(fused_world)

    # ── Build realm-level → topological-position index from the DAG ──────────
    # networkx validates the graph; the LLM is never asked "is this valid?".
    realm_topo_index: Dict[int, int] = _build_realm_topo_index(fused_world)

    reassembled: List[ReassembledEvent] = []
    prev_realm_level = 0

    for node in skeleton.nodes:
        # --- Bridge gap if realm skips (DAG-aware check) ---
        if _realm_gap_exists(prev_realm_level, node.realm_level, realm_topo_index):
            bridge = _create_bridge_event(
                client, kb, node, protagonist_desc, realm_system, prev_realm_level
            )
            reassembled.append(bridge)

        adapted = _adapt_node(client, kb, node, protagonist_desc, realm_system)
        reassembled.append(adapted)
        prev_realm_level = max(prev_realm_level, node.realm_level)

    print(f"[Step 5] Reassembled {len(reassembled)} events "
          f"(including {sum(1 for e in reassembled if e.is_bridge)} bridges)")
    return reassembled


# ── Step 5 persistence ────────────────────────────────────────────────────────

_STEP5_FILENAME = "step5_reassembled_plot.json"


def save_step5_output(reassembled: List[ReassembledEvent]) -> Path:
    """Serialise *reassembled* to ``intermediate_dir/step5_reassembled_plot.json``."""
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _STEP5_FILENAME
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in reassembled], f, ensure_ascii=False, indent=2)
    print(f"[Step 5] Intermediate output saved → {out_path}")
    return out_path


def load_step5_output(intermediate_dir: str | Path | None = None) -> List[ReassembledEvent]:
    """Load previously saved Step 5 output from ``intermediate_dir/step5_reassembled_plot.json``."""
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)
    in_path = inter_dir / _STEP5_FILENAME
    if not in_path.exists():
        raise FileNotFoundError(
            f"Step 5 intermediate file not found: {in_path}\n"
            "Run the pipeline from Step 5 first to generate it."
        )
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[Step 5] Loaded intermediate output from {in_path}")
    return [ReassembledEvent(**e) for e in data]


# ── DAG-aware realm gap helpers ───────────────────────────────────────────────

def _build_realm_topo_index(fused_world: FusedWorld) -> Dict[int, int]:
    """
    Return a mapping {realm_level: topological_position} derived from the
    cultivation-system DAG.

    networkx.is_directed_acyclic_graph() and networkx.topological_sort() are
    the sole validators – we never send the graph to the LLM to ask whether it
    is valid.  If the DAG is absent or invalid the function falls back to a
    simple level-equals-position mapping.
    """
    dag = fused_world.realm_dag
    if dag is None or not nx.is_directed_acyclic_graph(dag):
        # Fallback: treat each realm's numeric level as its own index.
        # Duplicate levels are resolved by keeping the first occurrence so
        # that the mapping is deterministic (realms are already level-sorted).
        index: Dict[int, int] = {}
        for r in fused_world.cultivation_realms:
            index.setdefault(r.level, r.level)
        return index

    topo_order = list(nx.topological_sort(dag))
    realm_name_to_level = {r.name: r.level for r in fused_world.cultivation_realms}
    index: Dict[int, int] = {}
    for topo_pos, name in enumerate(topo_order):
        if name in realm_name_to_level:
            index[realm_name_to_level[name]] = topo_pos
    return index


def _realm_gap_exists(
    prev_level: int, next_level: int, topo_index: Dict[int, int]
) -> bool:
    """
    Return True when the skeleton skips at least one intermediate realm level,
    meaning a bridge event is needed.

    Uses the topological-position index built from the DAG (not LLM logic).
    Falls back to simple arithmetic when either level is missing from the index.
    """
    if prev_level == 0:
        return False
    prev_pos = topo_index.get(prev_level)
    next_pos = topo_index.get(next_level)
    if prev_pos is None or next_pos is None:
        # Unknown realm level – fall back to simple arithmetic
        return next_level > prev_level + 1
    return next_pos > prev_pos + 1


# ── Internal helpers ──────────────────────────────────────────────────────────

def _format_protagonist(protagonist: Character) -> str:
    return (
        f"姓名：{protagonist.name}\n"
        f"道心/核心执念：{protagonist.dao_heart}\n"
        f"战斗风格：{protagonist.combat_style}\n"
        f"性格缺陷：{protagonist.personality_flaw}\n"
        f"特质：{', '.join(protagonist.traits)}"
    )


def _format_realm_system(fused_world: FusedWorld) -> str:
    lines = [f"世界名：{fused_world.world_name}", "修炼境界体系："]
    for realm in fused_world.cultivation_realms:
        lines.append(f"  第{realm.level}境 {realm.name}：{realm.breakthrough_condition}")
    return "\n".join(lines)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _adapt_node(
    client,
    kb: KnowledgeBase,
    node: SkeletonNode,
    protagonist_desc: str,
    realm_system: str,
) -> ReassembledEvent:
    # Retrieve similar events from other novels as inspiration
    similar_events = kb.query_events(
        query=node.original_summary or node.pacing_role,
        n_results=3,
        arc_filter=node.arc_name if node.arc_name else None,
    )
    retrieved_text = "\n".join(
        f"参考事件{i+1}：{e['document']}" for i, e in enumerate(similar_events)
    )
    source_ids = [e["id"] for e in similar_events]

    prompt = (
        "你是修仙小说情节改写专家，负责将原有情节骨架适配到新主角和新体系。\n\n"
        f"【新主角设定】\n{protagonist_desc}\n\n"
        f"【新修炼体系】\n{realm_system}\n\n"
        f"【当前节奏定位】境界弧：{node.arc_name}，叙事功能：{node.pacing_role}\n\n"
        f"【原骨架摘要】\n{node.original_summary}\n\n"
        f"【来自其他小说的参考事件（仅作灵感，禁止直接抄袭）】\n{retrieved_text}\n\n"
        "请按照新主角的性格和道心，将此情节节点重新设计：\n"
        "1. 行动逻辑必须符合主角道心和性格缺陷\n"
        "2. 修炼元素必须使用新修炼体系中的名称\n"
        "3. 输出200字以内的情节摘要\n"
        "4. 若主角行为与原骨架有出入，需给出符合人设的合理解释\n\n"
        '以JSON输出：{"adapted_summary": "情节摘要", "adaptation_note": "人设适配说明"}'
    )
    raw = chat_completion_json(
        client,
        system="你是专业的修仙小说情节人设对齐专家，只输出合法JSON。",
        user=prompt,
        json_mode=True,
    )
    try:
        data = json.loads(raw)
        adapted_summary = data.get("adapted_summary", node.original_summary)
    except (json.JSONDecodeError, AttributeError):
        adapted_summary = node.original_summary

    return ReassembledEvent(
        event_id=f"adapted_{node.node_id}",
        arc_name=node.arc_name,
        realm_level=node.realm_level,
        pacing_role=node.pacing_role,
        adapted_summary=adapted_summary,
        source_atom_ids=source_ids,
        is_bridge=False,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _create_bridge_event(
    client,
    kb: KnowledgeBase,
    next_node: SkeletonNode,
    protagonist_desc: str,
    realm_system: str,
    current_realm_level: int,
) -> ReassembledEvent:
    """Generate a breakthrough bridge event to fill a realm gap."""
    breakthrough_results = kb.query_breakthrough_opportunities(
        query=f"从第{current_realm_level}境突破到更高境界",
        n_results=3,
    )
    breakthrough_text = "\n".join(
        f"参考机缘{i+1}：{r['document']}" for i, r in enumerate(breakthrough_results)
    )

    prompt = (
        "你是修仙小说过渡情节设计师。\n\n"
        f"【新主角设定】\n{protagonist_desc}\n\n"
        f"【新修炼体系】\n{realm_system}\n\n"
        f"当前主角处于第{current_realm_level}境，"
        f"下一个剧情节点需要主角处于第{next_node.realm_level}境（{next_node.arc_name}）。\n"
        f"存在境界跳跃，需要设计一个合理的过渡机缘情节。\n\n"
        f"【可参考的突破机缘模板】\n{breakthrough_text}\n\n"
        "请设计一个符合主角道心的突破机缘情节（150字以内），要求：\n"
        "1. 突破过程有明确的前置因果\n"
        "2. 不能凭空掉落天材地宝\n"
        "3. 与主角性格和道心高度契合\n\n"
        '以JSON输出：{"bridge_summary": "过渡机缘情节摘要"}'
    )
    raw = chat_completion_json(
        client,
        system="你是专业的修仙小说过渡情节设计师，只输出合法JSON。",
        user=prompt,
        json_mode=True,
    )
    try:
        data = json.loads(raw)
        bridge_summary = data.get("bridge_summary", "主角经历磨砺，感悟突破。")
    except (json.JSONDecodeError, AttributeError):
        bridge_summary = "主角经历关键磨砺，感悟大道，突破至更高境界。"

    return ReassembledEvent(
        event_id=f"bridge_{next_node.node_id}",
        arc_name=next_node.arc_name,
        realm_level=current_realm_level + 1,
        pacing_role="境界突破过渡",
        adapted_summary=bridge_summary,
        source_atom_ids=[r["id"] for r in breakthrough_results],
        is_bridge=True,
    )
