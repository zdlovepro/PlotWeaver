"""
step5_reassembly.py – Character-driven Plot Reassembly (improved)

Key improvements over previous version:
1) Adds continuity memory (previous summary + story state) for each node.
2) Adds anti-repetition constraints and lightweight novelty checks.
3) Keeps DAG-aware ordering and bridge insertion.
4) Preserves output compatibility with Step 6 (ReassembledEvent unchanged).
5) Optional targeted regeneration support (only_event_ids + previous_events).

Output:
  List[ReassembledEvent] – the new plot sequence ready for generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

import config
from pipeline.step3_knowledge_base import KnowledgeBase, FusedWorld
from pipeline.step4_role_casting import NarrativeSkeleton, SkeletonNode, Character
from pipeline.utils import get_deepseek_client, chat_completion_json


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ReassembledEvent:
    event_id: str
    arc_name: str
    realm_level: int
    pacing_role: str
    adapted_summary: str
    source_atom_ids: List[str] = field(default_factory=list)
    is_bridge: bool = False


# ── Public API ────────────────────────────────────────────────────────────────

def reassemble_plot(
    skeleton: NarrativeSkeleton,
    kb: KnowledgeBase,
    fused_world: FusedWorld,
    only_event_ids: Optional[Set[str]] = None,
    previous_events: Optional[List[ReassembledEvent]] = None,
) -> List[ReassembledEvent]:
    """
    Build a new plot sequence by aligning skeleton nodes with retrieved events
    and the new protagonist's character.

    If only_event_ids is provided, only those nodes are regenerated;
    others are reused from previous_events when available.
    """
    client = get_deepseek_client()
    protagonist = skeleton.character_sheet.protagonist
    protagonist_desc = _format_protagonist(protagonist)
    realm_system = _format_realm_system(fused_world)

    # continuity memory
    prev_summary = ""
    story_state = ""
    recent_summaries: List[str] = []
    used_keywords: Set[str] = set()

    # for targeted regeneration
    prev_event_map = {e.event_id: e for e in (previous_events or [])}

    # Build realm topological index
    realm_topo_index: Dict[int, int] = _build_realm_topo_index(fused_world)

    # Pre-fix invalid realm_level=0 where possible (without mutating original objects deeply)
    fixed_nodes = [_fix_node_realm_level(n, fused_world) for n in skeleton.nodes]

    # Sort nodes by topological position
    sorted_nodes = sorted(
        fixed_nodes,
        key=lambda n: realm_topo_index.get(n.realm_level, n.realm_level),
    )

    reassembled: List[ReassembledEvent] = []
    prev_realm_level = 0

    for node in tqdm(sorted_nodes, desc="[Step 5] Reassembling plot", unit="node"):
        base_event_id = f"adapted_{node.node_id}"

        # targeted regeneration: reuse unchanged nodes
        if only_event_ids and node.node_id not in only_event_ids:
            if base_event_id in prev_event_map:
                reused = prev_event_map[base_event_id]
                reassembled.append(reused)
                prev_summary = reused.adapted_summary
                story_state = _tail_text(reused.adapted_summary, 80)
                recent_summaries.append(reused.adapted_summary)
                _update_used_keywords(used_keywords, reused.adapted_summary)
                prev_realm_level = max(prev_realm_level, node.realm_level)
                continue

        # Bridge if realm gap exists
        if _realm_gap_exists(prev_realm_level, node.realm_level, realm_topo_index):
            bridge = _create_bridge_event(
                client=client,
                kb=kb,
                next_node=node,
                protagonist_desc=protagonist_desc,
                realm_system=realm_system,
                current_realm_level=prev_realm_level,
                prev_summary=prev_summary,
                story_state=story_state,
                used_keywords=used_keywords,
            )
            reassembled.append(bridge)
            prev_summary = bridge.adapted_summary
            story_state = _tail_text(bridge.adapted_summary, 80)
            recent_summaries.append(bridge.adapted_summary)
            _update_used_keywords(used_keywords, bridge.adapted_summary)
            prev_realm_level = max(prev_realm_level, bridge.realm_level)

        adapted = _adapt_node(
            client=client,
            kb=kb,
            node=node,
            protagonist_desc=protagonist_desc,
            realm_system=realm_system,
            prev_summary=prev_summary,
            story_state=story_state,
            used_keywords=used_keywords,
            recent_summaries=recent_summaries,
        )
        reassembled.append(adapted)

        prev_summary = adapted.adapted_summary
        story_state = _tail_text(adapted.adapted_summary, 80)
        recent_summaries.append(adapted.adapted_summary)
        _update_used_keywords(used_keywords, adapted.adapted_summary)
        prev_realm_level = max(prev_realm_level, node.realm_level)

    tqdm.write(
        f"[Step 5] Reassembled {len(reassembled)} events "
        f"(including {sum(1 for e in reassembled if e.is_bridge)} bridges)"
    )
    return reassembled


# ── DAG helpers ───────────────────────────────────────────────────────────────

def _build_realm_topo_index(fused_world: FusedWorld) -> Dict[int, int]:
    dag = fused_world.realm_dag
    if dag is None or not nx.is_directed_acyclic_graph(dag):
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


def _realm_gap_exists(prev_level: int, next_level: int, topo_index: Dict[int, int]) -> bool:
    if prev_level == 0:
        return False
    prev_pos = topo_index.get(prev_level)
    next_pos = topo_index.get(next_level)
    if prev_pos is None or next_pos is None:
        return next_level > prev_level + 1
    return next_pos > prev_pos + 1


# ── Internal formatting / utility ─────────────────────────────────────────────

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


def _fix_node_realm_level(node: SkeletonNode, fused_world: FusedWorld) -> SkeletonNode:
    """
    If realm_level is 0, try to infer from arc_name against fused_world realm names.
    Fallback keeps original value.
    """
    if getattr(node, "realm_level", 0) != 0:
        return node

    inferred = 0
    arc = (node.arc_name or "").strip().lower()
    for r in fused_world.cultivation_realms:
        rn = (r.name or "").strip().lower()
        if rn and (rn in arc or arc in rn):
            inferred = r.level
            break

    if inferred == 0 and fused_world.cultivation_realms:
        # conservative fallback: first realm
        inferred = fused_world.cultivation_realms[0].level

    node.realm_level = inferred
    return node


def _tail_text(text: str, n: int) -> str:
    text = text or ""
    return text[-n:] if len(text) > n else text


def _tokens(text: str) -> Set[str]:
    words = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", text or "")
    return set(words)


def _jaccard(a: str, b: str) -> float:
    sa, sb = _tokens(a), _tokens(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _is_too_similar(candidate: str, recent: List[str], threshold: float = 0.45) -> bool:
    tail = recent[-2:] if len(recent) >= 2 else recent
    return any(_jaccard(candidate, r) >= threshold for r in tail)


def _update_used_keywords(used_keywords: Set[str], text: str) -> None:
    for w in _tokens(text):
        if len(w) >= 2:
            used_keywords.add(w)


def _forbidden_phrase_hits(text: str) -> int:
    # High-frequency bland phrases we want to suppress
    banned = ["逆天改命", "命运陷阱", "坚韧不拔", "聪慧地", "孤身探索", "发誓"]
    return sum(1 for b in banned if b in (text or ""))


# ── Core generation helpers ───────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _adapt_node(
    client,
    kb: KnowledgeBase,
    node: SkeletonNode,
    protagonist_desc: str,
    realm_system: str,
    prev_summary: str,
    story_state: str,
    used_keywords: Set[str],
    recent_summaries: List[str],
) -> ReassembledEvent:
    similar_events = kb.query_events(
        query=node.original_summary or node.pacing_role,
        n_results=3,
        arc_filter=node.arc_name if node.arc_name else None,
    )
    retrieved_text = "\n".join(
        f"参考事件{i+1}：{e['document']}" for i, e in enumerate(similar_events)
    )
    source_ids = [e["id"] for e in similar_events]

    avoid_kw = "、".join(sorted(list(used_keywords))[-20:]) if used_keywords else "无"

    prompt = (
        "你是修仙小说剧情重组专家。\n\n"
        "目标：把骨架节点改写为“有推进、有变化、不重复”的新剧情。\n"
        "必须满足：\n"
        "1) 与上一节点连续（承接状态）；\n"
        "2) 本节点必须引入至少1个“新信息”（新人物/新规则/新冲突/新线索）；\n"
        "3) 禁止复读模板化表达（如“逆天改命/命运陷阱”等空泛口号）；\n"
        "4) 禁止沿用原作专有名词；\n"
        "5) 字数 120-220 中文字。\n\n"
        f"【新主角设定】\n{protagonist_desc}\n\n"
        f"【新修炼体系】\n{realm_system}\n\n"
        f"【当前节点】境界弧：{node.arc_name}；叙事功能：{node.pacing_role}\n"
        f"【原骨架摘要】{node.original_summary}\n\n"
        f"【上一节点摘要】{prev_summary or '无'}\n"
        f"【当前故事状态】{story_state or '无'}\n\n"
        f"【参考事件（仅灵感）】\n{retrieved_text}\n\n"
        f"【应尽量避免复用关键词】{avoid_kw}\n\n"
        "仅输出JSON：\n"
        "{"
        "\"adapted_summary\":\"...\","
        "\"ending_state\":\"...\","
        "\"novelty_tags\":[\"新信息1\",\"新信息2\"],"
        "\"forbidden_reuse_detected\":false"
        "}"
    )

    raw = chat_completion_json(
        client,
        system="你是专业剧情重组助手，只输出合法JSON。",
        user=prompt,
        json_mode=True,
    )

    adapted_summary = node.original_summary or node.pacing_role
    try:
        data = json.loads(raw)
        adapted_summary = data.get("adapted_summary", adapted_summary) or adapted_summary
    except (json.JSONDecodeError, AttributeError):
        pass

    # lightweight anti-repetition rewrite
    if _is_too_similar(adapted_summary, recent_summaries) or _forbidden_phrase_hits(adapted_summary) >= 2:
        adapted_summary = _rewrite_with_novelty_bias(
            client=client,
            candidate=adapted_summary,
            prev_summary=prev_summary,
            story_state=story_state,
            pacing_role=node.pacing_role,
            arc_name=node.arc_name,
        )

    return ReassembledEvent(
        event_id=f"adapted_{node.node_id}",
        arc_name=node.arc_name,
        realm_level=node.realm_level,
        pacing_role=node.pacing_role,
        adapted_summary=adapted_summary,
        source_atom_ids=source_ids,
        is_bridge=False,
    )


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
def _rewrite_with_novelty_bias(
    client,
    candidate: str,
    prev_summary: str,
    story_state: str,
    pacing_role: str,
    arc_name: str,
) -> str:
    prompt = (
        "请重写下述剧情，使其更具体且不重复。\n"
        "硬性要求：\n"
        "1) 引入1个明确新线索或新冲突；\n"
        "2) 删除空话口号；\n"
        "3) 与上一节点有因果承接；\n"
        "4) 120-200字。\n\n"
        f"【上一节点】{prev_summary or '无'}\n"
        f"【状态】{story_state or '无'}\n"
        f"【当前定位】{arc_name} / {pacing_role}\n"
        f"【待重写文本】{candidate}\n\n"
        "仅输出JSON：{\"adapted_summary\":\"...\"}"
    )
    raw = chat_completion_json(
        client,
        system="你是剧情去重改写助手，只输出合法JSON。",
        user=prompt,
        json_mode=True,
    )
    try:
        data = json.loads(raw)
        return data.get("adapted_summary", candidate) or candidate
    except Exception:
        return candidate


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _create_bridge_event(
    client,
    kb: KnowledgeBase,
    next_node: SkeletonNode,
    protagonist_desc: str,
    realm_system: str,
    current_realm_level: int,
    prev_summary: str,
    story_state: str,
    used_keywords: Set[str],
) -> ReassembledEvent:
    breakthrough_results = kb.query_breakthrough_opportunities(
        query=f"从第{current_realm_level}境突破到更高境界",
        n_results=3,
    )
    breakthrough_text = "\n".join(
        f"参考机缘{i+1}：{r['document']}" for i, r in enumerate(breakthrough_results)
    )
    avoid_kw = "、".join(sorted(list(used_keywords))[-20:]) if used_keywords else "无"

    prompt = (
        "你是修仙小说过渡情节设计师。\n"
        "请设计“境界跨越桥段”，要求：\n"
        "1) 与上一节点有直接因果；\n"
        "2) 不可凭空机缘；\n"
        "3) 必须出现代价或风险；\n"
        "4) 120-180字，避免模板化词汇。\n\n"
        f"【新主角设定】\n{protagonist_desc}\n\n"
        f"【新修炼体系】\n{realm_system}\n\n"
        f"【当前境界】第{current_realm_level}境 -> 目标节点第{next_node.realm_level}境（{next_node.arc_name}）\n"
        f"【上一节点摘要】{prev_summary or '无'}\n"
        f"【当前故事状态】{story_state or '无'}\n"
        f"【避免复用关键词】{avoid_kw}\n\n"
        f"【可参考机缘模板】\n{breakthrough_text}\n\n"
        "仅输出JSON：{\"bridge_summary\":\"...\"}"
    )

    raw = chat_completion_json(
        client,
        system="你是专业过渡情节设计师，只输出合法JSON。",
        user=prompt,
        json_mode=True,
    )
    try:
        data = json.loads(raw)
        bridge_summary = data.get("bridge_summary", "主角经历代价与磨砺后完成突破。")
    except (json.JSONDecodeError, AttributeError):
        bridge_summary = "主角经历代价与磨砺后完成突破。"

    if _forbidden_phrase_hits(bridge_summary) >= 2:
        bridge_summary = bridge_summary.replace("逆天改命", "").replace("命运陷阱", "")

    return ReassembledEvent(
        event_id=f"bridge_{next_node.node_id}",
        arc_name=next_node.arc_name,
        realm_level=max(current_realm_level + 1, 1),
        pacing_role="境界突破过渡",
        adapted_summary=bridge_summary,
        source_atom_ids=[r["id"] for r in breakthrough_results],
        is_bridge=True,
    )


# ── Intermediate I/O ──────────────────────────────────────────────────────────

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
    if not in_path.exists():
        raise FileNotFoundError(
            f"Step 5 intermediate file not found: {in_path}\n"
            "Run the pipeline from Step 5 or earlier first to generate it."
        )
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = [ReassembledEvent(**e) for e in data]
    print(f"[Step 5] Loaded intermediate output from {in_path}")
    return events