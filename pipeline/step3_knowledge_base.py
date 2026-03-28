"""
step3_knowledge_base.py – RAG Knowledge Base & World Building

Responsibilities:
  - Populate four ChromaDB collections:
      * events          – PlotAtoms from all source novels
      * character_traits – Character role archetypes
      * breakthrough_opportunities – Realm breakthrough / bridge events
      * cultivation_systems – Realm names, rules, breakthroughs
  - Use DeepSeek to fuse a new unified Cultivation System.
  - Build a NetworkX DAG for realm progression (topological ordering).
  - Extract a one-sentence global theme.

Output:
  KnowledgeBase object exposing query helpers and the fused world data.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from pipeline.step2_extraction import PlotAtom
from pipeline.utils import get_deepseek_client, get_chromadb_client, chat_completion_json


# ── ChromaDB collection names ─────────────────────────────────────────────────
COL_EVENTS = "events"
COL_CHARACTER_TRAITS = "character_traits"
COL_BREAKTHROUGH = "breakthrough_opportunities"
COL_CULTIVATION = "cultivation_systems"


@dataclass
class CultivationRealm:
    name: str
    level: int
    breakthrough_condition: str
    special_abilities: List[str] = field(default_factory=list)


@dataclass
class FusedWorld:
    cultivation_realms: List[CultivationRealm] = field(default_factory=list)
    realm_dag: Optional[nx.DiGraph] = None
    global_theme: str = ""
    world_name: str = ""
    raw_system_text: str = ""


class KnowledgeBase:
    """Thin wrapper around ChromaDB collections with convenience query methods."""

    def __init__(self):
        self._chroma = get_chromadb_client()
        self._events = self._chroma.get_or_create_collection(COL_EVENTS)
        self._chars = self._chroma.get_or_create_collection(COL_CHARACTER_TRAITS)
        self._breakthroughs = self._chroma.get_or_create_collection(COL_BREAKTHROUGH)
        self._cultivation = self._chroma.get_or_create_collection(COL_CULTIVATION)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_events(self, atoms: List[PlotAtom]) -> None:
        if not atoms:
            return
        self._events.add(
            ids=[f"{a.atom_id}_{uuid.uuid4().hex[:8]}" for a in atoms],
            documents=[a.summary or a.core_action for a in atoms],
            metadatas=[
                {
                    "original_id": a.atom_id,
                    "novel_source": a.novel_source,
                    "arc_name": a.arc_name,
                    "conflict_type": a.conflict_type,
                    "narrative_function": a.narrative_function,
                    "tension_level": str(a.tension_level),
                    "cultivation_elements": json.dumps(
                        a.cultivation_elements, ensure_ascii=False
                    ),
                }
                for a in atoms
            ],
        )

    def add_character_traits(self, traits: List[Dict[str, Any]]) -> None:
        """traits: list of dicts with keys 'id', 'text', 'metadata'."""
        if not traits:
            return
        self._chars.add(
            ids=[t["id"] for t in traits],
            documents=[t["text"] for t in traits],
            metadatas=[t.get("metadata", {}) for t in traits],
        )

    def add_breakthrough_opportunities(self, items: List[Dict[str, Any]]) -> None:
        if not items:
            return
        self._breakthroughs.add(
            ids=[i["id"] for i in items],
            documents=[i["text"] for i in items],
            metadatas=[i.get("metadata", {}) for i in items],
        )

    def add_cultivation_system(self, realms: List[CultivationRealm]) -> None:
        if not realms:
            return
        self._cultivation.add(
            ids=[f"realm_{r.level}" for r in realms],
            documents=[
                f"{r.name}（第{r.level}境）：{r.breakthrough_condition}"
                for r in realms
            ],
            metadatas=[
                {
                    "name": r.name,
                    "level": str(r.level),
                    "abilities": json.dumps(r.special_abilities, ensure_ascii=False),
                }
                for r in realms
            ],
        )

    # ── Query helpers ─────────────────────────────────────────────────────────

    def query_events(
        self, query: str, n_results: int = 5, arc_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        where = {"arc_name": arc_filter} if arc_filter else None
        kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": n_results}
        if where:
            kwargs["where"] = where
        res = self._events.query(**kwargs)
        return _format_results(res)

    def query_character_traits(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        res = self._chars.query(query_texts=[query], n_results=n_results)
        return _format_results(res)

    def query_breakthrough_opportunities(
        self, query: str, n_results: int = 3
    ) -> List[Dict[str, Any]]:
        res = self._breakthroughs.query(query_texts=[query], n_results=n_results)
        return _format_results(res)

    def query_cultivation(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        res = self._cultivation.query(query_texts=[query], n_results=n_results)
        return _format_results(res)


# ── Public API ────────────────────────────────────────────────────────────────

def build_knowledge_base(
    all_atoms: dict[str, List[PlotAtom]],
) -> tuple[KnowledgeBase, FusedWorld]:
    """
    Populate the ChromaDB knowledge base from all extracted plot atoms,
    then fuse a new cultivation system and build the world.
    """
    kb = KnowledgeBase()
    client = get_deepseek_client()

    all_atoms_flat: List[PlotAtom] = [
        atom for atoms in all_atoms.values() for atom in atoms
    ]

    print(f"[Step 3] Adding {len(all_atoms_flat)} events to ChromaDB...")
    kb.add_events(all_atoms_flat)

    print("[Step 3] Extracting character traits...")
    char_traits = _extract_character_traits(client, all_atoms_flat)
    kb.add_character_traits(char_traits)

    print("[Step 3] Extracting breakthrough opportunities...")
    breakthroughs = _extract_breakthroughs(client, all_atoms_flat)
    kb.add_breakthrough_opportunities(breakthroughs)

    print("[Step 3] Fusing cultivation system...")
    fused_world = _fuse_cultivation_system(client, all_atoms_flat)
    kb.add_cultivation_system(fused_world.cultivation_realms)

    print("[Step 3] Extracting global theme...")
    fused_world.global_theme = _extract_global_theme(client, all_atoms_flat)

    print(f"[Step 3] Global theme: {fused_world.global_theme}")
    return kb, fused_world


# ── Internal helpers ──────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _extract_character_traits(
    client, atoms: List[PlotAtom]
) -> List[Dict[str, Any]]:
    all_chars: Dict[str, List[str]] = {}
    for atom in atoms:
        for ch in atom.characters:
            if ch not in all_chars:
                all_chars[ch] = []
            all_chars[ch].append(atom.core_action)

    char_summaries = [
        f"{name}：主要行动包括 {', '.join(actions[:3])}"
        for name, actions in list(all_chars.items())[:50]
    ]
    char_text = "\n".join(char_summaries)

    prompt = (
        "你是修仙小说角色分析师。\n"
        "以下是多本小说中出现的角色及其主要行动。\n"
        "请为每类角色提炼一个原型特质描述（50字以内），以JSON数组输出：\n"
        '[{"archetype": "冷傲天才型", "description": "...", "traits": ["冷漠","自负","天赋异禀"]}, ...]\n\n'
        f"角色行动数据：\n{char_text[:config.MAX_TEXT_CHUNK_LENGTH]}"
    )
    raw = chat_completion_json(
        client,
        system="你是修仙小说角色原型分析师，只输出合法JSON数组。",
        user=prompt,
        json_mode=True,
    )
    try:
        archetypes = json.loads(raw)
        if not isinstance(archetypes, list):
            archetypes = archetypes.get("archetypes", [])
    except (json.JSONDecodeError, AttributeError):
        archetypes = []

    return [
        {
            "id": f"archetype_{i}",
            "text": f"{a.get('archetype', '')}：{a.get('description', '')}",
            "metadata": {
                "archetype": a.get("archetype", ""),
                "traits": json.dumps(a.get("traits", []), ensure_ascii=False),
            },
        }
        for i, a in enumerate(archetypes)
    ]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _extract_breakthroughs(
    client, atoms: List[PlotAtom]
) -> List[Dict[str, Any]]:
    breakthrough_atoms = [
        a for a in atoms
        if any(kw in (a.narrative_function + a.cultivation_elements.__str__())
               for kw in ["突破", "晋级", "机缘", "传承", "天劫"])
    ][:30]

    summaries = "\n".join(
        f"- [{a.arc_name}] {a.summary[:100]}" for a in breakthrough_atoms
    )
    prompt = (
        "以下是多本修仙小说中与境界突破相关的情节片段。\n"
        "请提炼出10条最具代表性的'突破机缘模板'，每条包含：\n"
        "触发条件（前置状态）、机缘内容、突破方式。\n"
        "以JSON数组输出：\n"
        '[{"id": "breakthrough_0", "trigger": "主角被追杀、身受重伤", '
        '"opportunity": "坠入古墓发现前辈遗留传承", '
        '"breakthrough_method": "参悟传承感悟突破"}, ...]\n\n'
        f"情节片段：\n{summaries}"
    )
    raw = chat_completion_json(
        client,
        system="你是修仙小说突破机缘模板提炼专家，只输出合法JSON数组。",
        user=prompt,
        json_mode=True,
    )
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            items = items.get("breakthroughs", [])
    except (json.JSONDecodeError, AttributeError):
        items = []

    return [
        {
            "id": item.get("id", f"breakthrough_{i}"),
            "text": (
                f"触发条件：{item.get('trigger', '')}；"
                f"机缘：{item.get('opportunity', '')}；"
                f"突破方式：{item.get('breakthrough_method', '')}"
            ),
            "metadata": {
                "trigger": item.get("trigger", ""),
                "opportunity": item.get("opportunity", ""),
                "breakthrough_method": item.get("breakthrough_method", ""),
            },
        }
        for i, item in enumerate(items)
    ]


_CULTIVATION_SYSTEM_SCHEMA = """\
{
  "world_name": "新世界名称",
  "realms": [
    {
      "name": "境界名（原创，非直接抄袭）",
      "level": 1,
      "breakthrough_condition": "突破所需条件",
      "special_abilities": ["能力1", "能力2"]
    }
  ]
}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _fuse_cultivation_system(
    client, atoms: List[PlotAtom]
) -> FusedWorld:
    all_elements: List[str] = []
    for atom in atoms:
        all_elements.extend(atom.cultivation_elements)
    unique_elements = list(dict.fromkeys(all_elements))[:60]

    prompt = (
        "你是一位世界观设计大师。\n"
        f"以下是从多本修仙小说中收集的修炼元素：\n{unique_elements}\n\n"
        "请融合这些元素，设计一套全新的、内部逻辑自洽的修炼体系（至少8个境界）。\n"
        "要求：\n"
        "1. 境界名称必须原创（不直接沿用任何现有小说的境界名）\n"
        "2. 每个境界的突破条件要有逻辑递进\n"
        "3. 整体体系要有独特的世界观背景\n\n"
        f"请严格按照以下JSON Schema输出：\n{_CULTIVATION_SYSTEM_SCHEMA}"
    )
    raw = chat_completion_json(
        client,
        system="你是创意世界观设计师，只输出合法JSON。",
        user=prompt,
        json_mode=True,
    )
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, AttributeError):
        data = {}

    realms_data = data.get("realms", [])
    realms = [
        CultivationRealm(
            name=r.get("name", f"境界{r.get('level', i)}"),
            level=int(r.get("level", i + 1)),
            breakthrough_condition=r.get("breakthrough_condition", ""),
            special_abilities=r.get("special_abilities", []),
        )
        for i, r in enumerate(realms_data)
    ]
    realms.sort(key=lambda r: r.level)

    # Build DAG
    dag = nx.DiGraph()
    for realm in realms:
        dag.add_node(realm.name, level=realm.level)
    for i in range(len(realms) - 1):
        dag.add_edge(realms[i].name, realms[i + 1].name)

    # ── NetworkX validation: no LLM for graph-validity checks ─────────────────
    # LLM is only called here if networkx detects a cycle (broken link).
    if not nx.is_directed_acyclic_graph(dag):
        cycles = list(nx.simple_cycles(dag))
        print(f"[Step 3] WARNING: cultivation DAG has cycles {cycles}; using LLM to fix.")
        realms, dag = _fix_realm_dag_cycles(client, realms, cycles)
    else:
        # topological_sort confirms a valid linear progression; re-order realms
        # to match the canonical topological ordering from the graph.
        topo_names = list(nx.topological_sort(dag))
        realm_map = {r.name: r for r in realms}
        ordered = [realm_map[n] for n in topo_names if n in realm_map]
        if len(ordered) == len(realms):
            realms = ordered
        else:
            print(
                f"[Step 3] WARNING: topological ordering returned {len(ordered)} realms "
                f"but expected {len(realms)}; keeping level-sorted order."
            )

    return FusedWorld(
        cultivation_realms=realms,
        realm_dag=dag,
        world_name=data.get("world_name", "新世界"),
        raw_system_text=raw,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _fix_realm_dag_cycles(
    client,
    realms: List[CultivationRealm],
    cycles: List[List[str]],
) -> tuple[List[CultivationRealm], nx.DiGraph]:
    """
    Called ONLY when networkx.is_directed_acyclic_graph() returns False.
    Sends the specific broken-link information to the LLM and asks it to
    reassign level numbers so the progression is linear.  Validation itself
    is always done by networkx – we never ask the LLM "is this a DAG?".
    """
    cycle_desc = json.dumps(cycles, ensure_ascii=False)
    realm_desc = json.dumps(
        [{"name": r.name, "level": r.level} for r in realms],
        ensure_ascii=False,
    )
    prompt = (
        "以下修炼体系的境界进阶存在循环（由图论检测发现）：\n"
        f"循环路径：{cycle_desc}\n\n"
        f"当前境界列表：{realm_desc}\n\n"
        "请重新分配每个境界的level编号，消除循环，使境界进阶成为线性无回路序列（level从1开始递增）。\n"
        '以JSON数组输出修正后的列表：[{"name": "境界名", "level": 1}, ...]'
    )
    raw = chat_completion_json(
        client,
        system="你是修炼体系设计师，负责修正境界进阶图中的逻辑错误，只输出合法JSON。",
        user=prompt,
        json_mode=True,
    )
    try:
        fixed_data = json.loads(raw)
        if isinstance(fixed_data, dict):
            fixed_data = fixed_data.get("realms", fixed_data.get("levels", []))
    except (json.JSONDecodeError, AttributeError):
        fixed_data = []

    # Apply fixed levels back onto the existing realm objects
    fixed_levels: Dict[str, int] = {
        item["name"]: int(item["level"])
        for item in fixed_data
        if isinstance(item, dict) and "name" in item and "level" in item
    }
    for realm in realms:
        if realm.name in fixed_levels:
            realm.level = fixed_levels[realm.name]
    realms.sort(key=lambda r: r.level)

    # Rebuild a clean linear DAG from the fixed ordering
    dag = nx.DiGraph()
    for realm in realms:
        dag.add_node(realm.name, level=realm.level)
    for i in range(len(realms) - 1):
        dag.add_edge(realms[i].name, realms[i + 1].name)

    return realms, dag


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _extract_global_theme(client, atoms: List[PlotAtom]) -> str:
    motivations = list({a.motivation for a in atoms if a.motivation})[:20]
    prompt = (
        "以下是从多本修仙小说中提炼的角色动机列表：\n"
        f"{motivations}\n\n"
        "请从中提炼出一句话的全局核心主题（20字以内），"
        "要求富有哲理性，可以作为整部小说的精神内核。\n"
        "只输出这一句话，不要其他内容。"
    )
    result = chat_completion_json(
        client,
        system="你是文学主题提炼专家。",
        user=prompt,
        json_mode=False,
    )
    return result.strip().strip('"').strip("'")


def _format_results(chroma_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    docs = chroma_result.get("documents", [[]])[0]
    metas = chroma_result.get("metadatas", [[]])[0]
    ids = chroma_result.get("ids", [[]])[0]
    distances = chroma_result.get("distances", [[]])[0]
    for i, doc in enumerate(docs):
        items.append(
            {
                "id": ids[i] if i < len(ids) else "",
                "document": doc,
                "metadata": metas[i] if i < len(metas) else {},
                "distance": distances[i] if i < len(distances) else None,
            }
        )
    return items
