"""
step3_knowledge_base.py – RAG Knowledge Base & World Building (With Deep Templates)

Responsibilities:
  - Populate ChromaDB collections (Events, Characters, Breakthroughs, Cultivation).
  - EXTRACT AND ACCUMULATE Deep Narrative Templates using MULTIPLE PASSES over the novel.
  - Use DeepSeek to fuse a new unified Cultivation System.
  - Build a NetworkX DAG for realm progression.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
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
COL_TEMPLATES = "narrative_templates"


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
    world_background: str = ""
    power_source: str = ""
    major_factions: List[str] = field(default_factory=list)
    raw_system_text: str = ""
    narrative_templates: List[Dict[str, str]] = field(default_factory=list)


class KnowledgeBase:
    def __init__(self):
        self._chroma = get_chromadb_client()
        self._events = self._chroma.get_or_create_collection(COL_EVENTS)
        self._chars = self._chroma.get_or_create_collection(COL_CHARACTER_TRAITS)
        self._breakthroughs = self._chroma.get_or_create_collection(COL_BREAKTHROUGH)
        self._cultivation = self._chroma.get_or_create_collection(COL_CULTIVATION)
        self._templates = self._chroma.get_or_create_collection(COL_TEMPLATES)

    def add_events(self, atoms: List[PlotAtom]) -> None:
        if not atoms: return
        self._events.add(
            ids=[f"{a.atom_id}_{uuid.uuid4().hex[:8]}" for a in atoms],
            documents=[a.summary or a.core_action for a in atoms],
            metadatas=[{
                "original_id": a.atom_id, "novel_source": a.novel_source,
                "arc_name": a.arc_name, "conflict_type": a.conflict_type,
                "narrative_function": a.narrative_function, "tension_level": str(a.tension_level),
            } for a in atoms],
        )

    def add_character_traits(self, traits: List[Dict[str, Any]]) -> None:
        if not traits: return
        self._chars.add(
            ids=[t["id"] for t in traits], documents=[t["text"] for t in traits], metadatas=[t.get("metadata", {}) for t in traits],
        )

    def add_breakthrough_opportunities(self, items: List[Dict[str, Any]]) -> None:
        if not items: return
        self._breakthroughs.add(
            ids=[i["id"] for i in items], documents=[i["text"] for i in items], metadatas=[i.get("metadata", {}) for i in items],
        )

    def add_cultivation_system(self, realms: List[CultivationRealm]) -> None:
        if not realms: return
        self._cultivation.add(
            ids=[f"realm_{r.level}" for r in realms],
            documents=[f"{r.name}（第{r.level}境）：{r.breakthrough_condition}" for r in realms],
            metadatas=[{"name": r.name, "level": str(r.level)} for r in realms],
        )

    def add_narrative_templates(self, templates: List[Dict[str, str]]) -> None:
        if not templates: return
        self._templates.add(
            ids=[f"template_{uuid.uuid4().hex[:8]}" for _ in templates],
            documents=[f"【{t.get('name', '未知')}】\n{t.get('description', '')}" for t in templates],
            metadatas=[{"name": t.get("name", "未知")} for t in templates]
        )

    def query_events(self, query: str, n_results: int = 5, arc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        where = {"arc_name": arc_filter} if arc_filter else None
        kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": n_results}
        if where: kwargs["where"] = where
        return _format_results(self._events.query(**kwargs))

    def query_character_traits(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        return _format_results(self._chars.query(query_texts=[query], n_results=n_results))

    def query_breakthrough_opportunities(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        return _format_results(self._breakthroughs.query(query_texts=[query], n_results=n_results))

    def query_cultivation(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        return _format_results(self._cultivation.query(query_texts=[query], n_results=n_results))

    def query_narrative_templates(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        return _format_results(self._templates.query(query_texts=[query], n_results=n_results))


# ── Public API ──────────────────────────────────────────────────────────

def build_knowledge_base(all_atoms: dict[str, List[PlotAtom]]) -> tuple[KnowledgeBase, FusedWorld]:
    kb = KnowledgeBase()
    client = get_deepseek_client()

    all_atoms_flat: List[PlotAtom] = [atom for atoms in all_atoms.values() for atom in atoms]

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

    print("[Step 3] Extracting Deep Narrative Templates in Multiple Passes...")
    fused_world.narrative_templates = _extract_narrative_templates(client, all_atoms_flat)
    kb.add_narrative_templates(fused_world.narrative_templates)

    print(f"[Step 3] Global theme: {fused_world.global_theme}")
    return kb, fused_world


def connect_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()


# ── Intermediate I/O ────────────────────────────────────────────────────────

_STEP3_WORLD_FILENAME = "step3_fused_world.json"
_STEP3_TEMPLATES_FILENAME = "step3_narrative_templates.json"


def save_step3_output(fused_world: FusedWorld) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    world_path = out_dir / _STEP3_WORLD_FILENAME
    world_data = {
        "world_name": fused_world.world_name,
        "global_theme": fused_world.global_theme,
        "raw_system_text": fused_world.raw_system_text,
        "cultivation_realms": [
            {"name": r.name, "level": r.level, "breakthrough_condition": r.breakthrough_condition, "special_abilities": r.special_abilities}
            for r in fused_world.cultivation_realms
        ],
    }
    with open(world_path, "w", encoding="utf-8") as f:
        json.dump(world_data, f, ensure_ascii=False, indent=2)

    # Append-only logic for templates
    templates_path = out_dir / _STEP3_TEMPLATES_FILENAME
    existing_templates = []
    if templates_path.exists():
        try:
            with open(templates_path, "r", encoding="utf-8") as f:
                existing_templates = json.load(f).get("templates", [])
        except Exception:
            pass

    merged_dict = {t["name"]: t for t in existing_templates}
    for t in fused_world.narrative_templates:
        if t.get("name"): merged_dict[t["name"]] = t

    final_templates = list(merged_dict.values())

    with open(templates_path, "w", encoding="utf-8") as f:
        json.dump({"total_count": len(final_templates), "templates": final_templates}, f, ensure_ascii=False, indent=2)

    print(f"[Step 3] Intermediate outputs saved → {world_path.name} & {templates_path.name} (Total Templates: {len(final_templates)})")
    return world_path


def load_step3_output(intermediate_dir: str | Path | None = None) -> FusedWorld:
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)

    world_path = inter_dir / _STEP3_WORLD_FILENAME
    if not world_path.exists(): raise FileNotFoundError(f"Step 3 file not found: {world_path}")
    with open(world_path, "r", encoding="utf-8") as f: data = json.load(f)

    realms = [CultivationRealm(name=r["name"], level=int(r["level"]), breakthrough_condition=r["breakthrough_condition"], special_abilities=r.get("special_abilities", [])) for r in data.get("cultivation_realms", [])]
    realms.sort(key=lambda r: r.level)

    dag = nx.DiGraph()
    for realm in realms: dag.add_node(realm.name, level=realm.level)
    for i in range(len(realms) - 1): dag.add_edge(realms[i].name, realms[i + 1].name)

    templates_path = inter_dir / _STEP3_TEMPLATES_FILENAME
    loaded_templates = []
    if templates_path.exists():
        with open(templates_path, "r", encoding="utf-8") as f:
            loaded_templates = json.load(f).get("templates", [])

    fused_world = FusedWorld(
        cultivation_realms=realms, realm_dag=dag, global_theme=data.get("global_theme", ""),
        world_name=data.get("world_name", "新世界"), raw_system_text=data.get("raw_system_text", ""),
        narrative_templates=loaded_templates
    )
    return fused_world


# ── Internal helpers ────────────────────────────────────────────────────────

def _extract_narrative_templates(client, atoms: List[PlotAtom]) -> List[Dict[str, str]]:
    """将故事切分为多段，分批次调用大模型提炼模板"""
    conflict_atoms = [a for a in atoms if a.summary and len(a.summary) > 20]
    if not conflict_atoms: return []

    # 最多分3次调用（例如：前期、中期、后期），每次取60个事件
    chunk_size = 60
    max_passes = 3
    chunks = []

    if len(conflict_atoms) <= chunk_size:
        chunks.append(conflict_atoms)
    else:
        step = len(conflict_atoms) // max_passes
        for i in range(max_passes):
            start = i * step
            end = start + chunk_size if i < max_passes - 1 else len(conflict_atoms)
            chunks.append(conflict_atoms[start:min(start+chunk_size, len(conflict_atoms))])

    all_templates = []
    seen_names = set()

    for idx, chunk in enumerate(chunks):
        print(f"[Step 3] Extracting templates pass {idx+1}/{len(chunks)}...")
        summaries = "\n".join(f"- {a.summary[:150]}" for a in chunk)
        templates_from_chunk = _call_llm_for_templates(client, summaries)

        for t in templates_from_chunk:
            name = t.get("name", "").strip()
            if name and name not in seen_names:
                seen_names.add(name)
                all_templates.append(t)

    return all_templates


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _call_llm_for_templates(client, summaries: str) -> List[Dict[str, str]]:
    prompt = (
        "你是顶尖的网文大纲拆解专家与心理学大师。\n"
        "任务：从以下原著片段中，提炼出 5 到 8 个【深度情节交互模板】。\n\n"
        "【标杆级模板示范：藏锋者】\n"
        "描述：藏锋者一开始主动收敛锋芒，外表平庸、举止谦卑，引诱轻视者上钩。轻视者变本加厉当众羞辱，设局逼其出丑，旁人附和嘲笑。藏锋者照单全收，暗中观察弱点。直到触及底线，藏锋者眼神突变，用远超预期的实力碾压展示。轻视者苍白崩溃，围观者惊恐献媚。形成极致的“弱变强”反差爽感。\n\n"
        "请参照以上标杆的【起承转合、交互心理、张力反转】，提炼这批情节中的交互模式。\n"
        "只以JSON数组形式输出，格式如：\n"
        '[{"name": "模板名称", "description": "详细的交互博弈与情绪拉扯过程..."}]\n\n'
        f"【情节片段】：\n{summaries}"
    )
    raw = chat_completion_json(client, system="你是网文架构大师，只输出JSON数组。", user=prompt, json_mode=True)
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else data.get("templates", [])
    except Exception:
        return []


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _extract_character_traits(client, atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    all_chars: Dict[str, List[str]] = {}
    for atom in atoms:
        for ch in atom.characters:
            if ch not in all_chars: all_chars[ch] = []
            all_chars[ch].append(atom.core_action)

    char_summaries = [f"{name}：主要行动包括 {', '.join(actions[:3])}" for name, actions in list(all_chars.items())[:50]]
    char_text = "\n".join(char_summaries)

    prompt = (
        "你是修仙小说角色分析师。\n请为每类角色提炼一个原型特质描述（50字以内），以JSON数组输出：\n"
        '[{"archetype": "冷傲天才型", "description": "...", "traits": ["冷漠","自负","天赋异禀"]}, ...]\n\n'
        f"角色行动数据：\n{char_text[:config.MAX_TEXT_CHUNK_LENGTH]}"
    )
    raw = chat_completion_json(client, system="只输出合法JSON数组。", user=prompt, json_mode=True)
    try:
        archetypes = json.loads(raw)
        if not isinstance(archetypes, list): archetypes = archetypes.get("archetypes", [])
    except Exception: archetypes = []

    return [{"id": f"archetype_{i}", "text": f"{a.get('archetype', '')}：{a.get('description', '')}", "metadata": {"archetype": a.get("archetype", ""), "traits": json.dumps(a.get("traits", []), ensure_ascii=False)}} for i, a in enumerate(archetypes)]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _extract_breakthroughs(client, atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    breakthrough_atoms = [a for a in atoms if any(kw in (a.narrative_function + a.cultivation_elements.__str__()) for kw in ["突破", "晋级", "机缘", "传承", "天劫"])][:30]
    summaries = "\n".join(f"- [{a.arc_name}] {a.summary[:100]}" for a in breakthrough_atoms)
    prompt = (
        "请提炼出10条最具代表性的'突破机缘模板'，每条包含触发条件、机缘、突破方式。\n"
        '以JSON数组输出：[{"id": "bt_0", "trigger": "重伤", "opportunity": "古墓", "breakthrough_method": "参悟"}]\n\n'
        f"情节：\n{summaries}"
    )
    raw = chat_completion_json(client, system="只输出合法JSON数组。", user=prompt, json_mode=True)
    try:
        items = json.loads(raw)
        if not isinstance(items, list): items = items.get("breakthroughs", [])
    except Exception: items = []

    return [{"id": item.get("id", f"breakthrough_{i}"), "text": f"触发：{item.get('trigger', '')}；机缘：{item.get('opportunity', '')}；方式：{item.get('breakthrough_method', '')}", "metadata": {"trigger": item.get("trigger", ""), "opportunity": item.get("opportunity", ""), "breakthrough_method": item.get("breakthrough_method", "")}} for i, item in enumerate(items)]


_CULTIVATION_SYSTEM_SCHEMA = """{
  "world_name": "新世界",
  "realms": [ {"name": "境界名", "level": 1, "breakthrough_condition": "条件", "special_abilities": ["能力"]} ]
}"""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _fuse_cultivation_system(client, atoms: List[PlotAtom]) -> FusedWorld:
    all_elements = []
    for atom in atoms: all_elements.extend(atom.cultivation_elements)
    unique_elements = list(dict.fromkeys(all_elements))[:60]

    prompt = (
        f"修仙元素：\n{unique_elements}\n\n请设计一套全新修炼体系（至少8境）。\n"
        f"以JSON输出：\n{_CULTIVATION_SYSTEM_SCHEMA}"
    )
    raw = chat_completion_json(client, system="你是世界观设计师，只输出JSON。", user=prompt, json_mode=True)
    try: data = json.loads(raw)
    except Exception: data = {}

    realms = [CultivationRealm(name=r.get("name", f"境界{i+1}"), level=int(r.get("level", i + 1)), breakthrough_condition=r.get("breakthrough_condition", ""), special_abilities=r.get("special_abilities", [])) for i, r in enumerate(data.get("realms", []))]
    realms.sort(key=lambda r: r.level)
    dag = nx.DiGraph()
    for realm in realms: dag.add_node(realm.name, level=realm.level)
    for i in range(len(realms) - 1): dag.add_edge(realms[i].name, realms[i + 1].name)

    return FusedWorld(cultivation_realms=realms, realm_dag=dag, world_name=data.get("world_name", "新世界"), raw_system_text=raw)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _extract_global_theme(client, atoms: List[PlotAtom]) -> str:
    motivations = list({a.motivation for a in atoms if a.motivation})[:20]
    result = chat_completion_json(client, system="你是文学专家。", user=f"角色动机：{motivations}\n提炼一句全局核心主题（20字内）。只输出这句话。", json_mode=False)
    return result.strip().strip('"').strip("'")


def _format_results(chroma_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    docs, metas, ids, dists = chroma_result.get("documents", [[]])[0], chroma_result.get("metadatas", [[]])[0], chroma_result.get("ids", [[]])[0], chroma_result.get("distances", [[]])[0]
    for i, doc in enumerate(docs): items.append({"id": ids[i] if i < len(ids) else "", "document": doc, "metadata": metas[i] if i < len(metas) else {}, "distance": dists[i] if i < len(dists) else None})
    return items