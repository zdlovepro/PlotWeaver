"""
step3_knowledge_base.py – RAG Knowledge Base & World Building (Full & Complete)

Responsibilities:
  - Populate ChromaDB collections.
  - EXTRACT MACRO-TROPES (大跨度套路) and PLOT THREADS (长线剧情).
  - EXTRACT MICRO-INTERACTIONS (微观导演级心理博弈).
  - Use DeepSeek to fuse a new unified Cultivation System.
  - Extract Global Theme.
  - Persist all extracted templates and patterns into append-only JSON files.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
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
COL_MICRO_INTERACTIONS = "micro_interactions"  # 存微观博弈模板的向量库


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

    macro_tropes: List[Dict[str, Any]] = field(default_factory=list)         # 宏观套路（如退婚流）
    plot_threads: List[Dict[str, Any]] = field(default_factory=list)         # 长线剧情（如感情线）
    micro_interactions: List[Dict[str, Any]] = field(default_factory=list)   # 微观博弈（导演级心理推拉）


class KnowledgeBase:
    def __init__(self):
        self._chroma = get_chromadb_client()
        self._events = self._chroma.get_or_create_collection(COL_EVENTS)
        self._chars = self._chroma.get_or_create_collection(COL_CHARACTER_TRAITS)
        self._breakthroughs = self._chroma.get_or_create_collection(COL_BREAKTHROUGH)
        self._cultivation = self._chroma.get_or_create_collection(COL_CULTIVATION)
        self._micro_interactions = self._chroma.get_or_create_collection(COL_MICRO_INTERACTIONS)

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
        self._chars.add(ids=[t["id"] for t in traits], documents=[t["text"] for t in traits], metadatas=[t.get("metadata", {}) for t in traits])

    def add_breakthrough_opportunities(self, items: List[Dict[str, Any]]) -> None:
        if not items: return
        self._breakthroughs.add(ids=[i["id"] for i in items], documents=[i["text"] for i in items], metadatas=[i.get("metadata", {}) for i in items])

    def add_cultivation_system(self, realms: List[CultivationRealm]) -> None:
        if not realms: return
        self._cultivation.add(
            ids=[f"realm_{r.level}" for r in realms],
            documents=[f"{r.name}（第{r.level}境）：{r.breakthrough_condition}" for r in realms],
            metadatas=[{"name": r.name, "level": str(r.level)} for r in realms],
        )

    def add_micro_interactions(self, interactions: List[Dict[str, Any]]) -> None:
        if not interactions: return
        self._micro_interactions.add(
            ids=[f"micro_{uuid.uuid4().hex[:8]}" for _ in interactions],
            documents=[f"【{t.get('interaction_name', '未知')}】\n适用场景：{t.get('applicable_scene', '')}" for t in interactions],
            metadatas=[{"name": t.get("interaction_name", "未知")} for t in interactions]
        )

    def query_events(self, query: str, n_results: int = 5, arc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        where = {"arc_name": arc_filter} if arc_filter else None
        kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": n_results}
        if where: kwargs["where"] = where
        return _format_results(self._events.query(**kwargs))


# ── Public API ──────────────────────────────────────────────────────────

def build_knowledge_base(all_atoms: dict[str, List[PlotAtom]]) -> tuple[KnowledgeBase, FusedWorld]:
    kb = KnowledgeBase()
    client = get_deepseek_client()
    all_atoms_flat: List[PlotAtom] = [atom for atoms in all_atoms.values() for atom in atoms]

    print(f"[Step 3] Adding {len(all_atoms_flat)} events to ChromaDB...")
    kb.add_events(all_atoms_flat)

    print("[Step 3] Extracting character traits & breakthrough opportunities...")
    kb.add_character_traits(_extract_character_traits(client, all_atoms_flat))
    kb.add_breakthrough_opportunities(_extract_breakthroughs(client, all_atoms_flat))

    print("[Step 3] Fusing cultivation system & extracting theme...")
    fused_world = _fuse_cultivation_system(client, all_atoms_flat)
    kb.add_cultivation_system(fused_world.cultivation_realms)
    fused_world.global_theme = _extract_global_theme(client, all_atoms_flat)

    print("[Step 3] Extracting Micro-Interactions (导演级心理与动作博弈)...")
    fused_world.micro_interactions = _extract_micro_interactions(client, all_atoms_flat)
    kb.add_micro_interactions(fused_world.micro_interactions)

    print("[Step 3] Extracting Macro-Tropes (大跨度套路模式)...")
    fused_world.macro_tropes = _extract_macro_tropes(client, all_atoms_flat)

    print("[Step 3] Extracting Plot Threads (长线剧情线路)...")
    fused_world.plot_threads = _extract_plot_threads(client, all_atoms_flat)

    return kb, fused_world

def connect_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()


# ── Intermediate I/O ────────────────────────────────────────────────────────

_STEP3_WORLD_FILENAME = "step3_fused_world.json"
_STEP3_MACRO_FILENAME = "step3_macro_patterns.json"
_STEP3_MICRO_FILENAME = "step3_micro_interactions.json"

def save_step3_output(fused_world: FusedWorld) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. World Bible
    world_path = out_dir / _STEP3_WORLD_FILENAME
    world_data = {
        "world_name": fused_world.world_name, "global_theme": fused_world.global_theme, "raw_system_text": fused_world.raw_system_text,
        "cultivation_realms": [{"name": r.name, "level": r.level, "breakthrough_condition": r.breakthrough_condition, "special_abilities": r.special_abilities} for r in fused_world.cultivation_realms],
    }
    with open(world_path, "w", encoding="utf-8") as f: json.dump(world_data, f, ensure_ascii=False, indent=2)

    # 2. Macro Patterns (Append-only)
    macro_path = out_dir / _STEP3_MACRO_FILENAME
    macro_existing = {"macro_tropes": [], "plot_threads": []}
    if macro_path.exists():
        try:
            with open(macro_path, "r", encoding="utf-8") as f: macro_existing = json.load(f)
        except Exception: pass
    merged_tropes = {t["name"]: t for t in macro_existing.get("macro_tropes", [])}
    for t in fused_world.macro_tropes: merged_tropes[t["name"]] = t
    merged_threads = {t["name"]: t for t in macro_existing.get("plot_threads", [])}
    for t in fused_world.plot_threads: merged_threads[t["name"]] = t
    with open(macro_path, "w", encoding="utf-8") as f:
        json.dump({"macro_tropes": list(merged_tropes.values()), "plot_threads": list(merged_threads.values())}, f, ensure_ascii=False, indent=2)

    # 3. Micro Interactions (Append-only)
    micro_path = out_dir / _STEP3_MICRO_FILENAME
    _append_json_list(micro_path, fused_world.micro_interactions, "micro_interactions", key_field="interaction_name")

    print(f"[Step 3] Outputs saved → World, Macro Patterns, and Micro Interactions.")
    return world_path

def _append_json_list(file_path: Path, new_items: List[Dict], list_key: str, key_field: str = "name"):
    existing = []
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f: existing = json.load(f).get(list_key, [])
        except Exception: pass
    merged = {t[key_field]: t for t in existing if key_field in t}
    for t in new_items:
        if key_field in t: merged[t[key_field]] = t
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"total_count": len(merged), list_key: list(merged.values())}, f, ensure_ascii=False, indent=2)

def load_step3_output(intermediate_dir: str | Path | None = None) -> FusedWorld:
    inter_dir = Path(intermediate_dir or config.INTERMEDIATE_DIR)
    world_path = inter_dir / _STEP3_WORLD_FILENAME
    if not world_path.exists(): raise FileNotFoundError(f"Step 3 file not found: {world_path}")

    with open(world_path, "r", encoding="utf-8") as f: data = json.load(f)
    realms = [CultivationRealm(**r) for r in data.get("cultivation_realms", [])]
    realms.sort(key=lambda r: r.level)
    dag = nx.DiGraph()
    for realm in realms: dag.add_node(realm.name, level=realm.level)
    for i in range(len(realms) - 1): dag.add_edge(realms[i].name, realms[i + 1].name)

    fused_world = FusedWorld(cultivation_realms=realms, realm_dag=dag, global_theme=data.get("global_theme", ""), world_name=data.get("world_name", "新世界"), raw_system_text=data.get("raw_system_text", ""))

    macro_path = inter_dir / _STEP3_MACRO_FILENAME
    if macro_path.exists():
        with open(macro_path, "r", encoding="utf-8") as f:
            m_data = json.load(f)
            fused_world.macro_tropes = m_data.get("macro_tropes", [])
            fused_world.plot_threads = m_data.get("plot_threads", [])

    micro_path = inter_dir / _STEP3_MICRO_FILENAME
    if micro_path.exists():
        with open(micro_path, "r", encoding="utf-8") as f:
            fused_world.micro_interactions = json.load(f).get("micro_interactions", [])

    return fused_world


# ── Internal Micro & Macro Extraction Helpers ───────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _extract_micro_interactions(client, atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    """提取微观的、具有极强戏剧张力的角色心理与行为博弈模型"""
    conflict_atoms = [a for a in atoms if a.conflict_type and len(a.summary) > 20][:60]
    summaries = "\n".join(f"- {a.summary[:150]}" for a in conflict_atoms)

    schema = """[
  {
    "interaction_name": "老狐狸与小狐狸的极限拉扯",
    "applicable_scene": "交易/谈判/套情报",
    "role_A": {
      "archetype": "掌控资源的老油条（如黑市掌柜）",
      "initial_psychology": "看似热情，实则想利用信息差榨干对方。"
    },
    "role_B": {
      "archetype": "扮猪吃虎者（主角）",
      "initial_psychology": "故意示弱暴露出一个假破绽，掩饰真正的诉求。"
    },
    "the_dance_of_interaction": {
      "phase_1_probing": "A热情推销次品试探底细；B假装心动暴露出假破绽。",
      "phase_2_escalation": "A以为鱼儿上钩暗中提价；B突然变脸，精准指出物品致命缺陷，反将一军。",
      "phase_3_reversal": "A心理防线被击穿收起伪善；B才抛出真正诉求与无法拒绝的筹码。",
      "phase_4_resolution": "双方达成合作，A对B产生深深忌惮。"
    },
    "bystander_effect": "旁观的伙计从嘲笑变成冷汗直流。"
  }
]"""

    prompt = (
        "你是顶尖的戏剧导演和网文人物互动大师。\n"
        "任务：从原著片段中，提取出 4 到 6 个【微观心理博弈与行为交互模型】。\n"
        "要求：不要泛泛的剧情梗概，我要的是极致细腻的“身份对立、心理推拉、动作拆招与情绪反转”。如上述示范。\n"
        f"请严格按以下 JSON Schema 数组输出：\n{schema}\n\n"
        f"【原著情节片段】：\n{summaries}"
    )

    raw = chat_completion_json(client, system="只输出合法JSON数组。", user=prompt, json_mode=True)
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else data.get("micro_interactions", [])
    except Exception:
        return []


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _extract_macro_tropes(client, atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    arc_summaries = defaultdict(list)
    for atom in atoms:
        if atom.arc_name and atom.summary: arc_summaries[atom.arc_name].append(atom.summary)

    bird_eye_view = ""
    for arc, sum_list in list(arc_summaries.items())[:10]:
        bird_eye_view += f"【{arc}】：\n" + " -> ".join(s[:30] for s in sum_list[:5]) + "\n\n"

    schema = """[
  {
    "name": "退婚流/三年之约",
    "description": "主角开局受辱，跨越多个地图最终复仇打脸的超长线结构。",
    "stages": [
      "第一阶段（受辱）：遭遇当众背叛，定下长期誓言。",
      "第二阶段（蛰伏）：离开原生环境，险境中获底牌。",
      "第三阶段（验证）：中型舞台小试牛刀。",
      "第四阶段（爆发）：赴约之战当众击溃旧敌。"
    ]
  }
]"""

    prompt = (
        "你是网文大纲架构师。\n任务：从以下原著的【卷目发展缩影】中，提取出 3-5 个跨度极大、贯穿多卷的【宏观套路模式（Macro-Tropes）】。\n"
        f"请严格按 JSON Schema 数组输出：\n{schema}\n\n【全书缩影】：\n{bird_eye_view}"
    )
    raw = chat_completion_json(client, system="只输出JSON数组。", user=prompt, json_mode=True)
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else data.get("macro_tropes", [])
    except Exception: return []


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _extract_plot_threads(client, atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    pair_interactions = defaultdict(list)
    for atom in atoms:
        chars = [c for c in atom.characters if len(c) > 1]
        if len(chars) >= 2:
            pair = tuple(sorted(chars[:2]))
            pair_interactions[pair].append(atom.summary)

    top_pairs = sorted(pair_interactions.items(), key=lambda x: len(x[1]), reverse=True)[:3]
    threads_data = ""
    for pair, summaries in top_pairs:
        threads_data += f"【角色：{pair[0]} 与 {pair[1]} 的交互线】：\n" + " -> ".join(s[:50] for s in summaries[:8]) + "\n\n"
    if not threads_data: return []

    schema = """[
  {
    "name": "势均力敌的生死相托",
    "thread_type": "感情线",
    "stages": [
      "阶段1（相识）：因争夺机缘不打不相识。",
      "阶段2（遇险）：被迫联手求生放下成见。",
      "阶段3（离别）：被迫分离，埋下长线念想。",
      "阶段4（重逢）：高阶位面重逢站在同一阵线。"
    ]
  }
]"""

    prompt = (
        "你是人物线编剧。\n任务：根据以下【角色组合交互时间线】，提取 2-4 条经典的长线剧情模式（如感情线、宿敌线），抽象出【关系演变阶段】。\n"
        f"请严格按 JSON Schema 输出：\n{schema}\n\n【时间线】：\n{threads_data}"
    )
    raw = chat_completion_json(client, system="只输出JSON数组。", user=prompt, json_mode=True)
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else data.get("plot_threads", [])
    except Exception: return []


# ── (Fully Re-implemented Standard Functions) ──

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
        "请提炼出10条最具代表性的'突破机缘模板'，以JSON数组输出：\n"
        '[{"id": "bt_0", "trigger": "重伤", "opportunity": "古墓", "breakthrough_method": "参悟"}]\n\n'
        f"情节片段：\n{summaries}"
    )
    raw = chat_completion_json(client, system="只输出合法JSON数组。", user=prompt, json_mode=True)
    try:
        items = json.loads(raw)
        if not isinstance(items, list): items = items.get("breakthroughs", [])
    except Exception: items = []
    return [{"id": item.get("id", f"breakthrough_{i}"), "text": f"触发：{item.get('trigger', '')}；机缘：{item.get('opportunity', '')}；方式：{item.get('breakthrough_method', '')}", "metadata": {"trigger": item.get("trigger", ""), "opportunity": item.get("opportunity", ""), "breakthrough_method": item.get("breakthrough_method", "")}} for i, item in enumerate(items)]

_CULTIVATION_SYSTEM_SCHEMA = """{
  "world_name": "新世界",
  "world_background": "世界背景简述",
  "power_source": "力量本源",
  "major_factions": ["势力A"],
  "realms": [ {"name": "境界名", "level": 1, "breakthrough_condition": "条件", "special_abilities": ["能力"]} ]
}"""
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _fuse_cultivation_system(client, atoms: List[PlotAtom]) -> FusedWorld:
    all_elements = []
    for atom in atoms: all_elements.extend(atom.cultivation_elements)
    unique_elements = list(dict.fromkeys(all_elements))[:100]

    prompt = (
        f"修仙元素：\n{unique_elements}\n\n"
        "请结合以上元素，设计一套全新世界观与修炼体系（至少8境）。\n"
        "【强制要求】：境界名称必须严格采用传统经典的修仙体系（如：炼气、筑基、金丹、元婴、化神、炼虚、合体、大乘、渡劫等），绝对不要自己发明奇奇怪怪的名字！\n"
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

    return FusedWorld(cultivation_realms=realms, realm_dag=dag, world_name=data.get("world_name", "新世界"), world_background=data.get("world_background", ""), power_source=data.get("power_source", ""), major_factions=data.get("major_factions", []), raw_system_text=raw)

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