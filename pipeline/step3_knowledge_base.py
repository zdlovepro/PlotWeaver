"""
step3_knowledge_base.py – RAG Knowledge Base & World Building (High-Volume Multi-Pass Extraction)

Responsibilities:
  - Clear old ChromaDB collections to prevent ghost IDs.
  - Populate ChromaDB collections.
  - Extract MICRO Interaction Dynamics (Multi-pass, relaxed filter).
  - Extract MACRO-TROPES and PLOT THREADS (Multi-pass).
  - Use DeepSeek to fuse a new unified Cultivation System.
  - Persist all extracted templates and patterns.
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


# ── ChromaDB collection names ──
COL_EVENTS = "events"
COL_CHARACTER_TRAITS = "character_traits"
COL_BREAKTHROUGH = "breakthrough_opportunities"
COL_CULTIVATION = "cultivation_systems"
COL_MICRO_INTERACTIONS = "micro_interactions"


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

    macro_tropes: List[Dict[str, Any]] = field(default_factory=list)
    plot_threads: List[Dict[str, Any]] = field(default_factory=list)
    micro_interactions: List[Dict[str, Any]] = field(default_factory=list)


class KnowledgeBase:
    def __init__(self):
        self._chroma = get_chromadb_client()
        # 强制清理老旧的幽灵数据
        for col_name in [COL_EVENTS, COL_CHARACTER_TRAITS, COL_BREAKTHROUGH, COL_CULTIVATION, COL_MICRO_INTERACTIONS]:
            try:
                self._chroma.delete_collection(col_name)
            except Exception:
                pass

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


# ── Public API ──

def build_knowledge_base(all_atoms: dict[str, List[PlotAtom]]) -> tuple[KnowledgeBase, FusedWorld]:
    kb = KnowledgeBase()
    client = get_deepseek_client()
    all_atoms_flat: List[PlotAtom] = [atom for atoms in all_atoms.values() for atom in atoms]

    print(f"[Step 3] Adding {len(all_atoms_flat)} events to ChromaDB (Old collections cleared)...")
    kb.add_events(all_atoms_flat)

    print("[Step 3] Extracting character traits & breakthrough opportunities...")
    kb.add_character_traits(_extract_character_traits(client, all_atoms_flat))
    kb.add_breakthrough_opportunities(_extract_breakthroughs(client, all_atoms_flat))

    print("[Step 3] Fusing cultivation system & extracting theme...")
    fused_world = _fuse_cultivation_system(client, all_atoms_flat)
    kb.add_cultivation_system(fused_world.cultivation_realms)
    fused_world.global_theme = _extract_global_theme(client, all_atoms_flat)

    print("[Step 3] Extracting Micro-Interactions (Multi-Pass)...")
    fused_world.micro_interactions = _extract_micro_interactions(client, all_atoms_flat)
    kb.add_micro_interactions(fused_world.micro_interactions)

    print("[Step 3] Extracting Macro-Tropes (Multi-Pass)...")
    fused_world.macro_tropes = _extract_macro_tropes(client, all_atoms_flat)

    print("[Step 3] Extracting Plot Threads (Multi-Pass)...")
    fused_world.plot_threads = _extract_plot_threads(client, all_atoms_flat)

    return kb, fused_world

def connect_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()


# ── Intermediate I/O ──

_STEP3_WORLD_FILENAME = "step3_fused_world.json"
_STEP3_MACRO_FILENAME = "step3_macro_patterns.json"
_STEP3_MICRO_FILENAME = "step3_micro_interactions.json"

def save_step3_output(fused_world: FusedWorld) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    world_path = out_dir / _STEP3_WORLD_FILENAME
    world_data = {
        "world_name": fused_world.world_name, "global_theme": fused_world.global_theme, "raw_system_text": fused_world.raw_system_text,
        "cultivation_realms": [{"name": r.name, "level": r.level, "breakthrough_condition": r.breakthrough_condition, "special_abilities": r.special_abilities} for r in fused_world.cultivation_realms],
    }
    with open(world_path, "w", encoding="utf-8") as f: json.dump(world_data, f, ensure_ascii=False, indent=2)

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


# ── Internal Multi-Pass Extraction Helpers ──
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _call_llm_for_micro(client, summaries: str) -> List[Dict[str, Any]]:
    schema = """[
  {
    "interaction_name": "坊市捡漏与势利眼打脸",
    "applicable_scene": "交易、拍卖、寻宝类事件",
    "role_A": {"archetype": "势利眼/傲慢反派", "initial_psychology": "以貌取人，试图踩压主角抬高自己。"},
    "role_B": {"archetype": "低调的主角", "initial_psychology": "不争一时口舌，只看重实际利益。"},
    "the_dance_of_interaction": {
      "phase_1_probing": "【起-铺垫冲突】：主角低调入场，被势利眼嘲讽排挤。",
      "phase_2_escalation": "【承-发现机缘】：出现一件所有人都看走眼的废品，唯独主角察觉其惊人内幕。",
      "phase_3_reversal": "【转-高调反击】：反派故意抬价或阻挠，主角以极具魄力的方式拿下物品。",
      "phase_4_resolution": "【合-爽感释放】：物品真实价值显露，反派懊悔吐血。"
    },
    "bystander_effect": "旁观者从看戏转变为极度震惊。"
  }
]"""
    prompt = (
        "你是最顶级的网文主编和桥段设计大师。\n"
        "【任务】：请从以下剧情摘要中，归纳提取出 3-5 个【经典桥段的“起承转合”填空模板】。\n"
        f"请严格按以下 JSON 数组格式输出：\n{schema}\n\n"
        f"【原著剧情摘要】：\n{summaries}"
    )

    raw = chat_completion_json(client, system="只输出合法的JSON数组。", user=prompt, json_mode=True)

    try:
        # 暴力清洗大模型可能携带的 Markdown 代码块残留
        raw_clean = raw.strip()
        if raw_clean.startswith("```json"): raw_clean = raw_clean[7:]
        if raw_clean.startswith("```"): raw_clean = raw_clean[3:]
        if raw_clean.endswith("```"): raw_clean = raw_clean[:-3]

        data = json.loads(raw_clean.strip())

        # 兼容处理：不论大模型返回的是 List 还是 Dict，统统强行兼容
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # 如果它非要包一层 Dict，我们把 Dict 里面长得像 List 的值挖出来
            for v in data.values():
                if isinstance(v, list): return v
            return [data]  # 最后的倔强
        return []
    except Exception as e:
        print(f"    [Warning] 微观模板解析失败: {e}\n模型原始返回: {raw[:150]}...")
        return []


def _extract_micro_interactions(client, atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    # 【提取所有有效情节】
    valid_texts = []
    for a in atoms:
        text = a.summary or a.core_action or ""
        if len(text.strip()) > 10:
            valid_texts.append(text.strip())

    if not valid_texts:
        print("    [Warning] 没有找到任何有效的原著事件文本，无法提取微观模板！")
        return []

    # 【暴力增加请求次数】：缩小 chunk_size 到 25，让模型读得更细！
    chunk_size = 20
    chunks = [valid_texts[i:i + chunk_size] for i in range(0, len(valid_texts), chunk_size)]

    # 防止几千章的小说把 API 费用刷爆，设置最高请求次数为 15 次
    max_passes = 15
    if len(chunks) > max_passes:
        # 均匀采样 15 个区块（覆盖开头、中间、结尾的所有不同套路）
        step = len(chunks) / max_passes
        chunks = [chunks[int(i * step)] for i in range(max_passes)]

    all_micro = []
    seen = set()

    print(f"  -> [Micro Extraction] 准备发起 {len(chunks)} 轮提取...")

    for idx, chunk in enumerate(chunks):
        print(f"    -> Pass {idx + 1}/{len(chunks)}: 分析 {len(chunk)} 个事件...")
        summaries = "\n".join(f"- {text[:150]}" for text in chunk)

        # 请求大模型
        extracted = _call_llm_for_micro(client, summaries)

        # 去重并加入总库
        added_in_pass = 0
        for t in extracted:
            name = t.get("interaction_name", "").strip()
            if name and name not in seen:
                seen.add(name)
                all_micro.append(t)
                added_in_pass += 1

        print(f"       本轮获得新模板: {added_in_pass} 个 (累计: {len(all_micro)} 个)")

    print(f"  -> [Micro Extraction]：成功提取了 {len(all_micro)} 个桥段模板")
    return all_micro



@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _call_llm_for_macro(client, bird_eye_view: str) -> List[Dict[str, Any]]:
    schema = """[
  {
    "name": "退婚流/三年之约",
    "description": "主角开局受辱，最终复仇的超长线结构。",
    "stages": ["受辱", "蛰伏", "小试牛刀", "终极爆发"]
  }
]"""
    prompt = (
        "你是大纲架构师。\n任务：从【卷目缩影】中，提取出 3-5 个跨越极大跨度的【宏观套路模式（Macro-Tropes）】。\n"
        f"按 JSON Schema 数组输出：\n{schema}\n\n【缩影】：\n{bird_eye_view}"
    )
    raw = chat_completion_json(client, system="只输出JSON数组。", user=prompt, json_mode=True)
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else data.get("macro_tropes", [])
    except Exception: return []

def _extract_macro_tropes(client, atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    arc_summaries = defaultdict(list)
    for atom in atoms:
        if atom.arc_name and atom.summary: arc_summaries[atom.arc_name].append(atom.summary)

    all_arcs = list(arc_summaries.items())
    chunk_size = 10
    chunks = [all_arcs[i:i + chunk_size] for i in range(0, len(all_arcs), chunk_size)]

    all_macro = []
    seen = set()
    for idx, chunk in enumerate(chunks):
        print(f"  -> [Macro Extraction] Pass {idx+1}/{len(chunks)}...")
        bird_eye_view = "".join([f"【{arc}】：\n" + " -> ".join(s[:30] for s in sums[:5]) + "\n\n" for arc, sums in chunk])
        extracted = _call_llm_for_macro(client, bird_eye_view)
        for t in extracted:
            name = t.get("name", "").strip()
            if name and name not in seen:
                seen.add(name)
                all_macro.append(t)
    return all_macro


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _call_llm_for_threads(client, threads_data: str) -> List[Dict[str, Any]]:
    schema = """[
  {
    "name": "生死相托的感情线",
    "thread_type": "感情线",
    "stages": ["不打不相识", "遇险联手", "生死羁绊", "分离", "重逢"]
  }
]"""
    prompt = (
        "任务：根据以下【角色交互时间线】，提取出 2-4 条经典的长线剧情模式（如感情线、宿敌线），抽象出【关系演变阶段】。\n"
        f"按 JSON Schema 输出：\n{schema}\n\n【时间线】：\n{threads_data}"
    )
    raw = chat_completion_json(client, system="只输出JSON数组。", user=prompt, json_mode=True)
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else data.get("plot_threads", [])
    except Exception: return []

def _extract_plot_threads(client, atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    pair_interactions = defaultdict(list)
    for atom in atoms:
        chars = [c for c in atom.characters if len(c) > 1]
        if len(chars) >= 2:
            pair = tuple(sorted(chars[:2]))
            pair_interactions[pair].append(atom.summary)

    top_pairs = sorted(pair_interactions.items(), key=lambda x: len(x[1]), reverse=True)[:12]
    chunk_size = 3
    chunks = [top_pairs[i:i + chunk_size] for i in range(0, len(top_pairs), chunk_size)]

    all_threads = []
    seen = set()
    for idx, chunk in enumerate(chunks):
        print(f"  -> [Thread Extraction] Pass {idx+1}/{len(chunks)}...")
        threads_data = "".join([f"【角色：{pair[0]} 与 {pair[1]} 的交互线】：\n" + " -> ".join(s[:50] for s in sums[:8]) + "\n\n" for pair, sums in chunk])
        extracted = _call_llm_for_threads(client, threads_data)
        for t in extracted:
            name = t.get("name", "").strip()
            if name and name not in seen:
                seen.add(name)
                all_threads.append(t)
    return all_threads


# ── Rest of the generic helpers ──
def _extract_character_traits(client, atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    return []
def _extract_breakthroughs(client, atoms: List[PlotAtom]) -> List[Dict[str, Any]]:
    return []
_CULTIVATION_SYSTEM_SCHEMA = """{
  "world_name": "新世界",
  "world_background": "简述",
  "power_source": "灵气",
  "major_factions": ["宗门"],
  "realms": [ {"name": "炼气", "level": 1, "breakthrough_condition": "条件", "special_abilities": ["能力"]} ]
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