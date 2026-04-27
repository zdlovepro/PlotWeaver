from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

import config
from pipeline.core.common_json import read_json_file, safe_json_load as _safe_json_load, write_json_file
from pipeline.core.common_text import (
    coerce_text as _coerce_text,
    dedupe_text_values as _dedupe_text_values,
    flatten_text_values as _flatten_text_values,
    normalize_text as _normalize_text,
)
from pipeline.core.utils import get_chromadb_client
from pipeline.step2_extraction import PlotAtom


COL_EVENTS = "events"
COL_CHARACTER_TRAITS = "character_traits"
COL_BREAKTHROUGH = "breakthrough_opportunities"
COL_CULTIVATION = "cultivation_systems"
COL_MICRO_INTERACTIONS = "micro_interactions"

_ROLE_SLOT_LIBRARY: List[Dict[str, Any]] = [
    {
        "name": "trial_competition",
        "keywords": ["试炼", "考核", "大比", "斗法", "擂台", "选拔"],
        "role_slots": ["守关者", "竞争对手", "暗中使绊者", "旁观长辈"],
        "conflict_engines": ["名额争夺", "规则压制", "公开较量"],
    },
    {
        "name": "market_transaction",
        "keywords": ["拍卖", "坊市", "交易", "竞价", "商会", "宝物"],
        "role_slots": ["掌柜主事", "竞价对手", "识货者", "维持秩序者"],
        "conflict_engines": ["信息差", "抬价截胡", "真假难辨"],
    },
    {
        "name": "secret_realm",
        "keywords": ["秘境", "遗迹", "机缘", "传承", "洞府", "禁地"],
        "role_slots": ["引路者", "临时盟友", "争夺者", "守护者"],
        "conflict_engines": ["抢夺机缘", "探索风险", "传承筛选"],
    },
    {
        "name": "hunt_escape",
        "keywords": ["追杀", "逃亡", "围杀", "伏击", "追捕", "通缉"],
        "role_slots": ["追杀者", "掩护者", "线人", "收容者"],
        "conflict_engines": ["围追堵截", "身份暴露", "代价换命"],
    },
    {
        "name": "inheritance_mentor",
        "keywords": ["师尊", "传承", "拜师", "授法", "前辈", "道统"],
        "role_slots": ["试探前辈", "同门竞争者", "护道者", "规则维护者"],
        "conflict_engines": ["理念考验", "资格筛选", "传承归属"],
    },
    {
        "name": "revenge_rivalry",
        "keywords": ["复仇", "宿敌", "旧怨", "家族", "清算", "报复"],
        "role_slots": ["宿敌", "宿敌手下", "见证者", "失控调停者"],
        "conflict_engines": ["旧怨升级", "权势压迫", "身份反转"],
    },
    {
        "name": "emotion_misalignment",
        "keywords": ["情感", "误会", "互救", "婚约", "心结", "同盟"],
        "role_slots": ["误解对象", "撮合者", "挑拨者", "共患难者"],
        "conflict_engines": ["立场错位", "信任拉扯", "救与负债"],
    },
    {
        "name": "default_pressure",
        "keywords": [],
        "role_slots": ["阻碍者", "消息提供者", "短期盟友", "旁观者"],
        "conflict_engines": ["资源竞争", "局势施压", "临时合作"],
    },
]

_LEGACY_WORLD_NAME_MAP = {
    "Merged Xianxia World": "融合修真世界",
    "Merged World": "融合修真世界",
    "New World": "新世界",
}


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
    volume_templates: List[Dict[str, Any]] = field(default_factory=list)
    event_templates: List[Dict[str, Any]] = field(default_factory=list)
    role_slot_templates: List[Dict[str, Any]] = field(default_factory=list)
    event_flow_templates: List[Dict[str, Any]] = field(default_factory=list)


class KnowledgeBase:
    def __init__(self, clear_existing: bool = True):
        self._chroma = get_chromadb_client()
        if clear_existing:
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
        if not atoms:
            return
        self._events.add(
            ids=[f"{atom.atom_id}_{uuid.uuid4().hex[:8]}" for atom in atoms],
            documents=[atom.summary or atom.core_action for atom in atoms],
            metadatas=[
                {
                    "original_id": atom.atom_id,
                    "novel_source": atom.novel_source,
                    "arc_name": atom.arc_name,
                    "conflict_type": _coerce_text(atom.conflict_type),
                    "narrative_function": _coerce_text(atom.narrative_function),
                    "tension_level": str(atom.tension_level),
                    "location": atom.location,
                }
                for atom in atoms
            ],
        )

    def add_character_traits(self, traits: List[Dict[str, Any]]) -> None:
        if not traits:
            return
        self._chars.add(ids=[item["id"] for item in traits], documents=[item["text"] for item in traits], metadatas=[item.get("metadata", {}) for item in traits])

    def add_breakthrough_opportunities(self, items: List[Dict[str, Any]]) -> None:
        if not items:
            return
        self._breakthroughs.add(ids=[item["id"] for item in items], documents=[item["text"] for item in items], metadatas=[item.get("metadata", {}) for item in items])

    def add_cultivation_system(self, realms: List[CultivationRealm]) -> None:
        if not realms:
            return
        self._cultivation.add(
            ids=[f"realm_{realm.level}" for realm in realms],
            documents=[f"{realm.name}（第{realm.level}境）：{realm.breakthrough_condition}" for realm in realms],
            metadatas=[{"name": realm.name, "level": str(realm.level)} for realm in realms],
        )

    def add_micro_interactions(self, interactions: List[Dict[str, Any]]) -> None:
        if not interactions:
            return
        self._micro_interactions.add(
            ids=[f"micro_{uuid.uuid4().hex[:8]}" for _ in interactions],
            documents=[f"{item.get('interaction_name', 'unknown')}: {item.get('applicable_scene', '')}" for item in interactions],
            metadatas=[{"name": item.get("interaction_name", "unknown")} for item in interactions],
        )

    def query_events(self, query: str, n_results: int = 5, arc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": n_results}
        if arc_filter:
            kwargs["where"] = {"arc_name": arc_filter}
        return _format_results(self._events.query(**kwargs))


def save_world_snapshot(filename: str, fused_world: FusedWorld) -> Path:
    out_dir = Path(config.INTERMEDIATE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    payload = {
        "world_name": fused_world.world_name,
        "global_theme": fused_world.global_theme,
        "world_background": fused_world.world_background,
        "power_source": fused_world.power_source,
        "major_factions": fused_world.major_factions,
        "raw_system_text": fused_world.raw_system_text,
        "cultivation_realms": [asdict(realm) for realm in fused_world.cultivation_realms],
        "macro_tropes": fused_world.macro_tropes,
        "plot_threads": fused_world.plot_threads,
        "micro_interactions": fused_world.micro_interactions,
        "volume_templates": fused_world.volume_templates,
        "event_templates": fused_world.event_templates,
        "role_slot_templates": fused_world.role_slot_templates,
        "event_flow_templates": fused_world.event_flow_templates,
    }
    return write_json_file(path, payload)


def load_world_snapshot(filename: str) -> FusedWorld:
    path = Path(config.INTERMEDIATE_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(f"World snapshot not found: {path}")
    data = read_json_file(path)
    realms = [CultivationRealm(**item) for item in data.get("cultivation_realms", [])]
    realms.sort(key=lambda item: item.level)
    dag = nx.DiGraph()
    for realm in realms:
        dag.add_node(realm.name, level=realm.level)
    for index in range(len(realms) - 1):
        dag.add_edge(realms[index].name, realms[index + 1].name)
    world_name = str(data.get("world_name", "新世界") or "新世界")
    world_name = _LEGACY_WORLD_NAME_MAP.get(world_name, world_name)
    return FusedWorld(
        cultivation_realms=realms,
        realm_dag=dag,
        global_theme=data.get("global_theme", ""),
        world_name=world_name,
        world_background=data.get("world_background", ""),
        power_source=data.get("power_source", ""),
        major_factions=data.get("major_factions", []),
        raw_system_text=data.get("raw_system_text", ""),
        macro_tropes=data.get("macro_tropes", []),
        plot_threads=data.get("plot_threads", []),
        micro_interactions=data.get("micro_interactions", []),
        volume_templates=data.get("volume_templates", []),
        event_templates=data.get("event_templates", []),
        role_slot_templates=data.get("role_slot_templates", []),
        event_flow_templates=data.get("event_flow_templates", []),
    )


def _top_values(values: List[Any], limit: int = 3) -> List[str]:
    counter = Counter(item for item in _flatten_text_values(values) if item)
    return [item for item, _count in counter.most_common(limit)]


def _clean_characters(values: List[Any]) -> List[str]:
    cleaned: List[str] = []
    for value in _flatten_text_values(values):
        name = str(value or "").split("(", 1)[0].split("（", 1)[0].strip()
        if name and name not in cleaned:
            cleaned.append(name)
    return cleaned


def _character_mentions(atom: PlotAtom) -> List[Dict[str, str]]:
    aliases = _clean_characters(getattr(atom, "characters", []))
    raws = _clean_characters(getattr(atom, "raw_characters", []))
    keys = _flatten_text_values([getattr(atom, "character_keys", [])])
    max_len = max(len(aliases), len(raws), len(keys))
    mentions: List[Dict[str, str]] = []
    seen_keys: set[str] = set()
    for index in range(max_len):
        display = aliases[index] if index < len(aliases) else (raws[index] if index < len(raws) else f"Character{index + 1}")
        raw_name = raws[index] if index < len(raws) else display
        key = keys[index] if index < len(keys) else f"{getattr(atom, 'novel_source', 'novel')}_{display}_{index + 1}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        mentions.append({"key": key, "display": display, "raw": raw_name})
    return mentions


def _match_role_slot_template(text: str) -> Dict[str, Any]:
    normalized = _normalize_text(text)
    for item in _ROLE_SLOT_LIBRARY[:-1]:
        if any(keyword and keyword in normalized for keyword in item.get("keywords", [])):
            return item
    return _ROLE_SLOT_LIBRARY[-1]


def _tension_band(bucket: List[PlotAtom]) -> str:
    if not bucket:
        return "medium"
    avg = sum(atom.tension_level for atom in bucket) / len(bucket)
    if avg >= 8:
        return "high"
    if avg <= 4:
        return "low"
    return "medium"


def _format_results(chroma_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    ids = chroma_result.get("ids", [[]])[0]
    documents = chroma_result.get("documents", [[]])[0]
    metadatas = chroma_result.get("metadatas", [[]])[0]
    distances = chroma_result.get("distances", [[]])[0] if chroma_result.get("distances") else []
    for index, item_id in enumerate(ids):
        items.append(
            {
                "id": item_id,
                "document": documents[index] if index < len(documents) else "",
                "metadata": metadatas[index] if index < len(metadatas) else {},
                "distance": distances[index] if index < len(distances) else None,
            }
        )
    return items
