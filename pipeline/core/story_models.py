from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkeletonNode:
    node_id: str
    arc_name: str
    realm_level: int
    pacing_role: str
    original_summary: str
    conflict_hint: str = ""
    function_hint: str = ""
    role_slots: List[str] = field(default_factory=list)
    template_hint: str = ""
    source_event_ids: List[str] = field(default_factory=list)
    chapter_start: int = 0
    chapter_end: int = 0
    chapter_count: int = 0
    chapter_blueprint: List[Dict[str, Any]] = field(default_factory=list)
    character_keys: List[str] = field(default_factory=list)
    source_novels: List[str] = field(default_factory=list)
    selection_ref: str = ""
    selection_source_novel: str = ""
    selection_source_index: int = -1
    logic_card: Dict[str, Any] = field(default_factory=dict)
    logic_notes: List[str] = field(default_factory=list)
    source_induced_event_ids: List[str] = field(default_factory=list)
    source_atom_ids: List[str] = field(default_factory=list)
    source_refs: List[Dict[str, Any]] = field(default_factory=list)
    stage: str = ""
    power_stage: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Character:
    name: str = ""
    role: str = ""
    bond_depth: str = ""
    entry_event: str = ""
    exit_event: str = ""
    return_event: str = ""
    dao_heart: str = ""
    combat_style: str = ""
    personality_flaw: str = ""
    background: str = ""
    tier: str = "supporting"
    primary_arc: str = ""
    role_slots: List[str] = field(default_factory=list)


@dataclass
class StageNetwork:
    stage: str = ""
    active_characters: List[str] = field(default_factory=list)
    relationship_status: str = ""


@dataclass
class EventRolePlan:
    event_id: str
    event_index: int
    arc_name: str
    pacing_role: str
    summary: str
    role_slots: List[str] = field(default_factory=list)
    candidate_names: List[str] = field(default_factory=list)


@dataclass
class CharacterSheet:
    protagonist: Character
    supporting: List[Character] = field(default_factory=list)
    relationship_networks: List[StageNetwork] = field(default_factory=list)
    event_role_plans: List[EventRolePlan] = field(default_factory=list)


@dataclass
class NarrativeSkeleton:
    base_novel: str
    nodes: List[SkeletonNode] = field(default_factory=list)
    character_sheet: Optional[CharacterSheet] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
