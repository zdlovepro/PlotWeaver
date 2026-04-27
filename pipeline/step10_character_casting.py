from __future__ import annotations

from pathlib import Path

from pipeline.core import skeleton_core as core
from pipeline.core.story_models import NarrativeSkeleton
from pipeline.core.utils import get_deepseek_client
from pipeline.core.world_building_core import FusedWorld


_STEP10_CAST_FILENAME = "step10_casted_skeleton.json"


def cast_characters(skeleton: NarrativeSkeleton, fused_world: FusedWorld) -> NarrativeSkeleton:
    client = get_deepseek_client()
    event_role_plans = core._build_event_role_plans(skeleton.nodes)
    character_sheet = core._cast_characters_multipass(client, fused_world, skeleton.nodes, event_role_plans)
    print(f"[Step 10] New protagonist: {character_sheet.protagonist.name}")
    print(f"[Step 10] Generated {len(character_sheet.supporting)} supporting characters across core, volume, and event tiers.")
    return NarrativeSkeleton(base_novel=skeleton.base_novel, nodes=skeleton.nodes, character_sheet=character_sheet)


def save_step10_output(skeleton: NarrativeSkeleton) -> Path:
    path = core.save_skeleton_snapshot(_STEP10_CAST_FILENAME, skeleton)
    print(f"[Step 10] Intermediate output saved -> {path.name}")
    return path


def load_step10_output() -> NarrativeSkeleton:
    return core.load_skeleton_snapshot(_STEP10_CAST_FILENAME)
