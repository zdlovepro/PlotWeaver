from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from pipeline.core import skeleton_core as core
from pipeline.core.story_models import NarrativeSkeleton
from pipeline.core.state_validator import validate_skeleton_sequence
from pipeline.core.utils import get_deepseek_client
from pipeline.core.world_building_core import FusedWorld


_STEP10_CAST_FILENAME = "step10_casted_skeleton.json"


def _state_issue_label(issue_dict: dict) -> str:
    node_id = issue_dict.get("node_id") or "n/a"
    return f"[{issue_dict.get('severity', 'unknown')}/{issue_dict.get('issue_type', 'unknown')}] {issue_dict.get('message', '')} (node={node_id})"


def _attach_fatal_issue_notes(skeleton: NarrativeSkeleton, fatal_issue_dicts: Iterable[dict]) -> None:
    node_map = {node.node_id: node for node in skeleton.nodes}
    all_nodes = list(node_map.values())
    for issue_dict in fatal_issue_dicts:
        note = f"Fatal state validation issue before Step10 casting: {_state_issue_label(issue_dict)}"
        target_node_id = issue_dict.get("node_id") or ""
        if target_node_id and target_node_id in node_map:
            if note not in node_map[target_node_id].logic_notes:
                node_map[target_node_id].logic_notes.append(note)
            continue
        for node in all_nodes:
            if note not in node.logic_notes:
                node.logic_notes.append(note)


def cast_characters(skeleton: NarrativeSkeleton, fused_world: FusedWorld) -> NarrativeSkeleton:
    issues = validate_skeleton_sequence(skeleton.nodes, fused_world)
    issue_dicts = [asdict(issue) for issue in issues]
    fatal_issue_dicts = [issue_dict for issue_dict in issue_dicts if issue_dict.get("severity") == "fatal"]
    metadata = dict(skeleton.metadata or {})
    metadata["state_validation_failed"] = bool(fatal_issue_dicts)
    metadata["state_validation_issues"] = issue_dicts
    skeleton.metadata = metadata

    if fatal_issue_dicts:
        print("[Step 10] Fatal skeleton state issues detected before character casting:")
        for issue_dict in fatal_issue_dicts:
            print(f"  - {_state_issue_label(issue_dict)}")
        _attach_fatal_issue_notes(skeleton, fatal_issue_dicts)
        raise RuntimeError("Step10 aborted: fatal skeleton state issues detected. Please repair the Step9 skeleton before character casting.")

    client = get_deepseek_client()
    event_role_plans = core._build_event_role_plans(skeleton.nodes)
    character_sheet = core._cast_characters_multipass(client, fused_world, skeleton.nodes, event_role_plans)
    print(f"[Step 10] New protagonist: {character_sheet.protagonist.name}")
    print(f"[Step 10] Generated {len(character_sheet.supporting)} supporting characters across core, volume, and event tiers.")
    return NarrativeSkeleton(
        base_novel=skeleton.base_novel,
        nodes=skeleton.nodes,
        character_sheet=character_sheet,
        metadata=dict(skeleton.metadata or {}),
    )


def save_step10_output(skeleton: NarrativeSkeleton) -> Path:
    path = core.save_skeleton_snapshot(_STEP10_CAST_FILENAME, skeleton)
    print(f"[Step 10] Intermediate output saved -> {path.name}")
    return path


def load_step10_output() -> NarrativeSkeleton:
    return core.load_skeleton_snapshot(_STEP10_CAST_FILENAME)
