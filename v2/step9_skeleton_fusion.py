from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import config
from pipeline.core import skeleton_core as core
from pipeline.core.story_models import NarrativeSkeleton
from pipeline.core.world_building_core import FusedWorld
from pipeline.step1_chunking import VolumeArc
from pipeline.step2_extraction import PlotAtom
from pipeline.step3_event_induction import InducedEvent


_STEP9_SKELETON_FILENAME = "step9_skeleton.json"


def build_skeleton(
    novel_arcs: Dict[str, List[VolumeArc]],
    all_atoms: Dict[str, List[PlotAtom]],
    fused_world: FusedWorld,
    induced_events_by_novel: Optional[Dict[str, List[InducedEvent]]] = None,
    source_skeletons: Optional[Dict[str, List[core.SkeletonNode]]] = None,
) -> NarrativeSkeleton:
    base_novel = core._select_base_novel(None, novel_arcs, fused_world)
    print(f"[Step 9] Selected pacing reference novel: {base_novel}")
    per_novel_nodes = {}
    if source_skeletons:
        for novel_name, nodes in source_skeletons.items():
            cloned_nodes = [core.SkeletonNode(**asdict(node)) for node in nodes]
            for node in cloned_nodes:
                node.source_novels = [novel_name]
            if cloned_nodes:
                per_novel_nodes[novel_name] = cloned_nodes
        print(
            f"[Step 9] Using Step 8 segmented skeletons: "
            f"{sum(len(nodes) for nodes in per_novel_nodes.values())} node(s) across {len(per_novel_nodes)} novel(s).",
            flush=True,
        )
    else:
        for novel_name, arcs in novel_arcs.items():
            induced_events = (induced_events_by_novel or {}).get(novel_name, [])
            if induced_events:
                nodes = core._nodes_from_induced_events(induced_events, fused_world)
            else:
                nodes = core._extract_skeleton_nodes(arcs, all_atoms.get(novel_name, []), fused_world)
            for node in nodes:
                node.source_novels = [novel_name]
            if nodes:
                per_novel_nodes[novel_name] = nodes
        print(
            f"[Step 9] Step 8 skeletons unavailable, rebuilt {sum(len(nodes) for nodes in per_novel_nodes.values())} "
            f"node(s) from upstream inputs.",
            flush=True,
        )

    fused_nodes = core._blend_source_skeleton_nodes(base_novel, per_novel_nodes, fused_world)
    fused_nodes = core._refine_skeleton_sequence(
        base_novel,
        fused_nodes,
        per_novel_nodes,
        fused_world,
        induced_events_by_novel=induced_events_by_novel,
        all_atoms=all_atoms,
    )
    print(
        f"[Step 9] Skeleton generated with {len(fused_nodes)} paced slots filled from "
        f"{len(per_novel_nodes)} source novel(s); the base novel only provides pacing."
    )
    return NarrativeSkeleton(base_novel=base_novel, nodes=fused_nodes, character_sheet=None)


def save_step9_output(skeleton: NarrativeSkeleton) -> Path:
    path = core.save_skeleton_snapshot(_STEP9_SKELETON_FILENAME, skeleton)
    print(f"[Step 9] Intermediate output saved -> {path.name}")
    return path


def load_step9_output() -> NarrativeSkeleton:
    return core.load_skeleton_snapshot(_STEP9_SKELETON_FILENAME)
