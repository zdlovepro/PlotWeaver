"""
step4_knowledge_base.py

Builds the retrieval index from extracted plot atoms.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pipeline.core.world_building_core import KnowledgeBase
from pipeline.step2_extraction import PlotAtom


def build_knowledge_base(all_atoms: Dict[str, List[PlotAtom]]) -> KnowledgeBase:
    kb = KnowledgeBase(clear_existing=True)
    all_atoms_flat = [atom for atoms in all_atoms.values() for atom in atoms]

    print(f"[Step 4] Adding {len(all_atoms_flat)} events to ChromaDB (old collections cleared)...")
    kb.add_events(all_atoms_flat)
    return kb


def connect_knowledge_base(all_atoms: Optional[Dict[str, List[PlotAtom]]] = None) -> KnowledgeBase:
    kb = KnowledgeBase(clear_existing=True)
    if all_atoms:
        all_atoms_flat = [atom for atoms in all_atoms.values() for atom in atoms]
        kb.add_events(all_atoms_flat)
    return kb
