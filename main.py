"""
main.py – PlotWeaver V2.0 Main Orchestrator

Chains the 7-step pipeline for fusing multiple Xianxia novel outlines into a
new, highly coherent, plagiarism-resistant novel outline.

Usage:
    python main.py

Configuration is loaded from config.yaml (or environment variables).
Place source novel .txt files in the INPUT_DIR configured in config.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

import config
from pipeline import (
    step1_chunking,
    step2_extraction,
    step3_knowledge_base,
    step4_role_casting,
    step5_reassembly,
    step6_generation,
    step7_validation,
)


def run_pipeline() -> None:
    """Execute the full 7-step PlotWeaver pipeline."""
    print("=" * 60)
    print("  PlotWeaver V2.0 – Xianxia Novel Outline Fusion Pipeline")
    print("=" * 60)

    # Validate configuration
    config.validate()

    input_dir = Path(config.INPUT_DIR)
    output_dir = Path(config.OUTPUT_DIR)

    # ── Step 1: Semantic Chunking & Arc Anchoring ─────────────────────────────
    print("\n[Pipeline] ── Step 1: Semantic Chunking & Arc Anchoring ──")
    novel_arcs = step1_chunking.process_all_novels(input_dir)
    if not novel_arcs:
        print(
            f"ERROR: No source novels found in '{input_dir}'. "
            "Please place .txt files there and retry."
        )
        sys.exit(1)
    print(f"[Pipeline] Processed {len(novel_arcs)} novel(s).")

    # ── Step 2: Dual-stage Plot Extraction ───────────────────────────────────
    print("\n[Pipeline] ── Step 2: Dual-stage Plot Extraction ──")
    all_atoms = step2_extraction.extract_all(novel_arcs)
    total_atoms = sum(len(v) for v in all_atoms.values())
    print(f"[Pipeline] Total plot atoms extracted: {total_atoms}")

    # ── Step 3: RAG Knowledge Base & World Building ───────────────────────────
    print("\n[Pipeline] ── Step 3: RAG Knowledge Base & World Building ──")
    kb, fused_world = step3_knowledge_base.build_knowledge_base(all_atoms)
    print(f"[Pipeline] Fused world: {fused_world.world_name}")
    print(f"[Pipeline] Cultivation realms: {len(fused_world.cultivation_realms)}")

    # ── Step 4: Skeleton Extraction & Role Casting ────────────────────────────
    print("\n[Pipeline] ── Step 4: Skeleton Extraction & Role Casting ──")
    skeleton = step4_role_casting.build_skeleton(
        novel_arcs, all_atoms, kb, fused_world
    )
    print(f"[Pipeline] Skeleton nodes: {len(skeleton.nodes)}")

    # ── Step 5 + 7 loop (retry up to MAX_RETRY_STEPS times) ──────────────────
    max_retries = config.MAX_RETRY_STEPS
    flagged_ids: list[str] = []

    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"\n[Pipeline] ── Step 5 Retry (attempt {attempt}) ──")
        else:
            print("\n[Pipeline] ── Step 5: Character-driven Plot Reassembly ──")

        reassembled = step5_reassembly.reassemble_plot(skeleton, kb, fused_world)

        # ── Step 6: Sliding Window Volume Generation ──────────────────────────
        if attempt == 0:
            print("\n[Pipeline] ── Step 6: Sliding Window Volume Generation ──")
        volumes = step6_generation.generate_volumes(
            reassembled, fused_world, skeleton.character_sheet
        )

        # ── Step 7: Adversarial Plagiarism Check & Output ─────────────────────
        if attempt == 0:
            print("\n[Pipeline] ── Step 7: Adversarial Plagiarism Check & Output ──")

        source_texts = _load_source_texts(input_dir)
        validation_result = step7_validation.validate_and_output(
            volumes=volumes,
            reassembled_events=reassembled,
            skeleton=skeleton,
            fused_world=fused_world,
            source_texts=source_texts,
            output_dir=output_dir,
        )

        if validation_result.passed:
            break

        if attempt < max_retries:
            print(
                f"[Pipeline] Validation failed – retrying Step 5 for "
                f"{len(validation_result.flagged_event_ids)} flagged events..."
            )
            flagged_ids = validation_result.flagged_event_ids
        else:
            print(
                "[Pipeline] Warning: validation did not fully pass after "
                f"{max_retries} retries. Review the validation_report.md in output."
            )

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Pipeline complete! Output files in: {output_dir.resolve()}")
    print("=" * 60)


def _load_source_texts(input_dir: Path) -> dict[str, str]:
    """Load all source .txt files into a dict for plagiarism checking."""
    texts: dict[str, str] = {}
    for p in sorted(input_dir.glob("*.txt")):
        texts[p.name] = p.read_text(encoding="utf-8")
    return texts


if __name__ == "__main__":
    run_pipeline()
