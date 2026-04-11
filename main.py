
"""
main.py – PlotWeaver V2.0 Main Orchestrator

Chains the 7-step pipeline for fusing multiple Xianxia novel outlines into a
new, highly coherent, plagiarism-resistant novel outline.

Usage:
    python main.py                  # run full pipeline from Step 1
    python main.py --start-step 2   # skip Step 1, load step1_chunks.json and resume from Step 2
    python main.py --start-step 3   # skip Steps 1-2, load step2_extracted_plots.json and resume
    python main.py --start-step 5   # skip Steps 1-4, load saved state JSONs and resume from Step 5
    python main.py --help           # show usage

Configuration is loaded from config.yaml (or environment variables).
Place source novel .txt files in the INPUT_DIR configured in config.yaml.
"""

from __future__ import annotations

import argparse
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="PlotWeaver V2.0 – Xianxia Novel Outline Fusion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Resume examples:
  python main.py                  Run the full pipeline from Step 1.
  python main.py --start-step 2   Load intermediate_data/step1_chunks.json and
                                  start from Step 2 (skip Step 1).
  python main.py --start-step 3   Load intermediate_data/step2_extracted_plots.json
                                  and start from Step 3 (skip Steps 1-2).
  python main.py --start-step 5   Load saved state for Steps 1-4 and resume
                                  from Step 5 (skips ChromaDB re-insertion).

Intermediate files are stored in the directory configured as ``paths.intermediate_dir``
in config.yaml (default: ./intermediate_data).
        """,
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        choices=range(1, 8),
        metavar="N",
        help="Step number to start from (1-7). Steps before N are skipped and "
             "their outputs are loaded from intermediate_data/. "
             "Default: 1 (full run).",
    )
    return parser.parse_args()


def run_pipeline(start_step: int = 1) -> None:
    """Execute the PlotWeaver pipeline, optionally resuming from *start_step*."""
    print("=" * 60, flush=True)
    print("  PlotWeaver V2.0 – Xianxia Novel Outline Fusion Pipeline", flush=True)
    if start_step > 1:
        print(f"  Resuming from Step {start_step}", flush=True)
    print("=" * 60, flush=True)

    # Validate configuration
    config.validate()

    input_dir = Path(config.INPUT_DIR)
    output_dir = Path(config.OUTPUT_DIR)

    # ── Step 1: Semantic Chunking & Arc Anchoring ─────────────────────────────
    if start_step <= 1:
        print("\n[Pipeline] ── Step 1: Semantic Chunking & Arc Anchoring ──", flush=True)
        novel_arcs = step1_chunking.process_all_novels(input_dir)
        if not novel_arcs:
            print(
                f"ERROR: No source novels found in '{input_dir}'. "
                "Please place .txt files there and retry.",
                flush=True,
            )
            sys.exit(1)
        print(f"[Pipeline] Processed {len(novel_arcs)} novel(s).", flush=True)
    else:
        print("\n[Pipeline] ── Step 1 skipped – loading from intermediate file ──", flush=True)
        novel_arcs = step1_chunking.load_step1_output()
        print(f"[Pipeline] Loaded {len(novel_arcs)} novel(s) from intermediate data.", flush=True)

    # ── Step 2: Dual-stage Plot Extraction ───────────────────────────────────
    if start_step <= 2:
        print("\n[Pipeline] ── Step 2: Dual-stage Plot Extraction ──", flush=True)
        all_atoms = step2_extraction.extract_all(novel_arcs)
        total_atoms = sum(len(v) for v in all_atoms.values())
        print(f"[Pipeline] Total plot atoms extracted: {total_atoms}", flush=True)
    else:
        print("\n[Pipeline] ── Step 2 skipped – loading from intermediate file ──", flush=True)
        all_atoms = step2_extraction.load_step2_output()
        total_atoms = sum(len(v) for v in all_atoms.values())
        print(f"[Pipeline] Loaded {total_atoms} plot atoms from intermediate data.", flush=True)

    # ── Step 3: RAG Knowledge Base & World Building ───────────────────────────
    if start_step <= 3:
        print("\n[Pipeline] ── Step 3: RAG Knowledge Base & World Building ──", flush=True)
        kb, fused_world = step3_knowledge_base.build_knowledge_base(all_atoms)
        step3_knowledge_base.save_step3_output(fused_world)
        print(f"[Pipeline] Fused world: {fused_world.world_name}", flush=True)
        print(f"[Pipeline] Cultivation realms: {len(fused_world.cultivation_realms)}", flush=True)
    else:
        print("\n[Pipeline] ── Step 3 skipped – loading from intermediate file ──", flush=True)
        fused_world = step3_knowledge_base.load_step3_output()
        kb = step3_knowledge_base.connect_knowledge_base()
        print(f"[Pipeline] Loaded fused world: {fused_world.world_name}", flush=True)
        print(f"[Pipeline] Cultivation realms: {len(fused_world.cultivation_realms)}", flush=True)

    # ── Step 4: Skeleton Extraction & Role Casting ────────────────────────────
    if start_step <= 4:
        print("\n[Pipeline] ── Step 4: Skeleton Extraction & Role Casting ──", flush=True)
        skeleton = step4_role_casting.build_skeleton(
            novel_arcs, all_atoms, kb, fused_world
        )
        step4_role_casting.save_step4_output(skeleton)
        print(f"[Pipeline] Skeleton nodes: {len(skeleton.nodes)}", flush=True)
    else:
        print("\n[Pipeline] ── Step 4 skipped – loading from intermediate file ──", flush=True)
        skeleton = step4_role_casting.load_step4_output()
        print(f"[Pipeline] Loaded skeleton: {len(skeleton.nodes)} nodes, "
              f"protagonist: {skeleton.character_sheet.protagonist.name if skeleton.character_sheet else 'N/A'}",
              flush=True)

    # ── Step 5 + 7 loop (retry up to MAX_RETRY_STEPS times) ──────────────────
    max_retries = config.MAX_RETRY_STEPS
    flagged_ids: list[str] = []
    reassembled: list[step5_reassembly.ReassembledEvent] = []

    for attempt in range(max_retries + 1):
        run_step5 = (start_step <= 5) or (attempt > 0)

        if attempt > 0:
            print(f"\n[Pipeline] ── Step 5 Retry (attempt {attempt}) – 局部重写涉嫌抄袭的节点 ──", flush=True)
        elif run_step5:
            print("\n[Pipeline] ── Step 5: Character-driven Plot Reassembly ──", flush=True)

        if run_step5:
            # 核心修复点：将前一次生成的事件和需要重写的节点ID传给 Step 5 进行局部精准重写！
            reassembled = step5_reassembly.reassemble_plot(
                skeleton=skeleton,
                kb=kb,
                fused_world=fused_world,
                only_event_ids=set(flagged_ids) if flagged_ids else None,
                previous_events=reassembled if flagged_ids else None
            )
            if attempt == 0:
                step5_reassembly.save_step5_output(reassembled)
        else:
            print("\n[Pipeline] ── Step 5 skipped – loading from intermediate file ──", flush=True)
            reassembled = step5_reassembly.load_step5_output()
            print(f"[Pipeline] Loaded {len(reassembled)} reassembled events.", flush=True)

        # ── Step 6: Sliding Window Volume Generation ──────────────────────────
        if attempt == 0:
            print("\n[Pipeline] ── Step 6: Sliding Window Volume Generation ──", flush=True)
        else:
            print("\n[Pipeline] ── Step 6: 局部剧情修正后，重新生成大纲 ──", flush=True)

        volumes = step6_generation.generate_volumes(
            reassembled, fused_world, skeleton.character_sheet
        )

        # ── Step 7: Adversarial Plagiarism Check & Output ─────────────────────
        if attempt == 0:
            print("\n[Pipeline] ── Step 7: Adversarial Plagiarism Check & Output ──", flush=True)
        else:
            print(f"\n[Pipeline] ── Step 7: 再次检测是否存在抄袭 (第 {attempt} 次重试) ──", flush=True)

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
                f"{len(validation_result.flagged_event_ids)} flagged events...",
                flush=True,
            )
            flagged_ids = validation_result.flagged_event_ids
            if not flagged_ids:
                print("[Pipeline] Warning: Check failed but no specific events flagged. Breaking loop.")
                break
        else:
            print(
                "[Pipeline] Warning: validation did not fully pass after "
                f"{max_retries} retries. Review the validation_report.md in output.",
                flush=True,
            )

    # ── Done ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print(f"  Pipeline complete! Output files in: {output_dir.resolve()}", flush=True)
    print("=" * 60, flush=True)


def _load_source_texts(input_dir: Path) -> dict[str, str]:
    """Load all source .txt files into a dict for plagiarism checking."""
    texts: dict[str, str] = {}
    for p in sorted(input_dir.glob("*.txt")):
        texts[p.name] = p.read_text(encoding="utf-8")
    return texts


if __name__ == "__main__":
    args = _parse_args()
    args.start_step=3
    run_pipeline(start_step=args.start_step)

