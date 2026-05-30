from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import config
from pipeline import (
    step1_chunking,
    step2_extraction,
    step3_event_induction,
    step4_knowledge_base,
    step5_world_fusion,
    step6_interaction_mining,
    step7_template_mining,
    step8_skeleton_extraction,
    step9_skeleton_fusion,
    step10_character_casting,
    step11_reassembly,
    step12_generation,
    step13_validation,
)


_MAX_STEP = 13


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="PlotWeaver V2.0 - Xianxia Novel Outline Fusion Pipeline",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        choices=range(1, _MAX_STEP + 1),
        metavar="N",
        help="Step number to start from (1-13).",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=None,
        choices=range(1, _MAX_STEP + 1),
        metavar="N",
        help="Step number to stop after (1-13). Defaults to Step 13.",
    )
    parser.add_argument(
        "--only-step",
        type=int,
        default=None,
        choices=range(1, _MAX_STEP + 1),
        metavar="N",
        help="Run only the specified step (equivalent to --start-step N --end-step N).",
    )
    args = parser.parse_args()

    if args.only_step is not None:
        args.start_step = args.only_step
        args.end_step = args.only_step
    elif args.end_step is not None and args.end_step < args.start_step:
        parser.exit(2, "ERROR: --end-step must be greater than or equal to --start-step.\n")

    return args


def _resolve_end_step(end_step: int | None) -> int:
    return _MAX_STEP if end_step is None else end_step


def _should_stop_after(current_step: int, end_step: int) -> bool:
    return current_step >= end_step


def _print_partial_stop_message(step: int, output_dir: Path) -> None:
    print(f"[Pipeline] Stopped after Step {step} as requested by --end-step.", flush=True)
    print("[Pipeline] This is a partial run. Later steps were not executed.", flush=True)
    print(f"[Pipeline] Intermediate files are in: {Path(config.INTERMEDIATE_DIR).resolve()}", flush=True)
    print(f"[Pipeline] Output files, if any, are in: {output_dir.resolve()}", flush=True)


def run_pipeline(start_step: int = 1, end_step: int | None = None) -> bool:
    end_step = _resolve_end_step(end_step)

    print("=" * 60, flush=True)
    print("  PlotWeaver V2.0 - Xianxia Novel Outline Fusion Pipeline", flush=True)
    print(f"  Start step: {start_step}", flush=True)
    print(f"  End step: {end_step}", flush=True)
    if start_step == end_step:
        print(f"  Single-step mode: Step {start_step}", flush=True)
    elif start_step > 1:
        print(f"  Resuming from Step {start_step}", flush=True)
    print("=" * 60, flush=True)

    config.validate()
    input_dir = Path(config.INPUT_DIR)
    output_dir = Path(config.OUTPUT_DIR)

    if start_step <= 1:
        print("\n[Pipeline] Step 1: Semantic Chunking & Arc Anchoring", flush=True)
        novel_arcs = step1_chunking.process_all_novels(input_dir)
        if not novel_arcs:
            print(f"ERROR: No source novels found in '{input_dir}'.", flush=True)
            sys.exit(1)
        print(f"[Pipeline] Processed {len(novel_arcs)} novel(s).", flush=True)
    else:
        print("\n[Pipeline] Step 1 skipped - loading from intermediate file", flush=True)
        novel_arcs = step1_chunking.load_step1_output()
        print(f"[Pipeline] Loaded {len(novel_arcs)} novel(s) from intermediate data.", flush=True)

    if _should_stop_after(1, end_step):
        _print_partial_stop_message(1, output_dir)
        return False

    if start_step <= 2:
        print("\n[Pipeline] Step 2: Dual-stage Plot Extraction", flush=True)
        all_atoms = step2_extraction.extract_all(novel_arcs)
        print(f"[Pipeline] Total plot atoms extracted: {sum(len(v) for v in all_atoms.values())}", flush=True)
    else:
        print("\n[Pipeline] Step 2 skipped - loading from intermediate file", flush=True)
        all_atoms = step2_extraction.load_step2_output()
        print(f"[Pipeline] Loaded {sum(len(v) for v in all_atoms.values())} plot atoms from intermediate data.", flush=True)

    if _should_stop_after(2, end_step):
        _print_partial_stop_message(2, output_dir)
        return False

    if start_step <= 3:
        print("\n[Pipeline] Step 3: Atom Linking & Event Induction", flush=True)
        induced_events = step3_event_induction.induce_all(all_atoms)
        print(f"[Pipeline] Induced {sum(len(v) for v in induced_events.values())} event(s).", flush=True)
    else:
        print("\n[Pipeline] Step 3 skipped - loading from intermediate file", flush=True)
        induced_events = step3_event_induction.load_step3_output()
        print(f"[Pipeline] Loaded {sum(len(v) for v in induced_events.values())} induced event(s).", flush=True)

    if _should_stop_after(3, end_step):
        _print_partial_stop_message(3, output_dir)
        return False

    if start_step <= 4:
        print("\n[Pipeline] Step 4: RAG Knowledge Base", flush=True)
        kb = step4_knowledge_base.build_knowledge_base(all_atoms)
        print("[Pipeline] Knowledge base indexed.", flush=True)
    else:
        print("\n[Pipeline] Step 4 skipped - reconnecting knowledge base", flush=True)
        kb = step4_knowledge_base.connect_knowledge_base(all_atoms)
        print("[Pipeline] Knowledge base reconnected.", flush=True)

    if _should_stop_after(4, end_step):
        _print_partial_stop_message(4, output_dir)
        return False

    if start_step <= 5:
        print("\n[Pipeline] Step 5: World Fusion", flush=True)
        fused_world = step5_world_fusion.build_world_base(all_atoms, kb)
        step5_world_fusion.save_step5_output(fused_world)
    else:
        print("\n[Pipeline] Step 5 skipped - loading from intermediate file", flush=True)
        fused_world = step5_world_fusion.load_step5_output()

    if _should_stop_after(5, end_step):
        _print_partial_stop_message(5, output_dir)
        return False

    if start_step <= 6:
        print("\n[Pipeline] Step 6: Interaction Mining", flush=True)
        fused_world = step6_interaction_mining.mine_story_patterns(all_atoms, kb, fused_world)
        step6_interaction_mining.save_step6_output(fused_world)
    else:
        print("\n[Pipeline] Step 6 skipped - loading from intermediate file", flush=True)
        fused_world = step6_interaction_mining.load_step6_output()

    if _should_stop_after(6, end_step):
        _print_partial_stop_message(6, output_dir)
        return False

    if start_step <= 7:
        print("\n[Pipeline] Step 7: Template Mining", flush=True)
        fused_world = step7_template_mining.derive_templates(all_atoms, fused_world)
        step7_template_mining.save_step7_output(fused_world)
    else:
        print("\n[Pipeline] Step 7 skipped - loading from intermediate file", flush=True)
        fused_world = step7_template_mining.load_step7_output()

    if _should_stop_after(7, end_step):
        _print_partial_stop_message(7, output_dir)
        return False

    if start_step <= 8:
        print("\n[Pipeline] Step 8: Source Skeleton Extraction", flush=True)
        source_skeletons = step8_skeleton_extraction.extract_source_skeletons(
            novel_arcs=novel_arcs,
            all_atoms=all_atoms,
            fused_world=fused_world,
            induced_events_by_novel=induced_events,
        )
        step8_skeleton_extraction.save_step8_output(source_skeletons)
        print(f"[Pipeline] Source skeleton novels: {len(source_skeletons)}", flush=True)
    else:
        print("\n[Pipeline] Step 8 skipped - loading from intermediate file", flush=True)
        source_skeletons = step8_skeleton_extraction.load_step8_output()
        print(f"[Pipeline] Loaded source skeleton novels: {len(source_skeletons)}", flush=True)

    if _should_stop_after(8, end_step):
        _print_partial_stop_message(8, output_dir)
        return False

    if start_step <= 9:
        print("\n[Pipeline] Step 9: Skeleton Fusion", flush=True)
        skeleton = step9_skeleton_fusion.build_skeleton(
            novel_arcs=novel_arcs,
            all_atoms=all_atoms,
            fused_world=fused_world,
            induced_events_by_novel=induced_events,
            source_skeletons=source_skeletons,
        )
        step9_skeleton_fusion.save_step9_output(skeleton)
    else:
        print("\n[Pipeline] Step 9 skipped - loading from intermediate file", flush=True)
        skeleton = step9_skeleton_fusion.load_step9_output()

    if _should_stop_after(9, end_step):
        _print_partial_stop_message(9, output_dir)
        return False

    if start_step <= 10:
        print("\n[Pipeline] Step 10: Character Casting & Relationship Weaving", flush=True)
        skeleton = step10_character_casting.cast_characters(skeleton, fused_world)
        step10_character_casting.save_step10_output(skeleton)
    else:
        print("\n[Pipeline] Step 10 skipped - loading from intermediate file", flush=True)
        skeleton = step10_character_casting.load_step10_output()

    if _should_stop_after(10, end_step):
        _print_partial_stop_message(10, output_dir)
        return False

    max_retries = config.MAX_RETRY_STEPS
    flagged_ids: List[str] = []
    reassembled: List[step11_reassembly.ReassembledEvent] = []

    for attempt in range(max_retries + 1):
        run_step11 = (start_step <= 11) or (attempt > 0)

        if run_step11 and attempt == 0:
            print("\n[Pipeline] Step 11: Character-driven Plot Reassembly", flush=True)
        elif run_step11:
            print(f"\n[Pipeline] Step 11 Retry (attempt {attempt})", flush=True)

        if run_step11:
            reassembled = step11_reassembly.reassemble_plot(
                skeleton=skeleton,
                kb=kb,
                fused_world=fused_world,
                only_event_ids=set(flagged_ids) if flagged_ids else None,
                previous_events=reassembled if flagged_ids else None,
            )
            if attempt == 0:
                step11_reassembly.save_step11_output(reassembled)
        else:
            print("\n[Pipeline] Step 11 skipped - loading from intermediate file", flush=True)
            reassembled = step11_reassembly.load_step11_output()

        if _should_stop_after(11, end_step):
            _print_partial_stop_message(11, output_dir)
            return False

        if start_step <= 12 or attempt > 0:
            print("\n[Pipeline] Step 12: Sliding Window Volume Generation", flush=True)
            volumes = step12_generation.generate_volumes(reassembled, fused_world, skeleton.character_sheet)
        else:
            print("\n[Pipeline] Step 12 skipped - loading from intermediate file", flush=True)
            volumes = step12_generation.load_step12_output()

        if _should_stop_after(12, end_step):
            print("[Pipeline] Partial run stopped before validation.", flush=True)
            _print_partial_stop_message(12, output_dir)
            return False

        print("\n[Pipeline] Step 13: Validation & Output", flush=True)
        source_texts = _load_source_texts(input_dir)
        validation_result = step13_validation.validate_and_output(
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
                f"[Pipeline] Validation failed - retrying Step 11 for "
                f"{len(validation_result.flagged_event_ids)} flagged events...",
                flush=True,
            )
            flagged_ids = validation_result.flagged_event_ids
            if not flagged_ids:
                print("[Pipeline] Warning: validation failed but no events were flagged.", flush=True)
                break
        else:
            print(
                f"[Pipeline] Warning: validation did not fully pass after {max_retries} retries. "
                "Review validation_report.md in output.",
                flush=True,
            )

    print("\n[Pipeline] Full pipeline complete.", flush=True)
    print("=" * 60, flush=True)
    print(f"  Pipeline complete! Output files in: {output_dir.resolve()}", flush=True)
    print("=" * 60, flush=True)
    return True


def archive_current_run(partial_run: bool = False) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(f"history/run_{timestamp}")
    print("\n==================================================", flush=True)
    if partial_run:
        print(" Archiving partial run outputs...", flush=True)
    else:
        print(" Archiving this run's intermediate and final outputs...", flush=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    inter_dir = Path(config.INTERMEDIATE_DIR)
    out_dir = Path(config.OUTPUT_DIR)
    if inter_dir.exists():
        shutil.copytree(inter_dir, archive_dir / "intermediate", dirs_exist_ok=True)
    if out_dir.exists():
        shutil.copytree(out_dir, archive_dir / "output", dirs_exist_ok=True)
    print(f"Archive complete: {archive_dir}", flush=True)
    print("==================================================\n", flush=True)


def _load_source_texts(input_dir: Path) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    for path in sorted(input_dir.glob("*.txt")):
        texts[path.name] = path.read_text(encoding="utf-8")
    return texts


if __name__ == "__main__":
    args = _parse_args()
    args.start_step = 9
    args.end_step = 12
    completed_full = run_pipeline(start_step=args.start_step, end_step=args.end_step)

    if completed_full:
        archive_current_run(partial_run=False)
    else:
        print("[Pipeline] Partial run complete. Archiving available intermediate outputs.", flush=True)
        archive_current_run(partial_run=True)
