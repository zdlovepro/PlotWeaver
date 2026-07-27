from __future__ import annotations

import argparse
from pathlib import Path

from .annotate import (
    DEFAULT_ANNOTATION_INPUT_CHARS,
    DEFAULT_ANNOTATION_LIMIT,
    DEFAULT_ANNOTATION_OVERLAP_UNITS,
    annotate_work,
)
from .continuity import DEFAULT_CONTINUITY_LIMIT, build_continuity_work
from .narrative_graph import DEFAULT_GRAPH_LIMIT, build_narrative_graph_work
from .ingest import ingest_directory, ingest_file
from .pipeline import PipelineOptions, run_pipeline
from .paths import ROOT, validate_author_id
from .template_mining import DEFAULT_TEMPLATE_BATCH_CHAPTERS, DEFAULT_TEMPLATE_LIMIT, mine_template_library
from .style_distillation import DEFAULT_STYLE_BATCH_CHAPTERS, DEFAULT_STYLE_LIMIT, distil_style_profile
from .skill_compiler import DEFAULT_SKILL_LIMIT, compile_author_skill
from .skill_runtime import prepare_draft_prompt, prepare_plan_prompt, prepare_validation_prompt
from .scene_compiler import compile_chapter_program_work, plan_expansion_program_work, plan_expansion_programs_work
from .scene_generation import DEFAULT_SCENE_MAX_REPAIRS, generate_from_program
from .program_sequence import generate_program_sequence
from .program_fidelity import evaluate_program_sequence
from .model_probe import probe_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python main.py", description="Generic novel-author narrative distillation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--author-id", required=True)
    ingest.add_argument("--source-dir", type=Path, default=ROOT / "input")
    ingest.add_argument("--source", type=Path)
    ingest.add_argument("--work-id")
    probe = sub.add_parser("probe-model", help="在处理小说前验证模型能返回非空严格 JSON")
    probe.add_argument("--author-id", required=True); probe.add_argument("--run-id", default="model-probe")
    annotate = sub.add_parser("annotate", help="Extract typed, source-evidenced narrative annotations")
    annotate.add_argument("--author-id", required=True); annotate.add_argument("--work-id", required=True)
    annotate.add_argument("--limit", type=int, default=DEFAULT_ANNOTATION_LIMIT)
    annotate.add_argument("--max-input-chars", type=int, default=DEFAULT_ANNOTATION_INPUT_CHARS)
    annotate.add_argument("--overlap-units", type=int, default=DEFAULT_ANNOTATION_OVERLAP_UNITS)
    annotate.add_argument("--no-resume", action="store_true")
    continuity = sub.add_parser("continuity", help="Fuse evidence-first annotations into a cross-chapter continuity graph")
    continuity.add_argument("--author-id", required=True); continuity.add_argument("--work-id", required=True)
    continuity.add_argument("--limit", type=int, default=DEFAULT_CONTINUITY_LIMIT)
    graph = sub.add_parser("graph", help="Build local narrative_graph.json from accepted annotations and continuity")
    graph.add_argument("--author-id", required=True); graph.add_argument("--work-id", required=True)
    graph.add_argument("--limit", type=int, default=DEFAULT_GRAPH_LIMIT)
    templates = sub.add_parser("templates", help="Mine evidence-supported multi-level templates from the new continuity graph")
    templates.add_argument("--author-id", required=True); templates.add_argument("--work-id", required=True)
    templates.add_argument("--limit", type=int, default=DEFAULT_TEMPLATE_LIMIT)
    templates.add_argument("--batch-size", type=int, default=DEFAULT_TEMPLATE_BATCH_CHAPTERS)
    templates.add_argument("--offline", action="store_true")
    style = sub.add_parser("style", help="Distil quantified, evidence-backed author style constraints")
    style.add_argument("--author-id", required=True); style.add_argument("--work-id", required=True)
    style.add_argument("--limit", type=int, default=DEFAULT_STYLE_LIMIT)
    style.add_argument("--batch-size", type=int, default=DEFAULT_STYLE_BATCH_CHAPTERS)
    style.add_argument("--offline", action="store_true")
    compile_skill_cmd = sub.add_parser("compile-skill", help="Compile stages 4-6 into a standalone author skill package")
    compile_skill_cmd.add_argument("--author-id", required=True); compile_skill_cmd.add_argument("--work-id", required=True)
    compile_skill_cmd.add_argument("--limit", type=int, default=DEFAULT_SKILL_LIMIT)
    prepare_plan = sub.add_parser("prepare-plan", help="Render a compiled skill's chapter-planning prompt")
    prepare_plan.add_argument("--author-id", required=True); prepare_plan.add_argument("--story-state", type=Path, required=True)
    prepare_plan.add_argument("--chapter-brief", type=Path, required=True); prepare_plan.add_argument("--run-id", default="skill")
    prepare_draft = sub.add_parser("prepare-draft", help="Render a compiled skill's chapter-drafting prompt")
    prepare_draft.add_argument("--author-id", required=True); prepare_draft.add_argument("--story-state", type=Path, required=True)
    prepare_draft.add_argument("--chapter-plan", type=Path, required=True); prepare_draft.add_argument("--run-id", default="skill")
    prepare_validate = sub.add_parser("prepare-validation", help="Render a compiled skill's chapter-validation prompt")
    prepare_validate.add_argument("--author-id", required=True); prepare_validate.add_argument("--story-state", type=Path, required=True)
    prepare_validate.add_argument("--chapter-plan", type=Path, required=True); prepare_validate.add_argument("--draft", type=Path, required=True)
    prepare_validate.add_argument("--run-id", default="skill")
    chapter_program = sub.add_parser("compile-program", help="Compile one annotated chapter into facts, events, scenes and paragraph programs")
    chapter_program.add_argument("--author-id", required=True); chapter_program.add_argument("--work-id", required=True)
    chapter_program.add_argument("--chapter", type=int, required=True)
    chapter_program.add_argument("--limit", type=int, default=20)
    chapter_program.add_argument("--run-id", default="chapter-program")
    chapter_plan = sub.add_parser("plan-expansion", help="Compile one annotated chapter into a target-length controlled-expansion program")
    chapter_plan.add_argument("--author-id", required=True); chapter_plan.add_argument("--work-id", required=True)
    chapter_plan.add_argument("--chapter", type=int, required=True)
    chapter_plan.add_argument("--target-char-min", type=int, required=True)
    chapter_plan.add_argument("--target-char-max", type=int, required=True)
    chapter_plan.add_argument("--limit", type=int, default=20)
    chapter_plan.add_argument("--run-id", default="chapter-plan")
    chapter_plan_work = sub.add_parser("plan-expansion-work", help="Compile a sequential sample into controlled-expansion programs")
    chapter_plan_work.add_argument("--author-id", required=True); chapter_plan_work.add_argument("--work-id", required=True)
    chapter_plan_work.add_argument("--chapters", type=int, default=5)
    chapter_plan_work.add_argument("--target-char-min", type=int, required=True)
    chapter_plan_work.add_argument("--target-char-max", type=int, required=True)
    chapter_plan_work.add_argument("--annotation-limit", type=int, default=20)
    chapter_plan_work.add_argument("--run-id", default="chapter-plan-work")
    generate_program = sub.add_parser("generate-from-program", help="Generate a chapter scene by scene from a fact-event-scene-paragraph program")
    generate_program.add_argument("--author-id", required=True)
    generate_program.add_argument("--program", type=Path, required=True)
    generate_program.add_argument("--story-state", type=Path)
    generate_program.add_argument("--chapter-plan", type=Path)
    generate_program.add_argument("--style-budget", type=Path)
    generate_program.add_argument("--graph", type=Path, help="本次生成读取的 narrative_graph.json；受控扩写应始终传入该唯一事实底座")
    generate_program.add_argument("--run-id", default="program-generated")
    generate_program.add_argument("--max-repairs", type=int, default=DEFAULT_SCENE_MAX_REPAIRS)
    generate_program_work = sub.add_parser("generate-program-work", help="按规划顺序生成多章；仅向下一章传递已通过的出口状态")
    generate_program_work.add_argument("--author-id", required=True)
    generate_program_work.add_argument("--index", type=Path, required=True)
    generate_program_work.add_argument("--run-id", default="program-sequence")
    generate_program_work.add_argument("--max-repairs", type=int, default=DEFAULT_SCENE_MAX_REPAIRS)
    evaluate_program = sub.add_parser("evaluate-program-fidelity", help="以匿名结构和量化文风评测受控扩写生成序列")
    evaluate_program.add_argument("--author-id", required=True); evaluate_program.add_argument("--work-id", required=True)
    evaluate_program.add_argument("--index", type=Path, required=True); evaluate_program.add_argument("--sequence", type=Path, required=True)
    evaluate_program.add_argument("--annotation-limit", type=int, default=DEFAULT_ANNOTATION_LIMIT)
    evaluate_program.add_argument("--run-id", default="program-fidelity")
    run_cmd = sub.add_parser("run", help="Run the numbered V3 pipeline")
    run_cmd.add_argument("--author-id", required=True)
    run_cmd.add_argument("--run-id", default="pipeline")
    run_cmd.add_argument("--source-dir", type=Path, default=ROOT / "input")
    run_cmd.add_argument("--work-id", help="只运行指定的已导入作品；多作品时编译 skill 必填")
    run_cmd.add_argument("--offline", action="store_true")
    run_cmd.add_argument("--start-stage", type=int, default=1)
    run_cmd.add_argument("--end-stage", type=int, default=7)
    run_cmd.add_argument("--chapter-limit", type=int, default=DEFAULT_ANNOTATION_LIMIT)
    run_cmd.add_argument("--annotation-input-chars", type=int, default=DEFAULT_ANNOTATION_INPUT_CHARS)
    run_cmd.add_argument("--annotation-overlap-units", type=int, default=DEFAULT_ANNOTATION_OVERLAP_UNITS)
    run_cmd.add_argument("--template-batch-size", type=int, default=DEFAULT_TEMPLATE_BATCH_CHAPTERS)
    run_cmd.add_argument("--style-batch-size", type=int, default=DEFAULT_STYLE_BATCH_CHAPTERS)
    run_cmd.add_argument("--expansion-chapters", type=int, default=0)
    run_cmd.add_argument("--target-char-min", type=int, default=0)
    run_cmd.add_argument("--target-char-max", type=int, default=0)
    run_cmd.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    author_id = validate_author_id(args.author_id)
    if args.command == "ingest":
        result = ingest_file(author_id, args.source, args.work_id) if args.source else ingest_directory(author_id, args.source_dir)
        print(result)
    elif args.command == "probe-model":
        print({"probe": str(probe_model(author_id, run_id=args.run_id))})
    elif args.command == "annotate":
        annotations, quality = annotate_work(
            author_id, args.work_id, limit=args.limit, max_input_chars=args.max_input_chars,
            overlap_units=args.overlap_units, resume=not args.no_resume,
        )
        print({"annotations": str(annotations), "quality": str(quality)})
    elif args.command == "continuity":
        continuity, quality = build_continuity_work(args.author_id, args.work_id, limit=args.limit)
        print({"continuity": str(continuity), "quality": str(quality)})
    elif args.command == "graph":
        graph, quality = build_narrative_graph_work(args.author_id, args.work_id, limit=args.limit)
        print({"graph": str(graph), "quality": str(quality)})
    elif args.command == "templates":
        templates, quality = mine_template_library(
            args.author_id, args.work_id, limit=args.limit, batch_size=args.batch_size, offline=args.offline,
        )
        print({"templates": str(templates), "quality": str(quality)})
    elif args.command == "style":
        cards, profile, quality = distil_style_profile(
            args.author_id, args.work_id, limit=args.limit, batch_size=args.batch_size, offline=args.offline,
        )
        print({"style_cards": str(cards), "profile": str(profile), "quality": str(quality)})
    elif args.command == "compile-skill":
        skill, quality = compile_author_skill(args.author_id, args.work_id, limit=args.limit)
        print({"skill": str(skill), "quality": str(quality)})
    elif args.command == "prepare-plan":
        print({"prompt": str(prepare_plan_prompt(args.author_id, args.story_state, args.chapter_brief, args.run_id))})
    elif args.command == "prepare-draft":
        print({"prompt": str(prepare_draft_prompt(args.author_id, args.story_state, args.chapter_plan, args.run_id))})
    elif args.command == "prepare-validation":
        print({"prompt": str(prepare_validation_prompt(args.author_id, args.story_state, args.chapter_plan, args.draft, args.run_id))})
    elif args.command == "compile-program":
        print(compile_chapter_program_work(args.author_id, args.work_id, args.chapter, limit=args.limit, run_id=args.run_id))
    elif args.command == "plan-expansion":
        print(plan_expansion_program_work(
            args.author_id, args.work_id, args.chapter,
            target_char_min=args.target_char_min, target_char_max=args.target_char_max,
            limit=args.limit, run_id=args.run_id,
        ))
    elif args.command == "plan-expansion-work":
        print({"index": str(plan_expansion_programs_work(
            args.author_id, args.work_id, chapter_limit=args.chapters,
            target_char_min=args.target_char_min, target_char_max=args.target_char_max,
            annotation_limit=args.annotation_limit, run_id=args.run_id,
        ))})
    elif args.command == "generate-from-program":
        print(generate_from_program(
            args.author_id, args.program, story_state_path=args.story_state, chapter_plan_path=args.chapter_plan,
            style_budget_path=args.style_budget, run_id=args.run_id, max_repairs=args.max_repairs,
            graph_path=args.graph,
        ))
    elif args.command == "generate-program-work":
        print({"summary": str(generate_program_sequence(
            args.author_id, args.index, run_id=args.run_id, max_repairs=args.max_repairs,
        ))})
    elif args.command == "evaluate-program-fidelity":
        print({"summary": str(evaluate_program_sequence(
            args.author_id, args.work_id, args.index, args.sequence,
            annotation_limit=args.annotation_limit, run_id=args.run_id,
        ))})
    elif args.command == "run":
        print(run_pipeline(PipelineOptions(
            author_id=author_id, run_id=args.run_id, source_dir=args.source_dir, offline=args.offline,
            work_id=args.work_id, start_stage=args.start_stage, end_stage=args.end_stage, chapter_limit=args.chapter_limit,
            annotation_input_chars=args.annotation_input_chars, annotation_overlap_units=args.annotation_overlap_units,
            template_batch_size=args.template_batch_size, style_batch_size=args.style_batch_size,
            expansion_chapters=args.expansion_chapters, target_char_min=args.target_char_min,
            target_char_max=args.target_char_max, resume=not args.no_resume,
        )))
    return 0
