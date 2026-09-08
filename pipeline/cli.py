from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common.jsonio import read_json
from .common.paths import ROOT, runs_dir, validate_author_id
from .modules.module_00_ingestion.api import ingest_directory, ingest_file
from .orchestrator import StageContext, load_entrypoint


PRIVATE_SYNOPSIS_ENTRYPOINT = (
    "pipeline.modules.module_01_local_synopsis_private.api:run"
)
HIERARCHICAL_OUTLINE_ENTRYPOINT = (
    "pipeline.modules.module_02_hierarchical_outline.api:run"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="小说作者写作规律蒸馏流水线",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="导入小说并建立不可变章节文档")
    ingest.add_argument("--author-id", required=True)
    ingest.add_argument("--source-dir", type=Path, default=ROOT / "input")
    ingest.add_argument("--source", type=Path)
    ingest.add_argument("--work-id")

    synopsis = sub.add_parser(
        "extract-synopsis",
        help="运行第一模块：中等粒度局部剧情段与章节核心梗概",
    )
    synopsis.add_argument("--author-id", required=True)
    synopsis.add_argument("--work-id", required=True)
    synopsis.add_argument("--limit", type=int, default=5)
    synopsis.add_argument("--run-id", default="module-01")
    synopsis.add_argument("--no-resume", action="store_true")
    synopsis.add_argument(
        "--chapter-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅为全章导航草稿开启模型思考模式；其他单任务阶段保持关闭",
    )
    outline = sub.add_parser(
        "build-hierarchical-outline",
        help="运行第二模块：把已验收章纲聚合为故事弧、卷或全书大纲",
    )
    outline.add_argument("--author-id", required=True)
    outline.add_argument("--work-id", required=True)
    outline.add_argument(
        "--source-run-id",
        required=True,
        help="包含 module_01_local_synopsis 正式产物的运行 ID",
    )
    outline.add_argument(
        "--run-id",
        help="第二模块输出运行 ID；默认与 source-run-id 相同",
    )
    outline.add_argument("--no-resume", action="store_true")
    outline.add_argument(
        "--aggregation-ceiling",
        choices=("story_arc", "volume", "book"),
        help="聚合顶层；默认短样本为 story_arc，完整模式为 book",
    )
    outline.add_argument(
        "--boundary-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅为故事弧与语义卷边界判断开启思考模式",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    author_id = validate_author_id(args.author_id)
    if args.command == "ingest":
        result = (
            ingest_file(author_id, args.source, args.work_id)
            if args.source
            else ingest_directory(author_id, args.source_dir)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0

    if args.command == "extract-synopsis":
        entrypoint = load_entrypoint(PRIVATE_SYNOPSIS_ENTRYPOINT)
        result = entrypoint(StageContext(
            author_id=author_id,
            work_id=args.work_id,
            run_id=args.run_id,
            profile="short_validation" if args.limit <= 20 else "full_book",
            workspace_root=ROOT,
            options={
                "chapter_limit": args.limit,
                "resume": not args.no_resume,
                "chapter_thinking": args.chapter_thinking,
            },
        ))
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if result.accepted else 2
    if args.command == "build-hierarchical-outline":
        source_stage = (
            runs_dir(author_id, args.source_run_id)
            / "module_01_local_synopsis"
        )
        source_manifest = read_json(source_stage / "manifest.json")
        if not isinstance(source_manifest, dict):
            raise ValueError("module-01 manifest must be a JSON object")
        source_profile = str(source_manifest.get("profile", "short_validation"))
        aggregation_ceiling = args.aggregation_ceiling or (
            "book" if source_profile == "full_book" else "story_arc"
        )
        entrypoint = load_entrypoint(HIERARCHICAL_OUTLINE_ENTRYPOINT)
        result = entrypoint(StageContext(
            author_id=author_id,
            work_id=args.work_id,
            run_id=args.run_id or args.source_run_id,
            profile=source_profile,
            workspace_root=ROOT,
            input_paths={
                "synopsis_manifest": source_stage / "manifest.json",
                "synopsis_bundles": (
                    source_stage / "output" / "chapter_synopsis_bundles.jsonl"
                ),
            },
            options={
                "aggregation_ceiling": aggregation_ceiling,
                "resume": not args.no_resume,
                "boundary_thinking": args.boundary_thinking,
            },
        ))
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if result.accepted else 2
    return 2
