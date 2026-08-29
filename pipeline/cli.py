from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common.paths import ROOT, validate_author_id
from .modules.module_00_ingestion.api import ingest_directory, ingest_file
from .orchestrator import StageContext, load_entrypoint


PRIVATE_SYNOPSIS_ENTRYPOINT = (
    "pipeline.modules.module_01_local_synopsis_private.api:run"
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
    synopsis.add_argument("--target-chars", type=int, default=2600)
    synopsis.add_argument("--no-resume", action="store_true")
    synopsis.add_argument("--local-model-for-chapter", action="store_true")
    synopsis.add_argument(
        "--chapter-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="章级状态终审默认关闭思考模式；可显式开启进行对照实验",
    )
    synopsis.add_argument("--no-semantic-review", action="store_true")
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
                "target_chars": args.target_chars,
                "resume": not args.no_resume,
                "use_quality_model": not args.local_model_for_chapter,
                "chapter_thinking": args.chapter_thinking,
                "semantic_review": not args.no_semantic_review,
            },
        ))
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if result.accepted else 2
    return 2
