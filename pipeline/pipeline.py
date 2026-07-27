"""The numbered, evidence-first V3 pipeline entry point.

This module deliberately composes only the V3 contracts.  Retired extractor,
story-state and generic-reconstruction modules remain callable for archive
inspection, but cannot be reached from ``python main.py run``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .annotate import (
    DEFAULT_ANNOTATION_INPUT_CHARS,
    DEFAULT_ANNOTATION_LIMIT,
    DEFAULT_ANNOTATION_OVERLAP_UNITS,
    annotate_work,
)
from .continuity import build_continuity_work
from .narrative_graph import build_narrative_graph_work
from .ingest import ingest_directory
from .manifest import initialize_manifest, record_stage
from .model_probe import probe_model
from .paths import ROOT, active_work_ids, corpus_dir, output_dir
from .scene_compiler import plan_expansion_programs_work
from .skill_compiler import compile_author_skill
from .style_distillation import DEFAULT_STYLE_BATCH_CHAPTERS, distil_style_profile
from .template_mining import DEFAULT_TEMPLATE_BATCH_CHAPTERS, mine_template_library


@dataclass(frozen=True)
class PipelineOptions:
    """Configuration for one reproducible V3 author-skill pipeline run."""

    author_id: str
    run_id: str = "pipeline"
    source_dir: Path = ROOT / "input"
    work_id: str | None = None
    offline: bool = False
    start_stage: int = 1
    end_stage: int = 7
    chapter_limit: int = DEFAULT_ANNOTATION_LIMIT
    annotation_input_chars: int = DEFAULT_ANNOTATION_INPUT_CHARS
    annotation_overlap_units: int = DEFAULT_ANNOTATION_OVERLAP_UNITS
    template_batch_size: int = DEFAULT_TEMPLATE_BATCH_CHAPTERS
    style_batch_size: int = DEFAULT_STYLE_BATCH_CHAPTERS
    expansion_chapters: int = 0
    target_char_min: int = 0
    target_char_max: int = 0
    resume: bool = True


def _run_stage(
    number: int,
    options: PipelineOptions,
    action: Callable[[], list[Path]],
    name: str,
    details: dict[str, object],
) -> list[Path]:
    if not options.start_stage <= number <= options.end_stage:
        return []
    print(f"[pipeline] stage {number}/7: {name}", flush=True)
    artifacts = action()
    record_stage(options.author_id, options.run_id, number, name, artifacts, details)
    return artifacts


def _selected_work_ids(options: PipelineOptions) -> list[str]:
    available = active_work_ids(options.author_id)
    if options.work_id is None:
        return available
    if options.work_id not in available:
        raise FileNotFoundError(f"work_id is not active for {options.author_id}: {options.work_id}")
    return [options.work_id]


def _skill_work_id(work_ids: list[str]) -> str:
    """Protect the author-level output name from silently overwriting a work.

    The current V3 profile/template contracts are explicitly work-scoped.  A
    later author-wide merge stage may combine several such validated packages;
    until then callers must choose one work instead of receiving a misleading
    last-work-wins skill.
    """

    if len(work_ids) != 1:
        raise ValueError(
            "编译单一小说作者 skill 前必须只选择一部作品；请传 --work-id，"
            "或先分别完成各作品的 V3 产物。"
        )
    return work_ids[0]


def _validate_options(options: PipelineOptions) -> None:
    if not 1 <= options.start_stage <= options.end_stage <= 7:
        raise ValueError("stages must satisfy 1 <= start_stage <= end_stage <= 7")
    if options.chapter_limit <= 0:
        raise ValueError("chapter_limit must be positive")
    if options.annotation_input_chars < 400:
        raise ValueError("annotation_input_chars must be at least 400")
    if options.annotation_overlap_units < 0:
        raise ValueError("annotation_overlap_units must not be negative")
    if options.offline and options.start_stage <= 2 <= options.end_stage:
        raise ValueError("高密度证据标注需要模型；--offline 只能用于已完成标注后的结构性阶段")
    if options.template_batch_size <= 0 or options.style_batch_size <= 0:
        raise ValueError("template_batch_size and style_batch_size must be positive")
    if options.expansion_chapters < 0:
        raise ValueError("expansion_chapters must not be negative")
    has_target = bool(options.target_char_min or options.target_char_max)
    if bool(options.expansion_chapters) != has_target:
        raise ValueError("扩写规划需要同时设置 expansion_chapters 与完整的目标字数范围")
    if has_target and not 0 < options.target_char_min <= options.target_char_max:
        raise ValueError("目标字数范围必须满足 0 < target_char_min <= target_char_max")


def run_pipeline(options: PipelineOptions) -> Path:
    """Run V3 stages in order and return the generic author-skill destination.

    Stage 7 is intentionally planning-only.  It never calls the text model:
    generation is a separately resumable, checked command and cannot publish a
    candidate draft into the skill package.
    """

    _validate_options(options)
    initialize_manifest(options.author_id, options.run_id, {
        "pipeline_version": "v3",
        "offline": options.offline,
        "source_dir": str(options.source_dir.resolve()),
        "work_id": options.work_id,
        "chapter_limit": options.chapter_limit,
        "annotation_input_chars": options.annotation_input_chars,
        "annotation_overlap_units": options.annotation_overlap_units,
        "template_batch_size": options.template_batch_size,
        "style_batch_size": options.style_batch_size,
        "expansion_chapters": options.expansion_chapters,
        "target_char_min": options.target_char_min,
        "target_char_max": options.target_char_max,
        "resume": options.resume,
    })

    def ingest() -> list[Path]:
        ingest_directory(options.author_id, options.source_dir)
        return [corpus_dir(options.author_id, work_id) / "metadata.json" for work_id in _selected_work_ids(options)]

    _run_stage(1, options, ingest, "导入语料", {"source_dir": str(options.source_dir)})
    work_ids = _selected_work_ids(options)
    if not work_ids:
        raise FileNotFoundError(f"no active works found for {options.author_id}")

    def annotate() -> list[Path]:
        artifacts: list[Path] = [probe_model(options.author_id, run_id=f"{options.run_id}-annotation-probe")]
        for work_id in work_ids:
            annotations, quality = annotate_work(
                options.author_id,
                work_id,
                limit=options.chapter_limit,
                max_input_chars=options.annotation_input_chars,
                overlap_units=options.annotation_overlap_units,
                resume=options.resume,
            )
            artifacts.extend((annotations, quality))
        return artifacts

    _run_stage(2, options, annotate, "高密度证据标注", {
        "chapter_limit": options.chapter_limit,
        "max_input_chars": options.annotation_input_chars,
        "overlap_units": options.annotation_overlap_units,
    })

    def continuity() -> list[Path]:
        artifacts: list[Path] = []
        for work_id in work_ids:
            continuity_graph, continuity_quality = build_continuity_work(options.author_id, work_id, limit=options.chapter_limit)
            narrative_graph, narrative_quality = build_narrative_graph_work(options.author_id, work_id, limit=options.chapter_limit)
            artifacts.extend((continuity_graph, continuity_quality, narrative_graph, narrative_quality))
        return artifacts

    _run_stage(3, options, continuity, "构建跨章节连续性与剧情事实图谱", {"chapter_limit": options.chapter_limit})

    def templates() -> list[Path]:
        artifacts: list[Path] = []
        for work_id in work_ids:
            library, quality = mine_template_library(
                options.author_id,
                work_id,
                limit=options.chapter_limit,
                batch_size=options.template_batch_size,
                offline=options.offline,
            )
            artifacts.extend((library, quality))
        return artifacts

    _run_stage(4, options, templates, "多层叙事模板挖掘", {
        "chapter_limit": options.chapter_limit,
        "batch_size": options.template_batch_size,
    })

    def style() -> list[Path]:
        artifacts: list[Path] = []
        for work_id in work_ids:
            cards, profile, quality = distil_style_profile(
                options.author_id,
                work_id,
                limit=options.chapter_limit,
                batch_size=options.style_batch_size,
                offline=options.offline,
            )
            artifacts.extend((cards, profile, quality))
        return artifacts

    _run_stage(5, options, style, "量化文风蒸馏", {
        "chapter_limit": options.chapter_limit,
        "batch_size": options.style_batch_size,
    })

    def compile_skill() -> list[Path]:
        skill, quality = compile_author_skill(options.author_id, _skill_work_id(work_ids), limit=options.chapter_limit)
        return [skill / "SKILL.md", quality]

    _run_stage(6, options, compile_skill, "编译小说作者 skill", {"output_dir": str(output_dir(options.author_id))})

    def plan_expansion() -> list[Path]:
        if options.expansion_chapters == 0:
            return []
        work_id = _skill_work_id(work_ids)
        index = plan_expansion_programs_work(
            options.author_id,
            work_id,
            chapter_limit=min(options.expansion_chapters, options.chapter_limit),
            target_char_min=options.target_char_min,
            target_char_max=options.target_char_max,
            annotation_limit=options.chapter_limit,
            run_id=options.run_id,
        )
        return [index]

    _run_stage(7, options, plan_expansion, "受控扩写章节规划（可选）", {
        "expansion_chapters": options.expansion_chapters,
        "target_char_min": options.target_char_min,
        "target_char_max": options.target_char_max,
        "generation": "not_run_by_pipeline",
    })
    return output_dir(options.author_id)
