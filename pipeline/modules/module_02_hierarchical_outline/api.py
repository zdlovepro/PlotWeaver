from pathlib import Path

from ...common.jsonio import read_json
from ...common.model import ModelSettings
from ...common.paths import runs_dir
from ...contracts import OUTLINE_SCHEMA_VERSION
from ...orchestrator import StageContext, StageResult
from . import MODULE_VERSION
from .artifacts import write_outline_artifacts
from .input_loader import load_accepted_synopsis_input
from .internal import AcceptedSynopsisInput
from .service import build_hierarchical_outline


SYNOPSIS_MANIFEST_INPUT = "synopsis_manifest"
SYNOPSIS_BUNDLES_INPUT = "synopsis_bundles"


def prepare_input(context: StageContext) -> AcceptedSynopsisInput:
    missing = tuple(
        name for name in (SYNOPSIS_MANIFEST_INPUT, SYNOPSIS_BUNDLES_INPUT)
        if name not in context.input_paths
    )
    if missing:
        raise ValueError(
            "module 02 requires explicit input_paths: " + ", ".join(missing)
        )
    return load_accepted_synopsis_input(
        Path(context.input_paths[SYNOPSIS_MANIFEST_INPUT]),
        Path(context.input_paths[SYNOPSIS_BUNDLES_INPUT]),
        author_id=context.author_id,
        work_id=context.work_id,
        profile=context.profile,
    )


def run(context: StageContext) -> StageResult:
    accepted_input = prepare_input(context)
    aggregation_ceiling = str(
        context.options.get("aggregation_ceiling", "story_arc")
    )
    if aggregation_ceiling not in {"story_arc", "volume", "book"}:
        raise ValueError(f"unsupported aggregation ceiling: {aggregation_ceiling}")
    settings = ModelSettings.from_environment()
    stage_dir = runs_dir(context.author_id, context.run_id) / "module_02_hierarchical_outline"
    result = build_hierarchical_outline(
        accepted_input,
        aggregation_ceiling=aggregation_ceiling,
        settings=settings,
        checkpoint_dir=stage_dir / "checkpoints",
        resume=bool(context.options.get("resume", True)),
        boundary_thinking=bool(context.options.get("boundary_thinking", False)),
    )
    paths = write_outline_artifacts(
        stage_dir,
        accepted_input,
        result,
        run_id=context.run_id,
        model_policy={
            "generation_model": settings.review_model or settings.model,
            "review_model": settings.review_model or settings.model,
            "boundary_thinking": bool(context.options.get("boundary_thinking", False)),
            "aggregation_ceiling": aggregation_ceiling,
            "max_tokens": settings.max_tokens,
        },
    )
    quality = read_json(paths["quality"], {})
    accepted = bool(isinstance(quality, dict) and quality.get("passed"))
    return StageResult(
        module="module_02_hierarchical_outline",
        module_version=MODULE_VERSION,
        schema_version=OUTLINE_SCHEMA_VERSION,
        accepted=accepted,
        output_paths={
            key: value for key, value in paths.items()
            if key != "quality" and (key != "outline" or accepted)
        },
        quality_report=paths["quality"],
        diagnostics=(
            () if result.failure is None
            else (str(result.failure.get("message", "module 02 failed")),)
        ),
    )
