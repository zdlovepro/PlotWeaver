"""Write accepted module-02 artifacts and explicit failure reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...common.jsonio import write_json
from ...contracts import OUTLINE_SCHEMA_VERSION
from . import MODULE_VERSION
from .internal import AcceptedSynopsisInput, OutlineExecutionResult
from .prompts import PROMPT_REVISION
from .quality import outline_quality_report


def write_outline_artifacts(
    stage_dir: Path,
    input_set: AcceptedSynopsisInput,
    result: OutlineExecutionResult,
    *,
    run_id: str,
    model_policy: dict[str, Any],
) -> dict[str, Path]:
    output_dir = stage_dir / "output"
    paths = {
        "outline": output_dir / "hierarchical_outline.json",
        "quality": stage_dir / "quality_report.json",
        "failures": stage_dir / "failures.json",
        "manifest": stage_dir / "manifest.json",
    }
    quality = outline_quality_report(result)
    if result.accepted and result.bundle is not None:
        write_json(paths["outline"], result.bundle.to_dict())
    write_json(paths["quality"], quality)
    write_json(paths["failures"], {
        "failure_count": 0 if result.failure is None else 1,
        "failures": [] if result.failure is None else [result.failure],
        "blocking_issues": [issue.to_dict() for issue in result.review_issues],
    })
    outputs = {
        "quality": str(paths["quality"].relative_to(stage_dir)),
        "failures": str(paths["failures"].relative_to(stage_dir)),
    }
    if result.accepted:
        outputs["outline"] = str(paths["outline"].relative_to(stage_dir))
    write_json(paths["manifest"], {
        "module": "module_02_hierarchical_outline",
        "module_version": MODULE_VERSION,
        "schema_version": OUTLINE_SCHEMA_VERSION,
        "prompt_revision": PROMPT_REVISION,
        "author_id": input_set.author_id,
        "work_id": input_set.work_id,
        "run_id": run_id,
        "profile": input_set.profile,
        "aggregation_ceiling": result.aggregation_ceiling,
        "input_scope": {
            "available_narrative_chapter_count": (
                input_set.available_narrative_chapter_count
            ),
            "selected_narrative_chapter_count": (
                input_set.selected_narrative_chapter_count
            ),
            "complete_work": input_set.complete_work,
        },
        "chapter_ids": list(input_set.chapter_ids),
        "input_hashes": dict(zip(input_set.chapter_ids, input_set.synopsis_hashes)),
        "input_manifest": str(input_set.manifest_path),
        "model_policy": model_policy,
        "outputs": outputs,
        "accepted": bool(quality["passed"]),
    })
    return paths
