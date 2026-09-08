"""Quality report assembly for module 02."""

from __future__ import annotations

from typing import Any

from .internal import OutlineExecutionResult
from .validation import assess_outline_structure


def outline_quality_report(result: OutlineExecutionResult) -> dict[str, Any]:
    structure = (
        assess_outline_structure(result.bundle).to_dict()
        if result.bundle is not None
        else {
            "passed": False,
            "node_count": 0,
            "root_count": 0,
            "level_counts": {"story_arc": 0, "volume": 0, "book": 0},
            "issues": [],
        }
    )
    nodes = result.bundle.nodes if result.bundle is not None else ()
    story_arcs = tuple(node for node in nodes if node.level == "story_arc")
    return {
        "passed": result.accepted and bool(structure["passed"]),
        "processing_completed": result.accepted,
        "chapter_count": len(result.chapter_ids),
        "aggregation_ceiling": result.aggregation_ceiling,
        "story_arc_count": len(story_arcs),
        "volume_count": sum(node.level == "volume" for node in nodes),
        "book_count": sum(node.level == "book" for node in nodes),
        "singleton_story_arc_count": sum(
            len(node.chapter_ids) == 1 for node in story_arcs
        ),
        "boundary_count": len(result.boundary_decisions),
        "volume_boundary_count": len(result.volume_boundary_decisions),
        "structure": structure,
        "blocking_issues": [issue.to_dict() for issue in result.review_issues],
        "failure": result.failure,
        "request_metrics": result.request_metrics,
    }
