from __future__ import annotations

from .builders import (
    boundary_detection_prompt,
    boundary_review_prompt,
    hierarchy_review_prompt,
    node_repair_prompt,
    node_review_prompt,
    node_synthesis_prompt,
    parent_node_repair_prompt,
    parent_node_review_prompt,
    parent_node_synthesis_prompt,
    top_down_review_prompt,
    volume_boundary_detection_prompt,
    volume_boundary_review_prompt,
)
from .registry import PROMPT_REVISION, load_template, template_revision


OUTLINE_SYSTEM_PROMPT = load_template("system")

__all__ = [name for name in globals() if not name.startswith("_")]
