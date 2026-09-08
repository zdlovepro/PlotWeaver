"""Build public outline nodes from validated model semantics."""

from __future__ import annotations

import hashlib
from typing import Any

from ...contracts import OutlineNode
from .internal import ParentSpan, StoryArcSpan


def story_arc_id(work_id: str, chapter_ids: tuple[str, ...]) -> str:
    if not chapter_ids:
        raise ValueError("story arc id requires chapters")
    digest = hashlib.sha256("\n".join(chapter_ids).encode("utf-8")).hexdigest()[:12]
    return f"{work_id}:story_arc:{digest}"


def parent_outline_id(
    work_id: str, level: str, child_outline_ids: tuple[str, ...]
) -> str:
    if level not in {"volume", "book"}:
        raise ValueError(f"unsupported parent outline level: {level}")
    if not child_outline_ids:
        raise ValueError(f"{level} id requires child outlines")
    digest = hashlib.sha256(
        "\n".join(child_outline_ids).encode("utf-8")
    ).hexdigest()[:12]
    return f"{work_id}:{level}:{digest}"


def build_story_arc_node(
    work_id: str,
    span: StoryArcSpan,
    semantics: dict[str, Any],
) -> OutlineNode:
    payload = {
        **semantics,
        "outline_id": story_arc_id(work_id, span.chapter_ids),
        "level": "story_arc",
        "order": span.order,
        "chapter_ids": list(span.chapter_ids),
        "child_outline_ids": [],
    }
    node = OutlineNode.from_dict(payload)
    node.validate()
    return node


def build_parent_node(
    work_id: str,
    span: ParentSpan,
    semantics: dict[str, Any],
) -> OutlineNode:
    payload = {
        **semantics,
        "outline_id": parent_outline_id(
            work_id, span.target_level, span.child_outline_ids
        ),
        "level": span.target_level,
        "order": span.order,
        "chapter_ids": list(span.chapter_ids),
        "child_outline_ids": list(span.child_outline_ids),
    }
    node = OutlineNode.from_dict(payload)
    node.validate()
    return node
