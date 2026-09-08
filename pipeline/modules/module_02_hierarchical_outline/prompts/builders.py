from __future__ import annotations

import json
from typing import Any

from ....contracts import OutlineNode
from ..internal import (
    BoundaryDecision,
    ChapterCard,
    NodeReviewIssue,
    OutlineUnitCard,
    ParentBoundaryDecision,
    ParentSpan,
    StoryArcSpan,
)
from .registry import render_template


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def compact_chapter_cards(cards: tuple[ChapterCard, ...]) -> list[dict[str, Any]]:
    return [{
        "chapter_id": card.chapter_id,
        "order": card.order,
        "opening_state": card.opening_state,
        "chapter_summary": card.chapter_summary,
        "ending_state": card.ending_state,
    } for card in cards]


def detailed_chapter_cards(cards: tuple[ChapterCard, ...]) -> list[dict[str, Any]]:
    return [{
        **compact_chapter_cards((card,))[0],
        "local_segments": [{
            "segment_id": segment.segment_id,
            "order": segment.order,
            "summary": segment.summary,
        } for segment in card.local_segments],
    } for card in cards]


def boundary_detection_prompt(cards: tuple[ChapterCard, ...]) -> str:
    return render_template(
        "boundary_detection",
        CHAPTER_IDS_JSON=_json([card.chapter_id for card in cards]),
        CHAPTER_CARDS_JSON=_json(compact_chapter_cards(cards)),
    )


def boundary_review_prompt(
    cards: tuple[ChapterCard, ...], decisions: tuple[BoundaryDecision, ...]
) -> str:
    return render_template(
        "boundary_review",
        CHAPTER_CARDS_JSON=_json(compact_chapter_cards(cards)),
        BOUNDARIES_JSON=_json([{
            "left_chapter_id": item.left_chapter_id,
            "right_chapter_id": item.right_chapter_id,
            "decision": item.decision,
            "reason": item.reason,
        } for item in decisions]),
    )


def node_synthesis_prompt(span: StoryArcSpan) -> str:
    return render_template(
        "node_synthesis",
        CHAPTER_IDS_JSON=_json(list(span.chapter_ids)),
        CHAPTER_CARDS_JSON=_json(detailed_chapter_cards(span.cards)),
    )


def node_review_prompt(
    template: str, span: StoryArcSpan, candidate: OutlineNode
) -> str:
    return render_template(
        template,
        CHAPTER_CARDS_JSON=_json(detailed_chapter_cards(span.cards)),
        CANDIDATE_JSON=_json(candidate.to_dict()),
    )


def node_repair_prompt(
    span: StoryArcSpan,
    candidate: OutlineNode,
    issues: tuple[NodeReviewIssue, ...],
) -> str:
    return render_template(
        "node_repair",
        CHAPTER_IDS_JSON=_json(list(span.chapter_ids)),
        CHAPTER_CARDS_JSON=_json(detailed_chapter_cards(span.cards)),
        CANDIDATE_JSON=_json(candidate.to_dict()),
        ISSUES_JSON=_json([issue.to_dict() for issue in issues]),
    )


def top_down_review_prompt(
    cards: tuple[ChapterCard, ...], nodes: tuple[OutlineNode, ...]
) -> str:
    return render_template(
        "top_down_review",
        CHAPTER_CARDS_JSON=_json(compact_chapter_cards(cards)),
        NODES_JSON=_json([node.to_dict() for node in nodes]),
    )


def outline_unit_cards(cards: tuple[OutlineUnitCard, ...]) -> list[dict[str, Any]]:
    return [card.to_dict() for card in cards]


def volume_boundary_detection_prompt(cards: tuple[OutlineUnitCard, ...]) -> str:
    return render_template(
        "volume_boundary_detection",
        OUTLINE_IDS_JSON=_json([card.outline_id for card in cards]),
        STORY_ARC_CARDS_JSON=_json(outline_unit_cards(cards)),
    )


def volume_boundary_review_prompt(
    cards: tuple[OutlineUnitCard, ...],
    decisions: tuple[ParentBoundaryDecision, ...],
) -> str:
    return render_template(
        "volume_boundary_review",
        STORY_ARC_CARDS_JSON=_json(outline_unit_cards(cards)),
        BOUNDARIES_JSON=_json([{
            "left_outline_id": item.left_outline_id,
            "right_outline_id": item.right_outline_id,
            "decision": item.decision,
            "reason": item.reason,
        } for item in decisions]),
    )


def parent_node_synthesis_prompt(span: ParentSpan) -> str:
    return render_template(
        "parent_node_synthesis",
        TARGET_LEVEL=span.target_level,
        CHILD_LEVEL=span.children[0].level,
        CHILD_OUTLINE_IDS_JSON=_json(list(span.child_outline_ids)),
        CHILD_CARDS_JSON=_json(outline_unit_cards(span.children)),
    )


def parent_node_review_prompt(
    template: str, span: ParentSpan, candidate: OutlineNode
) -> str:
    return render_template(
        template,
        TARGET_LEVEL=span.target_level,
        CHILD_CARDS_JSON=_json(outline_unit_cards(span.children)),
        CANDIDATE_JSON=_json(candidate.to_dict()),
    )


def parent_node_repair_prompt(
    span: ParentSpan,
    candidate: OutlineNode,
    issues: tuple[NodeReviewIssue, ...],
) -> str:
    return render_template(
        "parent_node_repair",
        TARGET_LEVEL=span.target_level,
        CHILD_OUTLINE_IDS_JSON=_json(list(span.child_outline_ids)),
        CHILD_CARDS_JSON=_json(outline_unit_cards(span.children)),
        CANDIDATE_JSON=_json(candidate.to_dict()),
        ISSUES_JSON=_json([issue.to_dict() for issue in issues]),
    )


def hierarchy_review_prompt(
    target_level: str,
    child_nodes: tuple[OutlineNode, ...],
    parent_nodes: tuple[OutlineNode, ...],
) -> str:
    return render_template(
        "hierarchy_review",
        TARGET_LEVEL=target_level,
        CHILD_NODES_JSON=_json([node.to_dict() for node in child_nodes]),
        PARENT_NODES_JSON=_json([node.to_dict() for node in parent_nodes]),
    )
