"""Private, deterministic input views used by module 02.

These records deliberately contain synopsis text only.  They must never grow a
source-text field or become a public cross-module contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...contracts import ChapterSynopsisBundle, HierarchicalOutlineBundle, OutlineNode


@dataclass(frozen=True)
class LocalSegmentCard:
    segment_id: str
    order: int
    summary: str


@dataclass(frozen=True)
class ChapterCard:
    chapter_id: str
    order: int
    synopsis_hash: str
    opening_state: str
    local_segments: tuple[LocalSegmentCard, ...]
    chapter_summary: str
    ending_state: str


@dataclass(frozen=True)
class AcceptedSynopsisInput:
    author_id: str
    work_id: str
    profile: str
    manifest_path: Path
    bundles_path: Path
    chapter_ids: tuple[str, ...]
    synopsis_hashes: tuple[str, ...]
    bundles: tuple[ChapterSynopsisBundle, ...]
    chapter_cards: tuple[ChapterCard, ...]
    available_narrative_chapter_count: int
    selected_narrative_chapter_count: int
    complete_work: bool


@dataclass(frozen=True)
class BoundaryDecision:
    left_chapter_id: str
    right_chapter_id: str
    decision: str
    reason: str


@dataclass(frozen=True)
class BoundaryIssue:
    left_chapter_id: str
    right_chapter_id: str
    problem: str
    reason: str


@dataclass(frozen=True)
class ParentBoundaryDecision:
    left_outline_id: str
    right_outline_id: str
    decision: str
    reason: str


@dataclass(frozen=True)
class ParentBoundaryIssue:
    left_outline_id: str
    right_outline_id: str
    problem: str
    reason: str


@dataclass(frozen=True)
class StoryArcSpan:
    order: int
    cards: tuple[ChapterCard, ...]

    @property
    def chapter_ids(self) -> tuple[str, ...]:
        return tuple(card.chapter_id for card in self.cards)


@dataclass(frozen=True)
class OutlineUnitCard:
    outline_id: str
    level: str
    order: int
    chapter_ids: tuple[str, ...]
    title: str
    summary: str
    opening_situation: str
    central_goal: str
    central_conflict: str
    causal_chain: tuple[str, ...]
    turning_points: tuple[str, ...]
    ending_change: str
    open_threads: tuple[str, ...]
    character_arcs: tuple[dict[str, Any], ...]

    @classmethod
    def from_node(cls, node: OutlineNode) -> "OutlineUnitCard":
        return cls(
            outline_id=node.outline_id,
            level=node.level,
            order=node.order,
            chapter_ids=node.chapter_ids,
            title=node.title,
            summary=node.summary,
            opening_situation=node.opening_situation,
            central_goal=node.central_goal,
            central_conflict=node.central_conflict,
            causal_chain=node.causal_chain,
            turning_points=node.turning_points,
            ending_change=node.ending_change,
            open_threads=node.open_threads,
            character_arcs=tuple(item.to_dict() for item in node.character_arcs),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "outline_id": self.outline_id,
            "level": self.level,
            "order": self.order,
            "chapter_ids": list(self.chapter_ids),
            "title": self.title,
            "summary": self.summary,
            "opening_situation": self.opening_situation,
            "central_goal": self.central_goal,
            "central_conflict": self.central_conflict,
            "causal_chain": list(self.causal_chain),
            "turning_points": list(self.turning_points),
            "ending_change": self.ending_change,
            "open_threads": list(self.open_threads),
            "character_arcs": list(self.character_arcs),
        }


@dataclass(frozen=True)
class ParentSpan:
    target_level: str
    order: int
    children: tuple[OutlineUnitCard, ...]

    @property
    def child_outline_ids(self) -> tuple[str, ...]:
        return tuple(child.outline_id for child in self.children)

    @property
    def chapter_ids(self) -> tuple[str, ...]:
        return tuple(
            chapter_id for child in self.children for chapter_id in child.chapter_ids
        )


@dataclass(frozen=True)
class NodeReviewIssue:
    review_kind: str
    code: str
    field: str
    message: str
    chapter_ids: tuple[str, ...]
    segment_ids: tuple[str, ...] = ()
    outline_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_kind": self.review_kind,
            "code": self.code,
            "field": self.field,
            "message": self.message,
            "chapter_ids": list(self.chapter_ids),
            "segment_ids": list(self.segment_ids),
            "outline_ids": list(self.outline_ids),
        }


@dataclass(frozen=True)
class OutlineExecutionResult:
    chapter_ids: tuple[str, ...]
    aggregation_ceiling: str = "story_arc"
    bundle: HierarchicalOutlineBundle | None = None
    boundary_decisions: tuple[BoundaryDecision, ...] = ()
    volume_boundary_decisions: tuple[ParentBoundaryDecision, ...] = ()
    candidate_nodes: tuple[OutlineNode, ...] = ()
    review_issues: tuple[NodeReviewIssue, ...] = ()
    failure: dict[str, Any] | None = None
    request_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.bundle is not None and self.failure is None
