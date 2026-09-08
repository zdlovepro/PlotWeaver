"""Deterministic conversion between semantic boundary decisions and arc spans."""

from __future__ import annotations

from .internal import (
    BoundaryDecision,
    BoundaryIssue,
    ChapterCard,
    OutlineUnitCard,
    ParentBoundaryDecision,
    ParentBoundaryIssue,
    ParentSpan,
    StoryArcSpan,
)


def apply_boundary_issues(
    decisions: tuple[BoundaryDecision, ...],
    issues: tuple[BoundaryIssue, ...],
) -> tuple[BoundaryDecision, ...]:
    by_pair = {(issue.left_chapter_id, issue.right_chapter_id): issue for issue in issues}
    repaired = []
    for item in decisions:
        issue = by_pair.get((item.left_chapter_id, item.right_chapter_id))
        if issue is None:
            repaired.append(item)
            continue
        repaired.append(BoundaryDecision(
            item.left_chapter_id,
            item.right_chapter_id,
            "continue" if issue.problem == "false_split" else "split",
            f"边界审校修正：{issue.reason}",
        ))
    return tuple(repaired)


def story_arc_spans(
    cards: tuple[ChapterCard, ...],
    decisions: tuple[BoundaryDecision, ...],
) -> tuple[StoryArcSpan, ...]:
    if not cards:
        raise ValueError("story arc segmentation requires chapter cards")
    if tuple(card.order for card in cards) != tuple(range(len(cards))):
        raise ValueError("chapter card orders must be consecutive")
    expected_pairs = tuple(
        (cards[index].chapter_id, cards[index + 1].chapter_id)
        for index in range(len(cards) - 1)
    )
    actual_pairs = tuple(
        (item.left_chapter_id, item.right_chapter_id) for item in decisions
    )
    if actual_pairs != expected_pairs:
        raise ValueError("boundary decisions do not match adjacent chapter cards")

    groups: list[tuple[ChapterCard, ...]] = []
    start = 0
    for index, decision in enumerate(decisions):
        if decision.decision not in {"continue", "split"}:
            raise ValueError(f"unsupported boundary decision: {decision.decision}")
        if decision.decision == "split":
            groups.append(cards[start:index + 1])
            start = index + 1
    groups.append(cards[start:])
    return tuple(StoryArcSpan(order, group) for order, group in enumerate(groups))


def apply_parent_boundary_issues(
    decisions: tuple[ParentBoundaryDecision, ...],
    issues: tuple[ParentBoundaryIssue, ...],
) -> tuple[ParentBoundaryDecision, ...]:
    by_pair = {
        (issue.left_outline_id, issue.right_outline_id): issue for issue in issues
    }
    repaired = []
    for item in decisions:
        issue = by_pair.get((item.left_outline_id, item.right_outline_id))
        if issue is None:
            repaired.append(item)
            continue
        repaired.append(ParentBoundaryDecision(
            item.left_outline_id,
            item.right_outline_id,
            "continue" if issue.problem == "false_split" else "split",
            f"边界审校修正：{issue.reason}",
        ))
    return tuple(repaired)


def parent_spans(
    cards: tuple[OutlineUnitCard, ...],
    decisions: tuple[ParentBoundaryDecision, ...],
    *,
    target_level: str,
) -> tuple[ParentSpan, ...]:
    if target_level not in {"volume", "book"}:
        raise ValueError(f"unsupported parent level: {target_level}")
    if not cards:
        raise ValueError(f"{target_level} aggregation requires child cards")
    if tuple(card.order for card in cards) != tuple(range(len(cards))):
        raise ValueError("child outline orders must be consecutive")
    if target_level == "book" and decisions:
        raise ValueError("book aggregation does not accept boundary decisions")

    expected_pairs = tuple(
        (cards[index].outline_id, cards[index + 1].outline_id)
        for index in range(len(cards) - 1)
    )
    actual_pairs = tuple(
        (item.left_outline_id, item.right_outline_id) for item in decisions
    )
    if target_level == "volume" and actual_pairs != expected_pairs:
        raise ValueError("volume boundary decisions do not match adjacent story arcs")

    groups: list[tuple[OutlineUnitCard, ...]] = []
    if target_level == "book":
        groups.append(cards)
    else:
        start = 0
        for index, decision in enumerate(decisions):
            if decision.decision not in {"continue", "split"}:
                raise ValueError(
                    f"unsupported parent boundary decision: {decision.decision}"
                )
            if decision.decision == "split":
                groups.append(cards[start:index + 1])
                start = index + 1
        groups.append(cards[start:])
    return tuple(
        ParentSpan(target_level=target_level, order=order, children=group)
        for order, group in enumerate(groups)
    )
