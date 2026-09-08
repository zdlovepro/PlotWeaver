"""Reviewed story-arc, volume, and whole-book semantic aggregation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ...common.model import ModelSettings, complete_json
from ...contracts import HierarchicalOutlineBundle, OutlineNode
from .aggregation import build_parent_node, build_story_arc_node
from .internal import (
    AcceptedSynopsisInput,
    BoundaryDecision,
    NodeReviewIssue,
    OutlineExecutionResult,
    OutlineUnitCard,
    ParentBoundaryDecision,
    ParentSpan,
    StoryArcSpan,
)
from .model_io import CheckpointedOutlineModel, JsonCompletion
from .parser import (
    COVERAGE_CODES,
    FIDELITY_CODES,
    HIERARCHY_CODES,
    PARENT_COVERAGE_CODES,
    PARENT_FIDELITY_CODES,
    TOP_DOWN_CODES,
    parse_boundary_decisions,
    parse_boundary_issues,
    parse_node_review,
    parse_node_semantics,
    parse_parent_boundary_decisions,
    parse_parent_boundary_issues,
    parse_parent_node_review,
)
from .prompts import (
    boundary_detection_prompt,
    boundary_review_prompt,
    hierarchy_review_prompt,
    node_repair_prompt,
    node_review_prompt,
    node_synthesis_prompt,
    parent_node_repair_prompt,
    parent_node_review_prompt,
    parent_node_synthesis_prompt,
    template_revision,
    top_down_review_prompt,
    volume_boundary_detection_prompt,
    volume_boundary_review_prompt,
)
from .request_budget import OutlineRequestBudgetExceeded, OutlineRequestLedger
from .segmentation import (
    apply_boundary_issues,
    apply_parent_boundary_issues,
    parent_spans,
    story_arc_spans,
)


class OutlineSemanticReviewError(RuntimeError):
    code = "OUTLINE_SEMANTIC_REVIEW_FAILED"

    def __init__(self, message: str, *, issues: tuple[NodeReviewIssue, ...] = ()):
        super().__init__(message)
        self.issues = issues


class IncompleteBookScopeError(RuntimeError):
    code = "BOOK_REQUIRES_COMPLETE_WORK"


def _source_fingerprint(input_set: AcceptedSynopsisInput) -> str:
    return hashlib.sha256(
        "\n".join(input_set.synopsis_hashes).encode("utf-8")
    ).hexdigest()


def _source_character_count(input_set: AcceptedSynopsisInput) -> int:
    return sum(
        len(card.opening_state)
        + len(card.chapter_summary)
        + len(card.ending_state)
        + sum(len(segment.summary) for segment in card.local_segments)
        for card in input_set.chapter_cards
    )


def _segment_ids(span: StoryArcSpan) -> tuple[str, ...]:
    return tuple(
        segment.segment_id for card in span.cards for segment in card.local_segments
    )


def _review_story_arc_node(
    model: CheckpointedOutlineModel,
    span: StoryArcSpan,
    node: OutlineNode,
    *,
    round_name: str,
) -> tuple[NodeReviewIssue, ...]:
    segment_ids = _segment_ids(span)
    fidelity_payload = model.call(
        stage="node-fidelity-review",
        identity=f"{node.outline_id}:{round_name}",
        revision=template_revision("fidelity_review"),
        prompt=node_review_prompt("fidelity_review", span, node),
        max_tokens=4_000,
    )
    fidelity = parse_node_review(
        fidelity_payload,
        review_kind="fidelity",
        allowed_codes=FIDELITY_CODES,
        chapter_ids=span.chapter_ids,
        segment_ids=segment_ids,
    )
    coverage_payload = model.call(
        stage="node-coverage-review",
        identity=f"{node.outline_id}:{round_name}",
        revision=template_revision("coverage_review"),
        prompt=node_review_prompt("coverage_review", span, node),
        max_tokens=4_000,
    )
    coverage = parse_node_review(
        coverage_payload,
        review_kind="coverage",
        allowed_codes=COVERAGE_CODES,
        chapter_ids=span.chapter_ids,
        segment_ids=segment_ids,
    )
    return (*fidelity, *coverage)


def _build_reviewed_story_arc_node(
    model: CheckpointedOutlineModel,
    work_id: str,
    span: StoryArcSpan,
    candidate_nodes: list[OutlineNode],
) -> OutlineNode:
    payload = model.call(
        stage="node-synthesis",
        identity=f"story-arc:{span.order}",
        revision=template_revision("node_synthesis"),
        prompt=node_synthesis_prompt(span),
        max_tokens=model.settings.max_tokens,
    )
    node = build_story_arc_node(work_id, span, parse_node_semantics(payload))
    candidate_nodes.append(node)
    issues = _review_story_arc_node(model, span, node, round_name="initial")
    if not issues:
        return node

    repair_payload = model.call(
        stage="node-repair",
        identity=node.outline_id,
        revision=template_revision("node_repair"),
        prompt=node_repair_prompt(span, node, issues),
        max_tokens=model.settings.max_tokens,
    )
    repaired = build_story_arc_node(
        work_id, span, parse_node_semantics(repair_payload)
    )
    candidate_nodes.append(repaired)
    remaining = _review_story_arc_node(
        model, span, repaired, round_name="repaired"
    )
    if remaining:
        raise OutlineSemanticReviewError(
            f"story arc {node.outline_id} still has blocking issues after one repair",
            issues=remaining,
        )
    return repaired


def _story_arc_layer(
    model: CheckpointedOutlineModel,
    input_set: AcceptedSynopsisInput,
    candidate_nodes: list[OutlineNode],
    *,
    boundary_thinking: bool,
) -> tuple[tuple[OutlineNode, ...], tuple[BoundaryDecision, ...]]:
    cards = input_set.chapter_cards
    boundary_decisions: tuple[BoundaryDecision, ...] = ()
    if len(cards) > 1:
        boundary_payload = model.call(
            stage="boundary-detection",
            identity="all-chapters",
            revision=template_revision("boundary_detection"),
            prompt=boundary_detection_prompt(cards),
            max_tokens=4_000,
            thinking=boundary_thinking,
        )
        boundary_decisions = parse_boundary_decisions(
            boundary_payload, input_set.chapter_ids
        )
        review_payload = model.call(
            stage="boundary-review",
            identity="initial",
            revision=template_revision("boundary_review"),
            prompt=boundary_review_prompt(cards, boundary_decisions),
            max_tokens=4_000,
        )
        boundary_issues = parse_boundary_issues(review_payload, boundary_decisions)
        if boundary_issues:
            boundary_decisions = apply_boundary_issues(
                boundary_decisions, boundary_issues
            )
            recheck_payload = model.call(
                stage="boundary-review",
                identity="repaired",
                revision=template_revision("boundary_review"),
                prompt=boundary_review_prompt(cards, boundary_decisions),
                max_tokens=4_000,
            )
            remaining = parse_boundary_issues(recheck_payload, boundary_decisions)
            if remaining:
                raise OutlineSemanticReviewError(
                    "story arc boundaries still have blocking issues after one correction"
                )

    spans = story_arc_spans(cards, boundary_decisions)
    model.ledger.configure_arc_count(len(spans))
    nodes = tuple(
        _build_reviewed_story_arc_node(
            model, input_set.work_id, span, candidate_nodes
        )
        for span in spans
    )

    all_segment_ids = tuple(
        segment.segment_id for card in cards for segment in card.local_segments
    )
    top_down_payload = model.call(
        stage="top-down-review",
        identity="all-story-arcs",
        revision=template_revision("top_down_review"),
        prompt=top_down_review_prompt(cards, nodes),
        max_tokens=5_000,
    )
    top_down_issues = parse_node_review(
        top_down_payload,
        review_kind="top_down",
        allowed_codes=TOP_DOWN_CODES,
        chapter_ids=input_set.chapter_ids,
        segment_ids=all_segment_ids,
    )
    if top_down_issues:
        raise OutlineSemanticReviewError(
            "top-down story arc review found blocking issues",
            issues=top_down_issues,
        )
    return nodes, boundary_decisions


def _review_parent_node(
    model: CheckpointedOutlineModel,
    span: ParentSpan,
    node: OutlineNode,
    *,
    round_name: str,
) -> tuple[NodeReviewIssue, ...]:
    level = span.target_level
    fidelity_payload = model.call(
        stage=f"{level}-fidelity-review",
        identity=f"{node.outline_id}:{round_name}",
        revision=template_revision("parent_fidelity_review"),
        prompt=parent_node_review_prompt("parent_fidelity_review", span, node),
        max_tokens=4_000,
    )
    fidelity = parse_parent_node_review(
        fidelity_payload,
        review_kind=f"{level}_fidelity",
        allowed_codes=PARENT_FIDELITY_CODES,
        chapter_ids=span.chapter_ids,
        outline_ids=span.child_outline_ids,
    )
    coverage_payload = model.call(
        stage=f"{level}-coverage-review",
        identity=f"{node.outline_id}:{round_name}",
        revision=template_revision("parent_coverage_review"),
        prompt=parent_node_review_prompt("parent_coverage_review", span, node),
        max_tokens=4_000,
    )
    coverage = parse_parent_node_review(
        coverage_payload,
        review_kind=f"{level}_coverage",
        allowed_codes=PARENT_COVERAGE_CODES,
        chapter_ids=span.chapter_ids,
        outline_ids=span.child_outline_ids,
    )
    return (*fidelity, *coverage)


def _build_reviewed_parent_node(
    model: CheckpointedOutlineModel,
    work_id: str,
    span: ParentSpan,
    candidate_nodes: list[OutlineNode],
) -> OutlineNode:
    level = span.target_level
    payload = model.call(
        stage=f"{level}-node-synthesis",
        identity=f"{level}:{span.order}",
        revision=template_revision("parent_node_synthesis"),
        prompt=parent_node_synthesis_prompt(span),
        max_tokens=model.settings.max_tokens,
    )
    node = build_parent_node(work_id, span, parse_node_semantics(payload))
    candidate_nodes.append(node)
    issues = _review_parent_node(model, span, node, round_name="initial")
    if not issues:
        return node

    repair_payload = model.call(
        stage=f"{level}-node-repair",
        identity=node.outline_id,
        revision=template_revision("parent_node_repair"),
        prompt=parent_node_repair_prompt(span, node, issues),
        max_tokens=model.settings.max_tokens,
    )
    repaired = build_parent_node(
        work_id, span, parse_node_semantics(repair_payload)
    )
    candidate_nodes.append(repaired)
    remaining = _review_parent_node(model, span, repaired, round_name="repaired")
    if remaining:
        raise OutlineSemanticReviewError(
            f"{level} {node.outline_id} still has blocking issues after one repair",
            issues=remaining,
        )
    return repaired


def _review_hierarchy(
    model: CheckpointedOutlineModel,
    target_level: str,
    child_nodes: tuple[OutlineNode, ...],
    parent_nodes: tuple[OutlineNode, ...],
) -> None:
    payload = model.call(
        stage=f"{target_level}-hierarchy-review",
        identity=f"all-{target_level}-nodes",
        revision=template_revision("hierarchy_review"),
        prompt=hierarchy_review_prompt(target_level, child_nodes, parent_nodes),
        max_tokens=5_000,
    )
    chapter_ids = tuple(
        chapter_id for node in child_nodes for chapter_id in node.chapter_ids
    )
    outline_ids = tuple(
        node.outline_id for node in (*child_nodes, *parent_nodes)
    )
    issues = parse_parent_node_review(
        payload,
        review_kind=f"{target_level}_hierarchy",
        allowed_codes=HIERARCHY_CODES,
        chapter_ids=chapter_ids,
        outline_ids=outline_ids,
    )
    if issues:
        raise OutlineSemanticReviewError(
            f"{target_level} hierarchy review found blocking issues",
            issues=issues,
        )


def _volume_layer(
    model: CheckpointedOutlineModel,
    work_id: str,
    story_arc_nodes: tuple[OutlineNode, ...],
    candidate_nodes: list[OutlineNode],
    *,
    boundary_thinking: bool,
) -> tuple[tuple[OutlineNode, ...], tuple[ParentBoundaryDecision, ...]]:
    cards = tuple(OutlineUnitCard.from_node(node) for node in story_arc_nodes)
    decisions: tuple[ParentBoundaryDecision, ...] = ()
    if len(cards) > 1:
        model.ledger.enable_volume_boundaries()
        payload = model.call(
            stage="volume-boundary-detection",
            identity="all-story-arcs",
            revision=template_revision("volume_boundary_detection"),
            prompt=volume_boundary_detection_prompt(cards),
            max_tokens=4_000,
            thinking=boundary_thinking,
        )
        decisions = parse_parent_boundary_decisions(
            payload, tuple(card.outline_id for card in cards)
        )
        review_payload = model.call(
            stage="volume-boundary-review",
            identity="initial",
            revision=template_revision("volume_boundary_review"),
            prompt=volume_boundary_review_prompt(cards, decisions),
            max_tokens=4_000,
        )
        issues = parse_parent_boundary_issues(review_payload, decisions)
        if issues:
            decisions = apply_parent_boundary_issues(decisions, issues)
            recheck_payload = model.call(
                stage="volume-boundary-review",
                identity="repaired",
                revision=template_revision("volume_boundary_review"),
                prompt=volume_boundary_review_prompt(cards, decisions),
                max_tokens=4_000,
            )
            remaining = parse_parent_boundary_issues(recheck_payload, decisions)
            if remaining:
                raise OutlineSemanticReviewError(
                    "volume boundaries still have blocking issues after one correction"
                )

    spans = parent_spans(cards, decisions, target_level="volume")
    model.ledger.configure_parent_count("volume", len(spans))
    nodes = tuple(
        _build_reviewed_parent_node(model, work_id, span, candidate_nodes)
        for span in spans
    )
    _review_hierarchy(model, "volume", story_arc_nodes, nodes)
    return nodes, decisions


def _book_layer(
    model: CheckpointedOutlineModel,
    work_id: str,
    volume_nodes: tuple[OutlineNode, ...],
    candidate_nodes: list[OutlineNode],
) -> tuple[OutlineNode, ...]:
    cards = tuple(OutlineUnitCard.from_node(node) for node in volume_nodes)
    spans = parent_spans(cards, (), target_level="book")
    model.ledger.configure_parent_count("book", 1)
    nodes = (
        _build_reviewed_parent_node(model, work_id, spans[0], candidate_nodes),
    )
    _review_hierarchy(model, "book", volume_nodes, nodes)
    return nodes


def build_hierarchical_outline(
    input_set: AcceptedSynopsisInput,
    *,
    aggregation_ceiling: str,
    settings: ModelSettings,
    checkpoint_dir: Path,
    resume: bool = True,
    completion: JsonCompletion = complete_json,
    boundary_thinking: bool = False,
) -> OutlineExecutionResult:
    """Build a reviewed hierarchy without reading source prose."""

    if aggregation_ceiling not in {"story_arc", "volume", "book"}:
        raise ValueError(f"unsupported aggregation ceiling: {aggregation_ceiling}")
    ledger = OutlineRequestLedger(_source_character_count(input_set))
    boundary_decisions: tuple[BoundaryDecision, ...] = ()
    volume_boundary_decisions: tuple[ParentBoundaryDecision, ...] = ()
    candidate_nodes: list[OutlineNode] = []
    active_stage = "input-scope"
    try:
        if aggregation_ceiling == "book" and not input_set.complete_work:
            raise IncompleteBookScopeError(
                "book aggregation requires a module-01 manifest that confirms complete_work=true"
            )
        model = CheckpointedOutlineModel(
            checkpoint_dir=checkpoint_dir,
            source_fingerprint=_source_fingerprint(input_set),
            settings=settings.for_quality(),
            resume=resume,
            ledger=ledger,
            completion=completion,
        )

        active_stage = "story-arc-layer"
        story_arc_nodes, boundary_decisions = _story_arc_layer(
            model,
            input_set,
            candidate_nodes,
            boundary_thinking=boundary_thinking,
        )
        all_nodes: list[OutlineNode] = list(story_arc_nodes)
        roots = story_arc_nodes

        if aggregation_ceiling in {"volume", "book"}:
            active_stage = "volume-layer"
            volume_nodes, volume_boundary_decisions = _volume_layer(
                model,
                input_set.work_id,
                story_arc_nodes,
                candidate_nodes,
                boundary_thinking=boundary_thinking,
            )
            all_nodes.extend(volume_nodes)
            roots = volume_nodes

        if aggregation_ceiling == "book":
            active_stage = "book-layer"
            book_nodes = _book_layer(
                model, input_set.work_id, roots, candidate_nodes
            )
            all_nodes.extend(book_nodes)
            roots = book_nodes

        bundle = HierarchicalOutlineBundle(
            author_id=input_set.author_id,
            work_id=input_set.work_id,
            profile=input_set.profile,
            source_chapter_ids=input_set.chapter_ids,
            source_synopsis_hashes=input_set.synopsis_hashes,
            nodes=tuple(all_nodes),
            root_outline_ids=tuple(node.outline_id for node in roots),
            aggregation_ceiling=aggregation_ceiling,
        )
        bundle.validate()
        return OutlineExecutionResult(
            chapter_ids=input_set.chapter_ids,
            aggregation_ceiling=aggregation_ceiling,
            bundle=bundle,
            boundary_decisions=boundary_decisions,
            volume_boundary_decisions=volume_boundary_decisions,
            candidate_nodes=tuple(candidate_nodes),
            request_metrics=ledger.to_dict(),
        )
    except Exception as exc:
        issues = tuple(getattr(exc, "issues", ()))
        failed_stage = str(getattr(exc, "failed_stage", active_stage))
        code = str(getattr(exc, "code", type(exc).__name__))
        if isinstance(exc, OutlineRequestBudgetExceeded) and failed_stage.startswith(
            "book-"
        ):
            code = "BOOK_CONTEXT_LIMIT_EXCEEDED"
        return OutlineExecutionResult(
            chapter_ids=input_set.chapter_ids,
            aggregation_ceiling=aggregation_ceiling,
            boundary_decisions=boundary_decisions,
            volume_boundary_decisions=volume_boundary_decisions,
            candidate_nodes=tuple(candidate_nodes),
            review_issues=issues,
            failure={
                "code": code,
                "failed_stage": failed_stage,
                "message": str(exc),
            },
            request_metrics=ledger.to_dict(),
        )


def build_story_arc_outline(
    input_set: AcceptedSynopsisInput,
    *,
    settings: ModelSettings,
    checkpoint_dir: Path,
    resume: bool = True,
    completion: JsonCompletion = complete_json,
    boundary_thinking: bool = False,
) -> OutlineExecutionResult:
    """Backward-compatible story-arc-only entrypoint."""

    return build_hierarchical_outline(
        input_set,
        aggregation_ceiling="story_arc",
        settings=settings,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        completion=completion,
        boundary_thinking=boundary_thinking,
    )
