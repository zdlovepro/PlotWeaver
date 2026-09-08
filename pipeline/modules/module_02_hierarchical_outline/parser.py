"""Strict parsers for module-02 model responses."""

from __future__ import annotations

from typing import Any

from .internal import (
    BoundaryDecision,
    BoundaryIssue,
    NodeReviewIssue,
    ParentBoundaryDecision,
    ParentBoundaryIssue,
)


BOUNDARY_DECISIONS = frozenset({"continue", "split"})
BOUNDARY_PROBLEMS = frozenset({"false_split", "missed_split"})
NODE_FIELDS = frozenset({
    "title",
    "summary",
    "opening_situation",
    "central_goal",
    "central_conflict",
    "causal_chain",
    "turning_points",
    "ending_change",
    "open_threads",
    "character_arcs",
})
CHARACTER_ARC_FIELDS = frozenset({
    "character_name",
    "entry_state",
    "pursuit",
    "key_choices",
    "change",
    "exit_state",
})
FIDELITY_CODES = frozenset({
    "unsupported_claim",
    "subject_error",
    "sequence_error",
    "result_error",
    "causal_upgrade",
    "unsupported_resolution",
    "range_leak",
})
COVERAGE_CODES = frozenset({
    "missing_goal",
    "missing_conflict",
    "missing_turning_point",
    "missing_ending_change",
    "missing_open_thread",
    "incomplete_character_arc",
})
TOP_DOWN_CODES = frozenset({
    "duplicate_major_event",
    "gap_between_arcs",
    "unsupported_resolution",
    "invalid_boundary",
    "misordered_arcs",
})
PARENT_FIDELITY_CODES = frozenset({
    "unsupported_claim",
    "subject_error",
    "sequence_error",
    "result_error",
    "unsupported_cross_child_cause",
    "unsupported_resolution",
    "range_leak",
    "opening_mismatch",
    "ending_mismatch",
})
PARENT_COVERAGE_CODES = frozenset({
    "missing_goal",
    "missing_conflict",
    "missing_turning_point",
    "missing_ending_change",
    "lost_open_thread",
    "duplicated_progress",
    "incomplete_long_character_arc",
})
HIERARCHY_CODES = frozenset({
    "opening_mismatch",
    "ending_mismatch",
    "unsupported_cross_child_cause",
    "unsupported_resolution",
    "lost_open_thread",
    "duplicated_progress",
    "incomplete_long_character_arc",
    "misordered_children",
    "parent_child_coverage_gap",
})


class OutlineModelContractError(ValueError):
    """A model response violated a module-02 task contract."""


def _object(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OutlineModelContractError(f"{name} must be a JSON object")
    return payload


def _exact_keys(payload: dict[str, Any], expected: set[str] | frozenset[str], name: str) -> None:
    actual = set(payload)
    if actual != set(expected):
        raise OutlineModelContractError(
            f"{name} fields mismatch: expected={sorted(expected)}, actual={sorted(actual)}"
        )


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise OutlineModelContractError(f"{name} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise OutlineModelContractError(f"{name} must not be empty")
    return result


def _strings(value: Any, name: str, *, require_one: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise OutlineModelContractError(f"{name} must be a list")
    result = tuple(_text(item, f"{name} item") for item in value)
    if require_one and not result:
        raise OutlineModelContractError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise OutlineModelContractError(f"{name} must not contain duplicates")
    return result


def parse_boundary_decisions(
    payload: dict[str, Any], chapter_ids: tuple[str, ...]
) -> tuple[BoundaryDecision, ...]:
    payload = _object(payload, "boundary response")
    _exact_keys(payload, {"boundaries"}, "boundary response")
    rows = payload["boundaries"]
    if not isinstance(rows, list):
        raise OutlineModelContractError("boundaries must be a list")
    expected_pairs = tuple(zip(chapter_ids, chapter_ids[1:]))
    if len(rows) != len(expected_pairs):
        raise OutlineModelContractError("every adjacent chapter pair requires one boundary decision")
    result: list[BoundaryDecision] = []
    for index, (row, expected_pair) in enumerate(zip(rows, expected_pairs), start=1):
        row = _object(row, f"boundary {index}")
        _exact_keys(
            row,
            {"left_chapter_id", "right_chapter_id", "decision", "reason"},
            f"boundary {index}",
        )
        left = _text(row["left_chapter_id"], f"boundary {index} left_chapter_id")
        right = _text(row["right_chapter_id"], f"boundary {index} right_chapter_id")
        if (left, right) != expected_pair:
            raise OutlineModelContractError("boundary decisions must preserve adjacent chapter order")
        decision = _text(row["decision"], f"boundary {index} decision")
        if decision not in BOUNDARY_DECISIONS:
            raise OutlineModelContractError(f"unsupported boundary decision: {decision}")
        result.append(BoundaryDecision(
            left,
            right,
            decision,
            _text(row["reason"], f"boundary {index} reason"),
        ))
    return tuple(result)


def parse_boundary_issues(
    payload: dict[str, Any], decisions: tuple[BoundaryDecision, ...]
) -> tuple[BoundaryIssue, ...]:
    payload = _object(payload, "boundary review")
    _exact_keys(payload, {"issues"}, "boundary review")
    rows = payload["issues"]
    if not isinstance(rows, list):
        raise OutlineModelContractError("boundary review issues must be a list")
    by_pair = {(item.left_chapter_id, item.right_chapter_id): item for item in decisions}
    seen: set[tuple[str, str]] = set()
    result: list[BoundaryIssue] = []
    for index, row in enumerate(rows, start=1):
        row = _object(row, f"boundary issue {index}")
        _exact_keys(
            row,
            {"left_chapter_id", "right_chapter_id", "problem", "reason"},
            f"boundary issue {index}",
        )
        pair = (
            _text(row["left_chapter_id"], f"boundary issue {index} left_chapter_id"),
            _text(row["right_chapter_id"], f"boundary issue {index} right_chapter_id"),
        )
        if pair not in by_pair or pair in seen:
            raise OutlineModelContractError("boundary review refers to an unknown or duplicate pair")
        problem = _text(row["problem"], f"boundary issue {index} problem")
        if problem not in BOUNDARY_PROBLEMS:
            raise OutlineModelContractError(f"unsupported boundary problem: {problem}")
        current = by_pair[pair].decision
        if (problem == "false_split" and current != "split") or (
            problem == "missed_split" and current != "continue"
        ):
            raise OutlineModelContractError("boundary problem does not match the current decision")
        seen.add(pair)
        result.append(BoundaryIssue(
            pair[0], pair[1], problem,
            _text(row["reason"], f"boundary issue {index} reason"),
        ))
    return tuple(result)


def parse_parent_boundary_decisions(
    payload: dict[str, Any], outline_ids: tuple[str, ...]
) -> tuple[ParentBoundaryDecision, ...]:
    payload = _object(payload, "parent boundary response")
    _exact_keys(payload, {"boundaries"}, "parent boundary response")
    rows = payload["boundaries"]
    if not isinstance(rows, list):
        raise OutlineModelContractError("parent boundaries must be a list")
    expected_pairs = tuple(zip(outline_ids, outline_ids[1:]))
    if len(rows) != len(expected_pairs):
        raise OutlineModelContractError(
            "every adjacent child outline pair requires one boundary decision"
        )
    result: list[ParentBoundaryDecision] = []
    for index, (row, expected_pair) in enumerate(zip(rows, expected_pairs), start=1):
        row = _object(row, f"parent boundary {index}")
        _exact_keys(
            row,
            {"left_outline_id", "right_outline_id", "decision", "reason"},
            f"parent boundary {index}",
        )
        left = _text(row["left_outline_id"], f"parent boundary {index} left_outline_id")
        right = _text(
            row["right_outline_id"], f"parent boundary {index} right_outline_id"
        )
        if (left, right) != expected_pair:
            raise OutlineModelContractError(
                "parent boundary decisions must preserve adjacent child order"
            )
        decision = _text(row["decision"], f"parent boundary {index} decision")
        if decision not in BOUNDARY_DECISIONS:
            raise OutlineModelContractError(
                f"unsupported parent boundary decision: {decision}"
            )
        result.append(ParentBoundaryDecision(
            left,
            right,
            decision,
            _text(row["reason"], f"parent boundary {index} reason"),
        ))
    return tuple(result)


def parse_parent_boundary_issues(
    payload: dict[str, Any], decisions: tuple[ParentBoundaryDecision, ...]
) -> tuple[ParentBoundaryIssue, ...]:
    payload = _object(payload, "parent boundary review")
    _exact_keys(payload, {"issues"}, "parent boundary review")
    rows = payload["issues"]
    if not isinstance(rows, list):
        raise OutlineModelContractError("parent boundary review issues must be a list")
    by_pair = {
        (item.left_outline_id, item.right_outline_id): item for item in decisions
    }
    seen: set[tuple[str, str]] = set()
    result: list[ParentBoundaryIssue] = []
    for index, row in enumerate(rows, start=1):
        row = _object(row, f"parent boundary issue {index}")
        _exact_keys(
            row,
            {"left_outline_id", "right_outline_id", "problem", "reason"},
            f"parent boundary issue {index}",
        )
        pair = (
            _text(
                row["left_outline_id"],
                f"parent boundary issue {index} left_outline_id",
            ),
            _text(
                row["right_outline_id"],
                f"parent boundary issue {index} right_outline_id",
            ),
        )
        if pair not in by_pair or pair in seen:
            raise OutlineModelContractError(
                "parent boundary review refers to an unknown or duplicate pair"
            )
        problem = _text(row["problem"], f"parent boundary issue {index} problem")
        if problem not in BOUNDARY_PROBLEMS:
            raise OutlineModelContractError(
                f"unsupported parent boundary problem: {problem}"
            )
        current = by_pair[pair].decision
        if (problem == "false_split" and current != "split") or (
            problem == "missed_split" and current != "continue"
        ):
            raise OutlineModelContractError(
                "parent boundary problem does not match the current decision"
            )
        seen.add(pair)
        result.append(ParentBoundaryIssue(
            pair[0],
            pair[1],
            problem,
            _text(row["reason"], f"parent boundary issue {index} reason"),
        ))
    return tuple(result)


def parse_node_semantics(payload: dict[str, Any]) -> dict[str, Any]:
    payload = _object(payload, "outline node response")
    _exact_keys(payload, NODE_FIELDS, "outline node response")
    result: dict[str, Any] = {
        name: _text(payload[name], f"outline node {name}")
        for name in (
            "title", "summary", "opening_situation", "central_goal",
            "central_conflict", "ending_change",
        )
    }
    result["causal_chain"] = list(_strings(
        payload["causal_chain"], "outline node causal_chain", require_one=True
    ))
    result["turning_points"] = list(_strings(
        payload["turning_points"], "outline node turning_points"
    ))
    result["open_threads"] = list(_strings(
        payload["open_threads"], "outline node open_threads"
    ))
    character_rows = payload["character_arcs"]
    if not isinstance(character_rows, list):
        raise OutlineModelContractError("outline node character_arcs must be a list")
    characters = []
    seen_names: set[str] = set()
    for index, row in enumerate(character_rows, start=1):
        row = _object(row, f"character arc {index}")
        _exact_keys(row, CHARACTER_ARC_FIELDS, f"character arc {index}")
        character_name = _text(row["character_name"], f"character arc {index} character_name")
        if character_name in seen_names:
            raise OutlineModelContractError("character arcs must not repeat a surface name")
        seen_names.add(character_name)
        characters.append({
            "character_name": character_name,
            "entry_state": _text(row["entry_state"], f"character arc {index} entry_state"),
            "pursuit": _text(row["pursuit"], f"character arc {index} pursuit", allow_empty=True),
            "key_choices": list(_strings(row["key_choices"], f"character arc {index} key_choices")),
            "change": _text(row["change"], f"character arc {index} change"),
            "exit_state": _text(row["exit_state"], f"character arc {index} exit_state"),
        })
    result["character_arcs"] = characters
    return result


def parse_node_review(
    payload: dict[str, Any],
    *,
    review_kind: str,
    allowed_codes: frozenset[str],
    chapter_ids: tuple[str, ...],
    segment_ids: tuple[str, ...],
) -> tuple[NodeReviewIssue, ...]:
    payload = _object(payload, f"{review_kind} review")
    _exact_keys(payload, {"issues"}, f"{review_kind} review")
    rows = payload["issues"]
    if not isinstance(rows, list):
        raise OutlineModelContractError(f"{review_kind} review issues must be a list")
    known_chapters = set(chapter_ids)
    known_segments = set(segment_ids)
    result: list[NodeReviewIssue] = []
    identities: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
    for index, row in enumerate(rows, start=1):
        row = _object(row, f"{review_kind} issue {index}")
        _exact_keys(
            row,
            {"code", "field", "message", "chapter_ids", "segment_ids"},
            f"{review_kind} issue {index}",
        )
        code = _text(row["code"], f"{review_kind} issue {index} code")
        if code not in allowed_codes:
            raise OutlineModelContractError(f"unsupported {review_kind} issue code: {code}")
        cited_chapters = _strings(
            row["chapter_ids"], f"{review_kind} issue {index} chapter_ids", require_one=True
        )
        cited_segments = _strings(
            row["segment_ids"], f"{review_kind} issue {index} segment_ids"
        )
        if not set(cited_chapters).issubset(known_chapters):
            raise OutlineModelContractError(f"{review_kind} issue cites an unknown chapter")
        if not set(cited_segments).issubset(known_segments):
            raise OutlineModelContractError(f"{review_kind} issue cites an unknown segment")
        field = _text(row["field"], f"{review_kind} issue {index} field")
        identity = (code, field, cited_chapters, cited_segments)
        if identity in identities:
            raise OutlineModelContractError(f"{review_kind} review contains duplicate issues")
        identities.add(identity)
        result.append(NodeReviewIssue(
            review_kind=review_kind,
            code=code,
            field=field,
            message=_text(row["message"], f"{review_kind} issue {index} message"),
            chapter_ids=cited_chapters,
            segment_ids=cited_segments,
        ))
    return tuple(result)


def parse_parent_node_review(
    payload: dict[str, Any],
    *,
    review_kind: str,
    allowed_codes: frozenset[str],
    chapter_ids: tuple[str, ...],
    outline_ids: tuple[str, ...],
) -> tuple[NodeReviewIssue, ...]:
    payload = _object(payload, f"{review_kind} review")
    _exact_keys(payload, {"issues"}, f"{review_kind} review")
    rows = payload["issues"]
    if not isinstance(rows, list):
        raise OutlineModelContractError(f"{review_kind} review issues must be a list")
    known_chapters = set(chapter_ids)
    known_outlines = set(outline_ids)
    result: list[NodeReviewIssue] = []
    identities: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
    for index, row in enumerate(rows, start=1):
        row = _object(row, f"{review_kind} issue {index}")
        _exact_keys(
            row,
            {"code", "field", "message", "chapter_ids", "outline_ids"},
            f"{review_kind} issue {index}",
        )
        code = _text(row["code"], f"{review_kind} issue {index} code")
        if code not in allowed_codes:
            raise OutlineModelContractError(
                f"unsupported {review_kind} issue code: {code}"
            )
        cited_chapters = _strings(
            row["chapter_ids"],
            f"{review_kind} issue {index} chapter_ids",
            require_one=True,
        )
        cited_outlines = _strings(
            row["outline_ids"],
            f"{review_kind} issue {index} outline_ids",
            require_one=True,
        )
        if not set(cited_chapters).issubset(known_chapters):
            raise OutlineModelContractError(
                f"{review_kind} issue cites an unknown chapter"
            )
        if not set(cited_outlines).issubset(known_outlines):
            raise OutlineModelContractError(
                f"{review_kind} issue cites an unknown outline"
            )
        field = _text(row["field"], f"{review_kind} issue {index} field")
        identity = (code, field, cited_chapters, cited_outlines)
        if identity in identities:
            raise OutlineModelContractError(
                f"{review_kind} review contains duplicate issues"
            )
        identities.add(identity)
        result.append(NodeReviewIssue(
            review_kind=review_kind,
            code=code,
            field=field,
            message=_text(row["message"], f"{review_kind} issue {index} message"),
            chapter_ids=cited_chapters,
            outline_ids=cited_outlines,
        ))
    return tuple(result)
