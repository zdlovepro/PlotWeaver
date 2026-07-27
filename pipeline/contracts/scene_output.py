"""Strict contracts for paragraph-by-paragraph scene generation and checking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .program import SceneProgram


def _unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


def _as_string_tuple(payload: dict[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{context}.{key} 必须是字符串数组")
    return tuple(value)


def _require_exact_keys(payload: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(f"{context} 字段必须严格匹配；缺少={sorted(expected - actual)}，多出={sorted(actual - expected)}")


def _required_string(payload: dict[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} 必须是非空字符串")
    return value.strip()


def scene_fact_ids(scene: SceneProgram) -> set[str]:
    return {item.fact_id for item in scene.fact_contracts if item.must_realize}


def scene_event_ids(scene: SceneProgram) -> set[str]:
    return {item.event_id for item in scene.event_programs}


def scene_beat_ids(scene: SceneProgram) -> set[str]:
    return {item.beat_id for item in scene.narrative_beats}


@dataclass(frozen=True)
class ParagraphDraft:
    paragraph_id: str
    text: str
    claimed_fact_ids: tuple[str, ...] = field(default_factory=tuple)
    claimed_event_ids: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if not self.paragraph_id.strip():
            raise ValueError("paragraph draft requires paragraph_id")
        if not self.text.strip():
            raise ValueError("paragraph draft text must not be empty")
        _unique(self.claimed_fact_ids, "claimed_fact_ids")
        _unique(self.claimed_event_ids, "claimed_event_ids")

    def to_dict(self) -> dict[str, Any]:
        return {
            "paragraph_id": self.paragraph_id,
            "text": self.text,
            "claimed_fact_ids": list(self.claimed_fact_ids),
            "claimed_event_ids": list(self.claimed_event_ids),
        }


@dataclass(frozen=True)
class SceneDraft:
    scene_id: str
    paragraphs: tuple[ParagraphDraft, ...]

    def validate_for(self, scene: SceneProgram, *, expected_paragraph_ids: Iterable[str] | None = None, minimum_chars: int = 20) -> None:
        if self.scene_id != scene.scene_id:
            raise ValueError("scene draft scene_id does not match scene program")
        expected = tuple(expected_paragraph_ids) if expected_paragraph_ids is not None else tuple(item.paragraph_id for item in scene.paragraphs)
        actual = tuple(item.paragraph_id for item in self.paragraphs)
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError("scene draft must contain each requested paragraph exactly once")
        known_facts = scene_fact_ids(scene)
        known_events = scene_event_ids(scene)
        for paragraph in self.paragraphs:
            paragraph.validate()
            if len(paragraph.text.strip()) < minimum_chars:
                raise ValueError("scene draft paragraph is too short")
            if not set(paragraph.claimed_fact_ids).issubset(known_facts):
                raise ValueError("paragraph draft claims an unknown fact")
            if not set(paragraph.claimed_event_ids).issubset(known_events):
                raise ValueError("paragraph draft claims an unknown event")

    def to_dict(self) -> dict[str, Any]:
        return {"scene_id": self.scene_id, "paragraphs": [item.to_dict() for item in self.paragraphs]}


@dataclass(frozen=True)
class ParagraphIssue:
    paragraph_id: str
    problem: str
    repair: str

    def validate_for(self, scene: SceneProgram) -> None:
        if self.paragraph_id not in {item.paragraph_id for item in scene.paragraphs}:
            raise ValueError("paragraph issue references an unknown paragraph")
        if not self.problem.strip() or not self.repair.strip():
            raise ValueError("paragraph issue requires problem and repair")

    def to_dict(self) -> dict[str, str]:
        return {"paragraph_id": self.paragraph_id, "problem": self.problem, "repair": self.repair}


@dataclass(frozen=True)
class SceneValidation:
    scene_id: str
    reported_passed: bool
    realized_fact_ids: tuple[str, ...]
    missing_fact_ids: tuple[str, ...]
    realized_event_ids: tuple[str, ...]
    missing_event_ids: tuple[str, ...]
    premature_event_ids: tuple[str, ...]
    unsupported_claim_paragraph_ids: tuple[str, ...] = field(default_factory=tuple)
    paragraph_issues: tuple[ParagraphIssue, ...] = field(default_factory=tuple)
    repair_paragraph_ids: tuple[str, ...] = field(default_factory=tuple)

    def validate_for(self, scene: SceneProgram) -> None:
        if self.scene_id != scene.scene_id:
            raise ValueError("scene validation scene_id does not match scene program")
        for name, values in (
            ("realized_fact_ids", self.realized_fact_ids), ("missing_fact_ids", self.missing_fact_ids),
            ("realized_event_ids", self.realized_event_ids), ("missing_event_ids", self.missing_event_ids),
            ("premature_event_ids", self.premature_event_ids), ("unsupported_claim_paragraph_ids", self.unsupported_claim_paragraph_ids),
            ("repair_paragraph_ids", self.repair_paragraph_ids),
        ):
            _unique(values, name)
        expected_facts = scene_fact_ids(scene)
        expected_events = scene_event_ids(scene)
        if not set(self.realized_fact_ids).issubset(expected_facts) or not set(self.missing_fact_ids).issubset(expected_facts):
            raise ValueError("scene validation references an unknown fact")
        if set(self.realized_fact_ids) & set(self.missing_fact_ids):
            raise ValueError("a fact cannot be both realized and missing")
        if set(self.realized_fact_ids) | set(self.missing_fact_ids) != expected_facts:
            raise ValueError("scene validation must account for every required fact")
        if not set(self.realized_event_ids).issubset(expected_events) or not set(self.missing_event_ids).issubset(expected_events):
            raise ValueError("scene validation references an unknown event")
        if set(self.realized_event_ids) & set(self.missing_event_ids):
            raise ValueError("an event cannot be both realized and missing")
        if set(self.realized_event_ids) | set(self.missing_event_ids) != expected_events:
            raise ValueError("scene validation must account for every required event")
        if not set(self.premature_event_ids).issubset(scene.forbidden_event_ids):
            raise ValueError("scene validation reports an event that is not a future-event barrier")
        paragraph_ids = {item.paragraph_id for item in scene.paragraphs}
        if not set(self.unsupported_claim_paragraph_ids).issubset(paragraph_ids):
            raise ValueError("scene validation unsupported-claim target is unknown")
        if not set(self.repair_paragraph_ids).issubset(paragraph_ids):
            raise ValueError("scene validation repair target is unknown")
        if not set(self.unsupported_claim_paragraph_ids).issubset(self.repair_paragraph_ids):
            raise ValueError("scene validation must repair every unsupported-claim paragraph")
        for issue in self.paragraph_issues:
            issue.validate_for(scene)
        if self.reported_passed and (self.missing_fact_ids or self.missing_event_ids or self.premature_event_ids or self.unsupported_claim_paragraph_ids):
            raise ValueError("scene validation cannot pass with missing, premature, or unsupported content")

    @property
    def passed(self) -> bool:
        return not self.missing_fact_ids and not self.missing_event_ids and not self.premature_event_ids and not self.unsupported_claim_paragraph_ids and not self.paragraph_issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id, "reported_passed": self.reported_passed,
            "realized_fact_ids": list(self.realized_fact_ids), "missing_fact_ids": list(self.missing_fact_ids),
            "realized_event_ids": list(self.realized_event_ids), "missing_event_ids": list(self.missing_event_ids),
            "premature_event_ids": list(self.premature_event_ids),
            "unsupported_claim_paragraph_ids": list(self.unsupported_claim_paragraph_ids),
            "paragraph_issues": [item.to_dict() for item in self.paragraph_issues],
            "repair_paragraph_ids": list(self.repair_paragraph_ids), "passed": self.passed,
        }


@dataclass(frozen=True)
class SceneProseValidation:
    """An independent audit of whether a scene is enacted as prose.

    Structural validation proves that the correct events are present.  This
    contract separately proves that the generated paragraphs have staged the
    compiler's dramatic beats instead of merely naming their outcomes.
    """

    scene_id: str
    reported_passed: bool
    dramatized_beat_ids: tuple[str, ...]
    missing_beat_ids: tuple[str, ...]
    summary_paragraph_ids: tuple[str, ...] = field(default_factory=tuple)
    paragraph_issues: tuple[ParagraphIssue, ...] = field(default_factory=tuple)
    repair_paragraph_ids: tuple[str, ...] = field(default_factory=tuple)

    def validate_for(self, scene: SceneProgram) -> None:
        if self.scene_id != scene.scene_id:
            raise ValueError("scene prose validation scene_id does not match scene program")
        expected_beats = scene_beat_ids(scene)
        for name, values in (
            ("dramatized_beat_ids", self.dramatized_beat_ids), ("missing_beat_ids", self.missing_beat_ids),
            ("summary_paragraph_ids", self.summary_paragraph_ids), ("repair_paragraph_ids", self.repair_paragraph_ids),
        ):
            _unique(values, name)
        if not set(self.dramatized_beat_ids).issubset(expected_beats) or not set(self.missing_beat_ids).issubset(expected_beats):
            raise ValueError("scene prose validation references an unknown beat")
        if set(self.dramatized_beat_ids) & set(self.missing_beat_ids):
            raise ValueError("a beat cannot be both dramatized and missing")
        if set(self.dramatized_beat_ids) | set(self.missing_beat_ids) != expected_beats:
            raise ValueError("scene prose validation must account for every dramatic beat")
        paragraph_ids = {item.paragraph_id for item in scene.paragraphs}
        if not set(self.summary_paragraph_ids).issubset(paragraph_ids):
            raise ValueError("scene prose validation summary target is unknown")
        if not set(self.repair_paragraph_ids).issubset(paragraph_ids):
            raise ValueError("scene prose validation repair target is unknown")
        for issue in self.paragraph_issues:
            issue.validate_for(scene)
        if self.reported_passed and (self.missing_beat_ids or self.summary_paragraph_ids or self.paragraph_issues):
            raise ValueError("scene prose validation cannot pass with missing beats or summary paragraphs")

    @property
    def passed(self) -> bool:
        return not self.missing_beat_ids and not self.summary_paragraph_ids and not self.paragraph_issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id, "reported_passed": self.reported_passed,
            "dramatized_beat_ids": list(self.dramatized_beat_ids), "missing_beat_ids": list(self.missing_beat_ids),
            "summary_paragraph_ids": list(self.summary_paragraph_ids),
            "paragraph_issues": [item.to_dict() for item in self.paragraph_issues],
            "repair_paragraph_ids": list(self.repair_paragraph_ids), "passed": self.passed,
        }


def scene_draft_to_llm_dict(draft: SceneDraft) -> dict[str, Any]:
    """Render an internal scene draft with Chinese model-facing JSON keys."""

    return {
        "场景ID": draft.scene_id,
        "段落正文": [
            {
                "段落ID": item.paragraph_id,
                "正文": item.text,
                "自检兑现事实ID": list(item.claimed_fact_ids),
                "自检兑现事件ID": list(item.claimed_event_ids),
            }
            for item in draft.paragraphs
        ],
    }


def scene_validation_to_llm_dict(validation: SceneValidation) -> dict[str, Any]:
    """Render a semantic validation result with Chinese model-facing JSON keys."""

    return {
        "场景ID": validation.scene_id,
        "通过": validation.reported_passed,
        "已实现事实ID": list(validation.realized_fact_ids),
        "缺失事实ID": list(validation.missing_fact_ids),
        "已实现事件ID": list(validation.realized_event_ids),
        "缺失事件ID": list(validation.missing_event_ids),
        "提前泄露事件ID": list(validation.premature_event_ids),
        "未授权断言段落ID": list(validation.unsupported_claim_paragraph_ids),
        "段落问题": [
            {"段落ID": item.paragraph_id, "问题": item.problem, "修复要求": item.repair}
            for item in validation.paragraph_issues
        ],
        "修复段落ID": list(validation.repair_paragraph_ids),
    }


def scene_prose_validation_to_llm_dict(validation: SceneProseValidation) -> dict[str, Any]:
    """Render a prose-quality result using the model's Chinese JSON keys."""

    return {
        "场景ID": validation.scene_id,
        "正文性通过": validation.reported_passed,
        "已戏剧化节拍ID": list(validation.dramatized_beat_ids),
        "缺失节拍ID": list(validation.missing_beat_ids),
        "摘要化段落ID": list(validation.summary_paragraph_ids),
        "段落问题": [
            {"段落ID": item.paragraph_id, "问题": item.problem, "修复要求": item.repair}
            for item in validation.paragraph_issues
        ],
        "修复段落ID": list(validation.repair_paragraph_ids),
    }


def _paragraph_from_llm(payload: dict[str, Any], context: str) -> ParagraphDraft:
    _require_exact_keys(payload, {"段落ID", "正文", "自检兑现事实ID", "自检兑现事件ID"}, context)
    return ParagraphDraft(
        paragraph_id=_required_string(payload, "段落ID", context),
        text=_required_string(payload, "正文", context),
        claimed_fact_ids=_as_string_tuple(payload, "自检兑现事实ID", context),
        claimed_event_ids=_as_string_tuple(payload, "自检兑现事件ID", context),
    )


def scene_draft_from_llm(
    payload: dict[str, Any],
    scene: SceneProgram,
    *,
    minimum_chars: int = 20,
    expected_paragraph_ids: Iterable[str] | None = None,
) -> SceneDraft:
    """Parse a complete scene or one sequential, paragraph-bounded batch."""

    context = "场景正文"
    _require_exact_keys(payload, {"场景ID", "段落正文"}, context)
    if not isinstance(payload.get("段落正文"), list) or any(not isinstance(item, dict) for item in payload["段落正文"]):
        raise ValueError("场景正文.段落正文 必须是对象数组")
    draft = SceneDraft(
        scene_id=_required_string(payload, "场景ID", context),
        paragraphs=tuple(_paragraph_from_llm(item, "段落正文") for item in payload["段落正文"]),
    )
    draft.validate_for(scene, expected_paragraph_ids=expected_paragraph_ids, minimum_chars=minimum_chars)
    return draft


def scene_repair_from_llm(payload: dict[str, Any], scene: SceneProgram, target_ids: Iterable[str], *, minimum_chars: int = 20) -> SceneDraft:
    """Parse a repair that is allowed to replace only preselected paragraphs."""

    context = "段落修复"
    _require_exact_keys(payload, {"场景ID", "修复段落"}, context)
    if not isinstance(payload.get("修复段落"), list) or any(not isinstance(item, dict) for item in payload["修复段落"]):
        raise ValueError("段落修复.修复段落 必须是对象数组")
    draft = SceneDraft(
        scene_id=_required_string(payload, "场景ID", context),
        paragraphs=tuple(_paragraph_from_llm(item, "修复段落") for item in payload["修复段落"]),
    )
    draft.validate_for(scene, expected_paragraph_ids=tuple(target_ids), minimum_chars=minimum_chars)
    return draft


def scene_validation_from_llm(payload: dict[str, Any], scene: SceneProgram) -> SceneValidation:
    """Parse a semantic scene audit and force complete ID accounting."""

    context = "场景校验"
    _require_exact_keys(payload, {"场景ID", "通过", "已实现事实ID", "缺失事实ID", "已实现事件ID", "缺失事件ID", "提前泄露事件ID", "未授权断言段落ID", "段落问题", "修复段落ID"}, context)
    if not isinstance(payload.get("通过"), bool):
        raise ValueError("场景校验.通过 必须是布尔值")
    if not isinstance(payload.get("段落问题"), list) or any(not isinstance(item, dict) for item in payload["段落问题"]):
        raise ValueError("场景校验.段落问题 必须是对象数组")
    issues: list[ParagraphIssue] = []
    for item in payload["段落问题"]:
        _require_exact_keys(item, {"段落ID", "问题", "修复要求"}, "段落问题")
        issues.append(ParagraphIssue(
            _required_string(item, "段落ID", "段落问题"),
            _required_string(item, "问题", "段落问题"),
            _required_string(item, "修复要求", "段落问题"),
        ))
    validation = SceneValidation(
        scene_id=_required_string(payload, "场景ID", context), reported_passed=payload["通过"],
        realized_fact_ids=_as_string_tuple(payload, "已实现事实ID", context),
        missing_fact_ids=_as_string_tuple(payload, "缺失事实ID", context),
        realized_event_ids=_as_string_tuple(payload, "已实现事件ID", context),
        missing_event_ids=_as_string_tuple(payload, "缺失事件ID", context),
        premature_event_ids=_as_string_tuple(payload, "提前泄露事件ID", context),
        unsupported_claim_paragraph_ids=_as_string_tuple(payload, "未授权断言段落ID", context),
        paragraph_issues=tuple(issues), repair_paragraph_ids=_as_string_tuple(payload, "修复段落ID", context),
    )
    validation.validate_for(scene)
    return validation


def scene_prose_validation_from_llm(payload: dict[str, Any], scene: SceneProgram) -> SceneProseValidation:
    """Parse a strict prose-quality audit and force complete beat accounting."""

    context = "场景正文性校验"
    _require_exact_keys(payload, {"场景ID", "正文性通过", "已戏剧化节拍ID", "缺失节拍ID", "摘要化段落ID", "段落问题", "修复段落ID"}, context)
    if not isinstance(payload.get("正文性通过"), bool):
        raise ValueError("场景正文性校验.正文性通过 必须是布尔值")
    if not isinstance(payload.get("段落问题"), list) or any(not isinstance(item, dict) for item in payload["段落问题"]):
        raise ValueError("场景正文性校验.段落问题 必须是对象数组")
    issues: list[ParagraphIssue] = []
    for item in payload["段落问题"]:
        _require_exact_keys(item, {"段落ID", "问题", "修复要求"}, "场景正文性校验.段落问题")
        issues.append(ParagraphIssue(
            _required_string(item, "段落ID", "场景正文性校验.段落问题"),
            _required_string(item, "问题", "场景正文性校验.段落问题"),
            _required_string(item, "修复要求", "场景正文性校验.段落问题"),
        ))
    validation = SceneProseValidation(
        scene_id=_required_string(payload, "场景ID", context), reported_passed=payload["正文性通过"],
        dramatized_beat_ids=_as_string_tuple(payload, "已戏剧化节拍ID", context),
        missing_beat_ids=_as_string_tuple(payload, "缺失节拍ID", context),
        summary_paragraph_ids=_as_string_tuple(payload, "摘要化段落ID", context),
        paragraph_issues=tuple(issues), repair_paragraph_ids=_as_string_tuple(payload, "修复段落ID", context),
    )
    validation.validate_for(scene)
    return validation
