"""Deterministic chapter-program contracts between narrative facts and prose.

These contracts keep stable English identifiers for code.  The model-facing
representation is translated separately by ``program_llm`` into Chinese keys.
They record facts, events and scene-local paragraph jobs, never source prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .narrative import FACT_KINDS


FACT_EXPRESSION_MODES = frozenset({"narration", "action", "dialogue", "internal", "implicit"})
PARAGRAPH_FUNCTIONS = frozenset({
    "orientation", "action_progression", "obstacle", "perception_reaction",
    "dialogue_conflict", "turn", "decision", "transition", "aftermath",
})
BEAT_TYPES = frozenset({
    "setup", "action", "pressure", "reaction", "choice", "consequence", "transition", "aftermath",
})
GENERATION_MODES = frozenset({"faithful", "controlled_expansion"})
EXPANSION_LICENSE_LEVELS = frozenset({"source", "inference", "atmosphere"})


def _required(value: str, field_name: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


@dataclass(frozen=True)
class EntityBinding:
    """The compact entity dictionary required to make fact IDs writable."""

    entity_id: str
    canonical_name: str
    kind: str

    def validate(self) -> None:
        _required(self.entity_id, "entity_id")
        _required(self.canonical_name, "entity canonical_name")
        _required(self.kind, "entity kind")

    def to_dict(self) -> dict[str, Any]:
        return {"entity_id": self.entity_id, "canonical_name": self.canonical_name, "kind": self.kind}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EntityBinding":
        return cls(
            entity_id=str(payload.get("entity_id", "")),
            canonical_name=str(payload.get("canonical_name", "")),
            kind=str(payload.get("kind", "")),
        )


@dataclass(frozen=True)
class FactContract:
    """One fact that must be semantically realised in one scene's prose."""

    fact_id: str
    scene_id: str
    fact_kind: str
    subject_id: str
    predicate: str
    value: str
    object_id: str = ""
    expression_mode: str = "narration"
    must_realize: bool = True
    forbidden_inverse_meanings: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.fact_id, "fact_id")
        _required(self.scene_id, "scene_id")
        if self.fact_kind not in FACT_KINDS:
            raise ValueError(f"unsupported fact kind: {self.fact_kind}")
        _required(self.subject_id, "fact subject_id")
        _required(self.predicate, "fact predicate")
        if not str(self.value).strip() and not str(self.object_id).strip():
            raise ValueError("fact contract requires value or object_id")
        if self.expression_mode not in FACT_EXPRESSION_MODES:
            raise ValueError(f"unsupported fact expression mode: {self.expression_mode}")
        _unique(self.forbidden_inverse_meanings, "forbidden_inverse_meanings")
        for meaning in self.forbidden_inverse_meanings:
            _required(meaning, "forbidden inverse meaning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id, "scene_id": self.scene_id, "fact_kind": self.fact_kind,
            "subject_id": self.subject_id, "predicate": self.predicate, "value": self.value,
            "object_id": self.object_id, "expression_mode": self.expression_mode,
            "must_realize": self.must_realize,
            "forbidden_inverse_meanings": list(self.forbidden_inverse_meanings),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FactContract":
        return cls(
            fact_id=str(payload.get("fact_id", "")), scene_id=str(payload.get("scene_id", "")),
            fact_kind=str(payload.get("fact_kind", "")), subject_id=str(payload.get("subject_id", "")),
            predicate=str(payload.get("predicate", "")), value=str(payload.get("value", "")),
            object_id=str(payload.get("object_id", "")), expression_mode=str(payload.get("expression_mode", "narration")),
            must_realize=bool(payload.get("must_realize", True)),
            forbidden_inverse_meanings=tuple(str(item) for item in payload.get("forbidden_inverse_meanings", []) if str(item).strip()),
        )


@dataclass(frozen=True)
class EventProgram:
    """An event's explicit conditions and result facts inside a scene."""

    event_id: str
    scene_id: str
    summary: str
    action: str
    participant_ids: tuple[str, ...]
    precondition_fact_ids: tuple[str, ...]
    required_fact_ids: tuple[str, ...]
    outcome_fact_ids: tuple[str, ...]
    obstacle: str = ""
    decision: str = ""
    action_type: str = "other"
    actor_id: str = ""
    target_ids: tuple[str, ...] = field(default_factory=tuple)
    basis_fact_ids: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.event_id, "event_id")
        _required(self.scene_id, "scene_id")
        _required(self.summary, "event summary")
        _required(self.action, "event action")
        if not self.participant_ids:
            raise ValueError("event program requires participants")
        _unique(self.participant_ids, "event participant_ids")
        if self.action_type not in {"speech", "movement", "transfer", "perception", "decision", "confrontation", "state_change", "other"}:
            raise ValueError(f"unsupported event program action_type: {self.action_type}")
        if self.actor_id and self.actor_id not in self.participant_ids:
            raise ValueError("event program actor must be a participant")
        _unique(self.target_ids, "event target_ids")
        if any(item not in self.participant_ids for item in self.target_ids):
            raise ValueError("event program targets must be participants")
        _unique(self.basis_fact_ids, "event basis_fact_ids")
        if not self.outcome_fact_ids:
            raise ValueError("event program requires outcome facts")
        for name, values in (
            ("event precondition_fact_ids", self.precondition_fact_ids),
            ("event required_fact_ids", self.required_fact_ids),
            ("event outcome_fact_ids", self.outcome_fact_ids),
        ):
            _unique(values, name)
            for value in values:
                _required(value, name)
        if not set(self.outcome_fact_ids).issubset(self.required_fact_ids):
            raise ValueError("event outcomes must be included in required facts")
        if not set(self.basis_fact_ids).issubset(set(self.precondition_fact_ids) | set(self.required_fact_ids)):
            raise ValueError("event basis facts must be preconditions or required facts")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id, "scene_id": self.scene_id, "summary": self.summary, "action": self.action,
            "participant_ids": list(self.participant_ids), "obstacle": self.obstacle, "decision": self.decision,
            "precondition_fact_ids": list(self.precondition_fact_ids),
            "required_fact_ids": list(self.required_fact_ids), "outcome_fact_ids": list(self.outcome_fact_ids),
            "action_type": self.action_type, "actor_id": self.actor_id,
            "target_ids": list(self.target_ids), "basis_fact_ids": list(self.basis_fact_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EventProgram":
        participant_ids = tuple(str(item) for item in payload.get("participant_ids", []) if str(item).strip())
        required_fact_ids = tuple(str(item) for item in payload.get("required_fact_ids", []) if str(item).strip())
        has_frame = all(field in payload for field in ("action_type", "actor_id", "target_ids", "basis_fact_ids"))
        actor_id = str(payload.get("actor_id", "")).strip()
        basis_fact_ids = tuple(str(item) for item in payload.get("basis_fact_ids", []) if str(item).strip())
        if not has_frame:
            actor_id = actor_id or (participant_ids[0] if participant_ids else "")
            basis_fact_ids = basis_fact_ids or required_fact_ids
        return cls(
            event_id=str(payload.get("event_id", "")), scene_id=str(payload.get("scene_id", "")),
            summary=str(payload.get("summary", "")), action=str(payload.get("action", "")),
            participant_ids=participant_ids,
            precondition_fact_ids=tuple(str(item) for item in payload.get("precondition_fact_ids", []) if str(item).strip()),
            required_fact_ids=required_fact_ids,
            outcome_fact_ids=tuple(str(item) for item in payload.get("outcome_fact_ids", []) if str(item).strip()),
            obstacle=str(payload.get("obstacle", "")), decision=str(payload.get("decision", "")),
            action_type=str(payload.get("action_type", "other")), actor_id=actor_id,
            target_ids=tuple(str(item) for item in payload.get("target_ids", []) if str(item).strip()),
            basis_fact_ids=basis_fact_ids,
        )


@dataclass(frozen=True)
class NarrativeBeat:
    """One observable dramatic movement that prose must stage, not summarize.

    The fields are deliberately abstract: they may describe the required
    action and change, but never retain a sentence from the source chapter.
    A beat is small enough to be realised by one paragraph or a tightly
    coupled pair of paragraphs.
    """

    beat_id: str
    scene_id: str
    order: int
    beat_type: str
    actor_id: str
    objective: str
    pressure: str
    observable_action: str
    focal_detail: str
    state_change: str
    required_fact_ids: tuple[str, ...] = field(default_factory=tuple)
    event_ids: tuple[str, ...] = field(default_factory=tuple)
    dialogue_pressure: str = ""
    expansion_license: str = "source"
    support_fact_ids: tuple[str, ...] = field(default_factory=tuple)
    mechanism_ids: tuple[str, ...] = field(default_factory=tuple)
    narrative_function: str = ""
    counterfactual_guard: str = ""

    def validate(self) -> None:
        _required(self.beat_id, "beat_id")
        _required(self.scene_id, "beat scene_id")
        if self.order < 0:
            raise ValueError("beat order must be non-negative")
        if self.beat_type not in BEAT_TYPES:
            raise ValueError(f"unsupported beat type: {self.beat_type}")
        _required(self.actor_id, "beat actor_id")
        _required(self.objective, "beat objective")
        _required(self.observable_action, "beat observable_action")
        _required(self.focal_detail, "beat focal_detail")
        _required(self.state_change, "beat state_change")
        _unique(self.required_fact_ids, "beat required_fact_ids")
        _unique(self.event_ids, "beat event_ids")
        if self.expansion_license not in EXPANSION_LICENSE_LEVELS:
            raise ValueError(f"unsupported expansion license: {self.expansion_license}")
        _unique(self.support_fact_ids, "beat support_fact_ids")
        _unique(self.mechanism_ids, "beat mechanism_ids")
        if self.expansion_license != "source" and not self.support_fact_ids:
            raise ValueError("expanded beat requires at least one support fact")

    def to_dict(self) -> dict[str, Any]:
        return {
            "beat_id": self.beat_id, "scene_id": self.scene_id, "order": self.order,
            "beat_type": self.beat_type, "actor_id": self.actor_id, "objective": self.objective,
            "pressure": self.pressure, "observable_action": self.observable_action,
            "focal_detail": self.focal_detail, "state_change": self.state_change,
            "required_fact_ids": list(self.required_fact_ids), "event_ids": list(self.event_ids),
            "dialogue_pressure": self.dialogue_pressure,
            "expansion_license": self.expansion_license,
            "support_fact_ids": list(self.support_fact_ids),
            "mechanism_ids": list(self.mechanism_ids),
            "narrative_function": self.narrative_function,
            "counterfactual_guard": self.counterfactual_guard,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NarrativeBeat":
        return cls(
            beat_id=str(payload.get("beat_id", "")), scene_id=str(payload.get("scene_id", "")),
            order=int(payload.get("order", -1)), beat_type=str(payload.get("beat_type", "")),
            actor_id=str(payload.get("actor_id", "")), objective=str(payload.get("objective", "")),
            pressure=str(payload.get("pressure", "")), observable_action=str(payload.get("observable_action", "")),
            focal_detail=str(payload.get("focal_detail", "")), state_change=str(payload.get("state_change", "")),
            required_fact_ids=tuple(str(item) for item in payload.get("required_fact_ids", []) if str(item).strip()),
            event_ids=tuple(str(item) for item in payload.get("event_ids", []) if str(item).strip()),
            dialogue_pressure=str(payload.get("dialogue_pressure", "")),
            expansion_license=str(payload.get("expansion_license", "source")),
            support_fact_ids=tuple(str(item) for item in payload.get("support_fact_ids", []) if str(item).strip()),
            mechanism_ids=tuple(str(item) for item in payload.get("mechanism_ids", []) if str(item).strip()),
            narrative_function=str(payload.get("narrative_function", "")),
            counterfactual_guard=str(payload.get("counterfactual_guard", "")),
        )


@dataclass(frozen=True)
class ParagraphProgram:
    """A scene-local paragraph job, expressed without source wording."""

    paragraph_id: str
    scene_id: str
    order: int
    function: str
    required_fact_ids: tuple[str, ...] = field(default_factory=tuple)
    event_ids: tuple[str, ...] = field(default_factory=tuple)
    viewpoint_entity_id: str = ""
    dialogue_act: str = ""
    beat_ids: tuple[str, ...] = field(default_factory=tuple)
    target_chars: int = 0
    minimum_chars: int = 0

    def validate(self) -> None:
        _required(self.paragraph_id, "paragraph_id")
        _required(self.scene_id, "scene_id")
        if self.order < 0:
            raise ValueError("paragraph order must be non-negative")
        if self.function not in PARAGRAPH_FUNCTIONS:
            raise ValueError(f"unsupported paragraph function: {self.function}")
        _unique(self.required_fact_ids, "paragraph required_fact_ids")
        _unique(self.event_ids, "paragraph event_ids")
        _unique(self.beat_ids, "paragraph beat_ids")
        if self.target_chars < 0 or self.minimum_chars < 0:
            raise ValueError("paragraph character budgets must be non-negative")
        if self.target_chars and self.minimum_chars > self.target_chars:
            raise ValueError("paragraph minimum_chars cannot exceed target_chars")
        if not self.required_fact_ids and not self.event_ids and not self.beat_ids:
            raise ValueError("paragraph program must realise a fact, event or dramatic beat")

    def to_dict(self) -> dict[str, Any]:
        return {
            "paragraph_id": self.paragraph_id, "scene_id": self.scene_id, "order": self.order,
            "function": self.function, "required_fact_ids": list(self.required_fact_ids),
            "event_ids": list(self.event_ids), "viewpoint_entity_id": self.viewpoint_entity_id,
            "dialogue_act": self.dialogue_act, "beat_ids": list(self.beat_ids),
            "target_chars": self.target_chars, "minimum_chars": self.minimum_chars,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParagraphProgram":
        return cls(
            paragraph_id=str(payload.get("paragraph_id", "")), scene_id=str(payload.get("scene_id", "")),
            order=int(payload.get("order", -1)), function=str(payload.get("function", "")),
            required_fact_ids=tuple(str(item) for item in payload.get("required_fact_ids", []) if str(item).strip()),
            event_ids=tuple(str(item) for item in payload.get("event_ids", []) if str(item).strip()),
            viewpoint_entity_id=str(payload.get("viewpoint_entity_id", "")), dialogue_act=str(payload.get("dialogue_act", "")),
            beat_ids=tuple(str(item) for item in payload.get("beat_ids", []) if str(item).strip()),
            target_chars=int(payload.get("target_chars", 0) or 0),
            minimum_chars=int(payload.get("minimum_chars", 0) or 0),
        )


@dataclass(frozen=True)
class SceneProgram:
    """A state transition compiled into event and paragraph obligations."""

    scene_id: str
    chapter_id: str
    order: int
    objective: str
    participant_ids: tuple[str, ...]
    entry_state_fact_ids: tuple[str, ...]
    event_programs: tuple[EventProgram, ...]
    fact_contracts: tuple[FactContract, ...]
    paragraphs: tuple[ParagraphProgram, ...]
    exit_state_fact_ids: tuple[str, ...]
    forbidden_event_ids: tuple[str, ...] = field(default_factory=tuple)
    narrative_beats: tuple[NarrativeBeat, ...] = field(default_factory=tuple)
    # Facts established in an earlier scene and needed only to understand the
    # current action.  They are visible to the writer but must never be
    # realised again as if they were new narrative events.
    context_fact_contracts: tuple[FactContract, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        _required(self.scene_id, "scene_id")
        _required(self.chapter_id, "chapter_id")
        _required(self.objective, "scene objective")
        if not self.participant_ids:
            raise ValueError("scene program requires participants")
        _unique(self.participant_ids, "scene participant_ids")
        if self.order < 0:
            raise ValueError("scene program order must be non-negative")
        if not self.event_programs or not self.fact_contracts or not self.paragraphs:
            raise ValueError("scene program requires events, facts and paragraphs")
        event_ids = tuple(item.event_id for item in self.event_programs)
        fact_ids = tuple(item.fact_id for item in self.fact_contracts)
        context_fact_ids = tuple(item.fact_id for item in self.context_fact_contracts)
        paragraph_ids = tuple(item.paragraph_id for item in self.paragraphs)
        beat_ids = tuple(item.beat_id for item in self.narrative_beats)
        _unique(event_ids, "scene event ids")
        _unique(fact_ids, "scene fact ids")
        _unique(context_fact_ids, "scene context fact ids")
        _unique(paragraph_ids, "scene paragraph ids")
        _unique(beat_ids, "scene beat ids")
        _unique(self.entry_state_fact_ids, "scene entry_state_fact_ids")
        _unique(self.exit_state_fact_ids, "scene exit_state_fact_ids")
        _unique(self.forbidden_event_ids, "scene forbidden_event_ids")
        if set(event_ids) & set(self.forbidden_event_ids):
            raise ValueError("scene cannot require and forbid the same event")
        local_facts, context_facts, available_facts, local_events = set(fact_ids), set(context_fact_ids), set(fact_ids) | set(context_fact_ids), set(event_ids)
        if local_facts & context_facts:
            raise ValueError("scene context facts must not duplicate local facts")
        local_beats = set(beat_ids)
        for event in self.event_programs:
            event.validate()
            if event.scene_id != self.scene_id:
                raise ValueError("event program scene_id does not match its scene")
            if not set(event.required_fact_ids).issubset(local_facts):
                raise ValueError("event program refers to a non-local fact")
            if not set(event.precondition_fact_ids).issubset(available_facts):
                raise ValueError("event program refers to an unavailable precondition")
        for fact in self.fact_contracts:
            fact.validate()
            if fact.scene_id != self.scene_id:
                raise ValueError("fact contract scene_id does not match its scene")
        for fact in self.context_fact_contracts:
            fact.validate()
            if fact.scene_id != self.scene_id or fact.must_realize:
                raise ValueError("context fact must belong to the scene and be read-only")
        if self.narrative_beats:
            if tuple(item.order for item in self.narrative_beats) != tuple(range(len(self.narrative_beats))):
                raise ValueError("narrative beat orders must be consecutive and start at zero")
            for beat in self.narrative_beats:
                beat.validate()
                if beat.scene_id != self.scene_id:
                    raise ValueError("narrative beat scene_id does not match its scene")
                if not set(beat.required_fact_ids).issubset(local_facts):
                    raise ValueError("narrative beat refers to a non-local fact")
                if not set(beat.event_ids).issubset(local_events):
                    raise ValueError("narrative beat refers to an unknown event")
                if not set(beat.support_fact_ids).issubset(local_facts):
                    raise ValueError("narrative beat support refers to a non-local fact")
        if tuple(item.order for item in self.paragraphs) != tuple(range(len(self.paragraphs))):
            raise ValueError("paragraph program orders must be consecutive and start at zero")
        covered_facts: set[str] = set()
        covered_events: set[str] = set()
        covered_beats: set[str] = set()
        for paragraph in self.paragraphs:
            paragraph.validate()
            if paragraph.scene_id != self.scene_id:
                raise ValueError("paragraph program scene_id does not match its scene")
            if not set(paragraph.required_fact_ids).issubset(local_facts):
                raise ValueError("paragraph program refers to a non-local fact")
            if not set(paragraph.event_ids).issubset(local_events):
                raise ValueError("paragraph program refers to an unknown event")
            if not set(paragraph.beat_ids).issubset(local_beats):
                raise ValueError("paragraph program refers to an unknown dramatic beat")
            covered_facts.update(paragraph.required_fact_ids)
            covered_events.update(paragraph.event_ids)
            covered_beats.update(paragraph.beat_ids)
        if not {item.fact_id for item in self.fact_contracts if item.must_realize}.issubset(covered_facts):
            raise ValueError("every required fact contract must be bound to a paragraph")
        if not local_events.issubset(covered_events):
            raise ValueError("every event program must be bound to a paragraph")
        if self.narrative_beats and not local_beats.issubset(covered_beats):
            raise ValueError("every narrative beat must be bound to a paragraph")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id, "chapter_id": self.chapter_id, "order": self.order,
            "objective": self.objective, "participant_ids": list(self.participant_ids),
            "entry_state_fact_ids": list(self.entry_state_fact_ids),
            "event_programs": [item.to_dict() for item in self.event_programs],
            "fact_contracts": [item.to_dict() for item in self.fact_contracts],
            "context_fact_contracts": [item.to_dict() for item in self.context_fact_contracts],
            "paragraphs": [item.to_dict() for item in self.paragraphs],
            "exit_state_fact_ids": list(self.exit_state_fact_ids),
            "forbidden_event_ids": list(self.forbidden_event_ids),
            "narrative_beats": [item.to_dict() for item in self.narrative_beats],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SceneProgram":
        return cls(
            scene_id=str(payload.get("scene_id", "")), chapter_id=str(payload.get("chapter_id", "")), order=int(payload.get("order", -1)),
            objective=str(payload.get("objective", "")),
            participant_ids=tuple(str(item) for item in payload.get("participant_ids", []) if str(item).strip()),
            entry_state_fact_ids=tuple(str(item) for item in payload.get("entry_state_fact_ids", []) if str(item).strip()),
            event_programs=tuple(EventProgram.from_dict(item) for item in payload.get("event_programs", []) if isinstance(item, dict)),
            fact_contracts=tuple(FactContract.from_dict(item) for item in payload.get("fact_contracts", []) if isinstance(item, dict)),
            context_fact_contracts=tuple(FactContract.from_dict(item) for item in payload.get("context_fact_contracts", []) if isinstance(item, dict)),
            paragraphs=tuple(ParagraphProgram.from_dict(item) for item in payload.get("paragraphs", []) if isinstance(item, dict)),
            exit_state_fact_ids=tuple(str(item) for item in payload.get("exit_state_fact_ids", []) if str(item).strip()),
            forbidden_event_ids=tuple(str(item) for item in payload.get("forbidden_event_ids", []) if str(item).strip()),
            narrative_beats=tuple(NarrativeBeat.from_dict(item) for item in payload.get("narrative_beats", []) if isinstance(item, dict)),
        )


@dataclass(frozen=True)
class ChapterProgram:
    """The complete fact-to-paragraph program for one annotated chapter."""

    chapter_id: str
    source_hash: str
    scene_programs: tuple[SceneProgram, ...]
    schema_version: str = "1.0"
    entities: tuple[EntityBinding, ...] = field(default_factory=tuple)
    generation_mode: str = "faithful"
    target_char_min: int = 0
    target_char_max: int = 0

    def validate(self) -> None:
        _required(self.chapter_id, "chapter_id")
        _required(self.source_hash, "source_hash")
        _required(self.schema_version, "chapter program schema_version")
        if self.generation_mode not in GENERATION_MODES:
            raise ValueError(f"unsupported generation mode: {self.generation_mode}")
        if self.target_char_min < 0 or self.target_char_max < 0:
            raise ValueError("chapter character budget must be non-negative")
        if bool(self.target_char_min) != bool(self.target_char_max):
            raise ValueError("chapter character budget requires both minimum and maximum")
        if self.target_char_min and self.target_char_min > self.target_char_max:
            raise ValueError("chapter target_char_min cannot exceed target_char_max")
        if self.generation_mode == "controlled_expansion" and not self.target_char_min:
            raise ValueError("controlled expansion requires a chapter character budget")
        if not self.scene_programs:
            raise ValueError("chapter program requires scene programs")
        entity_ids = tuple(item.entity_id for item in self.entities)
        _unique(entity_ids, "chapter entity ids")
        for entity in self.entities:
            entity.validate()
        scene_ids = tuple(item.scene_id for item in self.scene_programs)
        _unique(scene_ids, "chapter scene ids")
        if tuple(item.order for item in self.scene_programs) != tuple(range(len(self.scene_programs))):
            raise ValueError("scene program orders must be consecutive and start at zero")
        all_fact_ids: set[str] = set()
        all_event_ids: set[str] = set()
        for scene in self.scene_programs:
            scene.validate()
            if scene.chapter_id != self.chapter_id:
                raise ValueError("scene program chapter_id does not match chapter")
            fact_ids = {item.fact_id for item in scene.fact_contracts}
            event_ids = {item.event_id for item in scene.event_programs}
            if all_fact_ids & fact_ids:
                raise ValueError("a fact contract may belong to only one scene")
            if all_event_ids & event_ids:
                raise ValueError("an event program may belong to only one scene")
            all_fact_ids.update(fact_ids)
            all_event_ids.update(event_ids)
        for scene in self.scene_programs:
            state_ids = set(scene.entry_state_fact_ids) | set(scene.exit_state_fact_ids)
            if not state_ids.issubset(all_fact_ids):
                raise ValueError("scene state references a fact without a chapter contract")
            if not set(scene.forbidden_event_ids).issubset(all_event_ids):
                raise ValueError("scene forbids an event without a chapter event program")
            for event in scene.event_programs:
                if not set(event.precondition_fact_ids).issubset(all_fact_ids):
                    raise ValueError("event preconditions reference a fact without a chapter contract")
            if not {fact.fact_id for fact in scene.context_fact_contracts}.issubset(all_fact_ids):
                raise ValueError("scene context references a fact without a chapter contract")
        if self.entities:
            known_entities = set(entity_ids)
            for scene in self.scene_programs:
                if not set(scene.participant_ids).issubset(known_entities):
                    raise ValueError("scene participants reference an unknown entity")
                for fact in scene.fact_contracts:
                    references = {fact.subject_id}
                    if fact.object_id:
                        references.add(fact.object_id)
                    if not references.issubset(known_entities):
                        raise ValueError("fact contract references an unknown entity")
                for event in scene.event_programs:
                    if not set(event.participant_ids).issubset(known_entities):
                        raise ValueError("event program references an unknown entity")
                for paragraph in scene.paragraphs:
                    if paragraph.viewpoint_entity_id and paragraph.viewpoint_entity_id not in known_entities:
                        raise ValueError("paragraph viewpoint references an unknown entity")
                for beat in scene.narrative_beats:
                    if beat.actor_id not in known_entities:
                        raise ValueError("narrative beat actor references an unknown entity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id, "source_hash": self.source_hash, "schema_version": self.schema_version,
            "entities": [item.to_dict() for item in self.entities],
            "scene_programs": [item.to_dict() for item in self.scene_programs],
            "generation_mode": self.generation_mode,
            "target_char_min": self.target_char_min,
            "target_char_max": self.target_char_max,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterProgram":
        return cls(
            chapter_id=str(payload.get("chapter_id", "")), source_hash=str(payload.get("source_hash", "")),
            schema_version=str(payload.get("schema_version", "1.0")),
            scene_programs=tuple(SceneProgram.from_dict(item) for item in payload.get("scene_programs", []) if isinstance(item, dict)),
            entities=tuple(EntityBinding.from_dict(item) for item in payload.get("entities", []) if isinstance(item, dict)),
            generation_mode=str(payload.get("generation_mode", "faithful")),
            target_char_min=int(payload.get("target_char_min", 0) or 0),
            target_char_max=int(payload.get("target_char_max", 0) or 0),
        )
