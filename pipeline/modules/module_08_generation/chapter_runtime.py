"""Provisional, paragraph-level state for a single generated chapter.

The immutable ``narrative_graph.json`` plus committed cross-chapter patches is
the factual base.  This module adds a deliberately short-lived working ledger:
one paragraph may update it only after a dedicated semantic audit passes.  The
next paragraph can therefore inherit verified local consequences without ever
receiving an entire scene draft or source chapter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import ChapterContract, ChapterProgram, ParagraphDraft, ParagraphProgram, ParagraphValidation, SceneProgram
from .jsonio import read_json, write_json


WORKING_LEDGER_SCHEMA = "1.0"


def _fingerprint(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item) for item in values if str(item).strip()))


def paragraph_obligations(scene: SceneProgram, paragraph: ParagraphProgram) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return only the facts/events this paragraph itself must account for.

    Beat support facts are evidence for an expansion beat, not necessarily new
    claims that must be restated in prose.  Required beat facts and explicitly
    assigned event IDs, by contrast, are executable obligations.
    """

    beats = {item.beat_id: item for item in scene.narrative_beats}
    fact_ids: list[str] = list(paragraph.required_fact_ids)
    event_ids: list[str] = list(paragraph.event_ids)
    for beat_id in paragraph.beat_ids:
        beat = beats.get(beat_id)
        if beat is not None:
            fact_ids.extend(beat.required_fact_ids)
            event_ids.extend(beat.event_ids)
    return _unique(fact_ids), _unique(event_ids)


def _ordered_paragraphs(program: ChapterProgram) -> tuple[ParagraphProgram, ...]:
    return tuple(
        paragraph
        for scene in sorted(program.scene_programs, key=lambda item: item.order)
        for paragraph in sorted(scene.paragraphs, key=lambda item: item.order)
    )


def _fact_map(program: ChapterProgram) -> dict[str, Any]:
    return {
        fact.fact_id: fact
        for scene in program.scene_programs
        for fact in scene.fact_contracts
    }


def _state_key(fact: Any) -> str:
    """A structural state slot; it deliberately does not inspect prose."""

    if fact.fact_kind == "location":
        return f"{fact.subject_id}|location:current"
    if fact.fact_kind in {"identity", "goal", "relationship", "progression", "resource", "knowledge", "information"}:
        return f"{fact.subject_id}|{fact.fact_kind}:{fact.predicate}"
    return f"{fact.subject_id}|{fact.fact_kind}:{fact.predicate}:{fact.object_id or fact.value}"


def _state_payload(fact: Any, *, paragraph_id: str) -> dict[str, Any]:
    return {
        "fact_id": fact.fact_id,
        "subject_id": fact.subject_id,
        "object_id": fact.object_id,
        "fact_kind": fact.fact_kind,
        "predicate": fact.predicate,
        "value": fact.value,
        "paragraph_id": paragraph_id,
    }


def _paragraph_entities(scene: SceneProgram, paragraph: ParagraphProgram) -> set[str]:
    facts = {item.fact_id: item for item in scene.fact_contracts}
    events = {item.event_id: item for item in scene.event_programs}
    fact_ids, event_ids = paragraph_obligations(scene, paragraph)
    entity_ids = set(scene.participant_ids)
    if paragraph.viewpoint_entity_id:
        entity_ids.add(paragraph.viewpoint_entity_id)
    for fact_id in fact_ids:
        fact = facts.get(fact_id)
        if fact is not None:
            entity_ids.add(fact.subject_id)
            if fact.object_id:
                entity_ids.add(fact.object_id)
    for event_id in event_ids:
        event = events.get(event_id)
        if event is not None:
            entity_ids.update(event.participant_ids)
    return entity_ids


def initialize_chapter_working_ledger(
    program: ChapterProgram,
    *,
    contract: ChapterContract | None = None,
) -> dict[str, Any]:
    """Create an empty, non-committed ledger for a chapter generation run."""

    return {
        "schema_version": WORKING_LEDGER_SCHEMA,
        "chapter_id": program.chapter_id,
        "source_hash": program.source_hash,
        "program_fingerprint": _fingerprint(program.to_dict()),
        "chapter_contract_fingerprint": _fingerprint(contract.to_dict()) if contract is not None else "",
        "committed_paragraph_ids": [],
        "realized_fact_ids": [],
        "realized_event_ids": [],
        "provisional_state": {},
        "handoffs": [],
        "completed": False,
    }


def load_or_initialize_chapter_working_ledger(
    path: Path,
    program: ChapterProgram,
    *,
    contract: ChapterContract | None = None,
) -> dict[str, Any]:
    """Load a resumable ledger only when it matches this exact program."""

    if not path.exists():
        return initialize_chapter_working_ledger(program, contract=contract)
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("chapter working ledger must be a JSON object")
    expected = initialize_chapter_working_ledger(program, contract=contract)
    for key in ("chapter_id", "source_hash", "program_fingerprint", "chapter_contract_fingerprint"):
        if str(payload.get(key, "")) != str(expected[key]):
            raise ValueError("chapter working ledger belongs to a different chapter program or contract")
    if not isinstance(payload.get("committed_paragraph_ids"), list) or not isinstance(payload.get("provisional_state"), dict) or not isinstance(payload.get("handoffs"), list):
        raise ValueError("chapter working ledger is incomplete")
    return {**expected, **payload, "schema_version": WORKING_LEDGER_SCHEMA}


def write_chapter_working_ledger(path: Path, ledger: dict[str, Any]) -> Path:
    write_json(path, ledger)
    return path


def _expected_next_paragraph(program: ChapterProgram, ledger: dict[str, Any]) -> str:
    committed = [str(item) for item in ledger.get("committed_paragraph_ids", [])]
    ordered = _ordered_paragraphs(program)
    if len(committed) >= len(ordered):
        return ""
    expected_prefix = [item.paragraph_id for item in ordered[:len(committed)]]
    if committed != expected_prefix:
        raise ValueError("chapter working ledger paragraph order is corrupted")
    return ordered[len(committed)].paragraph_id


def working_ledger_context(
    ledger: dict[str, Any],
    scene: SceneProgram,
    paragraph: ParagraphProgram,
) -> dict[str, Any]:
    """Return the bounded local hand-off allowed into the next writer packet."""

    entity_ids = _paragraph_entities(scene, paragraph)
    state = [
        {"状态键": key, **value}
        for key, value in ledger.get("provisional_state", {}).items()
        if isinstance(value, dict) and str(value.get("subject_id", "")) in entity_ids
    ]
    handoffs = [item for item in ledger.get("handoffs", []) if isinstance(item, dict)]
    latest = handoffs[-1] if handoffs else {}
    return {
        "已验证段落数": len(ledger.get("committed_paragraph_ids", [])),
        "本章暂存状态": state,
        "已兑现事实ID": list(ledger.get("realized_fact_ids", [])),
        "已兑现事件ID": list(ledger.get("realized_event_ids", [])),
        "上一段交接": {
            "段落ID": str(latest.get("paragraph_id", "")),
            "已兑现事实ID": list(latest.get("realized_fact_ids", [])),
            "已兑现事件ID": list(latest.get("realized_event_ids", [])),
            "结尾": str(latest.get("tail", "")),
        } if latest else {"段落ID": "", "已兑现事实ID": [], "已兑现事件ID": [], "结尾": ""},
        "约束": [
            "本章暂存状态只来自已通过段落校验的正文。",
            "只承接上一段的收束，不得复述或改写已经验证的段落。",
        ],
    }


def commit_paragraph_patch(
    ledger: dict[str, Any],
    program: ChapterProgram,
    scene: SceneProgram,
    paragraph: ParagraphProgram,
    draft: ParagraphDraft,
    validation: ParagraphValidation,
) -> dict[str, Any]:
    """Commit one audited paragraph or refuse to advance the chapter."""

    expected_paragraph_id = _expected_next_paragraph(program, ledger)
    if paragraph.paragraph_id != expected_paragraph_id:
        raise ValueError("paragraph patch is not the next executable paragraph")
    if draft.paragraph_id != paragraph.paragraph_id or validation.paragraph_id != paragraph.paragraph_id:
        raise ValueError("paragraph patch IDs do not match the paragraph program")
    fact_ids, event_ids = paragraph_obligations(scene, paragraph)
    validation.validate_for(scene, expected_fact_ids=fact_ids, expected_event_ids=event_ids)
    if not validation.passed:
        raise ValueError("failed paragraph validation cannot update the chapter working ledger")
    facts = _fact_map(program)
    state = {key: value for key, value in ledger.get("provisional_state", {}).items() if isinstance(value, dict)}
    for fact_id in validation.realized_fact_ids:
        fact = facts.get(fact_id)
        if fact is not None:
            state[_state_key(fact)] = _state_payload(fact, paragraph_id=paragraph.paragraph_id)
    tail = draft.text.strip()[-220:]
    return {
        **ledger,
        "committed_paragraph_ids": [*ledger.get("committed_paragraph_ids", []), paragraph.paragraph_id],
        "realized_fact_ids": list(_unique((*ledger.get("realized_fact_ids", []), *validation.realized_fact_ids))),
        "realized_event_ids": list(_unique((*ledger.get("realized_event_ids", []), *validation.realized_event_ids))),
        "provisional_state": state,
        "handoffs": [
            *ledger.get("handoffs", []),
            {
                "paragraph_id": paragraph.paragraph_id,
                "realized_fact_ids": list(validation.realized_fact_ids),
                "realized_event_ids": list(validation.realized_event_ids),
                "tail": tail,
            },
        ],
    }


def complete_chapter_working_ledger(ledger: dict[str, Any], program: ChapterProgram) -> dict[str, Any]:
    """Mark a ledger complete only after every planned paragraph committed."""

    if _expected_next_paragraph(program, ledger):
        raise ValueError("cannot complete chapter working ledger before every paragraph is validated")
    return {**ledger, "completed": True}

