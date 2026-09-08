"""Load the only legal input for module 02: accepted module-01 bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ...common.jsonio import read_json, read_jsonl
from ...contracts import SYNOPSIS_SCHEMA_VERSION, ChapterSynopsisBundle
from .internal import AcceptedSynopsisInput, ChapterCard, LocalSegmentCard


MODULE_01_NAME = "module_01_local_synopsis"


class SynopsisInputError(ValueError):
    """The supplied module-01 artifact is not a legal module-02 input."""


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SynopsisInputError(f"{name} must be a list")
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item for item in result):
        raise SynopsisInputError(f"{name} must contain non-empty values")
    if len(result) != len(set(result)):
        raise SynopsisInputError(f"{name} must not contain duplicates")
    return result


def canonical_synopsis_hash(bundle: ChapterSynopsisBundle) -> str:
    """Hash the accepted synopsis payload, not the copyrighted source text."""

    payload = json.dumps(
        bundle.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_declared_bundle_path(
    manifest: dict[str, Any], manifest_path: Path, bundles_path: Path
) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not str(outputs.get("bundles", "")).strip():
        raise SynopsisInputError("module-01 manifest does not declare its bundle output")
    declared = Path(str(outputs["bundles"]))
    if not declared.is_absolute():
        declared = manifest_path.parent / declared
    if declared.resolve() != bundles_path.resolve():
        raise SynopsisInputError("supplied bundles path does not match the accepted manifest")


def _manifest_scope(
    manifest: dict[str, Any], *, profile: str, selected_count: int
) -> tuple[int, int, bool]:
    scope = manifest.get("scope")
    if scope is None and profile == "short_validation":
        # Backward compatibility for accepted short-validation artifacts written
        # before module 01 declared whole-work coverage explicitly.
        return selected_count, selected_count, False
    if not isinstance(scope, dict):
        raise SynopsisInputError("module-01 manifest is missing its narrative scope")
    available = scope.get("available_narrative_chapter_count")
    selected = scope.get("selected_narrative_chapter_count")
    complete_work = scope.get("complete_work")
    if isinstance(available, bool) or not isinstance(available, int) or available < 1:
        raise SynopsisInputError("available narrative chapter count must be positive")
    if isinstance(selected, bool) or not isinstance(selected, int) or selected < 1:
        raise SynopsisInputError("selected narrative chapter count must be positive")
    if not isinstance(complete_work, bool):
        raise SynopsisInputError("complete_work must be a boolean")
    if selected != selected_count:
        raise SynopsisInputError("selected narrative chapter count does not match bundles")
    if available < selected:
        raise SynopsisInputError("available narrative chapter count is smaller than selection")
    if complete_work and available != selected:
        raise SynopsisInputError("complete_work conflicts with narrative chapter counts")
    return available, selected, complete_work


def _chapter_card(
    bundle: ChapterSynopsisBundle, order: int, synopsis_hash: str
) -> ChapterCard:
    segments_by_id = {
        segment.segment_id: segment
        for local in bundle.local_synopses
        for segment in local.segments
    }
    segments = tuple(
        LocalSegmentCard(segment_id, segment_order, segments_by_id[segment_id].summary)
        for segment_order, segment_id in enumerate(
            bundle.chapter_synopsis.chapter_summary.local_segment_ids
        )
    )
    return ChapterCard(
        chapter_id=bundle.chapter_id,
        order=order,
        synopsis_hash=synopsis_hash,
        opening_state=bundle.chapter_synopsis.opening_state.text,
        local_segments=segments,
        chapter_summary=bundle.chapter_synopsis.chapter_summary.text,
        ending_state=bundle.chapter_synopsis.ending_state.text,
    )


def load_accepted_synopsis_input(
    manifest_path: Path | str,
    bundles_path: Path | str,
    *,
    author_id: str,
    work_id: str,
    profile: str,
) -> AcceptedSynopsisInput:
    manifest_path = Path(manifest_path)
    bundles_path = Path(bundles_path)
    if not manifest_path.is_file():
        raise SynopsisInputError(f"module-01 manifest not found: {manifest_path}")
    if not bundles_path.is_file():
        raise SynopsisInputError(f"module-01 bundles not found: {bundles_path}")

    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise SynopsisInputError("module-01 manifest must be a JSON object")
    if manifest.get("module") != MODULE_01_NAME:
        raise SynopsisInputError("input manifest is not a module-01 synopsis manifest")
    if manifest.get("accepted") is not True:
        raise SynopsisInputError("module-01 manifest has not been accepted")
    if str(manifest.get("schema_version", "")) != SYNOPSIS_SCHEMA_VERSION:
        raise SynopsisInputError("module-01 synopsis schema is not supported")
    for name, expected in (
        ("author_id", author_id),
        ("work_id", work_id),
        ("profile", profile),
    ):
        if str(manifest.get(name, "")) != expected:
            raise SynopsisInputError(f"module-01 {name} does not match the stage context")
    _validate_declared_bundle_path(manifest, manifest_path, bundles_path)

    chapter_ids = _string_tuple(
        manifest.get("successful_chapter_ids"), "successful_chapter_ids"
    )
    attempted_ids = _string_tuple(manifest.get("chapter_ids"), "chapter_ids")
    if attempted_ids != chapter_ids:
        raise SynopsisInputError("accepted module-01 manifest must contain only successful chapters")
    failed_ids = manifest.get("failed_chapter_ids", [])
    if not isinstance(failed_ids, list) or failed_ids:
        raise SynopsisInputError("accepted module-01 manifest must not contain failed chapters")
    available_count, selected_count, complete_work = _manifest_scope(
        manifest, profile=profile, selected_count=len(chapter_ids)
    )

    rows = read_jsonl(bundles_path)
    if not rows:
        raise SynopsisInputError("module-01 bundle output is empty")
    bundles: list[ChapterSynopsisBundle] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise SynopsisInputError(
                f"invalid module-01 bundle at row {index}: expected a JSON object"
            )
        try:
            bundle = ChapterSynopsisBundle.from_dict(row)
            bundle.validate()
        except (TypeError, ValueError) as exc:
            raise SynopsisInputError(
                f"invalid module-01 bundle at row {index}: {exc}"
            ) from exc
        bundles.append(bundle)
    bundle_ids = tuple(bundle.chapter_id for bundle in bundles)
    if bundle_ids != chapter_ids:
        raise SynopsisInputError("bundle chapters do not match the accepted manifest order")

    input_hashes = manifest.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise SynopsisInputError("module-01 manifest is missing input_hashes")
    for bundle in bundles:
        if str(input_hashes.get(bundle.chapter_id, "")) != bundle.source_hash:
            raise SynopsisInputError(
                f"source hash mismatch for accepted chapter: {bundle.chapter_id}"
            )

    accepted_bundles = tuple(bundles)
    synopsis_hashes = tuple(canonical_synopsis_hash(bundle) for bundle in accepted_bundles)
    cards = tuple(
        _chapter_card(bundle, order, synopsis_hashes[order])
        for order, bundle in enumerate(accepted_bundles)
    )
    return AcceptedSynopsisInput(
        author_id=author_id,
        work_id=work_id,
        profile=profile,
        manifest_path=manifest_path.resolve(),
        bundles_path=bundles_path.resolve(),
        chapter_ids=chapter_ids,
        synopsis_hashes=synopsis_hashes,
        bundles=accepted_bundles,
        chapter_cards=cards,
        available_narrative_chapter_count=available_count,
        selected_narrative_chapter_count=selected_count,
        complete_work=complete_work,
    )
