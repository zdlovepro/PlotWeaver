from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


VALID_TAIL_STATUSES = {"open", "merged", "dropped_at_end"}


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _fail(message: str) -> None:
    print(message)
    raise SystemExit(1)


def _iter_records(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(payload, dict):
        _fail("Expected top-level JSON object mapping novel names to lists.")
    records: list[tuple[str, dict[str, Any]]] = []
    for novel_name, items in payload.items():
        if not isinstance(items, list):
            _fail(f"{novel_name}: expected list payload.")
        for item in items:
            if not isinstance(item, dict):
                _fail(f"{novel_name}: expected object item, got {type(item).__name__}.")
            records.append((str(novel_name), item))
    return records


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    intermediate_dir = repo_root / "intermediate_data"

    step1 = _load_json(intermediate_dir / "step1_chunks.json")
    windows_payload = _load_json(intermediate_dir / "step2_semantic_windows.json")
    atoms_payload = _load_json(intermediate_dir / "step2_extracted_plots.json")
    tails_payload = _load_json(intermediate_dir / "step2_tail_carry_log.json")

    if not isinstance(step1, dict) or not step1:
        _fail("step1_chunks.json must exist and contain at least one novel.")
    if not isinstance(windows_payload, dict) or not windows_payload:
        _fail("step2_semantic_windows.json must exist and be non-empty.")
    if not isinstance(atoms_payload, dict) or not atoms_payload:
        _fail("step2_extracted_plots.json must exist and be non-empty.")
    if not isinstance(tails_payload, dict):
        _fail("step2_tail_carry_log.json must exist and be a JSON object.")

    window_records = _iter_records(windows_payload)
    atom_records = _iter_records(atoms_payload)
    tail_records = _iter_records(tails_payload)

    if not window_records:
        _fail("step2_semantic_windows.json must contain at least one semantic window.")
    if not atom_records:
        _fail("step2_extracted_plots.json must contain at least one PlotAtom.")

    for novel_name, atom in atom_records:
        for key in ("atom_id", "summary", "chapter_start", "chapter_end", "semantic_window_id", "is_complete"):
            if key not in atom:
                _fail(f"{novel_name}: PlotAtom missing required key `{key}`.")

        if bool(atom.get("is_complete", True)):
            missing = []
            if not str(atom.get("outcome", "") or "").strip():
                missing.append("outcome")
            state_delta = atom.get("state_delta", {})
            if not isinstance(state_delta, dict) or not state_delta:
                missing.append("state_delta")
            if missing:
                print(
                    f"Warning: complete PlotAtom {atom.get('atom_id', '')} is missing "
                    f"{', '.join(missing)}."
                )

    for novel_name, tail in tail_records:
        status = str(tail.get("status", "") or "")
        if status not in VALID_TAIL_STATUSES:
            _fail(f"{novel_name}: invalid tail status `{status}`.")
        if status == "merged" and not str(tail.get("carried_to_semantic_window_id", "") or "").strip():
            _fail(f"{novel_name}: merged tail missing carried_to_semantic_window_id.")

    for novel_name, window in window_records:
        contains_tail = bool(window.get("contains_tail_from_previous", False))
        if contains_tail and not str(window.get("carried_tail_id", "") or "").strip():
            _fail(f"{novel_name}: semantic window with contains_tail_from_previous=True missing carried_tail_id.")

    print("Step2 atomization checks passed.")


if __name__ == "__main__":
    main()
