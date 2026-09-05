from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


REQUIRED_EXECUTABLE_TEMPLATE_KEYS = {
    "template_id",
    "template_name",
    "abstract_function",
    "conflict_engine",
    "role_slots",
    "beat_sequence",
    "state_delta",
    "variation_axes",
    "forbidden_source_details",
}


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_absent_keys(name: str, payload: dict[str, Any], keys: Iterable[str]) -> None:
    present = sorted(key for key in keys if key in payload)
    if present:
        raise AssertionError(f"{name} contains forbidden keys: {present}")


def _assert_present_keys(name: str, payload: dict[str, Any], keys: Iterable[str]) -> None:
    missing = sorted(key for key in keys if key not in payload)
    if missing:
        raise AssertionError(f"{name} is missing required keys: {missing}")


def _assert_executable_templates(payload: dict[str, Any]) -> None:
    templates = payload.get("executable_templates")
    if not isinstance(templates, list) or not templates:
        raise AssertionError("step7_templates.json must contain a non-empty executable_templates list")
    for index, template in enumerate(templates, start=1):
        if not isinstance(template, dict):
            raise AssertionError(f"Executable template #{index} is not a JSON object")
        missing = sorted(key for key in REQUIRED_EXECUTABLE_TEMPLATE_KEYS if key not in template)
        if missing:
            raise AssertionError(f"Executable template #{index} is missing keys: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PlotWeaver step artifact boundaries.")
    parser.add_argument("--dir", default="intermediate_data", help="Artifact directory. Defaults to ./intermediate_data")
    args = parser.parse_args()

    artifact_dir = Path(args.dir)
    step5 = _load_json(artifact_dir / "step5_world_base.json")
    step6 = _load_json(artifact_dir / "step6_interaction_patterns.json")
    step7 = _load_json(artifact_dir / "step7_templates.json")

    if not isinstance(step5, dict) or not isinstance(step6, dict) or not isinstance(step7, dict):
        raise AssertionError("All checked artifacts must be top-level JSON objects")

    _assert_absent_keys("step5_world_base.json", step5, {"macro_tropes", "plot_threads", "event_templates", "event_flow_templates"})
    _assert_absent_keys(
        "step6_interaction_patterns.json",
        step6,
        {"world_background", "cultivation_realms", "event_templates", "event_flow_templates"},
    )
    _assert_absent_keys("step7_templates.json", step7, {"world_background", "cultivation_realms", "power_source"})
    _assert_present_keys(
        "step7_templates.json",
        step7,
        {"role_slot_templates", "event_templates", "volume_templates", "event_flow_templates", "executable_templates"},
    )
    _assert_executable_templates(step7)

    print("Artifact checks passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Artifact check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

