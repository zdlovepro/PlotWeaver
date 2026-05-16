from __future__ import annotations

import sys
from pathlib import Path

import config
from pipeline.core.common_json import read_json_file


REQUIRED_KEYS = [
    "template_candidates",
    "template_clusters",
    "executable_templates",
    "coverage_report",
]

REQUIRED_EXECUTABLE_KEYS = [
    "template_id",
    "template_name",
    "abstract_function",
    "conflict_engine",
    "role_slots",
    "beat_sequence",
    "state_delta",
    "variation_axes",
    "forbidden_source_details",
]


def main() -> int:
    path = Path(config.INTERMEDIATE_DIR) / "step7_templates.json"
    if not path.exists():
        print(f"Missing file: {path}")
        return 1
    data = read_json_file(path)
    for key in REQUIRED_KEYS:
        if key not in data:
            print(f"Missing key: {key}")
            return 1

    executable_templates = data.get("executable_templates", [])
    for index, template in enumerate(executable_templates):
        for key in REQUIRED_EXECUTABLE_KEYS:
            if key not in template:
                print(f"Executable template {index} missing key: {key}")
                return 1

    if len(executable_templates) in {18, 24} and len(data.get("template_candidates", [])) > len(executable_templates):
        print("Executable template count still looks fixed at 18 or 24.")
        return 1

    if len(data.get("template_candidates", [])) > 20 and len(data.get("template_candidates", [])) <= len(executable_templates):
        print("Template candidates should noticeably exceed final executable templates when input is rich.")
        return 1

    print("Step7 template checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
