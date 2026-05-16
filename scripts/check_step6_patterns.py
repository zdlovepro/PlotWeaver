from __future__ import annotations

import sys
from pathlib import Path

import config
from pipeline.core.common_json import read_json_file


REQUIRED_KEYS = [
    "micro_interaction_candidates",
    "macro_trope_candidates",
    "plot_thread_candidates",
    "micro_interaction_clusters",
    "macro_trope_clusters",
    "plot_thread_clusters",
    "micro_interactions",
    "macro_tropes",
    "plot_threads",
    "coverage_report",
]


def main() -> int:
    path = Path(config.INTERMEDIATE_DIR) / "step6_interaction_patterns.json"
    if not path.exists():
        print(f"Missing file: {path}")
        return 1
    data = read_json_file(path)
    for key in REQUIRED_KEYS:
        if key not in data:
            print(f"Missing key: {key}")
            return 1

    candidate_counts = [
        len(data.get("micro_interaction_candidates", [])),
        len(data.get("macro_trope_candidates", [])),
        len(data.get("plot_thread_candidates", [])),
    ]
    if all(count in {3, 4, 5} for count in candidate_counts) and max(candidate_counts) > 0:
        print("Candidate counts look suspiciously fixed at 3-5.")
        return 1

    print("Step6 pattern checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
