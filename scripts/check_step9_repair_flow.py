from __future__ import annotations

from pathlib import Path
from typing import Any

import config
from pipeline.core.common_json import read_json_file


def _fail(message: str) -> int:
    print(message)
    return 1


def _repair_field(node: dict[str, Any], key: str) -> tuple[bool, Any]:
    if key in node:
        return True, node.get(key)
    metadata = node.get("metadata", {})
    if isinstance(metadata, dict) and key in metadata:
        return True, metadata.get(key)
    return False, None


def _is_bridge_node(node: dict[str, Any]) -> bool:
    node_id = str(node.get("node_id", "") or "")
    pacing_role = str(node.get("pacing_role", "") or "")
    if bool(node.get("is_bridge", False)):
        return True
    if node_id.startswith("bridge_"):
        return True
    return pacing_role == "bridge_transition"


def main() -> int:
    path = Path(config.INTERMEDIATE_DIR) / "step9_skeleton.json"
    if not path.exists():
        return _fail(f"Missing file: {path}")

    data = read_json_file(path)
    if not isinstance(data, dict):
        return _fail("step9_skeleton.json must be a JSON object.")

    nodes = data.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        return _fail("step9_skeleton.json must contain a non-empty `nodes` list.")

    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            return _fail(f"Node {index} must be an object, got {type(node).__name__}.")

        node_id = str(node.get("node_id", "") or "").strip()
        summary = str(node.get("original_summary", "") or "").strip()
        source_refs = node.get("source_refs", None)

        if not node_id:
            return _fail(f"Node {index} missing required field `node_id`.")
        if not summary:
            return _fail(f"Node {node_id} missing required field `original_summary`.")
        if not isinstance(source_refs, list):
            return _fail(f"Node {node_id} missing required field `source_refs` as list.")

        has_repair_action, repair_action = _repair_field(node, "repair_action")
        has_repair_notes, repair_notes = _repair_field(node, "repair_notes")
        has_repair_candidate_refs, repair_candidate_refs = _repair_field(node, "repair_candidate_refs")
        has_repair_validation_issues, repair_validation_issues = _repair_field(node, "repair_validation_issues")

        if has_repair_action and str(repair_action or "").strip() and not has_repair_notes:
            return _fail(f"Node {node_id} has `repair_action` but missing `repair_notes`.")

        if has_repair_notes and not isinstance(repair_notes, list):
            return _fail(f"Node {node_id} has invalid `repair_notes` (must be list).")

        if has_repair_candidate_refs:
            if not isinstance(repair_candidate_refs, list):
                return _fail(f"Node {node_id} has invalid `repair_candidate_refs` (must be list).")
            if not repair_candidate_refs:
                return _fail(f"Node {node_id} has empty `repair_candidate_refs`.")

        if has_repair_validation_issues:
            if not isinstance(repair_validation_issues, list):
                return _fail(f"Node {node_id} has invalid `repair_validation_issues` (must be list).")
            for issue_index, issue in enumerate(repair_validation_issues):
                if not isinstance(issue, dict):
                    return _fail(
                        f"Node {node_id} repair_validation_issues[{issue_index}] must be object."
                    )
                for key in ("severity", "issue_type", "message"):
                    if key not in issue:
                        return _fail(
                            f"Node {node_id} repair_validation_issues[{issue_index}] missing `{key}`."
                        )

        if _is_bridge_node(node) and not source_refs:
            return _fail(f"Bridge node {node_id} must have non-empty `source_refs`.")

    print("Step9 repair flow checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
