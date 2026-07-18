from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import config
from pipeline.core.common_json import read_json_file, write_json_file


MANIFEST_FILENAME = "pipeline_run_manifest.json"
MANIFEST_SCHEMA_VERSION = 1

# Step 4 rebuilds its local index from Step 2 every time, so it has no durable
# artifact to hash. The remaining steps have stable intermediate files.
STEP_ARTIFACTS: Dict[int, List[str]] = {
    1: ["step1_chunks.json"],
    2: ["step2_extracted_plots.json", "step2_semantic_windows.json", "step2_tail_carry_log.json"],
    3: ["step3_induced_events.json"],
    4: [],
    5: ["step5_world_fusion.json", "step5_world_base.json"],
    6: ["step6_interaction_mining.json", "step6_interaction_patterns.json"],
    7: ["step7_template_mining.json", "step7_templates.json", "pipeline_state_after_step7.json"],
    8: ["step8_source_skeletons.json"],
    9: ["step9_skeleton.json"],
    10: ["step10_casted_skeleton.json"],
    11: ["step11_reassembled_plot.json"],
    12: ["step12_volume_outlines.json"],
    13: ["step13_validation_result.json"],
}


def initialize_or_verify_run(start_step: int) -> Dict[str, Any]:
    """Create a fresh manifest for Step 1 or verify a resume request."""
    if start_step <= 1:
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": uuid4().hex,
            "created_at": _now(),
            "steps": {},
        }
        _save_manifest(manifest)
        return manifest

    manifest = _load_manifest()
    if manifest is None:
        if _legacy_resume_allowed():
            print("[Pipeline] Warning: resuming legacy artifacts without a run manifest.", flush=True)
            manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "run_id": f"legacy_{uuid4().hex}",
                "created_at": _now(),
                "legacy_unverified_steps": list(range(1, start_step)),
                "steps": {},
            }
            _save_manifest(manifest)
            return manifest
        raise RuntimeError(
            "Cannot resume because intermediate_data/pipeline_run_manifest.json is missing. "
            "Run from Step 1 to create a verified artifact set, or explicitly set "
            "PLOTWEAVER_ALLOW_LEGACY_RESUME=1 for one-time legacy recovery."
        )

    _verify_prior_steps(manifest, start_step)
    return manifest


def record_step_artifacts(step: int) -> None:
    manifest = _load_manifest()
    if manifest is None:
        return

    out_dir = Path(config.INTERMEDIATE_DIR)
    artifact_records: List[Dict[str, Any]] = []
    for filename in STEP_ARTIFACTS.get(step, []):
        path = out_dir / filename
        if not path.exists():
            raise RuntimeError(f"Step {step} completed but expected artifact is missing: {path}")
        artifact_records.append(
            {
                "filename": filename,
                "sha256": _file_sha256(path),
                "size": path.stat().st_size,
            }
        )

    steps = manifest.setdefault("steps", {})
    steps[str(step)] = {"recorded_at": _now(), "artifacts": artifact_records}
    _save_manifest(manifest)


def manifest_path() -> Path:
    return Path(config.INTERMEDIATE_DIR) / MANIFEST_FILENAME


def _verify_prior_steps(manifest: Dict[str, Any], start_step: int) -> None:
    steps = manifest.get("steps", {}) if isinstance(manifest.get("steps", {}), dict) else {}
    legacy_steps = {int(step) for step in manifest.get("legacy_unverified_steps", [])}
    for step in range(1, start_step):
        record = steps.get(str(step))
        if not isinstance(record, dict):
            if step in legacy_steps:
                print(f"[Pipeline] Warning: Step {step} uses an unverified legacy artifact.", flush=True)
                continue
            raise RuntimeError(f"Cannot resume at Step {start_step}: manifest has no verified record for Step {step}.")
        for artifact in record.get("artifacts", []):
            filename = str(artifact.get("filename", ""))
            path = Path(config.INTERMEDIATE_DIR) / filename
            if not path.exists():
                raise RuntimeError(f"Cannot resume: verified artifact is missing: {path}")
            if _file_sha256(path) != artifact.get("sha256"):
                raise RuntimeError(
                    f"Cannot resume: artifact changed since it was produced by this run: {path}. "
                    "Restart from the earliest changed step."
                )


def _load_manifest() -> Dict[str, Any] | None:
    path = manifest_path()
    if not path.exists():
        return None
    data = read_json_file(path)
    return data if isinstance(data, dict) else None


def _save_manifest(manifest: Dict[str, Any]) -> None:
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(path, manifest)


def _legacy_resume_allowed() -> bool:
    return str(os.getenv("PLOTWEAVER_ALLOW_LEGACY_RESUME", "")).strip().lower() in {"1", "true", "yes", "on"}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
