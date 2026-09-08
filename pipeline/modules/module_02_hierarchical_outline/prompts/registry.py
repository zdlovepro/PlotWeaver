from __future__ import annotations

import hashlib
import re
from pathlib import Path


TEMPLATE_DIR = Path(__file__).with_name("templates")
TEMPLATE_NAMES = (
    "system",
    "boundary_detection",
    "boundary_review",
    "node_synthesis",
    "fidelity_review",
    "coverage_review",
    "node_repair",
    "top_down_review",
    "volume_boundary_detection",
    "volume_boundary_review",
    "parent_node_synthesis",
    "parent_fidelity_review",
    "parent_coverage_review",
    "parent_node_repair",
    "hierarchy_review",
)
_PLACEHOLDER = re.compile(r"\[\[([A-Z][A-Z0-9_]*)\]\]")


def load_template(name: str) -> str:
    if name not in TEMPLATE_NAMES:
        raise KeyError(f"unknown prompt template: {name}")
    return (TEMPLATE_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


def template_revision(name: str) -> str:
    return hashlib.sha256(load_template(name).encode("utf-8")).hexdigest()[:12]


def render_template(name: str, **values: str) -> str:
    template = load_template(name)
    expected = set(_PLACEHOLDER.findall(template))
    supplied = set(values)
    if supplied != expected:
        raise ValueError(
            f"prompt {name} placeholders mismatch: "
            f"expected={sorted(expected)}, supplied={sorted(supplied)}"
        )
    for key, value in values.items():
        template = template.replace(f"[[{key}]]", str(value))
    if _PLACEHOLDER.search(template):
        raise ValueError(f"prompt {name} contains unresolved placeholders")
    return template


PROMPT_REVISION = "+".join(
    f"{name}:{template_revision(name)}" for name in TEMPLATE_NAMES
)
