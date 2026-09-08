"""Deterministic structure checks for module-02 output candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ...contracts import HierarchicalOutlineBundle


@dataclass(frozen=True)
class OutlineStructureIssue:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class OutlineStructureReport:
    passed: bool
    node_count: int
    root_count: int
    level_counts: dict[str, int]
    issues: tuple[OutlineStructureIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def assess_outline_structure(
    bundle: HierarchicalOutlineBundle,
) -> OutlineStructureReport:
    issues: list[OutlineStructureIssue] = []
    try:
        bundle.validate()
    except (TypeError, ValueError) as exc:
        issues.append(OutlineStructureIssue("invalid_outline_contract", str(exc)))
    level_counts = {
        level: sum(node.level == level for node in bundle.nodes)
        for level in ("story_arc", "volume", "book")
    }
    return OutlineStructureReport(
        passed=not issues,
        node_count=len(bundle.nodes),
        root_count=len(bundle.root_outline_ids),
        level_counts=level_counts,
        issues=tuple(issues),
    )
