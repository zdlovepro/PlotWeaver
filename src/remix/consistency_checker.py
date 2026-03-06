from typing import Optional

from ..models.plot_unit import AnnotatedPlotUnit
from ..models.character import CharacterMapping


class ConsistencyChecker:
    """Validates causal integrity, character consistency, and tension curve of remixed plots."""

    def __init__(self, min_tension_variance: float = 0.05):
        self.min_tension_variance = min_tension_variance

    def validate(
        self,
        remixed_units: list[AnnotatedPlotUnit],
        character_mappings: Optional[list[CharacterMapping]] = None,
    ) -> dict:
        """Run all consistency checks and return a validation report."""
        report = {
            "passed": True,
            "issues": [],
            "causal_integrity": self._check_causal_integrity(remixed_units),
            "character_consistency": self._check_character_consistency(remixed_units, character_mappings),
            "tension_curve": self._check_tension_curve(remixed_units),
            "narrative_arc": self._check_narrative_arc(remixed_units),
        }

        # Aggregate issues
        for section_name, section in report.items():
            if isinstance(section, dict) and not section.get("ok", True):
                report["passed"] = False
                for issue in section.get("issues", []):
                    report["issues"].append(f"[{section_name}] {issue}")

        return report

    def _check_causal_integrity(self, units: list[AnnotatedPlotUnit]) -> dict:
        """Check for obvious causal order violations."""
        issues = []
        narrative_order = ["开端", "发展", "转折", "高潮", "结局"]
        found_functions = [u.narrative_function for u in units]

        # Check that "结局" doesn't appear before "开端"
        if "结局" in found_functions and "开端" in found_functions:
            idx_end = found_functions.index("结局")
            idx_start = found_functions.index("开端")
            if idx_end < idx_start:
                issues.append("结局出现在开端之前，因果顺序异常")

        return {"ok": len(issues) == 0, "issues": issues}

    def _check_character_consistency(
        self,
        units: list[AnnotatedPlotUnit],
        mappings: Optional[list[CharacterMapping]],
    ) -> dict:
        """Check that mapped characters are referenced consistently."""
        issues = []
        if not mappings:
            return {"ok": True, "issues": []}

        mapped_ids = {m.source_character_id for m in mappings} | {m.target_character_id for m in mappings}
        # Basic check: ensure at least some characters appear in remixed plots
        all_chars = set()
        for u in units:
            all_chars.update(u.plot_unit.characters)

        if not all_chars:
            issues.append("混合后的情节中未检测到任何人物")

        return {"ok": len(issues) == 0, "issues": issues}

    def _check_tension_curve(self, units: list[AnnotatedPlotUnit]) -> dict:
        """Check that the tension curve has sufficient variance and a proper climax."""
        import numpy as np

        issues = []
        if not units:
            return {"ok": True, "issues": []}

        tensions = [u.tension_level for u in units]
        variance = float(np.var(tensions))
        if variance < self.min_tension_variance:
            issues.append(f"张力曲线方差过低 ({variance:.3f})，情节缺乏起伏")

        max_tension = max(tensions)
        max_idx = tensions.index(max_tension)
        n = len(tensions)
        if n > 4 and max_idx < n // 3:
            issues.append("最高张力点出现在情节前1/3处，高潮过早")

        return {"ok": len(issues) == 0, "issues": issues, "variance": variance}

    def _check_narrative_arc(self, units: list[AnnotatedPlotUnit]) -> dict:
        """Check that the narrative arc follows expected patterns."""
        issues = []
        functions = [u.narrative_function for u in units]
        has_start = any(f in ("开端", "发展") for f in functions)
        has_end = "结局" in functions

        if not has_start:
            issues.append("缺少开端/发展情节单元")
        if not has_end:
            issues.append("缺少结局情节单元")

        return {"ok": len(issues) == 0, "issues": issues}
