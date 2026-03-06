from typing import Optional

from ..models.plot_unit import AnnotatedPlotUnit
from .compatibility_scorer import CompatibilityScorer


class ReplacementEngine:
    """Greedy replacement engine: replaces plot units from donor novel into framework novel."""

    def __init__(
        self,
        scorer: Optional[CompatibilityScorer] = None,
        min_compatibility: float = 0.5,
        max_replacement_depth: int = 3,
    ):
        self.scorer = scorer or CompatibilityScorer()
        self.min_compatibility = min_compatibility
        self.max_replacement_depth = max_replacement_depth

    def remix(
        self,
        framework_units: list[AnnotatedPlotUnit],
        donor_units: list[AnnotatedPlotUnit],
    ) -> list[AnnotatedPlotUnit]:
        """Replace up to max_replacement_depth framework units with compatible donor units (greedy)."""
        result = list(framework_units)
        replaced_count = 0
        used_donor_ids = set()

        for i, fw_unit in enumerate(framework_units):
            if replaced_count >= self.max_replacement_depth:
                break

            best_donor = self._find_best_donor(fw_unit, donor_units, used_donor_ids)
            if best_donor is not None and self.scorer.score(fw_unit, best_donor) >= self.min_compatibility:
                result[i] = best_donor
                used_donor_ids.add(best_donor.plot_unit.id)
                replaced_count += 1

        return result

    def _find_best_donor(
        self,
        framework_unit: AnnotatedPlotUnit,
        donor_units: list[AnnotatedPlotUnit],
        used_ids: set,
    ) -> Optional[AnnotatedPlotUnit]:
        """Find the best compatible donor unit not already used."""
        best = None
        best_score = -1.0
        for donor in donor_units:
            if donor.plot_unit.id in used_ids:
                continue
            score = self.scorer.score(framework_unit, donor)
            if score > best_score:
                best_score = score
                best = donor
        return best
