import numpy as np

from ..models.plot_unit import AnnotatedPlotUnit


# Weight configuration for compatibility scoring
DEFAULT_WEIGHTS = {
    "narrative_function": 0.25,
    "emotion_tone": 0.20,
    "conflict_type": 0.20,
    "tension_level": 0.15,
    "themes": 0.20,
}

# Narrative function compatibility table (higher = more compatible)
_NARRATIVE_COMPAT: dict[tuple[str, str], float] = {
    ("开端", "发展"): 1.0,
    ("发展", "转折"): 1.0,
    ("转折", "高潮"): 1.0,
    ("高潮", "结局"): 1.0,
    ("铺垫", "高潮"): 0.9,
    ("悬念", "转折"): 0.9,
    ("铺垫", "转折"): 0.8,
}


def _narrative_compat(fn_a: str, fn_b: str) -> float:
    score = _NARRATIVE_COMPAT.get((fn_a, fn_b), _NARRATIVE_COMPAT.get((fn_b, fn_a), 0.3))
    return score


_EMOTION_COMPAT: dict[tuple[str, str], float] = {
    ("积极", "积极"): 1.0,
    ("消极", "消极"): 0.8,
    ("紧张", "高潮"): 0.9,
    ("悲伤", "结局"): 0.8,
}


def _emotion_compat(et_a: str, et_b: str) -> float:
    if et_a == et_b:
        return 1.0
    return _EMOTION_COMPAT.get((et_a, et_b), _EMOTION_COMPAT.get((et_b, et_a), 0.4))


def _themes_overlap(themes_a: list[str], themes_b: list[str]) -> float:
    if not themes_a or not themes_b:
        return 0.0
    set_a = set(themes_a)
    set_b = set(themes_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class CompatibilityScorer:
    """Computes weighted compatibility scores between plot units (no LLM)."""

    def __init__(self, weights: dict = None):
        self.weights = weights or DEFAULT_WEIGHTS

    def score(self, unit_a: AnnotatedPlotUnit, unit_b: AnnotatedPlotUnit) -> float:
        """Return compatibility score [0, 1] between two annotated plot units."""
        w = self.weights

        nf_score = _narrative_compat(unit_a.narrative_function, unit_b.narrative_function)
        et_score = _emotion_compat(unit_a.emotion_tone, unit_b.emotion_tone)
        ct_score = 1.0 if unit_a.conflict_type == unit_b.conflict_type else 0.3
        tl_score = 1.0 - abs(unit_a.tension_level - unit_b.tension_level)
        th_score = _themes_overlap(unit_a.themes, unit_b.themes)

        total = (
            w["narrative_function"] * nf_score
            + w["emotion_tone"] * et_score
            + w["conflict_type"] * ct_score
            + w["tension_level"] * tl_score
            + w["themes"] * th_score
        )
        return float(np.clip(total, 0.0, 1.0))

    def build_compatibility_matrix(
        self,
        units_a: list[AnnotatedPlotUnit],
        units_b: list[AnnotatedPlotUnit],
    ) -> list[list[float]]:
        """Return an n×m compatibility matrix."""
        return [[self.score(a, b) for b in units_b] for a in units_a]
