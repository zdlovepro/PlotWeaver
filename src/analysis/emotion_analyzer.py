from typing import Optional

import numpy as np

from ..models.plot_unit import AnnotatedPlotUnit


# Map emotion tones to numeric values for curve computation
_EMOTION_VALUES: dict[str, float] = {
    "喜悦": 1.0,
    "积极": 0.8,
    "平静": 0.5,
    "悲伤": -0.5,
    "消极": -0.6,
    "恐惧": -0.7,
    "愤怒": -0.4,
    "紧张": 0.2,
}


def _emotion_to_value(tone: str) -> float:
    """Convert emotion tone label to a numeric value [-1, 1]."""
    for key, val in _EMOTION_VALUES.items():
        if key in tone:
            return val
    return 0.0


class EmotionAnalyzer:
    """Analyzes emotion curves from annotated plot units using pure math (no LLM)."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def build_emotion_curve(self, annotated_units: list[AnnotatedPlotUnit]) -> dict:
        """Build a smoothed emotion curve and detect turning points."""
        if not annotated_units:
            return {"raw": [], "smoothed": [], "turning_points": []}

        raw_values = [_emotion_to_value(u.emotion_tone) for u in annotated_units]

        smoothed = self._smooth(raw_values)
        turning_points = self._detect_turning_points(smoothed)

        return {
            "raw": raw_values,
            "smoothed": smoothed,
            "turning_points": turning_points,
            "emotion_labels": [u.emotion_tone for u in annotated_units],
        }

    def _smooth(self, values: list[float]) -> list[float]:
        """Apply a centered moving-average smoothing."""
        if len(values) < self.window_size:
            return list(values)
        arr = np.array(values, dtype=float)
        kernel = np.ones(self.window_size) / self.window_size
        # Use 'same' mode and pad the boundaries with edge values
        padded = np.pad(arr, (self.window_size // 2, self.window_size // 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed[: len(values)].tolist()

    def _detect_turning_points(self, values: list[float]) -> list[int]:
        """Detect indices where the emotion curve changes direction."""
        if len(values) < 3:
            return []
        arr = np.array(values)
        diff = np.diff(arr)
        turning_points = []
        for i in range(1, len(diff)):
            if diff[i - 1] * diff[i] < 0:  # Sign change → local extremum
                turning_points.append(i)
        return turning_points
