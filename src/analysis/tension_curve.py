from typing import Optional

import numpy as np

try:
    from scipy.signal import savgol_filter, find_peaks  # type: ignore[import]
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from ..models.plot_unit import AnnotatedPlotUnit


class TensionCurve:
    """Computes tension curve with smoothing and peak detection (scipy/numpy)."""

    def __init__(self, smoothing_window: int = 7, polyorder: int = 3):
        self.smoothing_window = smoothing_window
        self.polyorder = polyorder

    def build_tension_curve(self, annotated_units: list[AnnotatedPlotUnit]) -> dict:
        """Build a smoothed tension curve and detect tension peaks."""
        if not annotated_units:
            return {"raw": [], "smoothed": [], "peaks": [], "valleys": []}

        raw = [u.tension_level for u in annotated_units]
        smoothed = self._smooth(raw)
        peaks, valleys = self._detect_peaks_valleys(smoothed)

        return {
            "raw": raw,
            "smoothed": smoothed,
            "peaks": peaks,
            "valleys": valleys,
        }

    def _smooth(self, values: list[float]) -> list[float]:
        """Apply Savitzky-Golay filter if scipy available, else simple moving average."""
        arr = np.array(values, dtype=float)
        n = len(arr)
        if n < 3:
            return list(arr)

        if _SCIPY_AVAILABLE and n >= self.smoothing_window:
            window = self.smoothing_window if self.smoothing_window % 2 == 1 else self.smoothing_window + 1
            window = min(window, n if n % 2 == 1 else n - 1)
            poly = min(self.polyorder, window - 1)
            try:
                smoothed = savgol_filter(arr, window_length=window, polyorder=poly)
                return smoothed.tolist()
            except Exception:
                pass

        # Fallback: simple moving average
        k = min(self.smoothing_window, n)
        kernel = np.ones(k) / k
        padded = np.pad(arr, (k // 2, k // 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed[:n].tolist()

    def _detect_peaks_valleys(self, values: list[float]) -> tuple[list[int], list[int]]:
        """Detect peaks and valleys in the tension curve."""
        if len(values) < 3:
            return [], []

        arr = np.array(values, dtype=float)

        if _SCIPY_AVAILABLE:
            peaks, _ = find_peaks(arr, prominence=0.1)
            valleys, _ = find_peaks(-arr, prominence=0.1)
            return peaks.tolist(), valleys.tolist()

        # Fallback: simple local maxima/minima
        peaks, valleys = [], []
        for i in range(1, len(arr) - 1):
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                peaks.append(i)
            elif arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
                valleys.append(i)
        return peaks, valleys
