"""Statistical Process Control (SPC) Shewhart & Western Electric Control Chart Monitor."""

from collections import deque
from typing import Dict, Any, List
import numpy as np


class SPCProcessMonitor:
    """
    Tracks inline inspection telemetry across rolling production windows
    and evaluates Western Electric / Shewhart 3-sigma control rules.
    """

    def __init__(self, window_size: int = 100, baseline_mean: float = 0.18, baseline_std: float = 0.05):
        self.window_size = window_size
        self.baseline_mean = baseline_mean
        self.baseline_std = baseline_std
        self.anomaly_scores: deque = deque(maxlen=window_size)
        self.defect_flags: deque = deque(maxlen=window_size)

    def record_frame(self, anomaly_score: float, has_defect: bool) -> None:
        self.anomaly_scores.append(float(anomaly_score))
        self.defect_flags.append(1 if has_defect else 0)

    def evaluate_spc(self) -> Dict[str, Any]:
        """Compute UCL, LCL, first-pass yield, and Western Electric out-of-control rules."""
        ucl_3sigma = round(self.baseline_mean + 3.0 * self.baseline_std, 4)
        ucl_2sigma = round(self.baseline_mean + 2.0 * self.baseline_std, 4)
        lcl_3sigma = round(max(0.0, self.baseline_mean - 3.0 * self.baseline_std), 4)

        n = len(self.anomaly_scores)
        if n == 0:
            return {
                "samples_in_window": 0,
                "process_in_control": True,
                "first_pass_yield_pct": 100.0,
                "current_mean_score": self.baseline_mean,
                "center_line": self.baseline_mean,
                "ucl_3sigma": ucl_3sigma,
                "lcl_3sigma": lcl_3sigma,
                "violated_rules": [],
            }

        scores = np.array(self.anomaly_scores, dtype=float)
        defects = np.array(self.defect_flags, dtype=int)
        yield_pct = float((1.0 - np.mean(defects)) * 100.0)
        violations: List[str] = []

        # Rule 1: Any single observation beyond 3-sigma UCL
        if scores[-1] > ucl_3sigma:
            violations.append("Western Electric Rule 1: Latest frame exceeds 3-sigma UCL")

        # Rule 2: 2 out of last 3 consecutive points beyond 2-sigma limit
        if n >= 3 and int(np.sum(scores[-3:] > ucl_2sigma)) >= 2:
            violations.append("Western Electric Rule 2: 2 of last 3 frames exceed 2-sigma warning limit")

        return {
            "samples_in_window": n,
            "process_in_control": len(violations) == 0,
            "first_pass_yield_pct": round(yield_pct, 2),
            "current_mean_score": round(float(np.mean(scores)), 4),
            "center_line": self.baseline_mean,
            "ucl_3sigma": ucl_3sigma,
            "lcl_3sigma": lcl_3sigma,
            "violated_rules": violations,
        }
