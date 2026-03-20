from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

HealthStatus = Literal["CRITICAL", "WARNING", "HEALTHY"]


CRITICAL_COLOR = "#E63946"
WARNING_COLOR = "#F4A261"
HEALTHY_COLOR = "#2A9D8F"



def rul_to_health_status(rul: float) -> HealthStatus:
    """
    Map a predicted RUL (in cycles) to a discrete health status.

    Thresholds follow the project spec:
    - CRITICAL:  RUL < 30
    - WARNING:   30 <= RUL <= 70
    - HEALTHY:   RUL > 70
    """
    if np.isnan(rul):
        return "CRITICAL"
    if rul < 30:
        return "CRITICAL"
    if rul <= 70:
        return "WARNING"
    return "HEALTHY"


def health_color(status: str) -> str:
    """
    Consistent color mapping for all charts and badges.
    """
    status_norm = (status or "").upper()
    if status_norm == "CRITICAL":
        return CRITICAL_COLOR
    if status_norm == "WARNING":
        return WARNING_COLOR
    if status_norm == "HEALTHY":
        return HEALTHY_COLOR
    return UNKNOWN_COLOR


@dataclass(frozen=True)
class DegradationSummary:
    slope: float
    label: str


def degradation_slope(cycles: np.ndarray, rul_series: np.ndarray) -> DegradationSummary:
    """
    Estimate degradation speed from a RUL-vs-cycle series.

    We fit a simple least-squares line to RUL over cycle and classify:
    - fast   : strongly negative slope
    - moderate: mildly negative slope
    - slow  : flat or positive slope (rare, indicates stable or improving)
    """
    if len(cycles) < 2 or len(rul_series) < 2:
        return DegradationSummary(slope=np.nan, label="unknown")

    x = np.asarray(cycles, dtype=float)
    y = np.asarray(rul_series, dtype=float)

    # Guard against degenerate inputs
    if np.allclose(x, x[0]) or np.all(np.isnan(y)):
        return DegradationSummary(slope=np.nan, label="unknown")

    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0])

    # Heuristic thresholds in cycles per cycle (dimensionless):
    # more negative = faster degradation.
    if slope <= -0.7:
        label = "fast"
    elif slope <= -0.25:
        label = "moderate"
    else:
        label = "slow"

    return DegradationSummary(slope=slope, label=label)

