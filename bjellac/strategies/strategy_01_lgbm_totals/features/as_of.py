"""Week-aware prior blending weights.

Tier 1 (Phase C) does not use these — ratings flow through directly.
Tier 2 (Phase D) plugs in here so EWMA(current-season) blends with last-season
terminal values. Tier 3 (Phase E) uses the same weights.

Returns (prior_weight, current_weight) summing to 1.0.

Heuristic from the strategy spec:
    weeks 1-3 : (1.0, 0.0)   — no current-season signal
    weeks 4-8 : (0.6, 0.4)
    weeks 9+  : (0.2, 0.8)
    bowl      : (0.2, 0.8)   — long layoff, but use current
"""
from __future__ import annotations


def blend_weights(week: int, season_type: str = "regular") -> tuple[float, float]:
    if week is None:
        return (1.0, 0.0)
    if week <= 3:
        return (1.0, 0.0)
    if week <= 8:
        return (0.6, 0.4)
    return (0.2, 0.8)
