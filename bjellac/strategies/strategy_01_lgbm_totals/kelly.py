"""Fractional Kelly sizing.

Standard Kelly for a binary-outcome wager at decimal odds `b` (net win per $1):

    f* = (b * p - q) / b,  q = 1 - p

We bet a fraction `kelly_fraction` of f* (default 0.25 = quarter-Kelly), capped
at `cap_pct_bankroll` of bankroll. Negative f* → no bet.

American odds are typically -110 on standard sides → b = 100/110 ≈ 0.909.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KellyStake:
    side: str             # "home_spread", "away_spread", "over", "under" — caller decides
    fraction_of_bankroll: float
    full_kelly: float
    win_prob: float
    decimal_b: float
    expected_value: float
    breakeven_prob: float


def american_to_b(american_odds: int) -> float:
    """Convert American odds to net decimal odds `b` (profit per $1 stake)."""
    if american_odds < 0:
        return 100.0 / abs(american_odds)
    return american_odds / 100.0


def kelly_stake(
    win_prob: float,
    american_odds: int = -110,
    kelly_fraction: float = 0.25,
    cap_pct_bankroll: float = 0.02,
    side: str = "",
) -> KellyStake:
    if not 0.0 <= win_prob <= 1.0:
        raise ValueError(f"win_prob out of range: {win_prob}")
    b = american_to_b(american_odds)
    q = 1.0 - win_prob
    full = (b * win_prob - q) / b if b > 0 else 0.0
    breakeven = q / (b + 1) if b > 0 else 1.0  # = 1/(b+1) in implied-prob form
    breakeven = 1.0 / (b + 1.0)
    sized = max(0.0, full) * kelly_fraction
    sized = min(sized, cap_pct_bankroll)
    ev = win_prob * b - q  # per $1 staked
    return KellyStake(
        side=side,
        fraction_of_bankroll=sized,
        full_kelly=full,
        win_prob=win_prob,
        decimal_b=b,
        expected_value=ev,
        breakeven_prob=breakeven,
    )
