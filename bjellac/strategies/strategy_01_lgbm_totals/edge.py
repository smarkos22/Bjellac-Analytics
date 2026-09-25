"""Convert model outputs + market lines into edge.

    edge_total = model_total - market_total
        > 0 → bet OVER
        < 0 → bet UNDER

Edge is translated to win probability via Gaussian CDF using residual_std from
walk-forward CV. That probability feeds Kelly sizing in `kelly.py`.
"""
from __future__ import annotations

from math import erf, sqrt


def gaussian_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))


def edge_total(model_total: float, market_total: float) -> float:
    return model_total - market_total


def over_probability(model_total: float, market_total: float, residual_std: float) -> float:
    """Probability the total goes OVER market_total."""
    return 1.0 - gaussian_cdf(market_total, model_total, residual_std)
