"""Sanity checks for the IsotonicCalibrator wrapper.

These don't depend on any data on disk — they exercise the public surface
(fit / transform / persistence-friendly thresholds) directly so a regression
in sklearn behavior is caught before the heavier walk-forward runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from bjellac.strategies.strategy_01_lgbm_totals.calibration import IsotonicCalibrator


def _toy_calibration_data(seed: int = 7, n: int = 2000):
    """Synthetic miscalibrated raw_p vs binary outcome.

    Underlying truth: P(y=1) = sigmoid(2 * (raw_p - 0.5)) — i.e. raw probs
    underestimate the extremes. The fitted calibrator should pull mid-range
    raw probs toward 0.5 and push the tails outward.
    """
    rng = np.random.default_rng(seed)
    raw_p = rng.uniform(0.05, 0.95, size=n)
    true_p = 1.0 / (1.0 + np.exp(-2.0 * (raw_p - 0.5)))
    y = (rng.uniform(size=n) < true_p).astype(float)
    return raw_p, y


def test_unfit_calibrator_refuses_transform():
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError):
        cal.transform(0.5)


def test_fit_then_transform_in_unit_interval():
    raw_p, y = _toy_calibration_data()
    cal = IsotonicCalibrator().fit(raw_p, y)
    grid = np.linspace(0.01, 0.99, 99)
    out = cal.transform(grid)
    assert out.shape == grid.shape
    assert np.all(out >= 0.001) and np.all(out <= 0.999)


def test_transform_is_monotonic():
    raw_p, y = _toy_calibration_data()
    cal = IsotonicCalibrator().fit(raw_p, y)
    grid = np.linspace(0.01, 0.99, 99)
    out = cal.transform(grid)
    diffs = np.diff(out)
    # Isotonic is non-decreasing — small float epsilon tolerance.
    assert np.all(diffs >= -1e-12)


def test_thresholds_round_trip_via_dict():
    raw_p, y = _toy_calibration_data()
    cal = IsotonicCalibrator().fit(raw_p, y)
    thr = cal.thresholds
    assert len(thr["x_thresholds"]) == len(thr["y_thresholds"])
    assert len(thr["x_thresholds"]) >= 2


def test_scalar_input_returns_scalar_output():
    raw_p, y = _toy_calibration_data()
    cal = IsotonicCalibrator().fit(raw_p, y)
    out = cal.transform(0.5)
    assert isinstance(out, float)


def test_empty_fit_raises():
    cal = IsotonicCalibrator()
    with pytest.raises(ValueError):
        cal.fit([], [])
