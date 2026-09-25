"""Probability calibration for the totals model.

The raw over/under probabilities in `edge.py` come from a Gaussian CDF around
the model's point prediction with cross-validated residual_std. Diagnostic
work showed this is not well-calibrated in the mid-range — errors are not
Gaussian.

This module fits an isotonic regression on out-of-sample (raw_p, outcome)
pairs collected via walk-forward across historical seasons, then applies the
fitted mapping at backtest / live-prediction time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from bjellac.strategies.strategy_01_lgbm_totals import ARTIFACTS_DIR, MANIFEST_PATH
from bjellac.strategies.strategy_01_lgbm_totals.edge import over_probability
from bjellac.strategies.strategy_01_lgbm_totals.models.train import (
    MODELS_DIR,
    features_for,
    walk_forward,
)

CALIBRATORS_DIR = ARTIFACTS_DIR / "calibrators"


class IsotonicCalibrator:
    """Thin wrapper around `sklearn.isotonic.IsotonicRegression`.

    Clips outputs to [0.001, 0.999] so calibrated probabilities stay strictly
    inside the unit interval — Kelly sizing divides by `(1 - p)` and degrades
    on hard 0/1 endpoints.
    """

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
        self._fitted = False

    def fit(self, raw_p: Iterable[float], outcome: Iterable[float]) -> "IsotonicCalibrator":
        x = np.asarray(list(raw_p), dtype=float)
        y = np.asarray(list(outcome), dtype=float)
        if x.size == 0:
            raise ValueError("Cannot fit calibrator on empty sample")
        self._iso.fit(x, y)
        self._fitted = True
        return self

    def transform(self, raw_p: float | np.ndarray) -> float | np.ndarray:
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fit")
        scalar = np.isscalar(raw_p)
        x = np.atleast_1d(np.asarray(raw_p, dtype=float))
        out = self._iso.transform(x)
        return float(out[0]) if scalar else out

    @property
    def thresholds(self) -> dict:
        if not self._fitted:
            return {"x_thresholds": [], "y_thresholds": []}
        return {
            "x_thresholds": self._iso.X_thresholds_.tolist(),
            "y_thresholds": self._iso.y_thresholds_.tolist(),
        }


def _calibration_set_for_total(
    season_rows: pl.DataFrame,
    total_pred: np.ndarray,
    total_std: float,
) -> tuple[list[float], list[float]]:
    """For one season, walk every game and emit (raw_p, outcome) pairs for
    the total market. Pushes are dropped.
    """
    total_p: list[float] = []
    total_y: list[float] = []

    for row, tp in zip(season_rows.to_dicts(), total_pred):
        market_total = float(row["market_total"])
        actual_total = float(row["target_total"])

        et = tp - market_total
        if et > 0:
            p = over_probability(tp, market_total, total_std)
            push = actual_total == market_total
            y = 0.5 if push else (1.0 if actual_total > market_total else 0.0)
        else:
            p = 1.0 - over_probability(tp, market_total, total_std)
            push = actual_total == market_total
            y = 0.5 if push else (0.0 if actual_total > market_total else 1.0)
        if y != 0.5:
            total_p.append(float(p))
            total_y.append(y)

    return total_p, total_y


def build_calibration_set(manifest: dict, train_seasons: list[int]) -> dict:
    """Walk-forward across `train_seasons`, training each fold on the manifest
    window and predicting on the fold season; collect (raw_p, outcome) pairs.
    """
    cal_manifest = json_clone(manifest)
    cal_manifest["train_test"]["test_seasons"] = list(train_seasons)
    train_results = walk_forward(cal_manifest)
    total_std = train_results["residual_std_total"]

    feats = features_for(manifest["features"]["tiers"], manifest["features"].get("families"))
    from bjellac.strategies.strategy_01_lgbm_totals.features.build import load

    df = load().filter(
        pl.col("completed")
        & pl.col("target_total").is_not_null()
        & pl.col("market_total").is_not_null()
    )

    total_p_all: list[float] = []
    total_y_all: list[float] = []
    n_per_season: dict[int, dict] = {}

    for s in train_seasons:
        season_home = df.filter((pl.col("season") == s) & pl.col("is_home"))
        if season_home.is_empty():
            n_per_season[s] = {"total": 0, "skipped": "no_rows"}
            continue
        total_path = MODELS_DIR / f"total_through_{s - 1}.txt"
        if not total_path.exists():
            n_per_season[s] = {"total": 0, "skipped": "no_model_artifact"}
            continue
        total_model = lgb.Booster(model_file=str(total_path))
        X = season_home.select(feats).to_numpy()
        total_pred = total_model.predict(X)
        tp_, ty = _calibration_set_for_total(season_home, total_pred, total_std)
        total_p_all += tp_
        total_y_all += ty
        n_per_season[s] = {"total": len(tp_)}

    return {
        "total": (np.array(total_p_all), np.array(total_y_all)),
        "residual_std_total": total_std,
        "per_season": n_per_season,
        "train_seasons": list(train_seasons),
    }


def fit_and_save(manifest: dict | None = None) -> dict:
    """Fit the total calibrator and persist to CALIBRATORS_DIR."""
    manifest = manifest or _load_manifest()
    cal_cfg = manifest.get("calibration") or {}
    train_seasons = cal_cfg.get("train_seasons")
    if not train_seasons:
        raise ValueError(
            "manifest['calibration']['train_seasons'] is required to fit calibrators"
        )

    cal_set = build_calibration_set(manifest, train_seasons)

    CALIBRATORS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict] = {}
    raw_p, y = cal_set["total"]
    if raw_p.size == 0:
        out["total"] = {"market": "total", "n_train": 0, "skipped": "empty_calibration_set"}
        return out
    cal = IsotonicCalibrator().fit(raw_p, y)
    meta = {
        "market": "total",
        "n_train": int(raw_p.size),
        "train_seasons": cal_set["train_seasons"],
        "per_season": cal_set["per_season"],
        "residual_std": cal_set["residual_std_total"],
        "thresholds": cal.thresholds,
        "raw_hit_rate": float(np.mean(y)),
    }
    (CALIBRATORS_DIR / "total.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )
    out["total"] = meta
    return out


def fit_per_fold(manifest: dict | None = None) -> dict:
    """Fit one calibrator per fold year and persist per-fold artifacts.

    For each fold year S, the calibrator is fit on walk-forward predictions
    from seasons (train_seasons ∩ years < S). This keeps the calibrator
    strictly causal — fold S never sees its own outcomes.

    The set of fold years fit equals train_seasons ∪ test_seasons, so a single
    `bjellac strategy calibrate --per-fold` run produces calibrators for both
    the in-sample range and any forward test seasons the manifest will backtest.

    Persists to CALIBRATORS_DIR/fold_{year}/total.json.
    """
    manifest = manifest or _load_manifest()
    cal_cfg = manifest.get("calibration") or {}
    train_seasons = cal_cfg.get("train_seasons")
    if not train_seasons:
        raise ValueError(
            "manifest['calibration']['train_seasons'] is required to fit calibrators"
        )
    test_seasons = (manifest.get("train_test") or {}).get("test_seasons") or []
    fold_years = sorted(set(train_seasons) | set(test_seasons))

    all_results: dict[int, dict] = {}

    for fold_year in fold_years:
        fold_train = [s for s in train_seasons if s < fold_year]
        if len(fold_train) < 2:
            all_results[fold_year] = {"skipped": "insufficient_prior_seasons", "n_prior": len(fold_train)}
            continue

        fold_cal_set = build_calibration_set(manifest, fold_train)
        fold_dir = CALIBRATORS_DIR / f"fold_{fold_year}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        raw_p, y = fold_cal_set["total"]
        if raw_p.size == 0:
            all_results[fold_year] = {"total": {"market": "total", "n_train": 0, "skipped": "empty_calibration_set"}}
            continue
        cal = IsotonicCalibrator().fit(raw_p, y)
        meta = {
            "market": "total",
            "fold_year": fold_year,
            "n_train": int(raw_p.size),
            "train_seasons": fold_train,
            "residual_std": fold_cal_set["residual_std_total"],
            "thresholds": cal.thresholds,
            "raw_hit_rate": float(np.mean(y)),
        }
        (fold_dir / "total.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )
        all_results[fold_year] = {"total": meta}

    return all_results


def _load_json_calibrator(path: Path) -> IsotonicCalibrator | None:
    if not path.exists():
        return None
    thresholds = json.loads(path.read_text())["thresholds"]
    x = np.asarray(thresholds["x_thresholds"], dtype=float)
    y = np.asarray(thresholds["y_thresholds"], dtype=float)
    if (x.ndim != 1 or y.ndim != 1 or x.size == 0 or x.shape != y.shape
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0)
            or np.any(y < 0.001) or np.any(y > 0.999)):
        raise ValueError("Invalid calibration thresholds")
    return IsotonicCalibrator().fit(x, y)


def load(market: str) -> IsotonicCalibrator | None:
    """Load validated JSON thresholds; executable pickle artifacts are unsupported."""
    if market != "total":
        raise ValueError("Only the total market is supported")
    return _load_json_calibrator(CALIBRATORS_DIR / "total.json")


def load_fold(market: str, fold_year: int) -> IsotonicCalibrator | None:
    """Load JSON thresholds for a particular evaluation fold."""
    if market != "total" or type(fold_year) is not int:
        raise ValueError("Expected the total market and an integer fold year")
    return _load_json_calibrator(CALIBRATORS_DIR / f"fold_{fold_year}" / "total.json")


def _load_manifest() -> dict:
    import yaml

    return yaml.safe_load(MANIFEST_PATH.read_text())


def json_clone(d: dict) -> dict:
    """Deep-copy a manifest dict via JSON round-trip (manifests are JSON-safe)."""
    return json.loads(json.dumps(d))
