"""Walk-forward training for the total model.

Train on seasons < S, validate on season S, for S in test_seasons. The total
model dedups team-game rows to one-row-per-game (home rows only), since the
total target is symmetric (home+away points).
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from bjellac.strategies.strategy_01_lgbm_totals import ARTIFACTS_DIR

MODELS_DIR = ARTIFACTS_DIR / "models"

_TIER1_FEATURES = [
    "is_home", "is_neutral", "week",
    "team_sp", "team_sp_off", "team_sp_def", "team_fpi", "team_srs", "team_elo",
    "opp_sp", "opp_sp_off", "opp_sp_def", "opp_fpi", "opp_srs", "opp_elo",
    "delta_sp", "delta_fpi", "delta_srs", "delta_elo",
]

_TIER2_STAT_COLS = [
    "off_ppa_overall", "def_ppa_overall",
    "off_ppa_passing", "def_ppa_passing",
    "off_ppa_rushing", "def_ppa_rushing",
    "off_ppa", "def_ppa",
    "off_success_rate", "def_success_rate",
    "off_explosiveness", "def_explosiveness",
    "off_plays",
    "def_havoc_rate", "off_havoc_rate_against",
]
_TIER2_RETURNING_COLS = [
    "returning_pct_ppa", "returning_pct_passing_ppa",
    "returning_pct_rushing_ppa", "returning_pct_receiving_ppa",
    "returning_usage",
]
_TIER2_TEAM_FEATURES = [f"t2_{c}" for c in _TIER2_STAT_COLS] + _TIER2_RETURNING_COLS
_TIER2_OPP_FEATURES = [f"opp_t2_{c}" for c in _TIER2_STAT_COLS] + [f"opp_{c}" for c in _TIER2_RETURNING_COLS]

_TIER3_FEATURES = [
    "team_talent", "opp_talent", "delta_talent",
    "team_portal_net_stars", "team_portal_net_n",
    "opp_portal_net_stars", "opp_portal_net_n",
    "delta_portal_stars",
    "team_coach_tenure", "opp_coach_tenure", "delta_coach_tenure",
    "venue_dome", "venue_grass", "venue_elevation_ft",
    "rest_days", "travel_distance_miles",
    "hour_of_day_local",
]

_TIER4_FEATURES = [
    "temp_kickoff_f",
    "wind_dir_kickoff",
    "pressure_kickoff_mb",
    "weather_code_kickoff",
    "wind_max_mph",
    "wind_gust_max_mph",
    "precip_total_in",
    "snow_total_in",
    "humidity_avg_pct",
]


def _campaign_features(families: list[str]) -> list[str]:
    """Model feature columns for the requested campaign families: each family's team-side
    columns + opp_* mirrors for the rolled/adjusted families. Centralized in campaign.py
    so the model column list can't drift from the built columns."""
    from bjellac.strategies.strategy_01_lgbm_totals.features import campaign
    feats: list[str] = []
    for fam in families:
        feats += campaign.FAMILY_COLUMNS.get(fam, [])
    feats += [f"opp_{c}" for c in campaign.mirror_columns(families)]
    return feats


def features_for(tiers: list[int], families: list[str] | None = None) -> list[str]:
    feats = list(_TIER1_FEATURES)
    if 2 in tiers:
        feats += _TIER2_TEAM_FEATURES + _TIER2_OPP_FEATURES
    if 3 in tiers:
        feats += _TIER3_FEATURES
    if 4 in tiers:
        feats += _TIER4_FEATURES
    if families:
        feats += _campaign_features(families)
    return feats


FEATURES = _TIER1_FEATURES


def _row_weight(season: int, covid_weight: float) -> float:
    return covid_weight if season == 2020 else 1.0


def _to_np(df: pl.DataFrame, target: str, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df.select(feats).to_numpy()
    y = df[target].to_numpy()
    return X, y


def fit_one(
    df_train: pl.DataFrame,
    target: str,
    params: dict,
    covid_weight: float,
    feats: list[str],
) -> lgb.Booster:
    X, y = _to_np(df_train, target, feats)
    w = np.array([_row_weight(s, covid_weight) for s in df_train["season"].to_list()])
    train_set = lgb.Dataset(X, label=y, weight=w, feature_name=feats)
    p = dict(params)
    n_estimators = p.pop("n_estimators", 800)
    booster = lgb.train(
        p,
        train_set,
        num_boost_round=n_estimators,
    )
    return booster


def walk_forward(manifest: dict) -> dict:
    from bjellac.strategies.strategy_01_lgbm_totals.features.build import load

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    feats = features_for(manifest["features"]["tiers"], manifest["features"].get("families"))
    df = load()
    df = df.filter(
        pl.col("completed")
        & pl.col("target_total").is_not_null()
    )

    tt = manifest["train_test"]
    test_seasons = list(tt["test_seasons"])
    window = tt.get("training_window_years")
    explicit_train = tt.get("train_seasons")
    covid_weight = tt.get("covid_2020_sample_weight", 1.0)
    total_params = manifest["model"]["params"]["total"]

    results = {"per_season": {}, "features": feats}

    def _train_seasons_for(test_year: int) -> set[int]:
        if window:
            return set(range(test_year - window, test_year))
        return set(explicit_train or [])

    for s in test_seasons:
        train_seasons = _train_seasons_for(s)
        train_df = df.filter((pl.col("season") < s) & pl.col("season").is_in(list(train_seasons)))
        val_df = df.filter(pl.col("season") == s)

        # Total: dedup to one row per game (home rows only).
        total_train = train_df.filter(pl.col("is_home"))
        total_val = val_df.filter(pl.col("is_home"))
        total_model = fit_one(total_train, "target_total", total_params, covid_weight, feats)
        total_pred = total_model.predict(total_val.select(feats).to_numpy())
        total_err = np.array(total_val["target_total"].to_list()) - total_pred

        season_metrics = {
            "n_train": train_df.height,
            "n_val": val_df.height,
            "total_mae": float(np.abs(total_err).mean()),
            "total_rmse": float(np.sqrt((total_err ** 2).mean())),
        }
        # Market baseline
        total_val_with_line = total_val.filter(pl.col("market_total").is_not_null())
        if total_val_with_line.height:
            mte = np.array(total_val_with_line["target_total"].to_list()) - np.array(total_val_with_line["market_total"].to_list())
            season_metrics["market_total_mae"] = float(np.abs(mte).mean())
            season_metrics["market_total_rmse"] = float(np.sqrt((mte ** 2).mean()))
            season_metrics["market_total_n"] = total_val_with_line.height

        results["per_season"][s] = season_metrics

        total_model.save_model(str(MODELS_DIR / f"total_through_{s - 1}.txt"))

    # Aggregate residual std across all val seasons combined.
    all_total_resid = []
    for s in test_seasons:
        val_df = df.filter(pl.col("season") == s)
        total_model = lgb.Booster(model_file=str(MODELS_DIR / f"total_through_{s - 1}.txt"))
        total_val = val_df.filter(pl.col("is_home"))
        all_total_resid += list(np.array(total_val["target_total"].to_list()) - total_model.predict(total_val.select(feats).to_numpy()))

    results["residual_std_total"] = float(np.std(all_total_resid))

    (MODELS_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    return results
