"""Top-level feature builder.

Reads canonical.* via DuckDB, applies registered tier modules per manifest, and
writes one parquet to `data/strategies/<name>/features/team_games.parquet`.

When Tier 2 (or higher) is enabled, opponent's features are joined as `opp_*`
via a self-join on (game_uid, opponent_uid).
"""
from __future__ import annotations

import polars as pl

from bjellac.db import connect
from bjellac.strategies.strategy_01_lgbm_totals import ARTIFACTS_DIR
from bjellac.strategies.strategy_01_lgbm_totals.features import (
    campaign,
    tier1_ratings,
    tier2_advanced,
    tier3_context,
    tier4_weather,
)

FEATURES_DIR = ARTIFACTS_DIR / "features"
OUTPUT = FEATURES_DIR / "team_games.parquet"


def build(tiers: list[int], families: list[str] | None = None) -> pl.DataFrame:
    if not isinstance(tiers, (list, tuple, set)) or not all(isinstance(t, int) for t in tiers):
        raise TypeError(
            f"build() expects tiers as list[int]; got {type(tiers).__name__}. "
            "If passing a manifest dict, use `tiers=manifest['features']['tiers']`."
        )
    families = list(families) if families else []
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    con = connect(read_only=True)
    try:
        df = tier1_ratings.build(con)

        if 2 in tiers:
            t2 = tier2_advanced.build(con)
            df = df.join(t2, on=["team_uid", "game_uid"], how="left")
            opp = t2.rename({c: f"opp_{c}" for c in t2.columns if c not in ("team_uid", "game_uid")})
            opp = opp.rename({"team_uid": "opponent_uid"})
            df = df.join(opp, on=["opponent_uid", "game_uid"], how="left")

        if 3 in tiers:
            t3 = tier3_context.build(con)
            df = df.join(t3, on=["team_uid", "game_uid"], how="left")

        if 4 in tiers:
            t4 = tier4_weather.build(con)
            df = df.join(t4, on="game_uid", how="left")

        # Campaign feature families (2026-07). Team-side join, then mirror the rolled
        # families as opp_* (same treatment tier2 gets); season families self-encode
        # the opponent via delta/matchup columns and are not mirrored.
        if families:
            camp = campaign.build(con, families=families)
            df = df.join(camp, on=["team_uid", "game_uid"], how="left")
            mirror = campaign.mirror_columns(families)
            if mirror:
                opp_c = camp.select(["team_uid", "game_uid"] + mirror).rename(
                    {c: f"opp_{c}" for c in mirror} | {"team_uid": "opponent_uid"})
                df = df.join(opp_c, on=["opponent_uid", "game_uid"], how="left")
    finally:
        con.close()

    from bjellac.storage import atomic_frame
    atomic_frame(df, OUTPUT)
    return df


def load() -> pl.DataFrame:
    return pl.read_parquet(OUTPUT)
