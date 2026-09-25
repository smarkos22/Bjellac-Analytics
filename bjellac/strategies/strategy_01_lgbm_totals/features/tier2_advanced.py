"""Tier 2 features: in-season form via causal EWMA.

For each (team_uid, game_uid, season, week), produce EWMA-as-of features over the
team's prior games in this season (strict: games BEFORE this game), with a
fallback to last-season terminal EWMA + returning-production prior in early
season weeks. Blend via `as_of.blend_weights(week)`.

Stats sourced from canonical.ppa_team_game and canonical.advanced_team_game.

EWMA half-life ≈ 5 games (alpha ≈ 0.13). Aggressive enough to react to in-season
form, conservative enough not to over-fit early weeks.
"""
from __future__ import annotations

import polars as pl

from bjellac.strategies.strategy_01_lgbm_totals.features.as_of import blend_weights

ALPHA = 0.13  # half-life ≈ 5 games

STAT_COLUMNS = [
    "off_ppa_overall", "def_ppa_overall",
    "off_ppa_passing", "def_ppa_passing",
    "off_ppa_rushing", "def_ppa_rushing",
    "off_ppa", "def_ppa",
    "off_success_rate", "def_success_rate",
    "off_explosiveness", "def_explosiveness",
    "off_plays",
    "def_havoc_rate", "off_havoc_rate_against",
]


def _team_game_stats(con) -> pl.DataFrame:
    """One row per (team_uid, game_uid) joining PPA + advanced + kickoff order."""
    sql = """
        SELECT
            tg.team_uid, tg.game_uid, tg.season, tg.week, tg.kickoff_utc,
            ppa.off_ppa_overall, ppa.def_ppa_overall,
            ppa.off_ppa_passing, ppa.def_ppa_passing,
            ppa.off_ppa_rushing, ppa.def_ppa_rushing,
            adv.off_ppa, adv.def_ppa,
            adv.off_success_rate, adv.def_success_rate,
            adv.off_explosiveness, adv.def_explosiveness,
            adv.off_plays,
            adv.def_havoc_rate,
            adv.off_havoc_rate_against
        FROM canonical.team_games tg
        LEFT JOIN canonical.ppa_team_game ppa
          ON ppa.team_uid = tg.team_uid AND ppa.game_uid = tg.game_uid
        LEFT JOIN canonical.advanced_team_game adv
          ON adv.team_uid = tg.team_uid AND adv.game_uid = tg.game_uid
    """
    return con.execute(sql).pl()


def _causal_ewma(stats: pl.DataFrame) -> pl.DataFrame:
    """For each stat column compute ewm_mean over (team_uid, season) ordered by
    kickoff, then shift(1) so the value reflects games STRICTLY BEFORE the
    current game."""
    stats = stats.sort(["team_uid", "season", "kickoff_utc", "game_uid"])
    out_cols = []
    for col in STAT_COLUMNS:
        ewm = pl.col(col).ewm_mean(alpha=ALPHA, ignore_nulls=True, adjust=False).over(
            ["team_uid", "season"]
        )
        out_cols.append(ewm.shift(1).over(["team_uid", "season"]).alias(f"ewm_{col}"))
    return stats.with_columns(out_cols)


def _last_season_terminal(stats_with_ewm: pl.DataFrame) -> pl.DataFrame:
    """For each team-season, the FINAL ewm_* value (last game's terminal EWMA)
    becomes the prior for next season."""
    last = (
        stats_with_ewm
        .sort(["team_uid", "season", "kickoff_utc", "game_uid"])
        .group_by(["team_uid", "season"], maintain_order=True)
        .agg([
            # Use the actual full-season EWMA (no shift) — the ewm value AT the
            # last game of season S is the terminal we want to feed into S+1.
            # drop_nulls() before last(): ewm_mean emits null AT null input rows
            # even with ignore_nulls=True, so a team whose FINAL game of the
            # season has no stats rows (e.g. a bowl missing from ppa/advanced —
            # New Mexico 2025) would get a null terminal and lose its entire t2
            # block for next season's early weeks. Take the last non-null EWMA
            # value instead (= EWMA through the last game that has stats).
            pl.col(col).ewm_mean(alpha=ALPHA, ignore_nulls=True, adjust=False)
            .drop_nulls().last()
            .alias(f"prev_ewm_{col}")
            for col in STAT_COLUMNS
        ])
    )
    last = last.with_columns((pl.col("season") + 1).alias("season"))  # shift to next season
    return last


RETURNING_COLS = [
    "returning_pct_ppa", "returning_pct_passing_ppa",
    "returning_pct_rushing_ppa", "returning_pct_receiving_ppa",
    "returning_usage",
]


def _returning_production(con) -> pl.DataFrame:
    sql = "SELECT team_uid, season, " + ", ".join(RETURNING_COLS) + " FROM canonical.returning_production"
    return con.execute(sql).pl()


def build(con) -> pl.DataFrame:
    stats = _team_game_stats(con)
    stats_ewm = _causal_ewma(stats)
    prev_terminal = _last_season_terminal(stats)
    returning = _returning_production(con)

    df = stats_ewm.join(prev_terminal, on=["team_uid", "season"], how="left")
    df = df.join(returning, on=["team_uid", "season"], how="left")

    # Blend weights
    weight_rows = df["week"].to_list()
    prior_w = [blend_weights(w)[0] for w in weight_rows]
    cur_w = [blend_weights(w)[1] for w in weight_rows]
    df = df.with_columns([
        pl.Series("blend_prior_w", prior_w),
        pl.Series("blend_cur_w", cur_w),
    ])

    blended_cols = []
    for col in STAT_COLUMNS:
        ewm_col = pl.col(f"ewm_{col}")
        prev_col = pl.col(f"prev_ewm_{col}")
        # If current-season EWMA is null (very early game), fall back to prev-season terminal.
        cur_filled = pl.when(ewm_col.is_not_null()).then(ewm_col).otherwise(prev_col)
        blended = (
            pl.col("blend_prior_w") * prev_col + pl.col("blend_cur_w") * cur_filled
        )
        # If prev is null too, use whatever current we have.
        blended = pl.when(prev_col.is_null()).then(cur_filled).otherwise(blended)
        blended_cols.append(blended.alias(f"t2_{col}"))

    df = df.with_columns(blended_cols)

    feat_cols = [f"t2_{c}" for c in STAT_COLUMNS] + RETURNING_COLS
    return df.select(["team_uid", "game_uid"] + feat_cols)
