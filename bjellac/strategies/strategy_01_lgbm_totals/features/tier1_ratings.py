"""Tier 1 features: ratings + market total line.

CAUSALITY GUARD — `ratings.{sp,fpi,srs}` from CFBD are END-OF-SEASON values, so
joining them on (team_uid, season) leaks the target. This module joins on
(team_uid, season - 1) so the prior is strictly the previous season's
finalized rating. In-season Elo signal comes from `home_pregame_elo` /
`away_pregame_elo` in `canonical.games`, which IS per-game and causal.

LINE COMPARATOR — The over/under line is NOT a model feature. It's attached
here purely as a comparator at backtest/live-bet time. We prefer opening lines
when CFBD has them (2021+), but fall back to closing lines. The `total_line_kind`
column flags which was used so metrics can be split by line type.

Sources joined per team-game:
  canonical.team_games
  canonical.ratings_team_season  (lagged season - 1)
  canonical.games               (home/away pregame Elo, kickoff-time)
  canonical.lines               (opening total preferred, closing fallback)

Output: one row per team-game with features + target `target_total`.
"""
from __future__ import annotations

import polars as pl


def build(con) -> pl.DataFrame:
    sql = """
        -- Pick one total line row per game_uid with deterministic provider
        -- preference. Prefer opening total when available; fall back to closing.
        WITH best_total AS (
            SELECT game_uid,
                   COALESCE(opening_total, closing_total) AS market_total,
                   provider AS total_provider,
                   CASE WHEN opening_total IS NOT NULL THEN 'open' ELSE 'close' END AS total_line_kind
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY game_uid
                           ORDER BY
                               CASE WHEN opening_total IS NOT NULL THEN 0 ELSE 1 END,
                               CASE provider
                                   WHEN 'Bovada' THEN 0
                                   WHEN 'DraftKings' THEN 1
                                   WHEN 'ESPN Bet' THEN 2
                                   WHEN 'consensus' THEN 3
                                   ELSE 9
                               END,
                               provider
                       ) AS rn
                FROM canonical.lines
                WHERE opening_total IS NOT NULL OR closing_total IS NOT NULL
            ) WHERE rn = 1
        )
        SELECT
            tg.team_uid, tg.opponent_uid, tg.game_uid,
            tg.season, tg.week, tg.season_type, tg.kickoff_utc,
            tg.is_home, tg.is_neutral, tg.completed,
            tg.points_for, tg.points_against,
            -- target
            (tg.points_for + tg.points_against) AS target_total,
            -- team ratings (LAGGED season - 1; end-of-season ratings -> strictly
            -- previous-season prior, no in-season leakage).
            r.sp_rating  AS team_sp,  r.sp_offense AS team_sp_off, r.sp_defense AS team_sp_def,
            r.fpi        AS team_fpi, r.srs        AS team_srs,
            -- opponent ratings (lagged)
            ro.sp_rating AS opp_sp,   ro.sp_offense AS opp_sp_off, ro.sp_defense AS opp_sp_def,
            ro.fpi       AS opp_fpi,  ro.srs        AS opp_srs,
            -- ratings deltas (team - opp)
            (r.sp_rating  - ro.sp_rating)  AS delta_sp,
            (r.fpi        - ro.fpi)        AS delta_fpi,
            (r.srs        - ro.srs)        AS delta_srs,
            -- in-season Elo: per-game pregame Elo from canonical.games (causal)
            CASE WHEN tg.is_home THEN g.home_pregame_elo ELSE g.away_pregame_elo END AS team_elo,
            CASE WHEN tg.is_home THEN g.away_pregame_elo ELSE g.home_pregame_elo END AS opp_elo,
            (CASE WHEN tg.is_home THEN g.home_pregame_elo ELSE g.away_pregame_elo END
             - CASE WHEN tg.is_home THEN g.away_pregame_elo ELSE g.home_pregame_elo END) AS delta_elo,
            -- market total line (for backtest comparison, NOT a feature)
            bt.market_total,
            bt.total_provider,
            bt.total_line_kind
        FROM canonical.team_games tg
        LEFT JOIN canonical.games g
            ON g.game_uid = tg.game_uid
        LEFT JOIN canonical.ratings_team_season r
            ON r.team_uid = tg.team_uid AND r.season = tg.season - 1
        LEFT JOIN canonical.ratings_team_season ro
            ON ro.team_uid = tg.opponent_uid AND ro.season = tg.season - 1
        LEFT JOIN best_total bt
            ON bt.game_uid = tg.game_uid
    """
    return con.execute(sql).pl()
