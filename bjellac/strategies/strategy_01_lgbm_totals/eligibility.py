"""Exclude first-year FBS teams from totals-model inference.

Eligibility comes from the schedule, not feature nullity. A missing opponent
profile can be a valid training pattern, so null checks alone cannot distinguish
an ineligible opponent from a temporary upstream data outage. Check both the
home and away team, and retain an explicit report of excluded games.
"""

from __future__ import annotations

import polars as pl

from bjellac.db import connect

__all__ = ["first_year_fbs_teams", "split_slate", "eligibility_report"]


def first_year_fbs_teams(season: int, con=None) -> set[str]:
    """team_uids playing FBS in `season` that played no FBS game in `season-1`.

    Both sides of every game are considered, so a promoted team is caught
    whether it is scheduled at home or away.
    """
    owned = con is None
    con = con or connect(read_only=True)
    try:
        rows = con.execute(
            """
            WITH fbs AS (
                SELECT season, home_team_uid AS team_uid FROM canonical.games
                 WHERE home_classification = 'fbs'
                UNION
                SELECT season, away_team_uid AS team_uid FROM canonical.games
                 WHERE away_classification = 'fbs'
            )
            SELECT DISTINCT c.team_uid
              FROM fbs c
             WHERE c.season = ?
               AND c.team_uid IS NOT NULL
               AND NOT EXISTS (
                   SELECT 1 FROM fbs p
                    WHERE p.season = ? AND p.team_uid = c.team_uid
               )
            """,
            [season, season - 1],
        ).fetchall()
    finally:
        if owned:
            con.close()
    return {r[0] for r in rows}


def split_slate(slate: pl.DataFrame,
                ineligible: set[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(eligible, excluded). A game is excluded when EITHER side is ineligible.

    The either-side test is the whole point — see the module docstring.
    """
    if not ineligible:
        return slate, slate.clear()
    mask = (pl.col("team_uid").is_in(list(ineligible))
            | pl.col("opponent_uid").is_in(list(ineligible)))
    return slate.filter(~mask), slate.filter(mask)


def eligibility_report(season: int, ineligible: set[str],
                       excluded: pl.DataFrame) -> dict:
    """What this pass refused to model, and why — recorded, never inferred.

    Written into meta.json so a reader can tell a deliberately unmodelled game
    from one that silently went missing.
    """
    games = []
    if excluded.height:
        have = [c for c in ("game_uid", "team_uid", "opponent_uid")
                if c in excluded.columns]
        for r in excluded.select(have).to_dicts():
            games.append({
                "game_uid": r.get("game_uid"),
                "ineligible": sorted(
                    {u for u in (r.get("team_uid"), r.get("opponent_uid"))
                     if u in ineligible}),
            })
    return {
        "rule": "no FBS game in the prior season (either side)",
        "season_compared_to": season - 1,
        "ineligible_teams": sorted(ineligible),
        "n_excluded": len(games),
        "excluded_games": games,
    }
