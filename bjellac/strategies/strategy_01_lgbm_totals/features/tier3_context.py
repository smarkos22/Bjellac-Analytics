"""Tier 3 features: roster + context.

Per team-game adds:
  talent (lagged season - 1)              — 247 composite roster talent
  portal_net_stars                        — net stars gained via portal (lagged)
  coach_tenure_years                      — coach seasons at this school
  is_dome                                 — venue dome flag
  is_grass                                — venue surface flag (turf=0, grass=1)
  elevation_ft                            — venue elevation in feet
  rest_days                               — days since this team's previous game
  travel_distance_miles                   — haversine miles from THIS team's home
                                            venue to the game venue (0 if home)
  hour_of_day_local                       — kickoff hour in the venue's local tz
                                            (noon vs prime-time effect)
Strict causality:
  - talent and portal_net are LAGGED to (season - 1) for the same reason as
    Tier 1 ratings: end-of-season composites might leak; safer to use prior.
  - coach_tenure_years is computed from canonical.coach_team_season for SEASON
    of the game (knowable pre-kickoff).
  - rest_days, travel: derived from canonical.team_games / venues at-or-before
    the current game.
  - hour_of_day_local, is_grass: venue/schedule attributes — fixed pre-kickoff.
"""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

import polars as pl


def _team_venues(con) -> pl.DataFrame:
    """One row per team-season with the team's primary venue lat/lng."""
    return con.execute("""
        SELECT t.team_uid, t.season,
               v.lat AS team_home_lat, v.lng AS team_home_lng
        FROM canonical.teams t
        LEFT JOIN canonical.venues v
          ON v.cfbd_id = t.venue_cfbd_id
    """).pl()


def _haversine_miles(lat1, lng1, lat2, lng2):
    if lat1 is None or lat2 is None or lng1 is None or lng2 is None:
        return None
    R = 3958.8
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlng / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def build(con) -> pl.DataFrame:
    sql = """
        SELECT
            tg.team_uid, tg.opponent_uid, tg.game_uid, tg.season, tg.week,
            tg.kickoff_utc, tg.is_home, tg.venue_uid,
            tal.talent                  AS team_talent,
            tal_o.talent                AS opp_talent,
            por.portal_net_stars        AS team_portal_net_stars,
            por.portal_net_n            AS team_portal_net_n,
            por_o.portal_net_stars      AS opp_portal_net_stars,
            por_o.portal_net_n          AS opp_portal_net_n,
            ct.tenure_years             AS team_coach_tenure,
            ct_o.tenure_years           AS opp_coach_tenure,
            v.dome                      AS venue_dome,
            v.grass                     AS venue_grass,
            v.elevation_ft              AS venue_elevation_ft,
            v.timezone                  AS venue_timezone,
        FROM canonical.team_games tg
        LEFT JOIN canonical.talent tal
          ON tal.team_uid = tg.team_uid AND tal.season = tg.season - 1
        LEFT JOIN canonical.talent tal_o
          ON tal_o.team_uid = tg.opponent_uid AND tal_o.season = tg.season - 1
        LEFT JOIN canonical.portal_net por
          ON por.team_uid = tg.team_uid AND por.season = tg.season - 1
        LEFT JOIN canonical.portal_net por_o
          ON por_o.team_uid = tg.opponent_uid AND por_o.season = tg.season - 1
        LEFT JOIN canonical.coach_team_season ct
          ON ct.team_uid = tg.team_uid AND ct.season = tg.season
        LEFT JOIN canonical.coach_team_season ct_o
          ON ct_o.team_uid = tg.opponent_uid AND ct_o.season = tg.season
        LEFT JOIN canonical.venues v
          ON v.venue_uid = tg.venue_uid
    """
    df = con.execute(sql).pl()

    # Talent gap is informative.
    df = df.with_columns([
        (pl.col("team_talent") - pl.col("opp_talent")).alias("delta_talent"),
        (pl.col("team_portal_net_stars") - pl.col("opp_portal_net_stars")).alias("delta_portal_stars"),
        (pl.col("team_coach_tenure") - pl.col("opp_coach_tenure")).alias("delta_coach_tenure"),
    ])

    # Rest days: days since this team's previous game (NULL for season opener).
    df = df.sort(["team_uid", "season", "kickoff_utc", "game_uid"])
    df = df.with_columns(
        ((pl.col("kickoff_utc") - pl.col("kickoff_utc").shift(1).over(["team_uid", "season"]))
         .dt.total_days()
         .alias("rest_days"))
    )

    # Travel distance: haversine from this team's primary home venue to the game venue.
    team_home = _team_venues(con)
    venue_loc = con.execute(
        "SELECT venue_uid, lat AS game_lat, lng AS game_lng FROM canonical.venues"
    ).pl()
    df = df.join(team_home, on=["team_uid", "season"], how="left")
    df = df.join(venue_loc, on="venue_uid", how="left")

    travel = []
    for r in df.iter_rows(named=True):
        if r["is_home"]:
            travel.append(0.0)
            continue
        d = _haversine_miles(r["team_home_lat"], r["team_home_lng"], r["game_lat"], r["game_lng"])
        travel.append(d)
    df = df.with_columns(pl.Series("travel_distance_miles", travel))

    # Cast booleans to int8 for LightGBM friendliness.
    df = df.with_columns([
        pl.col("venue_dome").cast(pl.Int8),
        pl.col("venue_grass").cast(pl.Int8),
    ])

    # hour_of_day_local: convert kickoff_utc to venue-local time, take hour.
    # Polars can't apply per-row timezone conversion natively, so build it
    # group-wise: split rows by venue_timezone, convert each group, recombine.
    parts = []
    for tz, sub in df.group_by("venue_timezone", maintain_order=True):
        tz_val = tz[0] if isinstance(tz, tuple) else tz
        if tz_val is None or sub.height == 0:
            sub = sub.with_columns(pl.lit(None).cast(pl.Int8).alias("hour_of_day_local"))
        else:
            sub = sub.with_columns(
                pl.col("kickoff_utc")
                  .dt.replace_time_zone("UTC")
                  .dt.convert_time_zone(tz_val)
                  .dt.hour()
                  .cast(pl.Int8)
                  .alias("hour_of_day_local")
            )
        parts.append(sub)
    df = pl.concat(parts) if parts else df.with_columns(
        pl.lit(None).cast(pl.Int8).alias("hour_of_day_local")
    )

    feat_cols = [
        "team_talent", "opp_talent", "delta_talent",
        "team_portal_net_stars", "team_portal_net_n",
        "opp_portal_net_stars", "opp_portal_net_n",
        "delta_portal_stars",
        "team_coach_tenure", "opp_coach_tenure", "delta_coach_tenure",
        "venue_dome", "venue_grass", "venue_elevation_ft",
        "rest_days", "travel_distance_miles",
        "hour_of_day_local",
    ]
    return df.select(["team_uid", "game_uid"] + feat_cols)
