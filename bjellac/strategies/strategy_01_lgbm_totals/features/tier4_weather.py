"""Tier 4 features: game-grain weather from canonical.weather_game.

Reads the Open-Meteo-derived weather table (22 seasons, 2004-2025) and
produces per-game weather features. Kickoff-point values are prioritized
over game-window aggregates — they represent what a bettor would see in a
weather forecast before the game starts.

Dome games get all weather columns nulled (weather is irrelevant indoors).

Joins on game_uid (weather is game-level, not team-level). Both home and
away rows of the same game receive identical weather values.
"""
from __future__ import annotations

import polars as pl

KICKOFF_FEATURES = [
    "temp_kickoff_f",
    "wind_dir_kickoff",
    "pressure_kickoff_mb",
    "weather_code_kickoff",
]

WINDOW_FEATURES = [
    "wind_max_mph",
    "wind_gust_max_mph",
    "precip_total_in",
    "snow_total_in",
    "humidity_avg_pct",
]

ALL_WEATHER_FEATURES = KICKOFF_FEATURES + WINDOW_FEATURES


def build(con) -> pl.DataFrame:
    sql = """
        SELECT
            w.game_uid,
            w.temp_kickoff_f,
            w.wind_dir_kickoff,
            w.pressure_kickoff_mb,
            w.weather_code_kickoff,
            w.wind_max_mph,
            w.wind_gust_max_mph,
            w.precip_total_in,
            w.snow_total_in,
            w.humidity_avg_pct,
            v.dome AS _venue_dome
        FROM canonical.weather_game w
        LEFT JOIN canonical.games g ON g.game_uid = w.game_uid
        LEFT JOIN canonical.venues v ON v.venue_uid = g.venue_uid
    """
    df = con.execute(sql).pl()

    # Null all weather features for dome games.
    dome_mask = pl.col("_venue_dome") == True  # noqa: E712
    df = df.with_columns([
        pl.when(dome_mask).then(None).otherwise(pl.col(c)).alias(c)
        for c in ALL_WEATHER_FEATURES
    ])

    return df.select(["game_uid"] + ALL_WEATHER_FEATURES)
