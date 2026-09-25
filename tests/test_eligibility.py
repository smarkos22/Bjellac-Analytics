"""Synthetic eligibility regressions covering both home and away teams.

A null opponent profile can be a normal pattern in historical features, so
eligibility must be checked independently of feature completeness.
"""
from __future__ import annotations

import duckdb
import polars as pl
import pytest

from bjellac.strategies.strategy_01_lgbm_totals import eligibility as elig


def _slate(rows):
    return pl.DataFrame(
        [{"game_uid": g, "team_uid": h, "opponent_uid": a} for g, h, a in rows],
        schema={"game_uid": pl.Utf8, "team_uid": pl.Utf8, "opponent_uid": pl.Utf8},
    )


# --------------------------------------------------------------- split_slate

def test_excludes_when_ineligible_team_is_home():
    slate = _slate([("g1", "new", "old"), ("g2", "old", "other")])
    keep, drop = elig.split_slate(slate, {"new"})
    assert keep["game_uid"].to_list() == ["g2"]
    assert drop["game_uid"].to_list() == ["g1"]


def test_excludes_when_ineligible_team_is_AWAY():
    """An ineligible away team must be excluded even if null features are allowed."""
    slate = _slate([("g1", "old", "new"), ("g2", "old", "other")])
    keep, drop = elig.split_slate(slate, {"new"})
    assert keep["game_uid"].to_list() == ["g2"]
    assert drop["game_uid"].to_list() == ["g1"], (
        "an ineligible AWAY team must exclude the game — this is the silent half"
    )


def test_excludes_when_both_sides_ineligible():
    slate = _slate([("g1", "new_a", "new_b"), ("g2", "old", "other")])
    keep, drop = elig.split_slate(slate, {"new_a", "new_b"})
    assert keep["game_uid"].to_list() == ["g2"]
    assert drop.height == 1


def test_empty_ineligible_set_is_a_noop_and_keeps_the_schema():
    slate = _slate([("g1", "a", "b")])
    keep, drop = elig.split_slate(slate, set())
    assert keep.height == 1
    assert drop.height == 0
    assert drop.columns == slate.columns


# ------------------------------------------------- first_year_fbs_teams (SQL)

@pytest.fixture()
def con():
    c = duckdb.connect(":memory:")
    c.execute("CREATE SCHEMA canonical")
    c.execute("""
        CREATE TABLE canonical.games (
            season BIGINT,
            home_team_uid VARCHAR, home_classification VARCHAR,
            away_team_uid VARCHAR, away_classification VARCHAR
        )
    """)
    return c


def _add(con, season, home, away, hc="fbs", ac="fbs"):
    con.execute("INSERT INTO canonical.games VALUES (?,?,?,?,?)",
                [season, home, hc, away, ac])


def test_first_year_team_is_detected_from_either_side(con):
    _add(con, 2025, "vet_a", "vet_b")
    _add(con, 2026, "vet_a", "vet_b")
    _add(con, 2026, "vet_a", "promoted")           # promoted appears only AWAY
    assert elig.first_year_fbs_teams(2026, con=con) == {"promoted"}


def test_returning_team_is_not_flagged(con):
    _add(con, 2025, "vet_a", "vet_b")
    _add(con, 2026, "vet_a", "vet_b")
    assert elig.first_year_fbs_teams(2026, con=con) == set()


def test_prior_season_FCS_appearance_does_not_count_as_fbs_history(con):
    """A team that played last season only as the FCS side is still first-year."""
    _add(con, 2025, "vet_a", "promoted", ac="fcs")
    _add(con, 2026, "promoted", "vet_a")
    assert elig.first_year_fbs_teams(2026, con=con) == {"promoted"}


def test_eligibility_is_read_from_the_schedule_not_feature_nullity(con):
    """An upstream outage must NOT shrink the slate.

    Sabotage: null every feature for a veteran team. Eligibility is computed
    from canonical.games, so the team stays eligible and the game keeps flowing
    to `preflight`, which is the loud path. Keying eligibility on nullity would
    silently drop it instead.
    """
    _add(con, 2025, "vet_a", "vet_b")
    _add(con, 2026, "vet_a", "vet_b")
    assert elig.first_year_fbs_teams(2026, con=con) == set()
    slate = _slate([("g1", "vet_a", "vet_b")])
    keep, drop = elig.split_slate(slate, set())
    assert keep.height == 1 and drop.height == 0


# ------------------------------------------------------------------- report

def test_report_records_what_was_excluded_and_why():
    slate = _slate([("g1", "new", "old"), ("g2", "old", "other")])
    _, drop = elig.split_slate(slate, {"new"})
    rep = elig.eligibility_report(2026, {"new"}, drop)
    assert rep["n_excluded"] == 1
    assert rep["season_compared_to"] == 2025
    assert rep["ineligible_teams"] == ["new"]
    assert rep["excluded_games"][0]["game_uid"] == "g1"
    assert rep["excluded_games"][0]["ineligible"] == ["new"]


def test_report_on_a_clean_slate_is_explicit_not_absent():
    slate = _slate([("g1", "a", "b")])
    _, drop = elig.split_slate(slate, set())
    rep = elig.eligibility_report(2026, set(), drop)
    assert rep["n_excluded"] == 0
    assert rep["excluded_games"] == []
    assert rep["ineligible_teams"] == []
