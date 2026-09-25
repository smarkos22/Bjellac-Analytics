"""Canonical team-name registry — exact or nothing, scoped forms only in scope."""
from __future__ import annotations

import polars as pl
import pytest

from bjellac.canonical import team_resolver as TR
from bjellac.canonical.team_resolver import Team, TeamRegistry, norm_key


def _t(uid, school, mascot, abbr, alts=(), cls="fbs"):
    return Team(team_uid=uid, school=school, mascot=mascot, abbreviation=abbr,
                alternate_names=tuple(alts), classification=cls)


@pytest.fixture
def reg():
    teams = [
        _t("cfbd:152", "NC State", "Wolfpack", "NCSU", ["North Carolina St.", "NCSU", "NC State"]),
        _t("cfbd:153", "North Carolina", "Tar Heels", "UNC", ["UNC", "North Carolina"]),
        _t("cfbd:2628", "TCU", "Horned Frogs", "TCU", ["TCU"]),
        _t("cfbd:23", "San José State", "Spartans", "SJSU", ["San Jose St.", "SJSU", "San José St"]),
        _t("cfbd:30", "USC", "Trojans", "USC", ["USC"]),
        _t("cfbd:62", "Hawai'i", "Rainbow Warriors", "HAW", ["HAW", "Hawai’i"]),
        _t("cfbd:16", "Sacramento State", "Hornets", "SAC", ["Cal State Sacramento", "SAC", "Sacramento St"]),
        _t("cfbd:2199", "Eastern Michigan", "Eagles", "EMU", ["EMU", "E Michigan"]),
        _t("cfbd:99", "LSU", "Tigers", "LSU", ["LSU"]),
        _t("cfbd:98", "Clemson", "Tigers", "CLEM", ["CLEM"]),
        _t("cfbd:333", "Alabama", "Crimson Tide", "ALA", ["ALA", "Bama"]),
        _t("cfbd:2390", "Miami", "Hurricanes", "MIA", ["Miami (FL)", "MIA", "Miami"]),
        _t("cfbd:193", "Miami (OH)", "RedHawks", "M-OH", ["M-OH", "Miami OH"]),
        _t("cfbd:2579", "South Carolina", "Gamecocks", "SC", ["SC", "South Carolina"]),
        _t("cfbd:9001", "Springfield", "Pumas", "SC", ["SC"], cls="iii"),
        _t("cfbd:2515", "St. Francis (PA)", "Red Flash", "SFPA", ["SFPA", "St Francis PA"], cls="fcs"),
    ]
    aliases = [{"source": "builtin", "source_name": k, "school": v}
               for k, v in TR.BUILTIN_ALIASES.items()]
    return TeamRegistry(teams, aliases)


# ------------------------------------------------------------ normalisation


def test_norm_key_is_deterministic_and_handles_the_known_spellings():
    assert norm_key("North Carolina St.") == "north carolina state"
    assert norm_key("N.C. State") == "nc state"
    assert norm_key("San José State") == norm_key("San Jose St") == "san jose state"
    assert norm_key("Hawai'i") == norm_key("Hawai’i") == norm_key("Hawaii") == "hawaii"
    assert norm_key("Texas A&M") == norm_key("Texas A & M") == "texas a and m"
    assert norm_key("St. Francis (PA)") == "saint francis pa", "a LEADING St is Saint"
    assert norm_key("University of Miami") == "miami"
    assert norm_key("The Ohio State University") == "ohio state"
    assert norm_key("") == "" and norm_key(None) == ""


# ------------------------------------------------------------ the trap this exists for


def test_nc_state_never_resolves_to_north_carolina(reg):
    for form in ("North Carolina St", "North Carolina St.", "North Carolina State",
                 "NC State", "NC St", "NCSU", "N.C. State"):
        r = reg.resolve(form)
        assert r.team_uid == "cfbd:152", (form, r)
    for form in ("North Carolina", "UNC"):
        assert reg.resolve(form).team_uid == "cfbd:153", form
    # and a near-miss is a MISS, never the neighbour
    assert reg.resolve("North Carolina Sta").team_uid is None
    assert reg.resolve("N Carolina").team_uid is None


def test_vip365_and_kalshi_spellings_resolve_exactly(reg):
    assert reg.resolve("San Jose St").team_uid == "cfbd:23"
    assert reg.resolve("San Jose St.").team_uid == "cfbd:23"
    assert reg.resolve("Hawaii").team_uid == "cfbd:62"
    assert reg.resolve("Sacramento St").team_uid == "cfbd:16"
    assert reg.resolve("Sac State").team_uid == "cfbd:16"
    assert reg.resolve("Miami (FL)").team_uid == "cfbd:2390"
    assert reg.resolve("Miami OH").team_uid == "cfbd:193"
    assert reg.resolve("Miami (Ohio)").team_uid == "cfbd:193"


def test_a_shared_school_name_is_never_broken_by_priority(reg):
    # Two schools whose SCHOOL keys collide would be ambiguous even if one is FBS.
    teams = [_t("a", "Union", "Bulldogs", "UN", cls="fbs"), _t("b", "Union", "Dutchmen", "UNI", cls="iii")]
    r = TeamRegistry(teams).resolve("Union")
    assert r.team_uid is None and r.method == "ambiguous" and set(r.candidates) == {"a", "b"}


def test_fbs_priority_applies_only_to_abbreviations_and_alternate_names(reg):
    r = reg.resolve("SC")
    assert r.team_uid == "cfbd:2579" and r.method.endswith("+fbs_priority")
    assert set(r.candidates) == {"cfbd:2579", "cfbd:9001"}


# ------------------------------------------------------------ scoped resolution


def test_mascots_resolve_only_inside_a_game_scope(reg):
    assert reg.resolve("Wolfpack").team_uid is None, "a mascot alone is not a global key"
    assert reg.resolve("Wolfpack", candidates=("cfbd:152", "cfbd:153")).team_uid == "cfbd:152"
    assert reg.side_for("Tar Heels", home_uid="cfbd:2628", away_uid="cfbd:153") == "away"
    assert reg.side_for("Horned Frogs", home_uid="cfbd:2628", away_uid="cfbd:153") == "home"


def test_an_ambiguous_mascot_stays_ambiguous_inside_scope(reg):
    r = reg.resolve("Tigers", candidates=("cfbd:99", "cfbd:98"))     # LSU vs Clemson
    assert r.team_uid is None and r.method == "ambiguous"
    assert reg.resolve("Tigers", candidates=("cfbd:99", "cfbd:333")).team_uid == "cfbd:99"


def test_scope_excludes_a_team_that_is_not_playing(reg):
    # "North Carolina St" on the UNC @ TCU game must NOT land on UNC.
    assert reg.side_for("North Carolina St", home_uid="cfbd:2628", away_uid="cfbd:153") is None
    assert reg.side_for("North Carolina", home_uid="cfbd:2628", away_uid="cfbd:153") == "away"


# ------------------------------------------------------------ aliases + review queue


def test_operator_aliases_and_pending_review_round_trip(tmp_path, reg, monkeypatch):
    monkeypatch.setattr(TR, "PENDING_REVIEW", tmp_path / "pending.parquet")
    monkeypatch.setattr(TR, "TEAM_ALIASES", tmp_path / "aliases.parquet")
    TR.set_registry(reg)
    try:
        assert reg.resolve("The Pack").team_uid is None
        assert TR.record_unresolved("sentiment", "The Pack", context="example:game") is True
        assert TR.record_unresolved("sentiment", "The Pack", context="again") is False, "dedupe"
        assert TR.pending_review(tmp_path / "pending.parquet").height == 1
        TR.add_alias("sentiment", "The Pack", "cfbd:152", path=tmp_path / "aliases.parquet")
        assert reg.resolve("The Pack").team_uid == "cfbd:152"
        assert TR.pending_review(tmp_path / "pending.parquet").height == 0, "cleared on alias"
        saved = pl.read_parquet(tmp_path / "aliases.parquet")
        assert saved.height == 1 and saved["team_uid"][0] == "cfbd:152"
    finally:
        TR.set_registry(None)


def test_snapshot_round_trip_preserves_every_key(reg):
    df = reg.to_frame()
    again = TeamRegistry.from_frame(df, [{"source": "builtin", "source_name": k, "school": v}
                                         for k, v in TR.BUILTIN_ALIASES.items()])
    for form, uid in (("NC St", "cfbd:152"), ("Hawaii", "cfbd:62"), ("EMU", "cfbd:2199")):
        assert again.resolve(form).team_uid == uid


def test_uid_for_school_is_exact_spelling_or_alternate_only(reg):
    assert reg.uid_for_school("NC State") == "cfbd:152"
    assert reg.uid_for_school("North Carolina St.") == "cfbd:152"
    assert reg.uid_for_school("Wolfpack") is None
