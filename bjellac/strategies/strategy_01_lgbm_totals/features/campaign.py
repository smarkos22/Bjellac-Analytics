"""Optional feature families for controlled ablation experiments.

Kept separate from tier1–4 modules so ablation can toggle each family independently.
Every column is prefixed with its family key; `FAMILY_COLUMNS` maps family → output columns
so `build(families=[...])` and the manifest can switch families on/off at family grain.

Implemented families:
  ADF  — drive efficiency / field position / ToP   (canonical.team_game_drives, rolled)
  BEGH — pace / tempo / tendency / downs / halves   (canonical.team_game_tempo, rolled)
  U    — fumble-recovery luck                        (canonical.team_game_turnover_luck, rolled)
  I    — roster experience (tenure)                  (canonical.team_roster_experience, season)
  Q    — QB-room concentration, season−1 lagged      (canonical.qb_concentration)
  DR   — NFL-draft attrition                         (canonical.draft_attrition, season)
  R    — poll rank / ranked-matchup                  (canonical.poll_ranks, season+week)
  L    — schedule-spot categoricals                  (derived from canonical.team_games)

Causality (NON-NEGOTIABLE, mirrors tier2/tier3): rolled families use a strictly-prior
EWMA (`ewm_mean(...).shift(1)` over (team_uid, season)); turnover luck uses a strictly-
prior cumulative rate; season families lag per their leakage class (I preseason-static →
season; Q → season−1; DR preseason-static → season; R = the in-effect poll for that week).
Opponent-adjustment (family J) and expected combined pace (family K) are cross-team and
live in a later pass, not here.
"""
from __future__ import annotations

import polars as pl
from bjellac.storage import sql_identifier

ALPHA = 0.13  # half-life ≈ 5 games — same as tier2

# --- rolled team-game efficiency columns (subset chosen for signal, not raw counts) ---
DRIVES_COLS = [
    "off_points_per_drive_clean", "def_points_per_drive_allowed_clean",
    "off_yards_per_drive", "def_yards_per_drive_allowed",
    "off_three_and_out_rate", "def_three_and_out_forced_rate",
    "off_available_yards_pct", "off_sec_per_play",
    "net_start_field_position",
]
TEMPO_COLS = [
    "off_scrimmage_plays", "off_dropback_rate", "off_early_down_dropback_rate",
    "off_success_rate_clean", "def_success_rate_allowed_clean",
    "off_explosiveness_clean", "off_ppa_per_play_clean", "def_ppa_per_play_allowed_clean",
    "off_sd_success_rate", "off_pd_success_rate", "off_second_half_adj",
]

FAMILY_COLUMNS: dict[str, list[str]] = {
    "ADF": [f"cADF_{c}" for c in DRIVES_COLS],
    "BEGH": [f"cBEGH_{c}" for c in TEMPO_COLS],
    "U": ["cU_off_fumble_keep_rate_todate", "cU_def_fumble_recovery_rate_todate"],
    "I": ["cI_avg_years_in_college", "cI_pct_veteran", "cI_ol_avg_weight",
          "cI_delta_years_in_college"],
    "Q": ["cQ_top_qb_pass_usage_prev", "cQ_qb_hhi_prev", "cQ_returning_starting_qb"],
    "DR": ["cDR_picks_lost", "cDR_first_round_picks_lost", "cDR_delta_picks_lost"],
    "R": ["cR_is_ranked", "cR_rank", "cR_opp_is_ranked", "cR_ranked_matchup",
          "cR_rank_advantage"],
    "L": ["cL_is_short_week", "cL_is_off_bye", "cL_tz_delta_from_home"],
    "J": ["cJ_off_ppa_vs_expected", "cJ_off_success_vs_expected", "cJ_def_ppa_vs_expected"],
    "K": ["cK_expected_total_plays", "cK_expected_game_pace"],
}

ALL_FAMILIES = list(FAMILY_COLUMNS.keys())

# Rolled team-side families whose columns should also be joined as opp_* (the opponent's
# rolled off/def tendencies) — same treatment tier2 gives its stats. Season families
# I/Q/DR/R/L already encode the opponent via internal delta/matchup columns, so they are
# NOT mirrored. J (opponent-adjusted realized) IS mirrored so the model sees both teams'
# adjusted output; K (expected combined pace) is symmetric across the two team rows of a
# game, so it is NOT mirrored.
MIRROR_FAMILIES = {"ADF", "BEGH", "U", "J"}


def mirror_columns(families: list[str]) -> list[str]:
    """Campaign columns (from the requested families) that build.py should mirror as opp_*."""
    out: list[str] = []
    for fam in families:
        if fam in MIRROR_FAMILIES:
            out.extend(FAMILY_COLUMNS[fam])
    return out


def _rolled_team_game(con, table: str, cols: list[str], prefix: str) -> pl.DataFrame:
    """Causal EWMA (shift(1)) of `cols` from a canonical team-game table, ordered by
    kickoff. Returns (team_uid, game_uid) + prefixed rolled columns."""
    table = sql_identifier(table)
    select_cols = ", ".join(f"src.{sql_identifier(c)}" for c in cols)
    sql = f"""
        SELECT tg.team_uid, tg.game_uid, tg.season, tg.kickoff_utc, {select_cols}
        FROM canonical.team_games tg
        LEFT JOIN canonical.{table} src
          ON src.team_uid = tg.team_uid AND src.game_uid = tg.game_uid
    """
    df = con.execute(sql).pl().sort(["team_uid", "season", "kickoff_utc", "game_uid"])
    rolled = [
        pl.col(c).ewm_mean(alpha=ALPHA, ignore_nulls=True, adjust=False)
        .over(["team_uid", "season"]).shift(1).over(["team_uid", "season"])
        .alias(f"{prefix}_{c}")
        for c in cols
    ]
    df = df.with_columns(rolled)
    return df.select(["team_uid", "game_uid"] + [f"{prefix}_{c}" for c in cols])


def _turnover_luck_todate(con) -> pl.DataFrame:
    """Strictly-prior cumulative fumble keep/recovery rates. Roll the COUNTS and divide
    the running sums (never average per-game rates — undefined on 0-fumble games)."""
    sql = """
        SELECT tg.team_uid, tg.game_uid, tg.season, tg.kickoff_utc,
               tl.off_fumbles, tl.off_fumbles_kept,
               tl.def_opp_fumbles, tl.def_fumbles_recovered
        FROM canonical.team_games tg
        LEFT JOIN canonical.team_game_turnover_luck tl
          ON tl.team_uid = tg.team_uid AND tl.game_uid = tg.game_uid
    """
    df = con.execute(sql).pl().sort(["team_uid", "season", "kickoff_utc", "game_uid"])
    grp = ["team_uid", "season"]
    df = df.with_columns([
        pl.col("off_fumbles").fill_null(0).cum_sum().over(grp).shift(1).over(grp).alias("_cum_fum"),
        pl.col("off_fumbles_kept").fill_null(0).cum_sum().over(grp).shift(1).over(grp).alias("_cum_kept"),
        pl.col("def_opp_fumbles").fill_null(0).cum_sum().over(grp).shift(1).over(grp).alias("_cum_dfum"),
        pl.col("def_fumbles_recovered").fill_null(0).cum_sum().over(grp).shift(1).over(grp).alias("_cum_drec"),
    ])
    df = df.with_columns([
        pl.when(pl.col("_cum_fum") > 0)
          .then(pl.col("_cum_kept") / pl.col("_cum_fum"))
          .alias("cU_off_fumble_keep_rate_todate"),
        pl.when(pl.col("_cum_dfum") > 0)
          .then(pl.col("_cum_drec") / pl.col("_cum_dfum"))
          .alias("cU_def_fumble_recovery_rate_todate"),
    ])
    return df.select(["team_uid", "game_uid",
                      "cU_off_fumble_keep_rate_todate", "cU_def_fumble_recovery_rate_todate"])


def _roster_experience(con) -> pl.DataFrame:
    """Family I — preseason-static tenure joined on (team_uid, season), + delta vs opp."""
    sql = """
        SELECT tg.team_uid, tg.game_uid,
               re.avg_years_in_college   AS cI_avg_years_in_college,
               re.pct_veteran            AS cI_pct_veteran,
               re.ol_avg_weight          AS cI_ol_avg_weight,
               re.avg_years_in_college - reo.avg_years_in_college AS cI_delta_years_in_college
        FROM canonical.team_games tg
        LEFT JOIN canonical.team_roster_experience re
          ON re.team_uid = tg.team_uid AND re.season = tg.season
        LEFT JOIN canonical.team_roster_experience reo
          ON reo.team_uid = tg.opponent_uid AND reo.season = tg.season
    """
    return con.execute(sql).pl()


def _qb_concentration(con) -> pl.DataFrame:
    """Family Q — season−1 QB concentration (causal) + returning-starter flag.

    returning_starting_qb: did season−1's top-usage QB appear on THIS season's roster
    (preseason-knowable via teams.roster id membership)."""
    sql = """
        SELECT tg.team_uid, tg.game_uid,
               q.top_qb_pass_usage       AS cQ_top_qb_pass_usage_prev,
               q.qb_pass_usage_hhi       AS cQ_qb_hhi_prev,
               CASE WHEN q.top_qb_id IS NULL THEN NULL
                    WHEN r.id IS NOT NULL THEN 1 ELSE 0 END AS cQ_returning_starting_qb
        FROM canonical.team_games tg
        LEFT JOIN canonical.qb_concentration q
          ON q.team_uid = tg.team_uid AND q.season = tg.season - 1
        LEFT JOIN canonical.teams t
          ON t.team_uid = tg.team_uid AND t.season = tg.season
        LEFT JOIN (SELECT DISTINCT team, season, id FROM teams.roster) r
          ON r.team = t.school AND r.season = tg.season AND r.id = q.top_qb_id
    """
    return con.execute(sql).pl()


def _draft_attrition(con) -> pl.DataFrame:
    """Family DR — preseason-static NFL-draft attrition into this season, + delta vs opp.
    Missing row = zero attrition (LEFT JOIN + fill 0)."""
    sql = """
        SELECT tg.team_uid, tg.game_uid,
               COALESCE(da.picks_lost, 0)             AS cDR_picks_lost,
               COALESCE(da.first_round_picks_lost, 0) AS cDR_first_round_picks_lost,
               COALESCE(da.picks_lost, 0) - COALESCE(dao.picks_lost, 0) AS cDR_delta_picks_lost
        FROM canonical.team_games tg
        LEFT JOIN canonical.draft_attrition da
          ON da.team_uid = tg.team_uid AND da.season = tg.season
        LEFT JOIN canonical.draft_attrition dao
          ON dao.team_uid = tg.opponent_uid AND dao.season = tg.season
    """
    return con.execute(sql).pl()


def _poll_rank(con) -> pl.DataFrame:
    """Family R — AP-poll rank in effect for this week (verified alignment), + matchup.
    Unranked → rank NULL, is_ranked 0. rank_advantage = opp_rank − team_rank with unranked
    treated as 30 (just outside the top 25) so a ranked team beats an unranked one."""
    sql = """
        WITH ap AS (
            SELECT team_uid, season, season_type, week, rank
            FROM canonical.poll_ranks WHERE poll = 'AP Top 25'
        )
        SELECT tg.team_uid, tg.game_uid,
               (a.rank IS NOT NULL)::int8 AS cR_is_ranked,
               a.rank                     AS cR_rank,
               (ao.rank IS NOT NULL)::int8 AS cR_opp_is_ranked,
               ((a.rank IS NOT NULL) AND (ao.rank IS NOT NULL))::int8 AS cR_ranked_matchup,
               COALESCE(ao.rank, 30) - COALESCE(a.rank, 30) AS cR_rank_advantage
        FROM canonical.team_games tg
        LEFT JOIN ap a  ON a.team_uid = tg.team_uid AND a.season = tg.season
                       AND a.season_type = tg.season_type AND a.week = tg.week
        LEFT JOIN ap ao ON ao.team_uid = tg.opponent_uid AND ao.season = tg.season
                       AND ao.season_type = tg.season_type AND ao.week = tg.week
    """
    return con.execute(sql).pl()


def _schedule_spots(con) -> pl.DataFrame:
    """Family L — rest-based schedule spots + body-clock tz delta. Structural, leakage-safe.
    rest_days from prior same-season kickoff; tz_delta = |game venue tz − home venue tz|."""
    sql = """
        SELECT tg.team_uid, tg.game_uid, tg.season, tg.kickoff_utc,
               vg.timezone AS game_tz, vh.timezone AS home_tz
        FROM canonical.team_games tg
        LEFT JOIN canonical.venues vg ON vg.venue_uid = tg.venue_uid
        LEFT JOIN canonical.teams th ON th.team_uid = tg.team_uid AND th.season = tg.season
        LEFT JOIN canonical.venues vh ON vh.cfbd_id = th.venue_cfbd_id
    """
    df = con.execute(sql).pl().sort(["team_uid", "season", "kickoff_utc", "game_uid"])
    df = df.with_columns(
        (pl.col("kickoff_utc") - pl.col("kickoff_utc").shift(1).over(["team_uid", "season"]))
        .dt.total_days().alias("_rest")
    )
    tz_offset = {  # US tz → UTC offset hours (DST-agnostic; body-clock delta only needs the gap)
        "America/New_York": -5, "America/Detroit": -5, "America/Indiana/Indianapolis": -5,
        "America/Chicago": -6, "America/Denver": -7, "America/Boise": -7,
        "America/Phoenix": -7, "America/Los_Angeles": -8,
    }
    df = df.with_columns([
        (pl.col("_rest") <= 5).cast(pl.Int8).alias("cL_is_short_week"),
        (pl.col("_rest") >= 12).cast(pl.Int8).alias("cL_is_off_bye"),
        pl.col("game_tz").replace_strict(tz_offset, default=None).alias("_goff"),
        pl.col("home_tz").replace_strict(tz_offset, default=None).alias("_hoff"),
    ])
    df = df.with_columns(
        (pl.col("_goff") - pl.col("_hoff")).abs().cast(pl.Int8).alias("cL_tz_delta_from_home")
    )
    return df.select(["team_uid", "game_uid", "cL_is_short_week", "cL_is_off_bye",
                      "cL_tz_delta_from_home"])


def _opponent_adjusted_and_pace(con) -> pl.DataFrame:
    """Families J (opponent-adjusted realized efficiency) + K (expected combined pace).

    THE subtle causality case (TIER1 §4.3). J adjusts a team's REALIZED output in each
    prior game by the OPPONENT's prior-games-to-date defensive baseline (rolling-vs-
    rolling), then EWMA-shifts that adjusted series so game G's feature uses only T's
    games before G. K sums both teams' to-date pace — a purely prior-games quantity.

    Mechanic: every team-game row carries the team's own EWMA-shift(1) to-date baselines.
    The OPPONENT's baseline "entering game G" is exactly the opponent's own row for G, so
    we self-join the frame on (opponent_uid ← team_uid, game_uid) to fetch it.
    """
    sql = """
        SELECT tg.team_uid, tg.opponent_uid, tg.game_uid, tg.season, tg.kickoff_utc,
               tp.off_ppa_per_play, tp.def_ppa_per_play_allowed,
               tp.off_success_rate, tp.def_success_rate_allowed,
               tp.off_scrimmage_plays,
               dr.off_sec_per_play
        FROM canonical.team_games tg
        LEFT JOIN canonical.team_game_tempo tp
          ON tp.team_uid = tg.team_uid AND tp.game_uid = tg.game_uid
        LEFT JOIN canonical.team_game_drives dr
          ON dr.team_uid = tg.team_uid AND dr.game_uid = tg.game_uid
    """
    df = con.execute(sql).pl().sort(["team_uid", "season", "kickoff_utc", "game_uid"])
    grp = ["team_uid", "season"]

    def _td(col):  # EWMA to-date, strictly prior (shift 1)
        return (pl.col(col).ewm_mean(alpha=ALPHA, ignore_nulls=True, adjust=False)
                .over(grp).shift(1).over(grp))

    # Each team's own to-date baselines entering each game.
    df = df.with_columns([
        _td("def_ppa_per_play_allowed").alias("_def_ppa_td"),
        _td("def_success_rate_allowed").alias("_def_succ_td"),
        _td("off_ppa_per_play").alias("_off_ppa_td"),
        _td("off_scrimmage_plays").alias("_off_plays_td"),
        _td("off_sec_per_play").alias("_off_sec_td"),
    ])

    # Opponent's to-date baselines entering the SAME game (self-join on the opponent's row).
    opp = df.select([
        pl.col("team_uid").alias("opponent_uid"), "game_uid",
        pl.col("_def_ppa_td").alias("_opp_def_ppa_td"),
        pl.col("_def_succ_td").alias("_opp_def_succ_td"),
        pl.col("_off_ppa_td").alias("_opp_off_ppa_td"),
        pl.col("_off_plays_td").alias("_opp_off_plays_td"),
        pl.col("_off_sec_td").alias("_opp_off_sec_td"),
    ])
    df = df.join(opp, on=["opponent_uid", "game_uid"], how="left")

    # J: per-game REALIZED adjusted vs the opponent's to-date expectation.
    df = df.with_columns([
        (pl.col("off_ppa_per_play") - pl.col("_opp_def_ppa_td")).alias("_adj_off_ppa"),
        (pl.col("off_success_rate") - pl.col("_opp_def_succ_td")).alias("_adj_off_succ"),
        # For defense: how much less PPA T allowed than O's offense typically produces
        # (negative = T's defense outperformed expectation).
        (pl.col("def_ppa_per_play_allowed") - pl.col("_opp_off_ppa_td")).alias("_adj_def_ppa"),
    ])
    # Then EWMA-shift the adjusted realized series → the causal feature. Re-sort first:
    # the opponent self-join above does not guarantee the frame stays kickoff-ordered.
    df = df.sort(["team_uid", "season", "kickoff_utc", "game_uid"])
    df = df.with_columns([
        _td("_adj_off_ppa").alias("cJ_off_ppa_vs_expected"),
        _td("_adj_off_succ").alias("cJ_off_success_vs_expected"),
        _td("_adj_def_ppa").alias("cJ_def_ppa_vs_expected"),
    ])

    # K: expected combined pace — team's to-date pace + opponent's to-date pace (both
    # already prior-games EWMA-shifted → causal, no further shift).
    df = df.with_columns([
        (pl.col("_off_plays_td") + pl.col("_opp_off_plays_td")).alias("cK_expected_total_plays"),
        ((pl.col("_off_sec_td") + pl.col("_opp_off_sec_td")) / 2).alias("cK_expected_game_pace"),
    ])
    return df.select(["team_uid", "game_uid"] + FAMILY_COLUMNS["J"] + FAMILY_COLUMNS["K"])


def build(con, families: list[str] | None = None) -> pl.DataFrame:
    """Return (team_uid, game_uid) + all requested families' columns.

    Cross-team families J (opponent-adjustment) and K (expected combined pace) are built
    at the merge layer in build.py, not here (they need both teams' rolled values).
    """
    families = list(families) if families is not None else ALL_FAMILIES
    base = con.execute("SELECT team_uid, game_uid FROM canonical.team_games").pl()

    joiners = {
        "ADF": lambda: _rolled_team_game(con, "team_game_drives", DRIVES_COLS, "cADF"),
        "BEGH": lambda: _rolled_team_game(con, "team_game_tempo", TEMPO_COLS, "cBEGH"),
        "U": lambda: _turnover_luck_todate(con),
        "I": lambda: _roster_experience(con),
        "Q": lambda: _qb_concentration(con),
        "DR": lambda: _draft_attrition(con),
        "R": lambda: _poll_rank(con),
        "L": lambda: _schedule_spots(con),
    }
    for fam in families:
        if fam in joiners:
            base = base.join(joiners[fam](), on=["team_uid", "game_uid"], how="left")

    # J and K share one computation (both need the opponent-row self-join).
    if "J" in families or "K" in families:
        jk = _opponent_adjusted_and_pace(con)
        keep = ["team_uid", "game_uid"]
        if "J" in families:
            keep += FAMILY_COLUMNS["J"]
        if "K" in families:
            keep += FAMILY_COLUMNS["K"]
        base = base.join(jk.select(keep), on=["team_uid", "game_uid"], how="left")
    return base
