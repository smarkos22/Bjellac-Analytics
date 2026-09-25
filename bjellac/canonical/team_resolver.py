"""ONE registry every intake path resolves a team name against.

Source lines can come from multiple sportsbooks and an
exchange, splits from ActionNetwork, schedules and scores from CFBD, and public
sentiment from Reddit threads and X posts — each spelling teams its own way.
"North Carolina State", "NC State", "North Carolina St.", "NC St", "NCSU",
"Wolfpack", "the Pack" are one team; "North Carolina" is a DIFFERENT one that
plays the same weekend. A loose match books the wrong game. So:

  * The source of truth is CFBD's `ref.teams` (school + abbreviation +
    `alternateNames` + mascot), keyed by `team_uid = "cfbd:<id>"` — the same uid
    `canonical.games` carries on every row.
  * Matching is EXACT after a deterministic normalisation (`norm_key`). There is
    no edit distance, no similarity score, no "closest". A name either maps to
    exactly one team or it does not map.
  * Aliases are explicit and reviewable: `BUILTIN_ALIASES` here (seed forms the
    data does not carry, e.g. "Sac State"), `data/entity_map/team_aliases.parquet`
    (operator-added, append-only, per source), plus the two per-source tables
    that predate this module (`config/kalshi_team_aliases.json`,
    `data/entity_map/action_network_teams.csv`) folded in as sources.
  * Anything that does not resolve lands in `pending_review.parquet` with the
    context it was seen in. It is never guessed.
  * SCOPED resolution: when the caller already knows the two teams in play (a
    comment attributed to a game, a betslip leg on a known matchup), the
    candidate set is those two uids and the looser forms become safe — a mascot
    ("Wolfpack"), or the school on its own. "Tigers" is ambiguous league-wide
    and unambiguous inside LSU @ Alabama. Scoped forms are NEVER used globally.

Consumers can use the same registry for text attribution and market matching.
Public school names and provider identifiers below are reference vocabulary;
they are not account records or personal data.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from bjellac.canonical import ENTITY_MAP_DIR, ensure_dirs
from bjellac.canonical.entity_map import _PENDING_REVIEW_SCHEMA, _TEAM_ALIAS_SCHEMA, _empty
from bjellac.config import CONFIG_DIR

REGISTRY_SNAPSHOT = ENTITY_MAP_DIR / "team_registry.parquet"
TEAM_ALIASES = ENTITY_MAP_DIR / "team_aliases.parquet"
PENDING_REVIEW = ENTITY_MAP_DIR / "pending_review.parquet"
KALSHI_ALIASES = Path(CONFIG_DIR) / "kalshi_team_aliases.json"
ACTION_CROSSWALK = ENTITY_MAP_DIR / "action_network_teams.csv"

# Seed aliases the source data does not carry. Each is a form that appeared in
# real inputs (r/CFB week-1 threads, the VIP365 board, Kalshi titles) and maps
# to exactly one team. Keys are matched after `norm_key`; values are CFBD
# `school` names, spelled exactly. Append here or in team_aliases.parquet —
# never widen the matcher.
BUILTIN_ALIASES: dict[str, str] = {
    "Sac State": "Sacramento State",
    "Hawaii": "Hawai'i",
    "Ole Miss Rebels": "Ole Miss",
    "Miami FL": "Miami",
    "Miami (Fla.)": "Miami",
    "Miami Hurricanes": "Miami",
    "Miami (Ohio)": "Miami (OH)",
    "Miami OH": "Miami (OH)",
    "Miami Ohio": "Miami (OH)",
    "Miami RedHawks": "Miami (OH)",
    "UConn": "Connecticut",
    "Pitt": "Pittsburgh",
    "Southern Cal": "USC",
    "Southern California": "USC",
    "LSU Tigers": "LSU",
    "Ohio St": "Ohio State",
    "Penn St": "Penn State",
    "Mississippi State": "Mississippi State",
    "Miss State": "Mississippi State",
    "Miss St": "Mississippi State",
    "Louisiana": "Louisiana",
    "Louisiana Lafayette": "Louisiana",
    "UL Lafayette": "Louisiana",
    "ULM": "UL Monroe",
    "Louisiana Monroe": "UL Monroe",
    "UMass": "Massachusetts",
    "App State": "App State",
    "Appalachian State": "App State",
    "Texas A&M": "Texas A&M",
    "Texas AM": "Texas A&M",
    "Central Florida": "UCF",
    "Brigham Young": "BYU",
    "Southern Methodist": "SMU",
    "Texas Christian": "TCU",
    "Nevada Las Vegas": "UNLV",
    "Texas San Antonio": "UTSA",
    "Texas El Paso": "UTEP",
    "Florida International": "Florida International",
    "FIU": "Florida International",
    "Middle Tennessee State": "Middle Tennessee",
    "MTSU": "Middle Tennessee",
    "Western Kentucky": "Western Kentucky",
    "WKU": "Western Kentucky",
    "Jax State": "Jacksonville State",
    "Jacksonville St": "Jacksonville State",
    "North Carolina St": "NC State",
    "North Carolina State": "NC State",
    "NC St": "NC State",
    "N.C. State": "NC State",
    "San Jose St": "San José State",
    "San Jose State": "San José State",
    "Cal": "California",
    "Georgia Tech Yellow Jackets": "Georgia Tech",
    "Virginia Tech Hokies": "Virginia Tech",
    "Boston College Eagles": "Boston College",
    "Fresno St": "Fresno State",
    "Boise St": "Boise State",
    "Kansas St": "Kansas State",
    "Oklahoma St": "Oklahoma State",
    "Oregon St": "Oregon State",
    "Washington St": "Washington State",
    "Arizona St": "Arizona State",
    "Colorado St": "Colorado State",
    "Utah St": "Utah State",
    "San Diego St": "San Diego State",
    "Michigan St": "Michigan State",
    "Iowa St": "Iowa State",
    "Florida St": "Florida State",
    "Arkansas St": "Arkansas State",
    "Georgia St": "Georgia State",
    "Kent St": "Kent State",
    "Ball St": "Ball State",
    "Kennesaw St": "Kennesaw State",
    "Sam Houston St": "Sam Houston",
    "New Mexico St": "New Mexico State",
    "North Dakota St": "North Dakota State",
    "South Dakota St": "South Dakota State",
    "Montana St": "Montana State",
    "Delaware St": "Delaware State",
    "Missouri St": "Missouri State",
    "Portland St": "Portland State",
    "Idaho St": "Idaho State",
    "Weber St": "Weber State",
    "Illinois St": "Illinois State",
    "Youngstown St": "Youngstown State",
    "Murray St": "Murray State",
    "Tennessee St": "Tennessee State",
    "Alcorn St": "Alcorn State",
    "Jackson St": "Jackson State",
    "Norfolk St": "Norfolk State",
    "Morgan St": "Morgan State",
    "Tarleton St": "Tarleton State",
    "Cal Poly": "Cal Poly",
    "Southern Miss": "Southern Miss",
    "Southern Mississippi": "Southern Miss",
    "Coastal": "Coastal Carolina",
    "Texas St": "Texas State",
    "Wake": "Wake Forest",
    "GT": "Georgia Tech",
    "VT": "Virginia Tech",
    "UVA": "Virginia",
    "Hoos": "Virginia",
}

_APOS = re.compile(r"[’‘'`ʼ]")
_PUNCT = re.compile(r"[^a-z0-9 ]+")


def norm_key(name: str | None) -> str:
    """Deterministic normalisation for EXACT matching.

    Lowercase; strip accents ("José" -> "jose"), apostrophes ("Hawai'i" ->
    "hawaii") and periods ("N.C. State" -> "nc state"); "&" -> "and";
    other punctuation -> space; a trailing "st"/"st." ->
    "state" (the abbreviation every book uses) while a LEADING "st" -> "saint"
    (St. Francis, St. John's); "univ"/"university of" dropped; leading "the"
    dropped. Nothing here is a similarity measure — two names either produce
    the same key or they do not.
    """
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _APOS.sub("", s).lower().replace(".", "")   # N.C. -> NC, St. -> St, Fla. -> Fla
    s = s.replace("&", " and ")
    s = _PUNCT.sub(" ", s)
    toks = [t for t in s.split() if t]
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "the" and not out:
            i += 1
            continue
        if t in ("univ", "university"):
            # "University of Miami" -> "miami"; "Miami University" -> "miami"
            if i + 1 < len(toks) and toks[i + 1] == "of":
                i += 2
            else:
                i += 1
            continue
        if t == "st":
            t = "saint" if not out else "state"
        out.append(t)
        i += 1
    return " ".join(out)


@dataclass(frozen=True)
class Resolution:
    """The outcome of one lookup. `team_uid` is None unless the match is exact
    and unique; `candidates` lists the uids an ambiguous key hit so the
    reviewer can pick, and `method` says which layer answered."""
    query: str
    key: str
    team_uid: str | None
    method: str
    candidates: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.team_uid is not None


@dataclass
class Team:
    team_uid: str
    school: str
    mascot: str | None
    abbreviation: str | None
    alternate_names: tuple[str, ...]
    classification: str | None
    conference: str | None = None
    keys: dict[str, str] = field(default_factory=dict)   # key -> layer


# Layers in order of authority. A key that maps to ONE uid resolves; a key
# that maps to several is ambiguous unless the FBS-priority rule below applies.
_LAYER_SCHOOL = "school"
_LAYER_ABBR = "abbreviation"
_LAYER_ALT = "alternate_name"
_LAYER_ALIAS = "alias"
_LAYER_MASCOT = "mascot"                # scoped only
_LAYER_SCHOOL_MASCOT = "school_mascot"  # global (unique by construction)


class TeamRegistry:
    def __init__(self, teams: list[Team], aliases: list[dict] | None = None):
        self.teams: dict[str, Team] = {t.team_uid: t for t in teams}
        self.by_school: dict[str, str] = {}
        # global index: key -> {uid: layer}
        self._index: dict[str, dict[str, str]] = {}
        for t in teams:
            self.by_school[t.school] = t.team_uid
            self._add(t.team_uid, t.school, _LAYER_SCHOOL)
            if t.abbreviation:
                self._add(t.team_uid, t.abbreviation, _LAYER_ABBR)
            for alt in t.alternate_names:
                self._add(t.team_uid, alt, _LAYER_ALT)
            if t.mascot:
                self._add(t.team_uid, f"{t.school} {t.mascot}", _LAYER_SCHOOL_MASCOT)
        self.alias_rows: list[dict] = []
        for a in aliases or []:
            self.add_alias_row(a)

    # ------------------------------------------------------------ building

    def _add(self, uid: str, form: str, layer: str) -> None:
        k = norm_key(form)
        if not k:
            return
        slot = self._index.setdefault(k, {})
        # An earlier (more authoritative) layer for the same uid wins.
        slot.setdefault(uid, layer)
        self.teams[uid].keys.setdefault(k, layer)

    def add_alias_row(self, row: dict) -> bool:
        """`{source, source_name, team_uid}` (or `school` instead of uid)."""
        uid = row.get("team_uid")
        if not uid and row.get("school"):
            uid = self.by_school.get(row["school"])
        if not uid or uid not in self.teams:
            return False
        self._add(uid, row["source_name"], f"{_LAYER_ALIAS}:{row.get('source') or 'builtin'}")
        self.alias_rows.append({**row, "team_uid": uid})
        return True

    # ------------------------------------------------------------ lookups

    def uid_for_school(self, school: str | None) -> str | None:
        """Exact `school` spelling only — the form canonical.games carries."""
        if school is None:
            return None
        uid = self.by_school.get(school)
        if uid:
            return uid
        r = self.resolve(school)
        return r.team_uid if r.method in (_LAYER_SCHOOL, _LAYER_ALT) else None

    def display(self, uid: str | None) -> dict | None:
        t = self.teams.get(uid or "")
        if not t:
            return None
        return {"team_uid": t.team_uid, "school": t.school, "mascot": t.mascot,
                "abbreviation": t.abbreviation, "classification": t.classification}

    def resolve(self, name: str | None, *, candidates=None, source: str | None = None) -> Resolution:
        """Exact, unique, or nothing.

        Global (`candidates=None`): school / abbreviation / alternate name /
        alias / "school mascot" keys. A key several teams share resolves only
        when exactly one of them is FBS and the form is an abbreviation or
        alternate name (the FBS slate is what every source here describes;
        a D-III "SC" never appears on it). A shared SCHOOL key is never
        broken that way.

        Scoped (`candidates=[uid, uid]`): the same keys restricted to the
        candidates, plus each candidate's mascot — unique among the candidates
        or nothing.
        """
        key = norm_key(name)
        if not key:
            return Resolution(str(name or ""), key, None, "empty")
        cands = tuple(c for c in (candidates or ()) if c in self.teams)
        hits = dict(self._index.get(key, {}))
        if candidates is not None:
            hits = {u: layer for u, layer in hits.items() if u in cands}
            for u in cands:
                t = self.teams[u]
                if t.mascot and norm_key(t.mascot) == key:
                    hits.setdefault(u, _LAYER_MASCOT)
        if not hits:
            return Resolution(str(name), key, None, "unresolved")
        if len(hits) == 1:
            (uid, layer), = hits.items()
            return Resolution(str(name), key, uid, layer)
        # Ambiguous. Global only: FBS priority on non-school forms.
        if candidates is None:
            fbs = [u for u in hits if (self.teams[u].classification or "").lower() == "fbs"]
            layers = {hits[u] for u in hits}
            if len(fbs) == 1 and _LAYER_SCHOOL not in layers:
                return Resolution(str(name), key, fbs[0], f"{hits[fbs[0]]}+fbs_priority",
                                  tuple(sorted(hits)))
        return Resolution(str(name), key, None, "ambiguous", tuple(sorted(hits)))

    def side_for(self, name: str | None, home_uid: str, away_uid: str) -> str | None:
        """'home' / 'away' / None for a name scoped to one game."""
        r = self.resolve(name, candidates=(home_uid, away_uid))
        if r.team_uid == home_uid:
            return "home"
        if r.team_uid == away_uid:
            return "away"
        return None

    # ------------------------------------------------------------ persistence

    def to_frame(self) -> pl.DataFrame:
        rows = [{"team_uid": t.team_uid, "school": t.school, "mascot": t.mascot,
                 "abbreviation": t.abbreviation, "alternate_names": list(t.alternate_names),
                 "classification": t.classification, "conference": t.conference}
                for t in self.teams.values()]
        return pl.DataFrame(rows, schema={"team_uid": pl.Utf8, "school": pl.Utf8, "mascot": pl.Utf8,
                                          "abbreviation": pl.Utf8, "alternate_names": pl.List(pl.Utf8),
                                          "classification": pl.Utf8, "conference": pl.Utf8})

    @classmethod
    def from_frame(cls, df: pl.DataFrame, aliases: list[dict] | None = None) -> "TeamRegistry":
        teams = [Team(team_uid=r["team_uid"], school=r["school"], mascot=r.get("mascot"),
                      abbreviation=r.get("abbreviation"),
                      alternate_names=tuple(r.get("alternate_names") or ()),
                      classification=r.get("classification"), conference=r.get("conference"))
                 for r in df.to_dicts()]
        return cls(teams, aliases)


# ---------------------------------------------------------------- loading


def _teams_from_db(con, season: int | None = None) -> pl.DataFrame:
    if season is None:
        season = con.execute("SELECT max(season) FROM ref.teams").fetchone()[0]
    rows = con.execute(
        """SELECT id, school, mascot, abbreviation, alternateNames, classification, conference
           FROM ref.teams WHERE season = ? ORDER BY school""", [season]).fetchall()
    return pl.DataFrame(
        [{"team_uid": f"cfbd:{r[0]}", "school": r[1], "mascot": r[2], "abbreviation": r[3],
          "alternate_names": list(r[4] or []), "classification": r[5], "conference": r[6]}
         for r in rows],
        schema={"team_uid": pl.Utf8, "school": pl.Utf8, "mascot": pl.Utf8, "abbreviation": pl.Utf8,
                "alternate_names": pl.List(pl.Utf8), "classification": pl.Utf8, "conference": pl.Utf8})


def _external_aliases() -> list[dict]:
    """Every alias layer on disk, as `{source, source_name, team_uid|school}`."""
    out: list[dict] = [{"source": "builtin", "source_name": k, "school": v}
                       for k, v in BUILTIN_ALIASES.items()]
    if TEAM_ALIASES.exists():
        try:
            for r in pl.read_parquet(TEAM_ALIASES).to_dicts():
                if r.get("valid_to") is None or r["valid_to"] >= date.today():
                    out.append({"source": r["source"], "source_name": r["source_name"],
                                "team_uid": r["team_uid"]})
        except Exception:  # noqa: BLE001 — a bad alias file must not take the resolver down
            pass
    if KALSHI_ALIASES.exists():
        try:
            for k, v in json.loads(KALSHI_ALIASES.read_text()).items():
                if not k.startswith("_"):
                    out.append({"source": "kalshi", "source_name": k, "school": v})
        except (json.JSONDecodeError, OSError):
            pass
    if ACTION_CROSSWALK.exists():
        try:
            for r in pl.read_csv(ACTION_CROSSWALK).to_dicts():
                if r.get("team_uid") and r.get("action_location"):
                    out.append({"source": "action_network", "source_name": r["action_location"],
                                "team_uid": r["team_uid"]})
        except Exception:  # noqa: BLE001
            pass
    return out


def build_registry(con=None, season: int | None = None, write_snapshot: bool = True) -> TeamRegistry:
    """From the DB (opens a read-only connection if none given); writes the
    parquet snapshot consumers read without touching the DB."""
    from bjellac.db import connect
    close_after = con is None
    con = con or connect(read_only=True)
    try:
        df = _teams_from_db(con, season)
    finally:
        if close_after:
            con.close()
    if write_snapshot:
        ensure_dirs()
        tmp = REGISTRY_SNAPSHOT.with_suffix(".parquet.tmp")
        df.write_parquet(tmp)
        tmp.replace(REGISTRY_SNAPSHOT)
    return TeamRegistry.from_frame(df, _external_aliases())


_REGISTRY: TeamRegistry | None = None


def get_registry(refresh: bool = False) -> TeamRegistry:
    """Process-wide registry. Snapshot parquet first (no DB, works inside a
    Sunday write-lock), DB fallback (and writes the snapshot)."""
    global _REGISTRY
    if _REGISTRY is not None and not refresh:
        return _REGISTRY
    if REGISTRY_SNAPSHOT.exists() and not refresh:
        _REGISTRY = TeamRegistry.from_frame(pl.read_parquet(REGISTRY_SNAPSHOT), _external_aliases())
    else:
        _REGISTRY = build_registry()
    return _REGISTRY


def set_registry(reg: TeamRegistry | None) -> None:
    """Test hook."""
    global _REGISTRY
    _REGISTRY = reg


def resolve_team(name: str | None, *, candidates=None, source: str | None = None) -> Resolution:
    return get_registry().resolve(name, candidates=candidates, source=source)


# ---------------------------------------------------------------- review queue + aliases


def record_unresolved(source: str, name: str, context: str | None = None,
                      candidates: tuple[str, ...] = (), path: Path | None = None) -> bool:
    """Append one row to pending_review (dedupe on source+name). Returns True
    when a new row was written. Never raises into the caller's pipeline."""
    p = path or PENDING_REVIEW
    try:
        ensure_dirs()
        df = pl.read_parquet(p) if p.exists() else _empty(_PENDING_REVIEW_SCHEMA)
        key = norm_key(name)
        if not key:
            return False
        dup = df.filter((pl.col("source") == source) & (pl.col("source_name") == name))
        if dup.height:
            return False
        row = pl.DataFrame([{
            "source": source, "source_id": context, "source_name": name,
            "candidate_uid": ",".join(candidates) if candidates else None,
            "score": None, "observed_at": datetime.now(timezone.utc).replace(tzinfo=None),
        }], schema=_PENDING_REVIEW_SCHEMA)
        tmp = p.with_suffix(".parquet.tmp")
        pl.concat([df, row], how="diagonal_relaxed").write_parquet(tmp)
        tmp.replace(p)
        return True
    except Exception:  # noqa: BLE001
        return False


def add_alias(source: str, source_name: str, team_uid: str, valid_from: date | None = None,
              path: Path | None = None) -> None:
    """Append-only. Resolves immediately for the running process too."""
    p = path or TEAM_ALIASES
    ensure_dirs()
    df = pl.read_parquet(p) if p.exists() else _empty(_TEAM_ALIAS_SCHEMA)
    row = pl.DataFrame([{"source": source, "source_id": None, "source_name": source_name,
                         "valid_from": valid_from or date.today(), "valid_to": None,
                         "team_uid": team_uid}], schema=_TEAM_ALIAS_SCHEMA)
    tmp = p.with_suffix(".parquet.tmp")
    pl.concat([df, row], how="diagonal_relaxed").write_parquet(tmp)
    tmp.replace(p)
    if _REGISTRY is not None:
        _REGISTRY.add_alias_row({"source": source, "source_name": source_name, "team_uid": team_uid})
    # Drop it from the review queue if it was waiting there.
    if PENDING_REVIEW.exists():
        try:
            pr = pl.read_parquet(PENDING_REVIEW)
            pr = pr.filter(~((pl.col("source") == source) & (pl.col("source_name") == source_name)))
            pr.write_parquet(PENDING_REVIEW)
        except Exception:  # noqa: BLE001
            pass


def pending_review(path: Path | None = None) -> pl.DataFrame:
    p = path or PENDING_REVIEW
    return pl.read_parquet(p) if p.exists() else _empty(_PENDING_REVIEW_SCHEMA)
