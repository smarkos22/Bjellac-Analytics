"""Entity-id minting + alias parquet readers/writers.

Phase A: CFBD-only. Every UID is deterministic from CFBD source ids:

    team_uid  = "cfbd:" + str(cfbd_team_id)
    venue_uid = "cfbd:" + str(cfbd_venue_id)
    game_uid  = "cfbd:" + str(cfbd_game_id)

Future sources land in `team_aliases.parquet` / `venue_aliases.parquet` with
deterministic shared-id matches first; ambiguous candidates go to
`pending_review.parquet` for operator confirmation. Aliases are append-only with
`valid_from` / `valid_to` to handle conference moves and renames.
"""
from __future__ import annotations

import polars as pl

from bjellac.canonical import ENTITY_MAP_DIR, ensure_dirs


def mint_uid(source: str, source_id: int | str) -> str:
    if source_id is None:
        raise ValueError("source_id cannot be None when minting UID")
    return f"{source}:{source_id}"


def mint_team_uid(source_id: int | str, source: str = "cfbd") -> str:
    return mint_uid(source, source_id)


def mint_venue_uid(source_id: int | str, source: str = "cfbd") -> str:
    return mint_uid(source, source_id)


def mint_game_uid(source_id: int | str, source: str = "cfbd") -> str:
    return mint_uid(source, source_id)


# Schemas for the alias tables. Even when only CFBD is connected, files exist
# with the full schema so future sources slot in without migration.
_ALIAS_SCHEMA = {
    "source": pl.Utf8,
    "source_id": pl.Utf8,
    "source_name": pl.Utf8,
    "valid_from": pl.Date,
    "valid_to": pl.Date,
}

_TEAM_ALIAS_SCHEMA = {**_ALIAS_SCHEMA, "team_uid": pl.Utf8}
_VENUE_ALIAS_SCHEMA = {**_ALIAS_SCHEMA, "venue_uid": pl.Utf8}
_PENDING_REVIEW_SCHEMA = {
    "source": pl.Utf8,
    "source_id": pl.Utf8,
    "source_name": pl.Utf8,
    "candidate_uid": pl.Utf8,
    "score": pl.Float64,
    "observed_at": pl.Datetime,
}


def _empty(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame({k: [] for k in schema}, schema=schema)


def init_alias_files() -> None:
    """Create empty alias parquet files (with schema) if missing."""
    ensure_dirs()
    pairs = [
        ("team_aliases.parquet", _TEAM_ALIAS_SCHEMA),
        ("venue_aliases.parquet", _VENUE_ALIAS_SCHEMA),
        ("pending_review.parquet", _PENDING_REVIEW_SCHEMA),
    ]
    for name, schema in pairs:
        path = ENTITY_MAP_DIR / name
        if not path.exists():
            _empty(schema).write_parquet(path)


def read_team_aliases() -> pl.DataFrame:
    return pl.read_parquet(ENTITY_MAP_DIR / "team_aliases.parquet")


def read_venue_aliases() -> pl.DataFrame:
    return pl.read_parquet(ENTITY_MAP_DIR / "venue_aliases.parquet")
