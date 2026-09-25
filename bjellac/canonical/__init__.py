"""Canonical layer — shared normalized tables.

Read from `data/raw/` (and `data/staging/` via `bjellac.pipeline`), write to
`data/canonical/<table>.parquet` plus a DuckDB view in schema `canonical.*`.

This portfolio includes entity resolution and feature consumers; acquisition
and canonical table builders remain in the operational project.
"""
from __future__ import annotations

from pathlib import Path

from bjellac.config import DATA_DIR
from bjellac.storage import sql_identifier

CANONICAL_DIR: Path = DATA_DIR / "canonical"
ENTITY_MAP_DIR: Path = DATA_DIR / "entity_map"

# Operational ingestion builders are outside this source selection.
BUILDERS: dict[str, str] = {}


def ensure_dirs() -> None:
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    ENTITY_MAP_DIR.mkdir(parents=True, exist_ok=True)


def register_view(con, view_name: str, parquet_path) -> None:
    """Register a parquet file as `canonical.<view_name>`.

    DuckDB can't bind parameters inside `read_parquet()`; the path is
    interpolated as a single-quoted literal. Path strings here come from our
    own code (no user input), so escaping is handled by replacing single quotes.
    """
    view_name = sql_identifier(view_name, qualified=True)
    safe_path = str(parquet_path).replace("'", "''")
    con.execute(f"CREATE OR REPLACE VIEW canonical.{view_name} AS SELECT * FROM read_parquet('{safe_path}')")
