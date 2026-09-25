import logging
import time
from pathlib import Path

import duckdb

from bjellac.config import DB_PATH

log = logging.getLogger(__name__)

SCHEMAS = [
    "ref",
    "calendar",
    "games",
    "drives",
    "plays",
    "teams",
    "players",
    "rankings",
    "ratings",
    "betting",
    "markets",
    "recruiting",
    "metrics",
    "stats",
    "adjusted",
    "draft",
    "catalog",
    "bets",  # Schema only; no account records are distributed.
]


def connect(
    read_only: bool = False,
    lock_retries: int = 5,
    lock_wait_s: float = 3.0,
    db_path: Path | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open the project DuckDB, retrying when another process holds the file lock.

    Retries use doubling backoff for transient database file locks. A writer's
    exclusive lock also blocks readers, so read-only connections retry too.
    Non-lock errors raise immediately.

    `db_path` overrides the default data/bjellac.duckdb so satellite DBs
    (film/film.duckdb) can share the retry behavior.
    """
    target = db_path or DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    wait = lock_wait_s
    for attempt in range(lock_retries + 1):
        try:
            return duckdb.connect(str(target), read_only=read_only)
        except duckdb.IOException as e:
            if "lock" not in str(e).lower() or attempt == lock_retries:
                raise
            log.warning(
                "DB locked by another process (attempt %d/%d) — retrying in %.0fs: %s",
                attempt + 1, lock_retries, wait, e,
            )
            time.sleep(wait)
            wait *= 2
    raise RuntimeError("unreachable")  # loop always returns or raises


def init_schemas(con: duckdb.DuckDBPyConnection | None = None):
    close_after = con is None
    if con is None:
        con = connect()
    try:
        for schema in SCHEMAS:
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        _init_catalog_tables(con)
    finally:
        if close_after:
            con.close()


def _init_catalog_tables(con: duckdb.DuckDBPyConnection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS catalog.ingestion_log (
            id TEXT PRIMARY KEY,
            source TEXT,
            endpoint TEXT,
            params TEXT,
            params_hash TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            status TEXT DEFAULT 'pending',
            row_count INTEGER,
            raw_file_path TEXT,
            error_message TEXT,
            duration_seconds DOUBLE
        )
    """)
