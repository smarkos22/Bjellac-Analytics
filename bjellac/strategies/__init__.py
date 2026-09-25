"""Bjellac strategy package.

Each subpackage is a single strategy. Code lives here (importable);
artifacts (manifest, features, models, predictions, backtests, notebooks)
live at `data/strategies/<name>/`.

A strategy must expose:
  - a `manifest.yaml` at `data/strategies/<name>/manifest.yaml`
  - a `cli` module with `register(subparsers)` so `bjellac strategy run`
    and `bjellac strategy backtest` can dispatch to it.
"""
from __future__ import annotations

import json
from pathlib import Path

from bjellac.config import DATA_DIR

STRATEGIES_DIR = DATA_DIR / "strategies"
PACKAGE_DIR = Path(__file__).resolve().parent
MANIFEST_SCHEMA_PATH = PACKAGE_DIR / "_manifest_schema.json"


def list_strategies() -> list[str]:
    if not STRATEGIES_DIR.exists():
        return []
    return sorted(p.name for p in STRATEGIES_DIR.iterdir() if (p / "manifest.yaml").exists())


def load_manifest_schema() -> dict:
    return json.loads(MANIFEST_SCHEMA_PATH.read_text())
