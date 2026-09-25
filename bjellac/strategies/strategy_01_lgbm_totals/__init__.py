"""Strategy 01 — LightGBM Totals (over/under, fractional Kelly).

Predict expected game total (home+away points) with LightGBM, compare to
over/under market lines, size positions with quarter-Kelly.

Training requires a user-supplied manifest and canonical data. These operational
artifacts are outside the public source selection.
"""
from __future__ import annotations

from pathlib import Path

from bjellac.config import DATA_DIR

NAME = "strategy_01_lgbm_totals"
ARTIFACTS_DIR: Path = DATA_DIR / "strategies" / NAME
MANIFEST_PATH: Path = ARTIFACTS_DIR / "manifest.yaml"
