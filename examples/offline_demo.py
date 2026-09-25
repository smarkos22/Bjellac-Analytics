"""Exercise the existing entity resolver, calibration, and sizing with toy inputs."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bjellac.canonical.team_resolver import Team, TeamRegistry
from bjellac.strategies.strategy_01_lgbm_totals.calibration import IsotonicCalibrator
from bjellac.strategies.strategy_01_lgbm_totals.kelly import kelly_stake


def main():
    registry = TeamRegistry([
        Team("demo:north", "North Example", "Owls", "NE", (), "fbs"),
        Team("demo:south", "South Example", "Owls", "SE", (), "fbs"),
    ], [])
    # Deliberately tiny toy calibration set: demonstrates the API, not accuracy.
    calibrator = IsotonicCalibrator().fit(
        [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        [0, 0, 1, 0, 1, 0, 1, 1],
    )
    probability = calibrator.transform(0.65)
    print(json.dumps({
        "data": "fictional; not a forecast or accuracy estimate",
        "exact_match": asdict(registry.resolve("North Example")),
        "ambiguous_mascot": asdict(registry.resolve("Owls")),
        "scoped_mascot": asdict(registry.resolve("Owls", candidates=["demo:north"])),
        "toy_calibrated_probability": probability,
        "hypothetical_sizing": asdict(kelly_stake(probability, side="over")),
    }, indent=2))


if __name__ == "__main__":
    main()
