"""Security and portability contracts using only generated temporary artifacts."""
import json

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from bjellac.canonical import register_view
from bjellac.storage import atomic_frame, sql_identifier
from bjellac.strategies.strategy_01_lgbm_totals import calibration as C
from bjellac.strategies.strategy_01_lgbm_totals.models import train


def test_json_calibration_round_trip_preserves_predictions(tmp_path, monkeypatch):
    cal = C.IsotonicCalibrator().fit([0.1, 0.3, 0.5, 0.7, 0.9], [0, 1, 0, 1, 1])
    monkeypatch.setattr(C, 'CALIBRATORS_DIR', tmp_path)
    content = json.dumps({'thresholds': cal.thresholds})
    (tmp_path / 'total.json').write_text(content)
    fold = tmp_path / 'fold_2030'
    fold.mkdir()
    (fold / 'total.json').write_text(content)
    grid = np.linspace(-0.1, 1.1, 100)
    for restored in [C.load('total'), C.load_fold('total', 2030)]:
        np.testing.assert_allclose(restored.transform(grid), cal.transform(grid), atol=1e-12)


@pytest.mark.parametrize('thresholds', [
    {'x_thresholds': [], 'y_thresholds': []},
    {'x_thresholds': [0.1, 0.1], 'y_thresholds': [0.2, 0.3]},
    {'x_thresholds': [0.1, 0.2], 'y_thresholds': [0.3, 0.2]},
    {'x_thresholds': [0.1], 'y_thresholds': [float('nan')]},
])
def test_invalid_calibration_artifacts_are_rejected(tmp_path, thresholds):
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps({'thresholds': thresholds}))
    with pytest.raises(ValueError):
        C._load_json_calibrator(path)


def test_native_model_round_trip(tmp_path):
    frame = pl.DataFrame({'season': [2020] * 6, 'x': [1., 2., 3., 4., 5., 6.],
                          'target': [2., 4., 6., 8., 10., 12.]})
    model = train.fit_one(frame, 'target', {'objective': 'regression', 'n_estimators': 2,
                         'min_data_in_leaf': 1, 'num_threads': 1, 'verbosity': -1}, 1., ['x'])
    path = tmp_path / 'model.txt'
    model.save_model(str(path))
    restored = lgb.Booster(model_file=str(path))
    np.testing.assert_allclose(restored.predict(frame.select('x').to_numpy()),
                               model.predict(frame.select('x').to_numpy()))


def test_atomic_frame_replaces_complete_file(tmp_path):
    path = tmp_path / 'data.parquet'
    atomic_frame(pl.DataFrame({'x': [1]}), path)
    atomic_frame(pl.DataFrame({'x': [2, 3]}), path)
    assert pl.read_parquet(path)['x'].to_list() == [2, 3]
    assert list(tmp_path.glob('*.tmp')) == []


def test_sql_identifiers_reject_injected_statements():
    assert sql_identifier('canonical.example', qualified=True) == 'canonical.example'
    with pytest.raises(ValueError):
        register_view(None, 'example; DROP TABLE unrelated', 'unused')
