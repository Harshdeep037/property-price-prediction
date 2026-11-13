"""Tests for training, saving and loading. Uses the random stand-in table,
so they check that the CODE works - not real-world accuracy."""
import json
import math

import joblib
import numpy as np
import pytest

from src import config
from src.modeling import evaluate, train_and_compare
from src.preprocessing import split_data
from train_model import main


def test_evaluate_known_values():
    m = evaluate(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["rmse"] == pytest.approx(math.sqrt(4 / 3))
    assert m["r2"] == pytest.approx(-1.0)


def test_train_and_compare_returns_both_models(fake_df):
    X_train, X_test, y_train, y_test = split_data(fake_df)
    results, fitted, best = train_and_compare(X_train, y_train, X_test, y_test, n_estimators=10, cv=2)
    assert set(results) == {"Linear Regression", "Random Forest"}
    assert best in results
    for r in results.values():
        assert all(math.isfinite(v) for v in r["test"].values())
    # the chosen model really has the lowest CV error
    assert results[best]["cv_rmse"] == min(r["cv_rmse"] for r in results.values())


def test_end_to_end_saves_and_reloads(fake_df, tmp_path):
    csv = tmp_path / "housing.csv"
    fake_df.to_csv(csv, index=False)
    metrics = main(data_path=csv, models_dir=tmp_path / "models",
                   images_dir=tmp_path / "img", n_estimators=10, cv=2)

    saved = json.loads((tmp_path / "models" / "metrics.json").read_text())
    assert saved["best_model"] == metrics["best_model"]
    assert (tmp_path / "img" / "model_comparison.png").exists()
    assert (tmp_path / "img" / "eda_location_map.png").exists()

    model = joblib.load(tmp_path / "models" / "model.joblib")
    one_row = fake_df.drop(columns=[config.TARGET]).iloc[[0]]   # raw, unprocessed input
    prediction = model.predict(one_row)
    assert prediction.shape == (1,) and np.isfinite(prediction).all()
