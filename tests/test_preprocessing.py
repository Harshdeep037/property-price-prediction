"""Tests for the preprocessing code.

NOTE: these tests use a small RANDOM stand-in table so they run without internet.
They check that the CODE works. They say nothing about real model accuracy.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.preprocessing import add_ratio_features, build_preprocessor, split_data


def test_ratio_features_and_zero_division():
    row = pd.DataFrame([{"total_rooms": 100, "total_bedrooms": 20, "population": 50, "households": 0}])
    out = add_ratio_features(row)
    assert np.isnan(out.loc[0, "rooms_per_household"])      # no crash, no inf
    assert out.loc[0, "bedrooms_per_room"] == pytest.approx(0.2)


def test_split_sizes_and_no_overlap(fake_df):
    X_train, X_test, y_train, y_test = split_data(fake_df)
    assert len(X_train) + len(X_test) == len(fake_df)
    assert len(X_test) == pytest.approx(0.2 * len(fake_df), abs=2)
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert config.TARGET not in X_train.columns


def test_pipeline_removes_nans_and_shapes(fake_df):
    X_train, X_test, _, _ = split_data(fake_df)
    pre = build_preprocessor().fit(X_train)
    out = pre.transform(X_test)
    assert not np.isnan(out).any()
    assert out.shape == (len(X_test), 11 + 5)               # 8 + 3 numeric, 5 one-hot


def test_imputer_uses_training_median_only(fake_df):
    """Leakage check: the fill value must come from the TRAIN set, not all data."""
    X_train, _, _, _ = split_data(fake_df)
    pre = build_preprocessor().fit(X_train)
    imputer = pre.named_steps["columns"].named_transformers_["num"].named_steps["imputer"]
    idx = (config.NUMERIC_FEATURES + ["rooms_per_household", "bedrooms_per_room",
                                       "population_per_household"]).index("total_bedrooms")
    assert imputer.statistics_[idx] == pytest.approx(X_train["total_bedrooms"].median())
    assert imputer.statistics_[idx] != pytest.approx(fake_df["total_bedrooms"].median())


def test_unknown_category_does_not_crash(fake_df):
    X_train, X_test, _, _ = split_data(fake_df)
    pre = build_preprocessor().fit(X_train)
    row = X_test.iloc[[0]].copy()
    row["ocean_proximity"] = "SOMEWHERE NEW"
    assert pre.transform(row).shape[0] == 1
