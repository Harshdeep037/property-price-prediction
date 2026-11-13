"""Shared test data: a small RANDOM stand-in table (no internet needed).

It lets us test that the CODE works. It says nothing about real accuracy.
"""
import numpy as np
import pandas as pd
import pytest

from src import config


@pytest.fixture
def fake_df():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "longitude": rng.uniform(-124, -114, n),
        "latitude": rng.uniform(32, 42, n),
        "housing_median_age": rng.integers(1, 52, n),
        "total_rooms": rng.integers(500, 5000, n),
        "total_bedrooms": rng.integers(100, 1000, n).astype(float),
        "population": rng.integers(300, 3000, n),
        "households": rng.integers(100, 800, n),
        "median_income": rng.uniform(0.5, 12, n),
        "ocean_proximity": rng.choice(config.OCEAN_CATEGORIES, n),
    })
    df["median_house_value"] = (
        df["median_income"] * 30_000
        + df["housing_median_age"] * 500
        - (df["ocean_proximity"] == "INLAND") * 40_000
        + rng.normal(0, 20_000, n)
    ).clip(lower=15_000)
    df.loc[rng.random(n) < 0.05, "total_bedrooms"] = np.nan
    return df
