"""Preprocessing: feature engineering, missing values, scaling, encoding, split.

Everything that transforms data lives inside ONE scikit-learn Pipeline.
That pipeline is fitted on the training set only (no data leakage) and is saved
together with the model, so the Flask app applies exactly the same steps.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config

ENGINEERED_FEATURES = [
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]


def add_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create three ratio features. Division by zero becomes NaN (then imputed)."""
    X = X.copy()
    households = X["households"].replace(0, np.nan)
    total_rooms = X["total_rooms"].replace(0, np.nan)
    X["rooms_per_household"] = X["total_rooms"] / households
    X["bedrooms_per_room"] = X["total_bedrooms"] / total_rooms
    X["population_per_household"] = X["population"] / households
    return X


def build_preprocessor() -> Pipeline:
    """Return an UNFITTED preprocessing pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_FEATURES + ENGINEERED_FEATURES),
            (
                "cat",
                OneHotEncoder(categories=[config.OCEAN_CATEGORIES], handle_unknown="ignore"),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("ratios", FunctionTransformer(add_ratio_features, validate=False)),
            ("columns", column_transformer),
        ]
    )


def split_data(df: pd.DataFrame):
    """Split into train/test. Stratified by income group so both sets look alike.

    The split happens BEFORE any fitting, which is what prevents data leakage.
    """
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    income_group = pd.cut(
        df["median_income"], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=False
    )
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=income_group,
    )
