"""Model building, evaluation and comparison."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import build_preprocessor


def evaluate(y_true, y_pred) -> dict:
    """Return MAE, RMSE and R2 as plain Python floats."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_models(n_estimators: int = 200) -> dict:
    """The two regressors we compare (Linear Regression is the baseline)."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def make_pipeline(regressor) -> Pipeline:
    """Preprocessing + model in ONE object, so they are always used together."""
    return Pipeline([("preprocessor", build_preprocessor()), ("model", regressor)])


def train_and_compare(X_train, y_train, X_test, y_test, n_estimators=200, cv=5):
    """Train every model and compare them.

    * Cross-validation runs on the TRAINING data only (the preprocessing is
      re-fitted inside every fold, so there is no leakage).
    * The best model is chosen by lowest cross-validated RMSE, NOT by looking
      at the test set. The test set is used once, for the final report.

    Returns (results, fitted_pipelines, best_name).
    """
    results, fitted = {}, {}
    for name, regressor in build_models(n_estimators).items():
        pipe = make_pipeline(regressor)
        cv_scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=("neg_root_mean_squared_error", "r2"),
        )
        pipe.fit(X_train, y_train)
        results[name] = {
            "test": evaluate(y_test, pipe.predict(X_test)),
            "cv_rmse": float(-cv_scores["test_neg_root_mean_squared_error"].mean()),
            "cv_r2": float(cv_scores["test_r2"].mean()),
        }
        fitted[name] = pipe
    best_name = min(results, key=lambda n: results[n]["cv_rmse"])
    return results, fitted, best_name


def top_feature_importances(pipeline, top_n: int = 10):
    """(feature, importance) pairs if the model supports it, else an empty list."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return []
    names = pipeline.named_steps["preprocessor"].named_steps["columns"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    pairs = sorted(zip(names, model.feature_importances_), key=lambda p: p[1], reverse=True)
    return [(n, float(v)) for n, v in pairs[:top_n]]
