"""Train, compare and save the models.

Run:   python train_model.py
Needs: data/housing.csv  (python -m src.data_loader downloads it)
"""
import argparse
import json
from datetime import datetime, timezone

import joblib

from src import config, visuals
from src.data_loader import load_dataset
from src.modeling import top_feature_importances, train_and_compare
from src.preprocessing import split_data


def main(skip_eda=False, data_path=config.DATA_PATH, models_dir=config.MODELS_DIR,
         images_dir=config.IMAGES_DIR, n_estimators=200, cv=5):
    df = load_dataset(data_path)
    visuals.print_data_summary(df)
    if not skip_eda:
        print("Creating EDA charts ...")
        visuals.make_eda_charts(df, images_dir)

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    print(f"Training models (with {cv}-fold cross-validation) - Random Forest can take a minute ...")
    results, fitted, best_name = train_and_compare(
        X_train, y_train, X_test, y_test, n_estimators=n_estimators, cv=cv)

    print("\n{:<20}{:>12}{:>12}{:>10}{:>14}".format("Model (test set)", "MAE ($)", "RMSE ($)", "R2", "CV RMSE ($)"))
    for name, r in results.items():
        t = r["test"]
        print("{:<20}{:>12,.0f}{:>12,.0f}{:>10.3f}{:>14,.0f}".format(
            name, t["mae"], t["rmse"], t["r2"], r["cv_rmse"]))
    print(f"\nBest model (lowest cross-validated RMSE on training data): {best_name}")

    best_pipeline = fitted[best_name]
    importances = top_feature_importances(best_pipeline)

    visuals.plot_model_comparison(results, images_dir)
    visuals.plot_actual_vs_predicted(y_test, best_pipeline.predict(X_test), best_name, images_dir)
    visuals.plot_feature_importance(importances, images_dir)

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, models_dir / "model.joblib")

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"rows": int(len(df)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "selection_criterion": "lowest cross-validated RMSE on the training set",
        "best_model": best_name,
        "models": results,
        "feature_importance": [{"feature": n, "importance": v} for n, v in importances],
        "feature_ranges": {
            c: {"min": float(X_train[c].min()), "max": float(X_train[c].max())}
            for c in config.NUMERIC_FEATURES
        },
    }
    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {models_dir / 'model.joblib'} and {models_dir / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the property price models.")
    parser.add_argument("--skip-eda", action="store_true", help="do not regenerate EDA charts")
    main(skip_eda=parser.parse_args().skip_eda)
