"""All charts: exploratory data analysis (EDA) and model results.

Charts are saved as PNG files in static/images/ so the website can show them.
"""
import matplotlib
matplotlib.use("Agg")  # draw to files, no pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid")


def _save(fig, images_dir, name):
    images_dir.mkdir(parents=True, exist_ok=True)
    path = images_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def print_data_summary(df):
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    print("Data types:\n", df.dtypes, "\n")
    print("Missing values:\n", df.isna().sum(), "\n")
    print("Statistics:\n", df.describe().round(2).T.to_string(), "\n")


def make_eda_charts(df, images_dir=config.IMAGES_DIR):
    """Create the six EDA charts. Returns the list of file paths."""
    paths = []

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[config.TARGET], bins=50, ax=ax, color="#2563eb")
    ax.set(title="Distribution of Median House Value", xlabel="Median house value ($)")
    paths.append(_save(fig, images_dir, "eda_target_distribution.png"))

    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[config.NUMERIC_FEATURES + [config.TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Between Numeric Features")
    paths.append(_save(fig, images_dir, "eda_correlation_heatmap.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["median_income"], df[config.TARGET], s=6, alpha=0.25, color="#7c3aed")
    ax.set(title="Median Income vs House Value",
           xlabel="Median income (tens of thousands of $)", ylabel="Median house value ($)")
    paths.append(_save(fig, images_dir, "eda_income_vs_value.png"))

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["longitude"], df["latitude"], c=df[config.TARGET],
                    cmap="viridis", s=6, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="Median house value ($)")
    ax.set(title="House Values by Location", xlabel="Longitude", ylabel="Latitude")
    paths.append(_save(fig, images_dir, "eda_location_map.png"))

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="ocean_proximity", y=config.TARGET,
                order=config.OCEAN_CATEGORIES, hue="ocean_proximity",
                hue_order=config.OCEAN_CATEGORIES, legend=False, ax=ax)
    ax.set(title="House Value by Ocean Proximity", xlabel="", ylabel="Median house value ($)")
    paths.append(_save(fig, images_dir, "eda_value_by_ocean.png"))

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, col in zip(axes.ravel(), config.NUMERIC_FEATURES):
        sns.histplot(df[col].dropna(), bins=40, ax=ax, color="#0891b2")
        ax.set(title=col, xlabel="", ylabel="")
    fig.suptitle("Feature Distributions")
    paths.append(_save(fig, images_dir, "eda_feature_distributions.png"))
    return paths


def plot_model_comparison(results, images_dir=config.IMAGES_DIR):
    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(axes, ["mae", "rmse", "r2"],
                              ["MAE ($) - lower is better", "RMSE ($) - lower is better",
                               "R2 - higher is better"]):
        values = [results[n]["test"][key] for n in names]
        ax.bar(names, values, color=["#94a3b8", "#2563eb"][: len(names)])
        ax.set_title(title)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:,.2f}" if key == "r2" else f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Model Comparison on the Test Set")
    return _save(fig, images_dir, "model_comparison.png")


def plot_actual_vs_predicted(y_true, y_pred, model_name, images_dir=config.IMAGES_DIR):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.3, color="#2563eb")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set(title=f"Actual vs Predicted ({model_name})",
           xlabel="Actual value ($)", ylabel="Predicted value ($)")
    ax.legend()
    return _save(fig, images_dir, "actual_vs_predicted.png")


def plot_feature_importance(pairs, images_dir=config.IMAGES_DIR):
    if not pairs:
        return None
    names, values = zip(*pairs[::-1])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, values, color="#7c3aed")
    ax.set(title="Top Feature Importances (best model)", xlabel="Importance")
    return _save(fig, images_dir, "feature_importance.png")
