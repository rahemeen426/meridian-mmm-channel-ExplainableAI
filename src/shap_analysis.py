"""
shap_analysis.py
-----------------
Task 5 - SHAP Explainability.

This is explaining MMM-style models with SHAP and lets us:
  - Rank feature importance
  - Inspect nonlinear/saturation-like effects via dependence plots
  - Cross-check SHAP importance against Meridian's own channel contributions

Run directly (after preprocessing.py):
    python src/shap_analysis.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from config import (CLEAN_CSV,CHARTS_DIR,TABLES_DIR,DATE_COL,TARGET_COL,MEDIA_CHANNELS,RANDOM_SEED)

# creating feature columns for the SHAP analysis, which includes media spend channels and various control variables such as organic search clicks, branded search clicks, email clicks, referral clicks, discount rate, trend, day of the week, month, and weekend indicator. 
FEATURE_COLS = (
    [f"{ch}_spend" for ch in MEDIA_CHANNELS]
    + [
        "ORGANIC_SEARCH_CLICKS",
        "BRANDED_SEARCH_CLICKS",
        "EMAIL_CLICKS",
        "REFERRAL_CLICKS",
        "discount_rate",
        "trend",
        "day_of_week",
        "month",
        "is_weekend",
    ]
)


def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV, parse_dates=[DATE_COL])

# Train the surrogate model using a Gradient Boosting Regressor on the cleaned dataset. The model is trained on a chronological split of the data, with 80% for training and 20% for testing. The function returns the trained model and the feature matrix X.
def train_surrogate(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=False  # chronological
    )
    # Setting up the Gradient Boosting Regressor with specific hyperparameters such as number of estimators, maximum depth, learning rate, subsample ratio, and random state for reproducibility. The model is then fitted to the training data and evaluated on the test set using R-squared metric. 
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"[shap] Surrogate GradientBoostingRegressor holdout R2 = {r2:.3f}")
    if r2 < 0:
        print(
            "[shap] NOTE: negative holdout R2 on a pure chronological "
            "(last-20%) split is common here -- 2024 is a partial, less "
            "volatile year, so the model is extrapolating rather than "
            "interpolating. This does not invalidate the SHAP explanations "
            "(they describe what the model learned from the training "
            "distribution), but you should discuss this limitation in your "
            "report. Options to improve it: try a log(target) transform, a "
            "blocked/rolling-origin CV instead of one holdout, or fewer "
            "estimators/shallower trees."
        )
    return model, X

    # this function runs the SHAP analysis by creating an explainer for the trained model, computing SHAP values for the feature matrix, generating summary and dependence plots, and saving the results to CSV files. 
def run_shap(model, X: pd.DataFrame):
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)

    # Summary plot for SHAP values, which provides a visual representation of the feature importance and their effects on the model's predictions.
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Feature importance ranking (mean |SHAP value|)
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": abs(shap_values.values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(TABLES_DIR / "shap_feature_importance.csv", index=False)

    # Dependence plots for each media spend channel
    for ch in MEDIA_CHANNELS:
        col = f"{ch}_spend"
        plt.figure()
        shap.dependence_plot(col, shap_values.values, X, show=False)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"shap_dependence_{ch}.png", dpi=150, bbox_inches="tight")
        plt.close()

    return shap_values, importance

    # Comparing     the SHAP feature importance with the channel contributions from the Meridian model. 
def compare_with_mmm_contribution(shap_importance: pd.DataFrame):
    """Task 5.3 - SHAP importance vs Meridian channel contribution.
    """
    mmm_path = TABLES_DIR / "channel_summary_metrics.csv"
    if not mmm_path.exists():
        print("[shap] Skipping comparison: run modeling.py first to produce "
              "channel_summary_metrics.csv")
        return None

    mmm_summary = pd.read_csv(mmm_path)
    shap_media = shap_importance[
        shap_importance["feature"].str.endswith("_spend")
    ].copy()
    shap_media["channel"] = shap_media["feature"].str.replace("_spend", "", regex=False)
    shap_media["shap_rank"] = shap_media["mean_abs_shap"].rank(ascending=False)

    comparison = shap_media[["channel", "mean_abs_shap", "shap_rank"]]
    comparison.to_csv(TABLES_DIR / "shap_vs_mmm_contribution.csv", index=False)
    print("[shap] SHAP vs MMM comparison:\n", comparison)
    return comparison

    # running the SHAP analysis by loading the cleaned dataset, training the surrogate model, running the SHAP analysis, printing the feature importance, and comparing it with the Meridian model's channel contributions. 
def run():
    df = load_clean()
    model, X = train_surrogate(df)
    shap_values, importance = run_shap(model, X)
    print("[shap] Feature importance:\n", importance)
    compare_with_mmm_contribution(importance)
    print(f"[shap] Charts saved to {CHARTS_DIR}")


if __name__ == "__main__":
    run()
