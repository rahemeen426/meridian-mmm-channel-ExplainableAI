"""
eda.py
------
Task 1 — Exploratory Data Analysis.

Produces:
  - Missing value inventory
  - Daily purchases plot
  - Media spend trend plot
  - Correlation heatmap
  - Spend concentration by channel (share of total)

Run directly (after preprocessing.py):
    python src/eda.py
"""

import matplotlib

matplotlib.use("Agg")  # safe for headless / VS Code "Run Python File"
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import CLEAN_CSV, CHARTS_DIR, TABLES_DIR, DATE_COL, TARGET_COL, MEDIA_SPEND_COLUMNS

sns.set_theme(style="whitegrid")

# This function is used by the EDA notebook and the eda.py script to load the cleaned dataset.
def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV, parse_dates=[DATE_COL])

# This function is used by the EDA notebook and the eda.py script to generate a missing value report. If miss is greater than 0, it is saved to TABLES_DIR / "missing_values.csv" and printed to the console.
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0].to_frame("missing_count")
    miss["missing_pct"] = (miss["missing_count"] / len(df) * 100).round(2)
    miss.to_csv(TABLES_DIR / "missing_values.csv")
    print("[eda] Missing values:\n", miss if len(miss) else "None remaining after preprocessing.")
    return miss

# This function is used by the EDA notebook and the eda.py script to plot daily purchases over time. It saves the plot to CHARTS_DIR / "daily_purchases.png".
def plot_daily_purchases(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[DATE_COL], df[TARGET_COL], color="#2563eb", linewidth=1)
    ax.set_title("Daily Purchases Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel(TARGET_COL)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "daily_purchases.png", dpi=150)
    plt.close(fig)

# This find out the media spend trends by channel and saves the plot to CHARTS_DIR / "media_spend_trends.png".
def plot_media_spend_trends(df: pd.DataFrame):
    spend_cols = [f"{ch}_spend" for ch in MEDIA_SPEND_COLUMNS]
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in spend_cols:
        ax.plot(df[DATE_COL], df[col], label=col.replace("_spend", ""), linewidth=1)
    ax.set_title("Media Spend Trends by Channel")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Spend")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "media_spend_trends.png", dpi=150)
    plt.close(fig)

# Correlation heatmap of media spend, target, and controls. Saves the plot to CHARTS_DIR / "correlation_heatmap.png". If there are any missing values, they are dropped before calculating the correlation matrix.
def plot_correlation_heatmap(df: pd.DataFrame):
    spend_cols = [f"{ch}_spend" for ch in MEDIA_SPEND_COLUMNS]
    cols = spend_cols + [TARGET_COL, "discount_rate", "trend"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap: Media Spend, Target, Controls")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

# This function is used by the EDA notebook and the eda.py script to plot spend concentration by channel (share of total spend). It saves the plot to CHARTS_DIR / "spend_concentration.png" and the share table to TABLES_DIR / "spend_concentration.csv".
def plot_spend_concentration(df: pd.DataFrame):
    spend_cols = [f"{ch}_spend" for ch in MEDIA_SPEND_COLUMNS]
    totals = df[spend_cols].sum().sort_values(ascending=False)
    share = (totals / totals.sum() * 100).round(1)
    share.to_csv(TABLES_DIR / "spend_concentration.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        [c.replace("_spend", "") for c in share.index],
        share.values,
        color="#0ea5e9",
    )
    ax.set_title("Spend Concentration by Channel (% of total)")
    ax.set_ylabel("% of total spend")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "spend_concentration.png", dpi=150)
    plt.close(fig)
    return share

# outliers detection is based on z-score method. It saves the outliers to TABLES_DIR / "target_outliers.csv" and prints the number of outliers detected to the console.
def detect_outliers(df: pd.DataFrame, col: str = TARGET_COL, z_thresh: float = 3.0) -> pd.DataFrame:
    z = (df[col] - df[col].mean()) / df[col].std()
    outliers = df.loc[z.abs() > z_thresh, [DATE_COL, col]]
    outliers.to_csv(TABLES_DIR / "target_outliers.csv", index=False)
    print(f"[eda] Detected {len(outliers)} outlier day(s) in {col} (|z| > {z_thresh}).")
    return outliers

# Based on the cleaned dataset, this function calculates the average purchases by day of week and by month. It saves the results to TABLES_DIR / "seasonality_day_of_week.csv" and TABLES_DIR / "seasonality_month.csv".
def seasonality_summary(df: pd.DataFrame) -> pd.DataFrame:
    by_dow = df.groupby(df[DATE_COL].dt.day_name())[TARGET_COL].mean().round(1)
    by_month = df.groupby(df[DATE_COL].dt.month)[TARGET_COL].mean().round(1)
    by_dow.to_csv(TABLES_DIR / "seasonality_day_of_week.csv")
    by_month.to_csv(TABLES_DIR / "seasonality_month.csv")
    return by_dow, by_month

# Loading all the EDA functions and running them in sequence. This function is used by the EDA notebook and the eda.py script.
def run_eda():
    df = load_clean()
    missing_value_report(df)
    plot_daily_purchases(df)
    plot_media_spend_trends(df)
    plot_correlation_heatmap(df)
    share = plot_spend_concentration(df)
    outliers = detect_outliers(df)
    by_dow, by_month = seasonality_summary(df)

    print("\n[eda] Spend concentration (%):\n", share)
    print("\n[eda] Avg purchases by day of week:\n", by_dow)
    print(f"\n[eda] Charts saved to {CHARTS_DIR}")
    print(f"[eda] Tables saved to {TABLES_DIR}")


if __name__ == "__main__":
    run_eda()
