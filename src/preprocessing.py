"""
preprocessing.py
-----------------
Task 2 - Data Preparation.

Loads the raw Conjura MMM CSV, filters to the selected organisation/territory, aggregates to a clean daily time series, engineers media/control/time
variables, handles missing values, and produces a train/validation split.

Run directly:
    python src/preprocessing.py
"""

import numpy as np
import pandas as pd

from config import (RAW_CSV,CLEAN_CSV,CLEAN_PARQUET,ORGANISATION_ID,TERRITORY_NAME,DATE_COL,TARGET_COL,MEDIA_SPEND_COLUMNS,MEDIA_IMPRESSION_COLUMNS,CONTROL_TRAFFIC_COLUMNS,DISCOUNT_CONTROL_COL,RANDOM_SEED,)


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    return df


# filter function restricts the dataset to a single organisation and territory, sorts by date, and resets the index. This is important for ensuring that the subsequent analysis is focused on the relevant subset of data.
def filter_organisation(df: pd.DataFrame) -> pd.DataFrame:
    """Task 2.1 - restrict to one organisation at the 'All Territories' grain."""
    mask = (df["ORGANISATION_ID"] == ORGANISATION_ID) & (
        df["TERRITORY_NAME"] == TERRITORY_NAME
    )
    out = df.loc[mask].copy()
    out = out.sort_values(DATE_COL).reset_index(drop=True)
    return out

    # this function builds a continuous daily series from the filtered dataset, ensuring that any gaps in the date range are filled with explicit rows. This is important for time series analysis, as it allows for proper handling of missing dates and ensures that the dataset is complete for modeling purposes.
def build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a continuous daily calendar so gaps become explicit rows."""
    full_range = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    df = df.set_index(DATE_COL).reindex(full_range)
    df.index.name = DATE_COL
    return df.reset_index()

    # media variables are engineered by filling missing spend and impression values with zeros. This is important for ensuring that the media variables are properly represented in the dataset, even when there are gaps in the data.
def engineer_media_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Task 2.3 - media spend + impressions, missing spend/impressions == 0."""
    for channel, spend_col in MEDIA_SPEND_COLUMNS.items():
        df[f"{channel}_spend"] = df[spend_col].fillna(0.0)
    for channel, impr_col in MEDIA_IMPRESSION_COLUMNS.items():
        df[f"{channel}_impressions"] = df[impr_col].fillna(0.0)
    return df

    # control variables are engineered by filling missing values for organic traffic columns with zeros and calculating the discount rate as a derived variable. This is important for ensuring that the control variables are properly represented in the dataset, even when there are gaps in the data.
def engineer_control_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Task 2.3 - organic traffic + discount rate controls."""
    for col in CONTROL_TRAFFIC_COLUMNS:
        df[col] = df[col].fillna(0.0)

    price = df["ALL_PURCHASES_ORIGINAL_PRICE"].fillna(0.0)
    discount = df["ALL_PURCHASES_GROSS_DISCOUNT"].fillna(0.0)
    df[DISCOUNT_CONTROL_COL] = np.where(price > 0, discount / price, 0.0)
    return df

    # time variables are engineered by extracting the day of the week, month, trend, and weekend flag from the date column. This is important for capturing temporal patterns in the data that may influence the target variable.
def engineer_time_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Task 2.3 - day-of-week, month, trend and weekend flag."""
    df["day_of_week"] = df[DATE_COL].dt.dayofweek
    df["month"] = df[DATE_COL].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["trend"] = np.arange(len(df))
    return df

# This function handle s missing values in the target column by interpolating short calendar gaps. It uses linear interpolation to fill in missing values and then backfills and forward fills any remaining NaN values. The number of missing values that were interpolated is printed to the console for reference.
def handle_target_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Task 2.4 - target gaps are interpolated (short calendar gaps only)."""
    n_missing = df[TARGET_COL].isna().sum()
    df[TARGET_COL] = df[TARGET_COL].interpolate(method="linear").bfill().ffill()
    print(f"[preprocessing] Interpolated {n_missing} missing '{TARGET_COL}' values.")
    return df

    # overall validation split function splits the cleaned dataset into training and validation sets based on a specified fraction. The split is chronological, meaning that the most recent data is used for validation. This is important for time series modeling, as it ensures that the model is evaluated on future data that it has not seen during training.
def train_val_split(df: pd.DataFrame, val_fraction: float = 0.15):
    """Task 2.5 - chronological split (no shuffling; MMM is a time series)."""
    n_val = int(len(df) * val_fraction)
    train_df = df.iloc[: len(df) - n_val].copy()
    val_df = df.iloc[len(df) - n_val :].copy()
    return train_df, val_df

    # The run_pipeline function orchestrates the entire preprocessing workflow.
def run_pipeline() -> pd.DataFrame:
    df = load_raw()
    df = filter_organisation(df)
    df = build_daily_series(df)
    df = engineer_media_variables(df)
    df = engineer_control_variables(df)
    df = engineer_time_variables(df)
    df = handle_target_missing(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    df.to_csv(CLEAN_CSV, index=False)
    try:
        df.to_parquet(CLEAN_PARQUET, index=False)
    except Exception as e:
        print(f"[preprocessing] Skipped parquet export ({e}); CSV saved instead.")

    print(f"[preprocessing] Clean dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"[preprocessing] Saved to {CLEAN_CSV}")
    return df


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    clean_df = run_pipeline()
    train_df, val_df = train_val_split(clean_df)
    print(f"[preprocessing] Train: {train_df.shape[0]} rows | Val: {val_df.shape[0]} rows")
