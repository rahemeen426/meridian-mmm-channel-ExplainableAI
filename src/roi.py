"""
roi.py
------
Task 4 - ROI and Marginal ROI Analysis.

This file oads the fitted Meridian model (pickled by modeling.py) and computes channel-level ROI and marginal ROI (mROI), flagging over/under-invested
channels by comparing mROI to the portfolio-average ROI.

Run directly (after modeling.py):
    python src/roi.py
"""

import pickle

import numpy as np
import pandas as pd

from config import TABLES_DIR, OUTPUTS_DIR, MEDIA_CHANNELS

MODEL_PATH = OUTPUTS_DIR / "fitted_meridian_model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def compute_roi_table(mmm) -> pd.DataFrame:
    """Task 4.1-4.3 - posterior-mean ROI and mROI per channel."""
    from meridian.analysis import analyzer

    an = analyzer.Analyzer(mmm)

    roi_tensor = an.roi(use_posterior=True, aggregate_geos=True)
    mroi_tensor = an.marginal_roi(use_posterior=True, aggregate_geos=True, incremental_increase=0.01)

    # Both tensors have shape (chains, draws, n_channels) -> average over
    # posterior draws for a point estimate, keep std for uncertainty.
    roi_np = np.array(roi_tensor)
    mroi_np = np.array(mroi_tensor)
    roi_flat = roi_np.reshape(-1, roi_np.shape[-1])
    mroi_flat = mroi_np.reshape(-1, mroi_np.shape[-1])

    # build dataframe based on the mean and std of ROI and mROI for each channel
    df = pd.DataFrame(
        {"channel": MEDIA_CHANNELS,
            "roi_mean": roi_flat.mean(axis=0),
            "roi_std": roi_flat.std(axis=0),
            "mroi_mean": mroi_flat.mean(axis=0),
            "mroi_std": mroi_flat.std(axis=0),
        }
    )
    return df

    # this function flags over/under-invested channels by comparing the marginal ROI (mROI) of each channel to the portfolio-average ROI. It categorizes channels as "over-invested (saturated)", "under-invested (room to grow)", or "near-optimal" based on their mROI relative to the portfolio average. This is important for guiding investment decisions and optimizing marketing spend across channels.
def flag_investment_status(df: pd.DataFrame) -> pd.DataFrame:
    """Task 4.4 - flag over/under-invested channels by comparing mROI to portfolio-average ROI."""
    portfolio_avg_roi = df["roi_mean"].mean()
    df = df.copy()
    df["vs_portfolio_avg_roi"] = df["mroi_mean"] - portfolio_avg_roi

    def status(row):
        if row["mroi_mean"] < 0.5 * portfolio_avg_roi:
            return "over-invested (saturated)"
        elif row["mroi_mean"] > 1.1 * portfolio_avg_roi:
            return "under-invested (room to grow)"
        return "near-optimal"

    # sort the dataframe by mROI in descending order and reset the index to ensure that the investment status is clearly presented alongside the corresponding channel information. 
    df["investment_status"] = df.apply(status, axis=1)
    df = df.sort_values("mroi_mean", ascending=False).reset_index(drop=True)
    return df

    # run the ROI and mROI analysis by loading the fitted Meridian model, computing the ROI table, flagging investment status, and saving the results to CSV files. The results are printed to the console for review.
def run():
    mmm = load_model()
    roi_df = compute_roi_table(mmm)
    roi_df.to_csv(TABLES_DIR / "roi_table.csv", index=False)
    
    # flag investment status and save the results to a CSV file. The results are printed to the console for review.
    full_df = flag_investment_status(roi_df)
    full_df.to_csv(TABLES_DIR / "roi_mroi_investment_status.csv", index=False)

    print("[roi] ROI / mROI table:\n", full_df)
    print(f"[roi] Saved to {TABLES_DIR / 'roi_mroi_investment_status.csv'}")
    return full_df


if __name__ == "__main__":
    run()
