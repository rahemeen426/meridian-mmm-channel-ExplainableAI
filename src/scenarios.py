"""
scenarios.py
-------------
Task 6 - Marketing Investment Scenario Simulation.

Loads the fitted Meridian model and uses meridian.analysis.optimizer to simulate three budget-allocation strategies over the historical spend
, then compares predicted incremental outcome (purchases) and ROI:

  1. Equal allocation across channels
  2. mROI-optimized allocation (Meridian's BudgetOptimizer, fixed total budget)
  3. Saturation-aware allocation (spend trimmed on the most-saturated channel, the savings reallocated to the channel with the highest mROI)


Run directly (after modeling.py and roi.py):
    python src/scenarios.py
"""

import pickle

import numpy as np
import pandas as pd

from config import TABLES_DIR, OUTPUTS_DIR, MEDIA_CHANNELS

MODEL_PATH = OUTPUTS_DIR / "fitted_meridian_model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_roi_table() -> pd.DataFrame:
    path = TABLES_DIR / "roi_mroi_investment_status.csv"
    if not path.exists():
        raise FileNotFoundError("Run roi.py first to produce roi_mroi_investment_status.csv")
    return pd.read_csv(path)


def get_historical_spend(mmm) -> dict:
    """Historical total spend per channel over the full modeled window,
    read off a baseline (unconstrained) optimize() call."""
    from meridian.analysis import optimizer

    opt = optimizer.BudgetOptimizer(mmm)
    baseline = opt.optimize(fixed_budget=True, use_optimal_frequency=False)
    spend = np.array(baseline.nonoptimized_data["spend"].values)
    return dict(zip(MEDIA_CHANNELS, spend)), baseline


def score_allocation(mmm, spend_by_channel: dict):
    """Score one specific spend split by pinning the optimizer to it
    (0% constraint band == "optimize" just evaluates this allocation)."""
    from meridian.analysis import optimizer

    total_budget = float(sum(spend_by_channel.values()))
    pct_of_spend = [
        spend_by_channel[ch] / total_budget if total_budget > 0 else 0.0
        for ch in MEDIA_CHANNELS
    ]
    opt = optimizer.BudgetOptimizer(mmm)
    results = opt.optimize(
        fixed_budget=True,
        budget=total_budget,
        pct_of_spend=pct_of_spend,
        spend_constraint_lower=0.0,
        spend_constraint_upper=0.0,
        use_optimal_frequency=False,
    )
    attrs = results.optimized_data.attrs
    return {
        "incremental_outcome": float(attrs["total_incremental_outcome"]),
        "roi": float(attrs["total_roi"]),
        "total_budget": total_budget,
    }


def optimize_allocation(mmm, total_budget: float):
    """Task 6 scenario 2 - let Meridian's optimizer find the mROI-maximizing
    split of the SAME total budget across channels."""
    from meridian.analysis import optimizer

    opt = optimizer.BudgetOptimizer(mmm)
    results = opt.optimize(fixed_budget=True, budget=total_budget, use_optimal_frequency=False)
    spend = np.array(results.optimized_data["spend"].values)
    attrs = results.optimized_data.attrs
    outcome = {
        "incremental_outcome": float(attrs["total_incremental_outcome"]),
        "roi": float(attrs["total_roi"]),
        "total_budget": total_budget,
    }
    return dict(zip(MEDIA_CHANNELS, spend)), outcome


def scenario_equal_allocation(total_budget: float) -> dict:
    n = len(MEDIA_CHANNELS)
    return {ch: total_budget / n for ch in MEDIA_CHANNELS}


def scenario_saturation_aware(historical_spend: dict, roi_df: pd.DataFrame, cut_pct: float = 0.3) -> dict:
    """Cut spend on the most-saturated (lowest mROI) channel by `cut_pct`
    and reallocate the freed budget to the channel with the highest mROI."""
    spend = dict(historical_spend)
    most_saturated = roi_df.sort_values("mroi_mean").iloc[0]["channel"]
    best_channel = roi_df.sort_values("mroi_mean", ascending=False).iloc[0]["channel"]

    freed = spend[most_saturated] * cut_pct
    spend[most_saturated] -= freed
    spend[best_channel] += freed
    return spend

    # this function runs the scenario analysis by loading the fitted Meridian model, computing the historical spend and baseline results, simulating three budget-allocation strategies, and comparing their predicted incremental outcomes and ROI then results are saved to a CSV file.
def run():
    mmm = load_model()
    roi_df = load_roi_table()
    historical_spend, baseline_results = get_historical_spend(mmm)
    total_budget = sum(historical_spend.values())
    baseline_incremental = float(baseline_results.nonoptimized_data.attrs["total_incremental_outcome"])

    # simulate the three budget-allocation strategies and store their results in a list of rows for comparison. 
    rows = []

    equal_alloc = scenario_equal_allocation(total_budget)
    equal_outcome = score_allocation(mmm, equal_alloc)
    rows.append(("Equal allocation", equal_outcome))

    _, mroi_outcome = optimize_allocation(mmm, total_budget)
    rows.append(("mROI optimized", mroi_outcome))

    # It store the results of the saturation-aware allocation scenario by trimming spend on the most-saturated channel and reallocating it to the channel with the highest mROI.
    sat_alloc = scenario_saturation_aware(historical_spend, roi_df)
    sat_outcome = score_allocation(mmm, sat_alloc)
    # scoring this allocation and appending the results to the comparison rows
    rows.append(("Saturation-aware", sat_outcome))

    comparison = pd.DataFrame(
        [{"Scenario": name,"Incremental Outcome (Purchases)": o["incremental_outcome"],"Incremental vs Historical Baseline": o["incremental_outcome"] - baseline_incremental,"ROI": o["roi"],}
            for name, o in rows
        ]
    )
    # Storing the results of the scenario comparison in a CSV file for further analysis and reporting.
    comparison.to_csv(TABLES_DIR / "scenario_comparison.csv", index=False)
    print("[scenarios] Historical baseline incremental outcome:", baseline_incremental)
    print("[scenarios] Scenario comparison:\n", comparison)
    print(f"[scenarios] Saved to {TABLES_DIR / 'scenario_comparison.csv'}")
    return comparison


if __name__ == "__main__":
    run()
