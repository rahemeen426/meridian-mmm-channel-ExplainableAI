"""
modeling.py
-----------
Task 3 - Meridian MMM Model Development.

This file builds a national (single-geo) Meridian model on the cleaned daily dataset,fits it with MCMC, and reports fit quality (R2 / MAPE / RMSE) plus
adstock/Hill saturation curves and channel contributions.

Run directly:
    python src/modeling.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from config import (CLEAN_CSV,TABLES_DIR,CHARTS_DIR,DATE_COL,TARGET_COL,MEDIA_CHANNELS,RANDOM_SEED,)

# Quick-run defaults 
N_CHAINS = 2
N_ADAPT = 300
N_BURNIN = 200
N_KEEP = 300

# Loading the cleaned dataset and returning it as a pandas DataFrame. This function is used by the modeling notebook and the modeling.py script.
def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV, parse_dates=[DATE_COL])


def build_input_data(df: pd.DataFrame):
    """Build Meridian's InputData for a national (single-geo) model.
    """
    from meridian.data import data_frame_input_data_builder as data_builder

    work = df.copy()
    work["geo"] = "national"
    work["population"] = 1.0
    work["time"] = work[DATE_COL].dt.strftime("%Y-%m-%d")

    media_cols = [f"{ch}_impressions" for ch in MEDIA_CHANNELS]
    spend_cols = [f"{ch}_spend" for ch in MEDIA_CHANNELS]
    control_cols = ["ORGANIC_SEARCH_CLICKS","BRANDED_SEARCH_CLICKS","EMAIL_CLICKS","REFERRAL_CLICKS","discount_rate","trend","is_weekend",]

    # This is the main function that builds the InputData object for the Meridian model. It uses the DataFrameInputDataBuilder to specify the KPI, population, controls, and media channels. The resulting InputData object is returned for use in model fitting.
    builder = data_builder.DataFrameInputDataBuilder(kpi_type="non_revenue")
    builder = builder.with_kpi(work, kpi_col=TARGET_COL, time_col="time", geo_col="geo")
    builder = builder.with_population(work, population_col="population", geo_col="geo")
    builder = builder.with_controls(work, control_cols=control_cols, time_col="time", geo_col="geo")
    builder = builder.with_media(
        work,
        media_cols=media_cols,
        media_spend_cols=spend_cols,
        media_channels=MEDIA_CHANNELS,
        time_col="time",
        geo_col="geo",
    )
    return builder.build()


def build_model_spec(n_knots: int = 24):
    """Task 3.1 - model configuration: weakly-informative ROI prior + knots.

    ~24 knots over ~4.4 years of daily data gives a roughly monthly-flexible baseline -- flexible enough to absorb slow trend/seasonality without
    stealing variance that should be attributed to media.
    """
    from meridian import constants
    from meridian.model import prior_distribution, spec
    import tensorflow_probability as tfp

    # Prior distribution for the ROI multiplier, using a log-normal distribution with specified mean and standard deviation. This prior is used to regularize the model and prevent overfitting.
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(0.2, 0.9, name=constants.ROI_M)
    )
    return spec.ModelSpec(prior=prior, knots=n_knots)

    # This function builds the model specification for the Meridian model, including the prior distribution for the ROI multiplier and the number of knots for the spline basis. The resulting ModelSpec object is returned for use in model fitting.
def fit_model(input_data, model_spec, seed: int = RANDOM_SEED):
    """Task 3.2 - sample_prior then sample_posterior (Bayesian MCMC fit)."""
    from meridian.model import model

    mmm = model.Meridian(input_data=input_data, model_spec=model_spec)
    mmm.sample_prior(500)
    mmm.sample_posterior(
        n_chains=N_CHAINS, n_adapt=N_ADAPT, n_burnin=N_BURNIN, n_keep=N_KEEP, seed=seed
    )
    return mmm

    # If evaluation is desired, the evaluate_fit function can be called after fitting the model to compute R2, MAPE, and RMSE metrics based on the expected vs actual KPI values. The results are saved to a CSV file for further analysis.
def evaluate_fit(mmm) -> dict:
    """Task 3.4 - R2 / MAPE / RMSE of expected vs actual KPI (posterior mean)."""
    from meridian.analysis import analyzer

    an = analyzer.Analyzer(mmm)
    ds = an.expected_vs_actual_data()
    df_fit = ds.to_dataframe().reset_index()

    # expected_vs_actual_data returns 'expected' and 'actual' columns aligned
    # by time (posterior-mean expected value).
    y_true = df_fit["actual"].values
    y_pred = df_fit["expected"].values

    # calculating evaluation metrics: R2, MAPE, RMSE and saving them to a CSV file for further analysis. 
    metrics = {
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    pd.Series(metrics).to_csv(TABLES_DIR / "model_fit_metrics.csv")
    print("[modeling] Fit metrics:", metrics)
    return metrics

    # This function evaluates the fit of the fitted Meridian model by comparing the expected KPI values (posterior mean) to the actual observed values. 
def save_contribution_and_saturation(mmm):
    """Task 3.3 - contribution summary + saturation (Hill) curves."""
    from meridian.analysis import analyzer, visualizer

    # analyzer is used to compute summary metrics for each media channel, including contribution and saturation curves. The results are saved to a CSV file for further analysis.
    an = analyzer.Analyzer(mmm)
    summary = an.summary_metrics()
    summary.to_dataframe().to_csv(TABLES_DIR / "channel_summary_metrics.csv")

    # In try and except blocks to handle any exceptions that may occur during the plotting of contribution and saturation curves. If an exception occurs, it is caught and a message is printed to the console, allowing the script to continue running without crashing.
    media_summary = visualizer.MediaSummary(mmm)
    try:
        chart = media_summary.plot_channel_contribution_bump_chart()
        chart.save(str(CHARTS_DIR / "channel_contribution.png"))
    except Exception as e:
        print(f"[modeling] Skipped contribution plot: {e}")

    media_effects = visualizer.MediaEffects(mmm)
    try:
        chart = media_effects.plot_response_curves()
        chart.save(str(CHARTS_DIR / "response_curves.png"))
    except Exception as e:
        print(f"[modeling] Skipped response-curve plot: {e}")
    try:
        chart = media_effects.plot_hill_curves()
        chart.save(str(CHARTS_DIR / "hill_saturation_curves.png"))
    except Exception as e:
        print(f"[modeling] Skipped Hill-curve plot: {e}")

    return summary

    # This function saves the contribution summary and saturation (Hill) curves for each media channel in the fitted Meridian model.
def run():
    df = load_clean()
    input_data = build_input_data(df)
    model_spec = build_model_spec()
    mmm = fit_model(input_data, model_spec)
    evaluate_fit(mmm)
    save_contribution_and_saturation(mmm)

    import pickle
    with open(TABLES_DIR.parent / "fitted_meridian_model.pkl", "wb") as f:
        pickle.dump(mmm, f)
    print("[modeling] Fitted model pickled to outputs/fitted_meridian_model.pkl")
    return mmm


if __name__ == "__main__":
    run()
