"""
main.py
-------
Runs the full MMM homework pipeline end to end, in the order specified by the README's suggested workflow:

    Data Loading -> EDA -> Feature Engineering -> Meridian MMM Training ->
    Contribution Analysis -> ROI/mROI -> SHAP -> Budget Scenarios

Run from the project root in VS Code (Run Python File, or `python src/main.py`).

"""

import time

# This function is used to time each step of the pipeline and print the elapsed time to the console.
def step(name, fn):
    print(f"\n{'=' * 70}\n[main] {name}\n{'=' * 70}")
    t0 = time.time()
    fn()
    print(f"[main] {name} done in {time.time() - t0:.1f}s")

# This function runs the full pipeline end to end, in the order specified by the README's suggested workflow.
def run_all():
    import preprocessing
    import eda
    import modeling
    import roi
    import shap_analysis
    import scenarios

    step("Task 2 - Preprocessing", preprocessing.run_pipeline)
    step("Task 1 - EDA", eda.run_eda)
    step("Task 3 - Meridian modeling (this is the slow one)", modeling.run)
    step("Task 4 - ROI / mROI", roi.run)
    step("Task 5 - SHAP explainability", shap_analysis.run)
    step("Task 6 - Budget scenarios", scenarios.run)

    print("\n[main] Pipeline complete. See outputs/charts, outputs/tables, outputs/reports.")


if __name__ == "__main__":
    run_all()
