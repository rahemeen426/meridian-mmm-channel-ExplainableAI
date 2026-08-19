"""
config.py
---------
Central configuration file for the MMM project, including file paths, selected organisation/territory, and media/control variable definitions.

I have selected a single organisation aggregated at the "All Territories" level because it has the longest and most complete daily data (about 1,610 records from 2020-01-06 to 2024-06-02). 
This organisation consistently uses only Google and Meta marketing channels, avoiding sparse channels such as TikTok. 
Aggregating all territories provides a stronger and more reliable signal for estimating media effects with Meridian's Bayesian MMM.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = OUTPUTS_DIR / "reports"

RAW_CSV = DATA_DIR / "conjura_mmm_data.csv"
DATA_DICTIONARY = DATA_DIR / "conjura_mmm_data_dictionary.xlsx"
CLEAN_PARQUET = DATA_DIR / "mmm_modeling_dataset.parquet"
CLEAN_CSV = DATA_DIR / "mmm_modeling_dataset.csv"

for d in (CHARTS_DIR, TABLES_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Organisation / territory selection
# ---------------------------------------------------------------------------
ORGANISATION_ID = "ba773ebd7ec0a08f1d042187d086ccb4"
TERRITORY_NAME = "All Territories"

# ---------------------------------------------------------------------------
# Modeling columns
# ---------------------------------------------------------------------------
DATE_COL = "DATE_DAY"
TARGET_COL = "ALL_PURCHASES"  # count of purchases; alt: ALL_PURCHASES_ORIGINAL_PRICE for revenue

# Media channels actually populated for the selected organisation
# (GOOGLE_DISPLAY/VIDEO/META_INSTAGRAM/META_OTHER/TIKTOK are ~empty for this org
# and are dropped -- see EDA notebook for the coverage check).
MEDIA_SPEND_COLUMNS = {"google_paid_search": "GOOGLE_PAID_SEARCH_SPEND","google_shopping": "GOOGLE_SHOPPING_SPEND","google_pmax": "GOOGLE_PMAX_SPEND","meta_facebook": "META_FACEBOOK_SPEND",
}

# Matching exposure (impressions) columns used by Meridian's media layer
MEDIA_IMPRESSION_COLUMNS = {
    "google_paid_search": "GOOGLE_PAID_SEARCH_IMPRESSIONS",
    "google_shopping": "GOOGLE_SHOPPING_IMPRESSIONS",
    "google_pmax": "GOOGLE_PMAX_IMPRESSIONS",
    "meta_facebook": "META_FACEBOOK_IMPRESSIONS",
}

MEDIA_CHANNELS = list(MEDIA_SPEND_COLUMNS.keys())

# Organic / control traffic columns (non-paid, used as controls not media)
CONTROL_TRAFFIC_COLUMNS = ["ORGANIC_SEARCH_CLICKS","BRANDED_SEARCH_CLICKS","EMAIL_CLICKS","REFERRAL_CLICKS",]


# Discount / promo control derived in preprocessing.py
DISCOUNT_CONTROL_COL = "discount_rate"

# Calendar controls engineered in preprocessing.py
TIME_CONTROLS = ["day_of_week", "month", "trend", "is_weekend"]

RANDOM_SEED = 42 # random seed for reproducibility of train/test split and model fitting
