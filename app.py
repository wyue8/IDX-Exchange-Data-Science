# Week 9 — Simple Prediction App (Streamlit)
# California Property Close Price Prediction
#
# Handbook deliverable: app.py — user inputs LivingArea, Beds, Baths, LotSize
# and gets a predicted close price from the trained model.
#
# Model: Week 7 champion — tuned LightGBM (max_depth 8, learning_rate 0.1,
# n_estimators 800), trained on the Week 6 feature set (44 encoded features).
# Test performance (Week 7 §3 / Week 8 §1, test n = 12,339, April 2026):
#   R2 0.8911 | RMSE $327,782 | MAE $167,600 | MAPE 13.07% | MdAPE 9.03%
#   +/-10% hit rate 54.5% (RF 56.8% — the champion wins the tail, RF wins
#   the typical house)
#
# How to run (from project folder, next to output_week6/ and
# output_week7/ — the same folder as the Week 3-8 notebooks):
#   pip install streamlit
#   streamlit run app.py
#
# Files the app loads (all already on disk from Weeks 6-7):
#   output_week6/preprocessing_pipeline_v2.joblib  (Week 6 encoder, fit on train only)
#   output_week7/gbm_tuned.joblib                 (Week 7 tuned LightGBM)
#
# How it works: the model needs all 44 encoded features, but the handbook only
# asks the user for four inputs. Everything else is filled in with the
# training median (numeric) or most frequent value (categorical), read from
# the fitted pipeline itself — so the defaults always match the model that
# was actually trained. Two features are derived exactly as in Week 6:
#   bed_bath_ratio = BedroomsTotal / BathroomsTotalInteger
#   property_age   = 2026 - YearBuilt (clipped at 0)
# Optional inputs (YearBuilt, City, PostalCode, Latitude, Longitude,
# DistrictName) are under "More details" — they matter (Latitude/Longitude
# are the #2/#3 most important features), so enter a real location for a
# meaningful price instead of the median-location default.
#
# IMPORTANT — why FrequencyEncoder is defined here: the Week 6 pipeline was
# saved with joblib, and it contains a custom FrequencyEncoder class defined
# in the Week 6 notebook. Unpickling looks that class up as
# `__main__.FrequencyEncoder`, so it must exist in this file, with the same
# name, BEFORE joblib.load runs. Do not delete or rename it. (This is the
# same class, copied verbatim from Week 6 §3.)

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Simple leakage-free encoding for high-cardinality cats (fit sees train only)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.cols_ = list(X.columns)
        self.maps_ = {c: X[c].value_counts(normalize=True) for c in self.cols_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cols_)
        out = pd.DataFrame(index=X.index)
        for c in self.cols_:
            out[c] = X[c].map(self.maps_[c]).fillna(0.0)
        return out.values

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "cols_", [])
        return np.asarray(list(input_features))


# ---------------------------------------------------------------- config
BASE_DIR = Path(__file__).parent
PIPELINE_PATH = BASE_DIR / "output_week6" / "preprocessing_pipeline_v2.joblib"
MODEL_PATH = BASE_DIR / "output_week7" / "gbm_tuned.joblib"

# Frozen test metrics (Week 7 §3, Week 8 §1) — shown for context, not recomputed.
TEST_R2 = 0.8911
TEST_RMSE = 327_782
TEST_MAE = 167_600
TEST_MAPE = 13.07
TEST_MDAPE = 9.03
HIT_10PCT = 54.5


@st.cache_resource
def load_artifacts():
    """Load the fitted Week 6 pipeline and the Week 7 champion model."""
    missing = [p for p in (PIPELINE_PATH, MODEL_PATH) if not p.exists()]
    if missing:
        st.error(
            "Could not find the trained model files. This app must live in the same "
            "folder as your Week 3–8 notebooks (next to `output_week6/` and "
            f"`output_week7/`).\n\nMissing: {', '.join(str(p) for p in missing)}"
        )
        st.stop()
    preprocess = joblib.load(PIPELINE_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocess, model


def pipeline_inputs(preprocess):
    """Read the exact raw columns the fitted pipeline expects, plus the
    training median (numeric) / mode (categorical) for each, straight from
    the fitted imputers — no hard-coded feature list to drift out of date.

    Returns (columns, defaults) where columns is the ordered list of raw
    input columns and defaults maps each column to its fill-in value.
    """
    columns, defaults = [], {}
    for name, transformer, cols in preprocess.transformers_:
        # sklearn >= 1.2 lists the dropped columns as a ('remainder', 'drop',
        # [...]) entry — skip it (and anything that is not a fitted Pipeline).
        if not hasattr(transformer, "named_steps"):
            continue
        cols = list(cols)
        imputer = transformer.named_steps["imputer"]
        stats = imputer.statistics_
        for col, stat in zip(cols, stats):
            columns.append(col)
            defaults[col] = stat
    return columns, defaults


def build_input_row(columns, defaults, user):
    """Build the one-row raw DataFrame the pipeline expects.

    `user` holds the values the user typed (already cleaned): LivingArea,
    BedroomsTotal, BathroomsTotalInteger, LotSizeSquareFeet, YearBuilt,
    Latitude, Longitude, City, PostalCode, DistrictName.
    Everything else falls back to the training median / mode.
    Derived features follow Week 6 exactly, and the `*_was_missing`
    indicators are recomputed from whether each raw value is missing.
    """
    row = {col: defaults[col] for col in columns}

    # --- the four handbook inputs ---
    row["LivingArea"] = user["LivingArea"]
    row["BedroomsTotal"] = user["BedroomsTotal"]
    row["BathroomsTotalInteger"] = user["BathroomsTotalInteger"]
    row["LotSizeSquareFeet"] = user["LotSizeSquareFeet"]

    # --- optional inputs (only overwrite when the user gave a value) ---
    if user.get("YearBuilt") is not None:
        row["YearBuilt"] = user["YearBuilt"]
    if user.get("Latitude") is not None:
        row["Latitude"] = user["Latitude"]
    if user.get("Longitude") is not None:
        row["Longitude"] = user["Longitude"]
    for col in ("City", "PostalCode", "DistrictName"):
        if user.get(col):
            row[col] = user[col]

    # --- derived features, exactly as in Week 6 §1 ---
    baths = row["BathroomsTotalInteger"]
    row["bed_bath_ratio"] = (
        row["BedroomsTotal"] / baths if baths not in (0, None) and not pd.isna(baths) else np.nan
    )
    row["property_age"] = max(0, 2026 - int(row["YearBuilt"])) if not pd.isna(row["YearBuilt"]) else np.nan

    # --- missing indicators: 1 only if the raw value is actually missing ---
    for col in columns:
        if col.endswith("_was_missing"):
            base = col[: -len("_was_missing")]
            row[col] = 1 if base not in row or pd.isna(row[base]) else 0

    return pd.DataFrame([row], columns=columns)


def band_note(pred):
    """One-line Week 8 context for the predicted price band."""
    if pred < 500_000:
        return (
            "Week 8 note: in the cheapest band (<$500k) the model overpredicted on "
            "average (mean residual −$40,007), so a prediction here tends to run high."
        )
    if pred < 1_000_000:
        return (
            "Week 8 note: the $500k–$1M band was the model's best (MdAPE 7.43%, "
            "±10% hit rate 62.4%)."
        )
    if pred < 2_000_000:
        return (
            "Week 8 note: in the $1M–$2M band the model underpredicted slightly on "
            "average (mean residual +$19,543)."
        )
    return (
        "Week 8 note: luxury homes (>$2M) were the weakest band — underpredicted "
        "by $240,637 on average, ±10% hit rate only 40.4%. Treat a luxury "
        "prediction as a floor, not a point estimate."
    )


# ---------------------------------------------------------------- page
st.set_page_config(page_title="California Close Price Predictor", layout="centered")
st.title("California Close Price Predictor")
st.caption(
    "Week 9 — Streamlit app. Tuned LightGBM (Week 7 champion) on the Week 6 "
    "feature set. Trained on CRMLS single-family sales, Apr 2024–Apr 2026."
)

preprocess, model = load_artifacts()
columns, defaults = pipeline_inputs(preprocess)

# --- the four handbook inputs (defaults = median training house) ---
st.subheader("Property details")
col_a, col_b = st.columns(2)
with col_a:
    living_area = st.number_input(
        "Living area (sq ft)",
        min_value=100, max_value=20_000,
        value=int(defaults["LivingArea"]),
        step=50,
        help="Above/below-grade finished living area.",
    )
    beds = st.number_input(
        "Bedrooms",
        min_value=0, max_value=15,
        value=int(defaults["BedroomsTotal"]),
        step=1,
    )
with col_b:
    baths = st.number_input(
        "Bathrooms",
        min_value=0, max_value=15,
        value=int(defaults["BathroomsTotalInteger"]),
        step=1,
    )
    lot = st.number_input(
        "Lot size (sq ft)",
        min_value=0, max_value=500_000,
        value=int(defaults["LotSizeSquareFeet"]),
        step=100,
        help="Leave 0 if unknown — the model will use the training median.",
    )

# --- optional inputs: these move the prediction a lot, so they are offered,
# --- but the app still works if they are left at their defaults.
with st.expander("More details (optional — improves the prediction)"):
    st.caption(
        "Left blank, each of these uses the training median / most common value. "
        "Location matters most: Latitude/Longitude are the #2/#3 most important "
        "features, so a real location beats the median-location default."
    )
    year_built = st.number_input(
        "Year built",
        min_value=1800, max_value=2026,
        value=int(defaults["YearBuilt"]),
        step=1,
        help="Used as property_age = 2026 − YearBuilt (Week 6 feature).",
    )
    lat = st.number_input(
        "Latitude", min_value=32.0, max_value=42.0,
        value=float(defaults["Latitude"]), step=0.0001, format="%.4f",
    )
    lon = st.number_input(
        "Longitude", min_value=-125.0, max_value=-114.0,
        value=float(defaults["Longitude"]), step=0.0001, format="%.4f",
    )
    city = st.text_input(
        "City",
        value=str(defaults["City"]),
        help="Must match a city name in the training data; anything else is "
        "treated as an unseen (rare) city.",
    )
    postal = st.text_input(
        "Postal code",
        value=str(defaults["PostalCode"]).split(".")[0],
        help="5-digit ZIP. Anything not in the training data is treated as unseen.",
    )
    district = st.text_input(
        "School district (DistrictName)",
        value=str(defaults["DistrictName"]),
        help="CA School District Areas 2024–25 name, e.g. the district the address "
        "falls in. Leave the default if unsure.",
    )

if st.button("Predict close price", type="primary"):
    # PostalCode was stored as a number in training: keep the type aligned
    # with the training mode so the frequency lookup can match.
    postal_value = postal.strip()
    mode_postal = defaults["PostalCode"]
    if isinstance(mode_postal, (int, float, np.integer, np.floating)) and postal_value.isdigit():
        postal_value = int(postal_value)

    user = {
        "LivingArea": living_area,
        "BedroomsTotal": beds,
        "BathroomsTotalInteger": baths,
        "LotSizeSquareFeet": lot if lot > 0 else np.nan,
        "YearBuilt": year_built,
        "Latitude": lat,
        "Longitude": lon,
        "City": city.strip(),
        "PostalCode": postal_value,
        "DistrictName": district.strip(),
    }
    X_row = build_input_row(columns, defaults, user)
    X_enc = preprocess.transform(X_row)
    pred = float(model.predict(X_enc)[0])

    st.divider()
    st.metric("Predicted close price", f"${pred:,.0f}")
    lo, hi = max(0, pred - TEST_RMSE), pred + TEST_RMSE
    st.caption(
        f"Rough range (prediction ± test RMSE ${TEST_RMSE:,.0f}): "
        f"${lo:,.0f} – ${hi:,.0f}. This is not a formal confidence interval — "
        "it just scales the model's typical test error."
    )
    st.info(band_note(pred))
    with st.expander("What went into this prediction"):
        st.write(
            f"Bed/bath ratio: **{X_row['bed_bath_ratio'].iloc[0]:.2f}** | "
            f"Property age: **{int(X_row['property_age'].iloc[0])} years** | "
            f"District: **{X_row['DistrictName'].iloc[0]}**"
        )
        st.write(
            "All other features (stories, parking, garage, fees, flooring, "
            "school districts, etc.) use the training median / most common value."
        )

with st.expander("How good is this model? (Week 7–8 test results)"):
    st.write(
        f"Test set (April 2026 sales, n = 12,339): R² **{TEST_R2}** | "
        f"RMSE **${TEST_RMSE:,.0f}** | MAE **${TEST_MAE:,.0f}** | "
        f"MAPE **{TEST_MAPE}%** | MdAPE **{TEST_MDAPE}%** | "
        f"within ±10% of the true price **{HIT_10PCT}%** of the time."
    )
    st.write(
        "Known limits: predictions shrink toward the middle — cheap homes tend "
        "to be overpredicted and expensive homes underpredicted. The model was "
        "trained on sales from Apr 2024–Apr 2026, when 30-year mortgage rates "
        "were near 6%; rates have since risen to ~7.2% (Nov 2026), so prices "
        "for today's market may run lower than the model expects (Week 10 "
        "limitations material)."
    )
