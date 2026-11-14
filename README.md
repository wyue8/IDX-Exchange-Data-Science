# California Property Close Price Prediction

A 12-week internship project: predict the **close price (final sale price)**
of any single-family residential property in California from its
characteristics at the time of the query — whether or not the home is
currently on the market.

**Champion model:** tuned LightGBM — test R² **0.8911**, RMSE **$327,782**,
MAPE **13.07%**, MdAPE **9.03%** (April 2026 test set, n = 12,339).

## Dataset

- **Source:** CRMLS (California Regional Multiple Listing Service) sold
  listings, provided through IDX Exchange (see the task handbook for access;
  do not commit access credentials to this repo).
- **Files:** 28 monthly `CRMLSSold` CSVs, Jan 2024 – Apr 2026, 80 columns.
- **Scope (per task prompt):** only `PropertyType = Residential` and
  `PropertySubType = SingleFamilyResidence`.
- **Target:** `ClosePrice`.
- **Size:** 615,707 raw rows → **309,370 rows** after the scope filter,
  validity / logic checks, coordinate fixes and deduplication.

## How to re-run

### Environment

- Python 3.11, PyCharm or any IDE. Install the packages used by the
  notebooks (`pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`,
  `joblib`, `matplotlib`, `streamlit`, `geopandas` for Week 6).
- **macOS note:** XGBoost / LightGBM need OpenMP. If the import fails with
  a missing `libomp.dylib` error, run `brew install libomp` and restart
  the kernel. No linker flags are needed (libomp is keg-only).

### Order

Run the notebooks in order, from the project folder, with the dataset at
`dataset/csv/CRMLSSold*.csv` (edit `DATA_GLOB` in `02_preprocessing.ipynb`
if path differs). Each notebook reads the previous week's `output_*`
folder and writes the next one — no notebook needs to be re-run out of
order.

| # | Notebook | Reads | Writes |
|---|----------|-------|--------|
| 1 | `01_exploration.ipynb` | raw CSVs | EDA plots |
| 2 | `02_preprocessing.ipynb` | raw CSVs | `output_week3/` |
| 3 | `03_baseline_model.ipynb` | `output_week3/` | `output_week4/` |
| 4 | `04_model_comparison.ipynb` | `output_week3/`, `output_week4/` | `output_week5/` |
| 5 | `05_feature_engineering.ipynb` | `output_week3/`, `output_week5/` | `output_week6/` |
| 6 | `06_advanced_models.ipynb` | `output_week6/` | `output_week7/` |
| 7 | `07_evaluation.ipynb` | `output_week6/`, `output_week7/` | `output_week8/` |
| — | `app.py` (Streamlit) | `output_week6/`, `output_week7/` | — |

```bash
streamlit run app.py        # launch the Week 9 prediction app
```

## Method

### Chronological split (no random shuffling)

The test set is the most recent month, so the evaluation mimics pricing a
home in the future rather than interpolating between known sales.

- **Train:** Apr 2025 – Feb 2026 — 105,974 rows
- **Validation:** Mar 2026 — 11,416 rows
- **Test:** Apr 2026 — 12,339 rows (scored once for model selection, Week 7)

Outlier cutoffs were learned on train only (0.5 / 99.5 percentiles, frozen
at **187k – 8.75M**) and applied to all splits, so no information from
validation or test leaked into the cutoffs.

### Preprocessing (Week 3)

- **Leakage removal (instructor rule):** the model must price *any*
  single-family address, including off-market homes, so anything that only
  exists because a home was listed or sold is excluded: `ListPrice`,
  `OriginalListPrice`, `DaysOnMarket` (and cumulative DOM),
  `ListingContractDate`, `PurchaseContractDate`, `ContractStatusChangeDate`,
  tax-assessed values (`TaxAnnualAmount`, `TaxYear` — also 100% missing),
  and listing/buyer agent and office fields. Anything derived from them
  (e.g. close-to-list ratios) is excluded too.
- **Missingness:** columns with >60% missing were dropped, except core
  features (`LivingArea`, `BedroomsTotal`, `BathroomsTotalInteger`,
  `LotSizeSquareFeet`, `Latitude`, `Longitude`), which were kept with
  imputation + a missing indicator. Missing indicators were built only for
  numeric columns with >1% missing in train.
- **Encoding (fit on train only):** numerics → median imputation +
  `StandardScaler`; low-cardinality categoricals (<50 values) → mode
  imputation + one-hot (`min_frequency=0.01`); high-cardinality
  categoricals (`City`, `PostalCode`, `MLSAreaMajor`, `CountyOrParish`,
  `HighSchoolDistrict`, `Flooring`) → mode imputation + frequency
  encoding. Row identifiers (`ListingKey`, `ListingId`, `UnparsedAddress`)
  were dropped.
- Result: **41 encoded features** (33 after Week 4 feature selection, used
  for linear models only; trees keep all 41).

### Feature engineering (Week 6)

Three new features, re-encoded with the identical Week 3 recipe so the
old-vs-new comparison is clean:

- `bed_bath_ratio = BedroomsTotal / BathroomsTotalInteger`
  (0 baths → NaN, median-imputed).
- `property_age = 2026 − YearBuilt` (negative values clipped to 0).
- `DistrictName` — a school-district layer from a spatial join of each
  property's coordinates against the CA School District Areas 2024–25
  boundaries (data.ca.gov, 937 districts). Match rates: train 99.9%,
  val 99.4%, test 99.3%. Overlapping districts were deduplicated
  deterministically (Unified first, then Elementary, then High) — see
  Limitations.

Result: **44 encoded features**. The three new features together carry
4.8% of Random Forest importance (`DistrictName` rank 10/44, 0.0203;
`property_age` rank 14; `bed_bath_ratio` rank 15).

## Models tested

All metrics are on the **April 2026 test set (n = 12,339)** unless noted.

| Model | Features | Test R² | Test RMSE | Test MAE | Test MAPE |
|---|---|---|---|---|---|
| Naive median (train) | — | −0.1247 | $1,053,302 | $614,448 | 49.72% |
| Linear Regression | 41 | 0.5569 | $661,120 | $419,225 | 40.68% |
| Linear Regression (selected) | 33 | 0.5570 | $661,068 | $419,194 | 40.67% |
| Decision Tree (tuned) | 33 | 0.7746 | $471,506 | $234,350 | 17.74% |
| Random Forest (tuned) | 41 | 0.8612 | $369,981 | $180,129 | 13.76% |
| Linear Regression (new feats) | 44 | 0.5617 | $657,514 | — | 39.86% |
| Random Forest (new feats) | 44 | 0.8617 | $369,380 | $179,843 | 13.76% |
| **LightGBM (tuned) — champion** | **44** | **0.8911** | **$327,782** | **$167,600** | **13.07%** |

Notes:

- **Week 4 (baseline):** Linear Regression. Feature selection dropped 8
  features and left test R² unchanged (0.5569 → 0.5570). Median residual
  −$32,266: systematic overprediction of ~3.5%.
- **Week 5:** trees beat linear by a wide margin. The tuned Decision Tree
  (max_depth unconstrained, min_samples_leaf 20) still overfits visibly
  (train R² 0.9997). The tuned Random Forest (200 trees) became the
  interim champion. Most important feature: `BathroomsTotalInteger`
  (0.2381).
- **Week 6:** the three engineered features helped Linear Regression
  (+0.0048 R² on test) but barely moved Random Forest (+0.0005) — the
  forest already had beds, baths, `YearBuilt` and coordinates to split on.
  A log-target variant for Linear Regression failed outright: healthy in
  log space (test R² 0.5801) but catastrophic in dollars (naive exp
  R² **−0.3666**, RMSE $1,161,063) — the top 10 errors alone were 64% of
  the squared error, all explosive overpredictions. Trees are immune
  (bounded leaf averages); linear + exponentiation is not.
- **Week 7 (advanced models):** XGBoost (lr 0.1, depth 6, n 500) and
  LightGBM (lr 0.1, 31 leaves, n 500) were essentially tied on validation
  (R² 0.8932 vs 0.8936), so only LightGBM was tuned: depth 8,
  learning rate 0.1, n_estimators 800 → validation R² 0.8980. The tuned
  model was scored on test **once** and beats the Random Forest on all
  four headline metrics.

## Evaluation beyond R² (Week 8)

The champion was evaluated on the saved Week 7 test predictions — no
retraining, no re-selection.

| Model | R² | RMSE | MAE | MAPE | **MdAPE** | Median residual | ±10% hit |
|---|---|---|---|---|---|---|---|
| LightGBM (tuned) | 0.8911 | $327,782 | $167,600 | 13.07% | **9.03%** | $8,289 | 54.5% |
| Random Forest (new) | 0.8617 | $369,380 | $179,843 | 13.76% | **8.33%** | $2,161 | 56.8% |

**The honest caveat:** the champion wins the tail-sensitive metrics (R²,
RMSE, MAE, MAPE) but *loses* the median-based ones — Random Forest is
more accurate on the typical house. Which model is "better" depends on
what error matters: avoiding big misses favours the GBM; typical-house
accuracy favours RF. The Week 7 champion decision stands (it was made on
the full metric set), but this is the trade-off to report.

### Price-band breakdown (champion, test n = 12,339)

| Band | Share | R² | MAPE | MdAPE | Mean residual | ±10% hit |
|---|---|---|---|---|---|---|
| <500k | 14.1% | −1.74 | 18.86% | 10.33% | −$40,007 | 49.0% |
| 500k–1M | 41.8% | 0.23 | 10.86% | 7.43% | — | 62.4% |
| 1M–2M | 30.5% | 0.13 | 12.42% | 9.54% | +$19,543 | — |
| >2M (luxury) | 13.6% | 0.62 | 15.33% | 12.30% | +$240,637 | 40.4% |

**Pattern: predictions shrink toward the middle** — cheap homes are
systematically overpredicted, expensive homes systematically
underpredicted. Within-band R² is misleading here (luxury's 0.62 looks
best but carries the largest dollar errors, MAE $519,787); compare bands
on MdAPE and hit rate instead.

## Prediction app (Week 9)

`app.py` is a Streamlit app around the champion model. It loads
`output_week6/preprocessing_pipeline_v2.joblib` and
`output_week7/gbm_tuned.joblib`, so it must run from the project folder
(next to `output_week6/` and `output_week7/`).

- **Inputs (handbook):** LivingArea, Bedrooms, Bathrooms, LotSize.
- **Optional inputs:** YearBuilt, City, PostalCode, Latitude, Longitude,
  School District — these matter (Latitude/Longitude are the #2/#3 most
  important features), so a real location beats the default.
- **Everything else** defaults to the training median / most frequent
  value, read from the fitted pipeline itself. `bed_bath_ratio` and
  `property_age` are derived exactly as in Week 6.
- The app shows the predicted price, a rough range (prediction ± test
  RMSE $327,782 — not a formal confidence interval), and the Week 8
  band note for the predicted price.
- **Verified 2026-11-14** on the real model files: a typical input
  (1,800 sq ft, 3 bd / 2 ba, 6,500 sq ft lot, built 1990, Los Angeles)
  predicts **883,921**; raising LivingArea to 3,000 sq ft raises the
  prediction to 1,020,327 — monotone and sensibly sized.
- **Implementation note:** the pipeline was saved with `joblib` from the
  Week 6 notebook, so it contains a custom `FrequencyEncoder` class.
  `app.py` defines that class (same name, before `joblib.load`) because
  unpickling looks it up as `__main__.FrequencyEncoder`. Do not delete
  or rename it.

## Limitations

1. **Time drift — the rate environment moved.** The model was trained on
   sales from Apr 2024 – Apr 2026, when 30-year mortgage rates were near
   6%. As of the Nov 12, 2026 Freddie Mac PMMS, the 30-year fixed rate
   averaged **7.11%** — down 7bp from 7.18% the prior week, the fourth
   consecutive weekly decline (−29bp cumulative from the 7.40% peak on
   Oct 15) and the lowest in seven weeks; the 15-year averaged 6.45%.
   The rate remains ~1.1pp above the training-window environment, so the
   drift narrative stands, with the easing nuance extended. Fannie Mae's
   October 2026 forecast has rates at 7.4% by end-2026 and 7.5% through 2027
2. **Caveat:** the October jobs beat (+165k vs ~65k expected,
   unemployment 4.1%) has economists expecting a December rate hike, so
   7.11% is a weekly print, not an established trend — and a model
   trained in a ~6% world will tend to overprice in a ~7% world.
   Existing-home sales were already soft (NAR: October sales fell 5.5%
   to a 3.75M annual pace, the lowest since Aug 2010).
2. **Shrink-to-the-middle.** Cheap homes are overpredicted (mean
   residual −40,007 below 500k); luxury homes are underpredicted
   (+240,637 above 2M). Treat luxury predictions as a floor, not a
   point estimate.
3. **One test month.** All headline metrics rest on a single month
   (April 2026). Seasonal effects and a one-month regime cannot be ruled
   out; a longer or rolling test window would be needed to confirm
   stability.
4. **District dedup is a simplification.** Overlapping Unified /
   Elementary / High districts were collapsed to one district per
   property by a fixed priority (Unified > Elementary > High) with no
   empirical basis. Splitting into separate elementary + high district
   features is an open refinement.
5. **A suspicious feature.** `StreetNumberNumeric` ranks #5 in LightGBM
   importance. A street number should not drive price; it likely proxies
   for location or data artefacts and needs review before the model is
   trusted for individual addresses.
6. **Log-target modelling failed** (see Week 6) — the linear + log
   route is not a viable alternative without a different back-transform.
7. **Outlier policy.** Deleting the worst test rows would mechanically
   lift test R² (e.g. to ≈ +0.50 for the log-target model), but that is
   post-hoc test-set editing, not a model improvement, and is never
   reported as performance. Legitimate outlier handling acts on train
   only and is re-evaluated on the full untouched test set.
8. **Frequency encoding, not target encoding.** High-cardinality
   categoricals use train frequencies; unseen categories map to 0.0
   (treated as rare). The app's City / PostalCode / DistrictName inputs
   behave the same way.

## Repository layout

```
01_exploration.ipynb          Week 2 — EDA
02_preprocessing.ipynb        Week 3 — cleaning, encoding, splits
03_baseline_model.ipynb       Week 4 — Linear Regression baseline
04_model_comparison.ipynb     Week 5 — Decision Tree + Random Forest
05_feature_engineering.ipynb  Week 6 — bed/bath ratio, property age, school districts
06_advanced_models.ipynb      Week 7 — XGBoost / LightGBM + light tuning
07_evaluation.ipynb           Week 8 — MdAPE, median residual, price bands
app.py                        Week 9 — Streamlit prediction app (optional)
output_week3/ … output_week8/ Saved matrices, pipelines, models, metrics, summaries
README.md                     This file (Week 10)
```

**Numbering note:** the handbook numbers the Week 7 file `05_` and the
Week 8 file `06_`, but those slots were already taken, so the files here
are `06_advanced_models.ipynb` and `07_evaluation.ipynb`.
