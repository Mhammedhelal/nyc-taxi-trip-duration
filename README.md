# NYC Taxi Trip Duration Prediction

A feature-rich regression pipeline for predicting NYC taxi trip durations. This project combines exploratory data analysis, feature engineering, and multiple modeling approaches to deliver accurate predictions on the [Kaggle NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) challenge.

**Key Idea:** Couple exploratory notebooks with reusable `src/` utilities to iterate quickly, then productionize validated approaches in `train.py` and `test.py`.

**Data:** [Download from Google Drive](https://drive.google.com/drive/folders/1OUo50pMZ1CaAx2CE8lh7Un3v2soWDTKw)

---

## Project Layout

```
├── data/                        # Raw CSVs from Kaggle
├── data_processed/
│   ├── train_engineered.parquet ← produced by feature_engineering.py
│   ├── train_stats.pkl          ← produced by feature_engineering.py
│   ├── val/
│   │   └── test_engineered.parquet
│   └── test/
│       └── test_engineered.parquet
├── notebooks/                   # EDA, feature engineering, and modeling experiments
├── src/                         # Reusable training/evaluation scripts
├── models/                      # Serialized pipelines (.pkl)
└── split/                       # Raw train/val/test CSV splits
```

Key notebooks:

- `EDA.ipynb` — distribution dives, hourly/weekday insights, distance sanity checks.
- `Feature Engineering.ipynb` — NYC bounding-box filter, congestion proxies, outlier analysis.
- `Modeling.ipynb` — benchmarks Ridge/Lasso/ElasticNet, RandomForest/GBMs, and KNN with shared preprocessing.

---

## Environment Setup

```bash
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
pip install -r requirements.txt
pip install xgboost pyarrow   # additional dependencies
```

---

## CLI Workflow

### Step 1 — Feature Engineering (run once)

Produces engineered Parquet files and a `train_stats.pkl` that all subsequent training and evaluation runs share.

```bash
# Train + validation + test in one shot (recommended)
python src/feature_engineering.py \
    --train_dataset  split/train.csv \
    --test_dataset   split/test.csv \
    --train_stats    data_processed/train_stats.pkl \
    --output_dir     data_processed \
    --iqr_factor     2.5
```

> **Note:** `--train_stats` must be passed when `--test_dataset` is provided. The script writes `train_stats.pkl` for the train split first, then reads it back to apply consistent statistics to the test split.

---

### Step 2 — Training

**Mode A — pre-engineered Parquet (recommended; skips FE on every run):**

```bash
# Ridge
python src/train.py \
    --engineered_dataset data_processed/train_engineered.parquet \
    --train_stats        data_processed/train_stats.pkl \
    --model_type         ridge \
    --poly_degree        1 \
    --ridge_alpha        1.0 \
    --no_feature_selection \
    --model_save_name    models/ridge_model_degree_1_alpha_1.pkl

# HistGradientBoosting (fast, recommended for large data)
python src/train.py \
    --engineered_dataset data_processed/train_engineered.parquet \
    --train_stats        data_processed/train_stats.pkl \
    --model_type         histgradientboosting \
    --hgb_max_depth      10 \
    --hgb_learning_rate  0.1 \
    --hgb_max_iter       100 \
    --model_save_name    models/hgb_model.pkl

# XGBoost
python src/train.py \
    --engineered_dataset data_processed/train_engineered.parquet \
    --train_stats        data_processed/train_stats.pkl \
    --model_type         xgboost \
    --xgb_n_estimators   200 \
    --xgb_learning_rate  0.1 \
    --xgb_max_depth      6 \
    --xgb_subsample      0.8 \
    --xgb_colsample_bytree 0.8 \
    --model_save_name    models/xgb_model.pkl
```

All available `--model_type` values: `ridge`, `lasso`, `elasticnet`, `knn`, `randomforest`, `gradientboosting`, `histgradientboosting`, `xgboost`.

**Mode B — raw CSV (FE runs inline; no pre-processing step needed):**

```bash
python src/train.py \
    --dataset         split/train.csv \
    --model_type      histgradientboosting \
    --model_save_name models/hgb_model.pkl
```

---

### Step 3 — Evaluation

```bash
# Raw CSV — FE applied inline using stored train_stats
python src/test.py \
    --model   models/ridge_model_degree_1_alpha_1.pkl \
    --dataset split/test.csv

# Pre-engineered Parquet — fast, skip FE
python src/test.py \
    --model          models/ridge_model_degree_1_alpha_1.pkl \
    --dataset        data_processed/test/test_engineered.parquet \
    --pre-engineered

# Evaluate every saved model in sequence
for MODEL in models/*.pkl; do
    echo "========== $MODEL =========="
    python src/test.py --model "$MODEL" --dataset split/test.csv
done
```

> **Note:** Use `--pre-engineered` only with `.parquet` files. Without the flag, the script reads a raw CSV and applies feature engineering inline.

---

## Feature Engineering

Key features engineered in the pipeline:

| Feature Group | Examples |
|---|---|
| **Distance Metrics** | haversine distance, Manhattan distance, log distance |
| **Temporal Features** | hour, day-of-week, month, holiday flag, rush-hour flag |
| **Congestion Proxies** | per-slot congestion rate, pickup/dropoff zone congestion |
| **Spatial Clustering** | MiniBatchKMeans pickup/dropoff clusters, corridor median duration |
| **Interaction Terms** | hour × weekday, vendor × hour speed median |
| **Anomaly Detection** | daily trip count z-score flag |

Outliers are removed using IQR thresholding (configurable via `--iqr_factor`, default 2.5), removing ~6.4% of training rows.

---

## Model Results

All metrics are on log-transformed trip duration (`log1p(trip_duration)`). Lower RMSE/MAE and higher R² is better.

Here is your updated table with the Gradient Boosting, HistGradientBoosting, and XGBoost results cleaned up and formatted to match the original structure. I also removed the empty separator rows to keep the final table clean and easy to read.

| Model | Params | Split | RMSE | R² | MAE |
| --- | --- | --- | --- | --- | --- |
| Ridge | degree=1, α=1.0 | Train | 0.1865 | 0.9241 | 0.1098 |
| Ridge | degree=1, α=1.0 | Val | 0.1861 | 0.9246 | 0.1098 |
| Ridge | degree=1, α=1.0 | Test | 0.1861 | 0.9244 | 0.1100 |
|||
|||
|||
| Ridge | degree=2, α=1.0 | Train | 0.1557 | 0.9471 | 0.0859 |
| Ridge | degree=2, α=1.0 | Val | 0.1540 | 0.9484 | 0.0856 |
| Ridge | degree=2, α=1.0 | Test | 0.1551 | 0.9475 | 0.0862 |
|||
|||
|||
| Lasso | α=5e-4, max_iter=5000 | Train | 0.1890 | 0.9221 | 0.1109 |
| Lasso | α=5e-4, max_iter=5000 | Val | 0.1886 | 0.9226 | 0.1108 |
| Lasso | α=5e-4, max_iter=5000 | Test | 0.1886 | 0.9224 | 0.1111 |
|||
|||
|||
| ElasticNet | α=5e-4, l1_ratio=0.3 | Train | 0.1890 | 0.9221 | 0.1109 |
| ElasticNet | α=5e-4, l1_ratio=0.3 | Val | 0.1886 | 0.9226 | 0.1108 |
| ElasticNet | α=5e-4, l1_ratio=0.3 | Test | 0.1886 | 0.9224 | 0.1111 |
|||
|||
|||
| Gradient Boosting | n_estimators=100, lr=0.1, max_depth=3 | Train | 0.0454 | 0.9955 | 0.0297 |
| Gradient Boosting | n_estimators=100, lr=0.1, max_depth=3 | Val | 0.0469 | 0.9952 | 0.0299 |
| Gradient Boosting | n_estimators=100, lr=0.1, max_depth=3 | Test | 0.0471 | 0.9952 | 0.0299 |
|||
|||
|||
| HistGradientBoosting | max_depth=10, lr=0.1, max_iter=100 | Train | 0.0562 | 0.9931 | 0.0190 |
| HistGradientBoosting | max_depth=10, lr=0.1, max_iter=100 | Val | 0.0625 | 0.9915 | 0.0193 |
| HistGradientBoosting | max_depth=10, lr=0.1, max_iter=100 | Test | 0.0613 | 0.9918 | 0.0193 |
|||
|||
|||
| XGBoost | n_estimators=200, lr=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8 | Train | 0.0474 | 0.9951 | 0.0160 |
| XGBoost | n_estimators=200, lr=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8 | Val | 0.0637 | 0.9912 | 0.0170 |
| XGBoost | n_estimators=200, lr=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8 | Test | 0.0629 | 0.9914 | 0.0170 |

---

## Notebooks

1. **EDA.ipynb** — Run top-to-bottom to refresh data checks and insight summaries. Highlights outlier thresholds used in the automated scripts.

2. **Feature Engineering.ipynb** — Mirrors the production `utils_data.apply_feature_engineering` implementation. Documents the impact of spatial clustering, temporal interactions, and IQR clipping.

3. **Modeling.ipynb** — Builds a common preprocessing stack and cross-validates Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, HistGradientBoosting, and KNN. Persists the best performer to `models/benchmark_<name>.pkl`.

---

## Next Steps

- Add weather API enrichment to improve congestion signals.
- Promote the best notebook configuration into `src/train.py` once validated on the val split.
- Automate experiment tracking (e.g., MLflow) to capture hyperparameters and scores per run.

---

## Troubleshooting

**Import errors?** Verify your environment is activated: `source ml_env/bin/activate`

**Model file not found?** Check that model paths are relative to the project root: `python src/train.py --engineered_dataset data_processed/train_engineered.parquet --model_save_name models/my_model.pkl`

**`test.py` crashes on `.csv` with `--pre-engineered`?** Remove the flag — `--pre-engineered` expects a `.parquet` file and skips feature engineering entirely.

**Feature mismatch at test time?** Ensure `--train_stats` points to the pkl produced when the training data was engineered. The stored statistics (cluster models, IQR bounds, aggregation tables) must match what was used during training.

**`feature_engineering.py` silently skips the test set?** Pass `--train_stats <output_dir>/train_stats.pkl` — without it the script exits early before processing the test CSV.
