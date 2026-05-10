# NYC Taxi Trip Duration Prediction

A feature-rich regression pipeline for predicting NYC taxi trip durations. This project combines exploratory data analysis, feature engineering, and multiple modeling approaches to deliver accurate predictions on the [Kaggle NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) challenge.

**Key Idea:** Couple exploratory notebooks with reusable `src/` utilities to iterate quickly, then productionize validated approaches in `train.py` and `test.py`.

**Data:** [Download from Google Drive](https://drive.google.com/drive/folders/1OUo50pMZ1CaAx2CE8lh7Un3v2soWDTKw)

---

## Project Layout

```
├── data/                 # Raw CSVs from Kaggle + processed artifacts
├── notebooks/            # EDA, feature engineering, and modeling experiments
├── src/                  # Reusable training/evaluation scripts
├── models/               # Serialized pipelines (.pkl)
└── split/                # Train/val/test splits used by CLI scripts
```

Key notebooks:

- `EDA.ipynb` — distribution dives, hourly/weekday insights, distance sanity checks.
- `Feature Engineering.ipynb` — NYC bounding-box filter, congestion proxies, outlier analysis.
- `Modeling.ipynb` — benchmarks Ridge/Lasso/ElasticNet, RandomForest/GBMs, and KNN with shared preprocessing.

---

## Environment Setup

### Quick Install

```bash
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
pip install -r requirements.txt
```

**Required packages:** pandas, numpy, scikit-learn, seaborn, holidays, nbformat

> If `requirements.txt` is missing, install packages manually with the command above.

---

## CLI Workflow

### Train

```bash
cd src
python train.py \
  --dataset ../split/train.csv \
  --model_save_name ../models/taxi_model.pkl \
  --poly_degree 4 \
  --ridge_alpha 1.0 \
  --model_type ridge \
  --iqr_factor 2.5
```

The script:

1. Applies the shared `apply_feature_engineering` routine (NYC bounds, congestion rates, week-hour stats).
2. Optionally applies Lasso-based feature selection (`--no_feature_selection` disables it).
3. Fits either a Ridge-on-Polynomial pipeline (`--model_type ridge`) or a standardized KNN regressor (`--model_type knn`, configurable via `--knn_neighbors/--knn_weights`) and serialises the model alongside feature metadata.

Example KNN run:

```bash
python train.py \
  --dataset ../split/train.csv \
  --model_save_name ../models/taxi_knn.pkl \
  --model_type knn \
  --knn_neighbors 15 \
  --knn_weights distance
```

### Evaluate

```bash
cd src
python test.py \
  --model ../models/taxi_model.pkl \
  --dataset ../split/val.csv
```

`test.py` reloads the saved pipeline, reapplies feature engineering with stored statistics, and prints RMSE/R²/MAE.

---

## Feature Engineering

Key features engineered in the pipeline:

| Feature | Description |
| --- | --- |
| **Distance Metrics** | Haversine distance, Manhattan distance approximations |
| **Temporal Features** | hour of day, day of week, week of year, holiday indicators |
| **Congestion Proxies** | aggregated trip patterns per hour/location |
| **Spatial Clustering** | pickup/dropoff within NYC bounds, zone categorization |
| **Interaction Terms** | week-hour combinations for seasonal patterns |

Outliers are clipped using IQR thresholding (configurable via `--iqr_factor`), removing ~6.4% of data.

---

1. **EDA.ipynb**  
   - Run top-to-bottom to refresh data checks and markdown insight summaries.
   - Highlights outlier thresholds used later in automated scripts.

2. **Feature Engineering.ipynb**  
   - Mirrors the production `utils_data.apply_feature_engineering` implementation.
   - Documents the impact of spatial clustering, temporal interactions, and IQR clipping (~6.4% rows removed).

3. **Modeling.ipynb**  
   - Builds a common preprocessing stack and cross-validates Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, HistGradientBoosting, and KNN (with dense+scaler adapters).
   - Persists the best performer to `models/benchmark_<name>.pkl` for quick smoke tests.

## Next Steps

- Add weather APIs to enrich congestion signals further.
- Promote the best modeling notebook configuration into `src/train.py` once validated on the validation split.
- Automate experiment tracking (e.g., MLflow) to capture hyperparameters and scores per run.

---

## Troubleshooting

**Import errors?** Verify your environment is activated: `source ml_env/bin/activate`

**Model file not found?** Check that model paths are relative to the `src/` directory: `python train.py --dataset ../split/train.csv --model_save_name ../models/my_model.pkl`

**Feature mismatch in test?** Ensure the training data includes the same features as the validation set. The feature engineering step uses train statistics for standardization.
