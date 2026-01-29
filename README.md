# NYC Taxi Trip Duration

Feature-rich regression pipeline for the Kaggle NYC Taxi Trip Duration challenge.  
The repo couples exploratory notebooks with reusable `src/` utilities so we can
iterate on ideas quickly and then productionise them inside `train.py`/`test.py`.

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

## Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # pandas, numpy, scikit-learn, seaborn, holidays, nbformat
```

> If a `requirements.txt` is not present, install the packages listed above manually.

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

## Notebook Guidance

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