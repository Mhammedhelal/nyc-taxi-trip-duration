#!/usr/bin/env bash
# =============================================================================
# commands.sh  —  Quick-reference for the NYC Taxi Trip Duration pipeline
# =============================================================================
# All commands are run from the PROJECT ROOT unless stated otherwise.
# Adjust paths as needed; defaults match the directory layout below:
#
#   project/
#   ├── split/
#   │   ├── train.csv
#   │   ├── test.csv
#   │   ├── train_engineered.parquet   ← produced by feature_engineering.py
#   |   ├── test
#   │       ├── test_engineered.parquet    ← produced by feature_engineering.py
#   |   ├── val
#   │       ├── test_engineered.parquet    ← produced by feature_engineering.py
#   │   └── train_stats.pkl            ← produced by feature_engineering.py
#   ├── models/
#   │   └── taxi_model.pkl
#   └── src/
#       ├── feature_engineering.py
#       ├── train.py
#       └── test.py
# =============================================================================


# =============================================================================
# 0. INSTALL DEPENDENCIES
# =============================================================================

pip install scikit-learn pandas numpy holidays xgboost pyarrow


# =============================================================================
# 1. FEATURE ENGINEERING  (run ONCE; reuse for all models)
# =============================================================================

# --- 1a. Train data only ---
python src/feature_engineering.py \
    --train_dataset  split/train.csv \
    --output_dir     split

# --- 1b. Train + test data in one shot (recommended) ---
#         --train_stats must point to the pkl produced in the same run.
#         Because feature_engineering.py writes train_stats.pkl before
#         processing the test set, pass the output path explicitly.
python src/feature_engineering.py \
    --train_dataset  split/train.csv \
    --test_dataset   split/test.csv \
    --train_stats    data_processed/train_stats.pkl \
    --output_dir     split \
    --iqr_factor     2.5


# =============================================================================
# 2. TRAINING  (two modes)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# MODE A: use pre-engineered Parquet  ← fast; skip FE on every run
#         Requires both --engineered_dataset and --train_stats.
# ─────────────────────────────────────────────────────────────────────────────

# Ridge (default: poly_degree=1, alpha=1.0, with Lasso feature selection)
python src/train.py \
    --engineered_dataset data_processed/train_engineered.parquet \
    --train_stats        data_processed/train_stats.pkl \
    --model_type         ridge \
    --poly_degree        1 \
    --ridge_alpha        1.0 \
    --model_save_name    models/ridge_model.pkl

# Ridge WITHOUT feature selection
python src/train.py \
    --engineered_dataset data_processed/train_engineered.parquet \
    --train_stats        data_processed/train_stats.pkl \
    --model_type         ridge \
    --no_feature_selection \
    --model_save_name    models/ridge_no_fs_model.pkl

# Lasso  (alpha=5e-4, max_iter=5000)
python src/train.py \
    --engineered_dataset data_processed/train_engineered.parquet \
    --train_stats        data_processed/train_stats.pkl \
    --model_type         lasso \
    --lasso_alpha        5e-4 \
    --lasso_max_iter     5000 \
    --model_save_name    models/lasso_model.pkl

# ElasticNet  (alpha=5e-4, l1_ratio=0.3)
python src/train.py \
    --engineered_dataset data_processed/train_engineered.parquet \
    --train_stats        data_processed/train_stats.pkl \
    --model_type         elasticnet \
    --en_alpha           5e-4 \
    --en_l1_ratio        0.3 \
    --en_max_iter        5000 \
    --model_save_name    models/elasticnet_model.pkl

# KNN  (k=10, distance-weighted)
python src/train.py \
    --engineered_dataset data_processed/train_engineered.parquet \
    --train_stats        data_processed/train_stats.pkl \
    --model_type         knn \
    --knn_neighbors      10 \
    --knn_weights        distance \
    --model_save_name    models/knn_model.pkl

# Random Forest  (200 trees, max_depth=20)
python src/train.py \
    --engineered_dataset data_processed/train_engineered.parquet \
    --train_stats        data_processed/train_stats.pkl \
    --model_type         randomforest \
    --rf_n_estimators    200 \
    --rf_max_depth       20 \
    --rf_n_jobs          -1 \
    --model_save_name    models/rf_model.pkl

# Gradient Boosting  (sklearn, slower but stable)
python src/train.py \
    --engineered_dataset data_processed/train_engineered.parquet \
    --train_stats        data_processed/train_stats.pkl \
    --model_type         gradientboosting \
    --gb_n_estimators    100 \
    --gb_learning_rate   0.1 \
    --gb_max_depth       3 \
    --model_save_name    models/gb_model.pkl

# HistGradientBoosting  (fast, recommended over GradientBoosting for large data)
python src/train.py \
    --engineered_dataset data_processed/train_engineered.parquet \
    --train_stats        data_processed/train_stats.pkl \
    --model_type         histgradientboosting \
    --hgb_max_depth      10 \
    --hgb_learning_rate  0.1 \
    --hgb_max_iter       100 \
    --model_save_name    models/hgb_model.pkl

# XGBoost  (requires: pip install xgboost)
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

# ─────────────────────────────────────────────────────────────────────────────
# MODE B: raw CSV  (FE runs inline; slower, no need to pre-run FE)
# ─────────────────────────────────────────────────────────────────────────────

python src/train.py \
    --dataset         split/train.csv \
    --model_type      histgradientboosting \
    --model_save_name models/hgb_model.pkl


# =============================================================================
# 3. EVALUATION / TESTING
# =============================================================================

# --- 3a. Test on raw CSV (FE applied inline using stored train_stats) ---
python src/test.py \
    --model   models/hgb_model.pkl \
    --dataset split/test.csv

# --- 3b. Test on pre-engineered Parquet (fast; skip FE) ---
python src/test.py \
    --model          models/hgb_model.pkl \
    --dataset        split/test_engineered.parquet \
    --pre-engineered

# --- 3c. Evaluate every saved model in sequence (raw CSV path) ---
for MODEL in models/*.pkl; do
    echo "========== $MODEL =========="
    python src/test.py --model "$MODEL" --dataset split/test.csv
done


# =============================================================================
# 4. TIPS
# =============================================================================
#
# • Run step 1b once (with --train_stats), then freely iterate over step 2
#   with different --model_type / hyperparameter flags.  FE is the bottleneck.
#
# • Step 1b requires --train_stats to be passed so the script knows where to
#   load the pkl it just wrote before processing the test set.
#
# • test.py reads the dataset with pd.read_parquet when --pre-engineered is
#   set; pass a .parquet file in that case (see 3b).  Without the flag it
#   reads a raw CSV and applies FE inline (see 3a).
#
# • For very large datasets, prefer --model_type histgradientboosting or
#   xgboost; both handle large feature matrices efficiently.
#
# • GradientBoostingRegressor (sklearn) is slow on large data.  Prefer
#   HistGradientBoosting or XGBoost for anything > 100 k rows.
#
# • All model pickles contain {'model', 'train_stats', 'feature_lists'},
#   so test.py works with any of them without extra flags.