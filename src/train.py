# src/train.py
"""
Model training script.

Supports two modes:
  1. Raw CSV  → runs feature engineering inline, then trains.
  2. Pre-engineered Parquet + train_stats.pkl  → skips FE, trains immediately.

Mode 2 is recommended: run feature_engineering.py once, then iterate over
different models/hyperparameters without paying the FE cost every time.
"""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from models_stratigies import (
    ElasticNetStrategy,
    GradientBoostingStrategy,
    HistGradientBoostingStrategy,
    KNNStrategy,
    LassoStrategy,
    RandomForestStrategy,
    RidgeStrategy,
    XGBoostStrategy,
)
from transformers import CyclicalFeatures
from utils_data import apply_feature_engineering, get_feature_lists


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def _build_preprocessor(feature_lists):
    """Build the sklearn preprocessing pipeline."""
    cyclical_encoder = CyclicalFeatures(
        cols    = feature_lists['cyclic'],
        periods = (24, 7, 12, 31),
    )

    column_transformer = ColumnTransformer([
        ('cyclical', cyclical_encoder,                               feature_lists['cyclic']),
        ('ohe',      OneHotEncoder(handle_unknown='ignore'),         feature_lists['categorical']),
        ('scaling',  QuantileTransformer(),                          feature_lists['numeric']),
        ('binary',   'passthrough',                                  feature_lists['binary']),
    ], remainder='drop')

    return Pipeline([('processor', column_transformer)])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(train, feature_lists, model_strategy):
    """Assemble and fit the full sklearn pipeline."""
    X_train = train[feature_lists['all']]
    y_train = train['log_trip_duration']

    steps = [('preprocessor', _build_preprocessor(feature_lists))]
    steps.extend(model_strategy.build_steps())

    full_pipeline = Pipeline(steps)
    full_pipeline.fit(X_train, y_train)
    return full_pipeline


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    project_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description='taxi_trip_train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data sources ---------------------------------------------------- #
    data_group = parser.add_argument_group('Data sources (choose one mode)')
    data_group.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to RAW training CSV. Feature engineering will be run inline.',
    )
    data_group.add_argument(
        '--engineered_dataset',
        type=str,
        default=None,
        help='Path to pre-engineered Parquet file (from feature_engineering.py). '
             'Skips FE; requires --train_stats.',
    )
    data_group.add_argument(
        '--train_stats',
        type=str,
        default=None,
        help='Path to train_stats.pkl produced by feature_engineering.py. '
             'Required when --engineered_dataset is used.',
    )
    data_group.add_argument(
        '--iqr_factor',
        type=float,
        default=2.5,
        help='IQR multiplier for outlier removal (only used with --dataset).',
    )

    # ---- Output ---------------------------------------------------------- #
    parser.add_argument(
        '--model_save_name',
        type=str,
        default=str(project_root / 'models' / 'taxi_model.pkl'),
        help='Destination path for the saved model pickle.',
    )

    # ---- Model selection ------------------------------------------------- #
    parser.add_argument(
        '--model_type',
        choices=['ridge', 'lasso', 'elasticnet', 'knn',
                 'randomforest', 'gradientboosting',
                 'histgradientboosting', 'xgboost'],
        default='ridge',
        help='Regression model to train.',
    )

    # ---- Ridge ----------------------------------------------------------- #
    ridge = parser.add_argument_group('Ridge options')
    ridge.add_argument('--poly_degree',  type=int,   default=4)
    ridge.add_argument('--ridge_alpha',  type=float, default=1.0)
    ridge.add_argument(
        '--feature_selection', dest='feature_selection', action='store_true',
        help='Enable Lasso-based feature selection before Ridge.',
    )
    ridge.add_argument(
        '--no_feature_selection', dest='feature_selection', action='store_false',
        help='Disable Lasso-based feature selection.',
    )
    parser.set_defaults(feature_selection=True)

    # ---- Lasso ----------------------------------------------------------- #
    lasso_g = parser.add_argument_group('Lasso / feature-selection options')
    lasso_g.add_argument('--lasso_alpha',    type=float, default=5e-4)
    lasso_g.add_argument('--lasso_max_iter', type=int,   default=5000)

    # ---- ElasticNet ------------------------------------------------------ #
    en = parser.add_argument_group('ElasticNet options')
    en.add_argument('--en_alpha',    type=float, default=5e-4)
    en.add_argument('--en_l1_ratio', type=float, default=0.3)
    en.add_argument('--en_max_iter', type=int,   default=5000)

    # ---- KNN ------------------------------------------------------------- #
    knn = parser.add_argument_group('KNN options')
    knn.add_argument('--knn_neighbors', type=int,   default=10)
    knn.add_argument('--knn_weights',   choices=['uniform', 'distance'], default='distance')

    # ---- Random Forest --------------------------------------------------- #
    rf = parser.add_argument_group('Random Forest options')
    rf.add_argument('--rf_n_estimators', type=int,   default=200)
    rf.add_argument('--rf_max_depth',    type=int,   default=20)
    rf.add_argument('--rf_n_jobs',       type=int,   default=-1)
    rf.add_argument('--rf_random_state', type=int,   default=42)

    # ---- GradientBoosting ------------------------------------------------ #
    gb = parser.add_argument_group('GradientBoosting options')
    gb.add_argument('--gb_n_estimators',  type=int,   default=100)
    gb.add_argument('--gb_learning_rate', type=float, default=0.1)
    gb.add_argument('--gb_max_depth',     type=int,   default=3)
    gb.add_argument('--gb_random_state',  type=int,   default=42)

    # ---- HistGradientBoosting -------------------------------------------- #
    hgb = parser.add_argument_group('HistGradientBoosting options')
    hgb.add_argument('--hgb_max_depth',     type=int,   default=10)
    hgb.add_argument('--hgb_learning_rate', type=float, default=0.1)
    hgb.add_argument('--hgb_max_iter',      type=int,   default=100)
    hgb.add_argument('--hgb_random_state',  type=int,   default=42)

    # ---- XGBoost --------------------------------------------------------- #
    xgb = parser.add_argument_group('XGBoost options')
    xgb.add_argument('--xgb_n_estimators',     type=int,   default=200)
    xgb.add_argument('--xgb_learning_rate',    type=float, default=0.1)
    xgb.add_argument('--xgb_max_depth',        type=int,   default=6)
    xgb.add_argument('--xgb_subsample',        type=float, default=0.8)
    xgb.add_argument('--xgb_colsample_bytree', type=float, default=0.8)
    xgb.add_argument('--xgb_n_jobs',           type=int,   default=-1)
    xgb.add_argument('--xgb_random_state',     type=int,   default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def build_strategy(args):
    t = args.model_type
    if t == 'ridge':
        print(f"Training Ridge (alpha={args.ridge_alpha}, poly_degree={args.poly_degree})...")
        return RidgeStrategy(
            poly_degree       = args.poly_degree,
            ridge_alpha       = args.ridge_alpha,
            feature_selection = args.feature_selection,
            lasso_alpha       = args.lasso_alpha,
            lasso_max_iter    = args.lasso_max_iter,
        )
    if t == 'lasso':
        print(f"Training Lasso (alpha={args.lasso_alpha}, max_iter={args.lasso_max_iter})...")
        return LassoStrategy(
            lasso_alpha    = args.lasso_alpha,
            lasso_max_iter = args.lasso_max_iter,
        )
    if t == 'elasticnet':
        print(f"Training ElasticNet (alpha={args.en_alpha}, l1_ratio={args.en_l1_ratio})...")
        return ElasticNetStrategy(
            en_alpha    = args.en_alpha,
            en_l1_ratio = args.en_l1_ratio,
            en_max_iter = args.en_max_iter,
        )
    if t == 'knn':
        print(f"Training KNN (neighbors={args.knn_neighbors}, weights={args.knn_weights})...")
        return KNNStrategy(
            knn_neighbors = args.knn_neighbors,
            knn_weights   = args.knn_weights,
        )
    if t == 'randomforest':
        print(f"Training RandomForest (n_estimators={args.rf_n_estimators}, max_depth={args.rf_max_depth})...")
        return RandomForestStrategy(
            rf_n_estimators = args.rf_n_estimators,
            rf_max_depth    = args.rf_max_depth,
            rf_n_jobs       = args.rf_n_jobs,
            rf_random_state = args.rf_random_state,
        )
    if t == 'gradientboosting':
        print(f"Training GradientBoosting (n_estimators={args.gb_n_estimators}, lr={args.gb_learning_rate})...")
        return GradientBoostingStrategy(
            gb_n_estimators  = args.gb_n_estimators,
            gb_learning_rate = args.gb_learning_rate,
            gb_max_depth     = args.gb_max_depth,
            gb_random_state  = args.gb_random_state,
        )
    if t == 'histgradientboosting':
        print(f"Training HistGradientBoosting (max_depth={args.hgb_max_depth}, lr={args.hgb_learning_rate})...")
        return HistGradientBoostingStrategy(
            hgb_max_depth     = args.hgb_max_depth,
            hgb_learning_rate = args.hgb_learning_rate,
            hgb_max_iter      = args.hgb_max_iter,
            hgb_random_state  = args.hgb_random_state,
        )
    if t == 'xgboost':
        print(f"Training XGBoost (n_estimators={args.xgb_n_estimators}, lr={args.xgb_learning_rate}, "
              f"max_depth={args.xgb_max_depth})...")
        return XGBoostStrategy(
            xgb_n_estimators     = args.xgb_n_estimators,
            xgb_learning_rate    = args.xgb_learning_rate,
            xgb_max_depth        = args.xgb_max_depth,
            xgb_subsample        = args.xgb_subsample,
            xgb_colsample_bytree = args.xgb_colsample_bytree,
            xgb_n_jobs           = args.xgb_n_jobs,
            xgb_random_state     = args.xgb_random_state,
        )
    raise ValueError(f"Unsupported model_type: {t}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Determine data loading mode ------------------------------------ #
    using_preengineered = args.engineered_dataset is not None

    if using_preengineered:
        if args.train_stats is None:
            raise ValueError(
                "--train_stats is required when using --engineered_dataset. "
                "Run feature_engineering.py first."
            )
        print(f"Loading pre-engineered data from: {args.engineered_dataset}")
        train_fe = pd.read_parquet(args.engineered_dataset)
        print(f"  Rows: {len(train_fe):,}")

        with open(args.train_stats, 'rb') as f:
            train_stats = pickle.load(f)
        print(f"  train_stats loaded from: {args.train_stats}")

    else:
        if args.dataset is None:
            raise ValueError("Provide either --dataset (raw CSV) or --engineered_dataset (Parquet).")
        print(f"Loading raw data from: {args.dataset}")
        train_raw = pd.read_csv(args.dataset)
        print(f"  Raw rows: {len(train_raw):,}")

        print("\nRunning feature engineering inline ...")
        train_fe, train_stats = apply_feature_engineering(
            train_raw, train_stats=None, iqr_factor=args.iqr_factor
        )
        print(f"  Engineered rows: {len(train_fe):,}")

    # ---- Build feature lists & strategy --------------------------------- #
    feature_lists = get_feature_lists()
    strategy      = build_strategy(args)

    # ---- Train ---------------------------------------------------------- #
    model = train_model(train_fe, feature_lists, strategy)

    # ---- Save ----------------------------------------------------------- #
    model_dict = {
        'model':         model,
        'train_stats':   train_stats,
        'feature_lists': feature_lists,
    }

    save_path = Path(args.model_save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(model_dict, f)

    print(f"\nModel saved to: {save_path}")


if __name__ == '__main__':
    main()