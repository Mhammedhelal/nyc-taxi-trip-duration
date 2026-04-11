import argparse
import pickle
from pathlib import Path

import pandas as pd

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor

from utils_data import apply_feature_engineering, get_feature_lists
from transformers import CyclicalFeatures, ToDenseTransformer


def _build_preprocessor(feature_lists):
    """Build the preprocessing pipeline."""
    cyclical_encoder = CyclicalFeatures(
        cols=feature_lists['cyclic'],
        periods=(24, 7, 12, 31)
    )
    
    column_transformer = ColumnTransformer([
        ('cyclical', cyclical_encoder, feature_lists['cyclic']),
        ('ohe', OneHotEncoder(handle_unknown='ignore'), feature_lists['categorical']),
        ('scaling', QuantileTransformer(), feature_lists['numeric']),
        ('binary', 'passthrough', feature_lists['binary'])
    ], remainder='drop')
    
    return Pipeline([('processor', column_transformer)])


def _build_ridge_pipeline(poly_degree, ridge_alpha):
    """Build Ridge regression pipeline steps."""
    return [
        ('polynomial', PolynomialFeatures(degree=poly_degree)),
        ('regression', Ridge(alpha=ridge_alpha))
    ]


def _build_knn_pipeline(knn_neighbors, knn_weights):
    """Build KNN regression pipeline steps."""
    return [
        ('to_dense', ToDenseTransformer()),
        ('scaler', StandardScaler()),
        ('model', KNeighborsRegressor(n_neighbors=knn_neighbors, weights=knn_weights))
    ]


def train_model(
    train,
    feature_lists,
    poly_degree,
    ridge_alpha,
    feature_selection,
    lasso_alpha,
    lasso_max_iter,
    model_type,
    knn_neighbors,
    knn_weights,
):
    """Train taxi trip duration prediction model."""
    
    # Prepare data
    X_train = train[feature_lists['all']]
    y_train = train['log_trip_duration']
    
    # Build full pipeline steps
    steps = [('preprocessor', _build_preprocessor(feature_lists))]
    
    # Add feature selection if applicable
    if feature_selection and model_type == 'ridge':
        print(f"Performing feature selection (Lasso alpha={lasso_alpha})...")
        lasso = Lasso(alpha=lasso_alpha, max_iter=lasso_max_iter)
        selector = SelectFromModel(estimator=lasso, prefit=False)
        steps.append(('selector', selector))
    
    # Add model-specific pipeline
    if model_type == 'ridge':
        print(f"Training model (Ridge alpha={ridge_alpha}, poly degree={poly_degree})...")
        steps.extend(_build_ridge_pipeline(poly_degree, ridge_alpha))
    elif model_type == 'knn':
        print(f"Training model (KNN neighbors={knn_neighbors}, weights={knn_weights})...")
        steps.extend(_build_knn_pipeline(knn_neighbors, knn_weights))
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    # Build and fit complete pipeline
    full_pipeline = Pipeline(steps)
    full_pipeline.fit(X_train, y_train)
    
    return full_pipeline


def parse_args():
    project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description='taxi_trip_train')
    parser.add_argument('--dataset', type=str, default=str(project_root / 'split' / 'train.csv'))
    parser.add_argument('--model_save_name', type=str, default=str(project_root / 'models' / 'taxi_model.pkl'))
    parser.add_argument('--poly_degree', type=int, default=4)
    parser.add_argument('--ridge_alpha', type=float, default=1.0)
    parser.add_argument('--model_type', choices=['ridge', 'knn'], default='ridge')
    parser.add_argument('--feature_selection', dest='feature_selection', action='store_true',
                        help='Enable Lasso-based feature selection prior to ridge regression.')
    parser.add_argument('--no_feature_selection', dest='feature_selection', action='store_false',
                        help='Disable Lasso-based feature selection.')
    parser.set_defaults(feature_selection=True)
    parser.add_argument('--lasso_alpha', type=float, default=0.01)
    parser.add_argument('--lasso_max_iter', type=int, default=10000)
    parser.add_argument('--knn_neighbors', type=int, default=10)
    parser.add_argument('--knn_weights', choices=['uniform', 'distance'], default='distance')
    parser.add_argument('--iqr_factor', type=float, default=2.5,
                        help='Outlier removal factor applied to IQR bounds during feature engineering.')
    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()
    
    # Load data
    train = pd.read_csv(args.dataset)
    
    # Feature engineering
    train_fe, train_stats = apply_feature_engineering(train, iqr_factor=args.iqr_factor)
    print(f"Training rows after feature engineering: {len(train_fe)}")
    
    feature_lists = get_feature_lists()
    
    # Train model
    model = train_model(
        train_fe,
        feature_lists,
        poly_degree=args.poly_degree,
        ridge_alpha=args.ridge_alpha,
        feature_selection=args.feature_selection,
        lasso_alpha=args.lasso_alpha,
        lasso_max_iter=args.lasso_max_iter,
        model_type=args.model_type,
        knn_neighbors=args.knn_neighbors,
        knn_weights=args.knn_weights
    )
    
    # Save model with stats
    model_dict = {
        'model': model,
        'train_stats': train_stats,
        'feature_lists': feature_lists
    }
    
    save_path = Path(args.model_save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_dict, f)
    
    print(f"\nModel saved to: {save_path}")


if __name__ == '__main__':
    main()