import argparse
import pickle
from pathlib import Path

import pandas as pd

from utils_eval import eval_model
from utils_data import apply_feature_engineering


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description='taxi_trip_test')
    parser.add_argument('--model', type=str, default=str(project_root / 'models' / 'taxi_model.pkl'))
    parser.add_argument('--dataset', type=str, default=str(project_root / 'split' / 'test.csv'))
    parser.add_argument('--pre-engineered', action='store_true', help='Skip feature engineering if data is pre-engineered')
    args = parser.parse_args()
    
    # Load model
    with open(args.model, 'rb') as f:
        loaded_model_dict = pickle.load(f)
    
    model = loaded_model_dict['model']
    train_stats = loaded_model_dict['train_stats']
    feature_lists = loaded_model_dict['feature_lists']
    
    # Load and preprocess data
    data = pd.read_csv(args.dataset)
    if not args.pre_engineered:
        data, _ = apply_feature_engineering(data, train_stats)
    
    # Evaluate
    eval_model(model, data, feature_lists)