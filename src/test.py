import argparse
import pickle
import pandas as pd

from utils_eval import eval_model
from utils_data import apply_feature_engineering


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='taxi_trip_test')
    parser.add_argument('--model', type=str, default='models/taxi_model.pkl')
    parser.add_argument('--dataset', type=str, default='split/test.csv')
    args = parser.parse_args()
    
    # Load model
    with open(args.model, 'rb') as f:
        loaded_model_dict = pickle.load(f)
    
    model = loaded_model_dict['model']
    train_stats = loaded_model_dict['train_stats']
    feature_lists = loaded_model_dict['feature_lists']
    
    # Load and preprocess data
    data = pd.read_csv(args.dataset)
    data, _ = apply_feature_engineering(data, train_stats)
    
    # Evaluate
    eval_model(model, data, feature_lists)