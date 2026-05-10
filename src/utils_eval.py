# src/utils_eval.py
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error


def eval_model(model, data, feature_lists):
    """
    Evaluate regression model on data.
    
    Args:
        model: Trained model pipeline
        data: DataFrame with features and target
        feature_lists: Dictionary of feature categories
    """
    X = data[feature_lists['all']]
    y = data['log_trip_duration']
    
    # Predict
    y_pred = model.predict(X)
    
    # Calculate metrics
    rmse = root_mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # Print results
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mae': mae
    }