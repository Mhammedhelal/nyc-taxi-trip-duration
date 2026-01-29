import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer for encoding cyclical features using sine/cosine transformation.
    
    Args:
        cols: List of column names to transform
        periods: List of periods corresponding to each column
    """
    def __init__(self, cols, periods):
        self.cols = cols
        self.periods = periods
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X.copy()
        for col, period in zip(self.cols, self.periods):
            sin_col = np.sin(2 * np.pi * X_[col] / period)
            cos_col = np.cos(2 * np.pi * X_[col] / period)
            X_[f"{col}_sin"] = sin_col
            X_[f"{col}_cos"] = cos_col
            X_.drop(columns=[col], inplace=True)
        return X_


class ToDenseTransformer(BaseEstimator, TransformerMixin):
    """Convert sparse matrices from preprocessing into dense arrays."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else X