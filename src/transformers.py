# src/transformers.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalFeatures(BaseEstimator, TransformerMixin):
    """
    Encode cyclical features using sine/cosine transformation.

    Args:
        cols:    List of column names to transform.
        periods: Tuple/list of periods corresponding to each column
                 (e.g. 24 for hour, 7 for day-of-week).

    Input contract
    --------------
    Accepts both a pandas DataFrame (column names intact) and a numpy array
    (columns addressed by positional index).  The latter occurs when sklearn's
    ColumnTransformer passes a sliced sub-array instead of a sub-DataFrame.
    Output is always a 2-D numpy float64 array with shape
    (n_samples, 2 * len(cols)), ordered [col0_sin, col0_cos, col1_sin, ...].
    """

    def __init__(self, cols, periods):
        self.cols    = cols
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Normalise input to a plain 2-D numpy array so positional indexing
        # always works regardless of whether X arrived as a DataFrame or ndarray.
        if isinstance(X, pd.DataFrame):
            arr = X.values.astype(float)
        else:
            arr = np.asarray(X, dtype=float)

        parts = []
        for i, period in enumerate(self.periods):
            col_vals = arr[:, i]
            parts.append(np.sin(2 * np.pi * col_vals / period))
            parts.append(np.cos(2 * np.pi * col_vals / period))

        return np.column_stack(parts)

    def get_feature_names_out(self, input_features=None):
        names = []
        for col in self.cols:
            names.extend([f"{col}_sin", f"{col}_cos"])
        return np.array(names, dtype=object)


class ToDenseTransformer(BaseEstimator, TransformerMixin):
    """Convert sparse matrices from preprocessing into dense numpy arrays."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X)