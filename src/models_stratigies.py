# src/models_stratigies.py
from abc import ABC, abstractmethod

from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from transformers import ToDenseTransformer


class ModelStrategy(ABC):
    """Base class for model strategies."""

    @abstractmethod
    def build_steps(self):
        """Build and return pipeline steps for the model."""
        pass


# ---------------------------------------------------------------------------
# Linear / regularised
# ---------------------------------------------------------------------------

class RidgeStrategy(ModelStrategy):
    """Strategy for Ridge regression with optional Lasso feature selection."""

    def __init__(self, poly_degree, ridge_alpha, feature_selection, lasso_alpha, lasso_max_iter):
        self.poly_degree       = poly_degree
        self.ridge_alpha       = ridge_alpha
        self.feature_selection = feature_selection
        self.lasso_alpha       = lasso_alpha
        self.lasso_max_iter    = lasso_max_iter

    def build_steps(self):
        steps = []

        if self.feature_selection:
            print(f"Performing feature selection (Lasso alpha={self.lasso_alpha})...")
            lasso_selector = Lasso(alpha=self.lasso_alpha, max_iter=self.lasso_max_iter)
            steps.append(('selector', SelectFromModel(estimator=lasso_selector, prefit=False)))

        steps.extend([
            ('to_dense',   ToDenseTransformer()),
            ('polynomial', PolynomialFeatures(degree=self.poly_degree)),
            ('regression', Ridge(alpha=self.ridge_alpha)),
        ])
        return steps


class LassoStrategy(ModelStrategy):
    """Strategy for Lasso regression."""

    def __init__(self, lasso_alpha, lasso_max_iter):
        self.lasso_alpha    = lasso_alpha
        self.lasso_max_iter = lasso_max_iter

    def build_steps(self):
        return [
            ('to_dense',   ToDenseTransformer()),
            ('regression', Lasso(alpha=self.lasso_alpha, max_iter=self.lasso_max_iter)),
        ]


class ElasticNetStrategy(ModelStrategy):
    """Strategy for ElasticNet regression."""

    def __init__(self, en_alpha, en_l1_ratio, en_max_iter):
        self.en_alpha    = en_alpha
        self.en_l1_ratio = en_l1_ratio
        self.en_max_iter = en_max_iter

    def build_steps(self):
        return [
            ('to_dense',   ToDenseTransformer()),
            ('regression', ElasticNet(
                alpha    = self.en_alpha,
                l1_ratio = self.en_l1_ratio,
                max_iter = self.en_max_iter,
            )),
        ]


# ---------------------------------------------------------------------------
# Neighbours
# ---------------------------------------------------------------------------

class KNNStrategy(ModelStrategy):
    """Strategy for KNN regression."""

    def __init__(self, knn_neighbors, knn_weights):
        self.knn_neighbors = knn_neighbors
        self.knn_weights   = knn_weights

    def build_steps(self):
        return [
            ('to_dense', ToDenseTransformer()),
            ('scaler',   StandardScaler()),
            ('model',    KNeighborsRegressor(
                n_neighbors = self.knn_neighbors,
                weights     = self.knn_weights,
            )),
        ]


# ---------------------------------------------------------------------------
# Tree-based ensembles
# ---------------------------------------------------------------------------

class RandomForestStrategy(ModelStrategy):
    """Strategy for Random Forest regression."""

    def __init__(self, rf_n_estimators, rf_max_depth, rf_n_jobs, rf_random_state):
        self.rf_n_estimators  = rf_n_estimators
        self.rf_max_depth     = rf_max_depth
        self.rf_n_jobs        = rf_n_jobs
        self.rf_random_state  = rf_random_state

    def build_steps(self):
        return [
            ('to_dense', ToDenseTransformer()),
            ('model',    RandomForestRegressor(
                n_estimators = self.rf_n_estimators,
                max_depth    = self.rf_max_depth,
                n_jobs       = self.rf_n_jobs,
                random_state = self.rf_random_state,
            )),
        ]


class GradientBoostingStrategy(ModelStrategy):
    """Strategy for sklearn GradientBoostingRegressor."""

    def __init__(self, gb_n_estimators, gb_learning_rate, gb_max_depth, gb_random_state):
        self.gb_n_estimators  = gb_n_estimators
        self.gb_learning_rate = gb_learning_rate
        self.gb_max_depth     = gb_max_depth
        self.gb_random_state  = gb_random_state

    def build_steps(self):
        return [
            ('to_dense', ToDenseTransformer()),
            ('model',    GradientBoostingRegressor(
                n_estimators  = self.gb_n_estimators,
                learning_rate = self.gb_learning_rate,
                max_depth     = self.gb_max_depth,
                random_state  = self.gb_random_state,
            )),
        ]


class HistGradientBoostingStrategy(ModelStrategy):
    """Strategy for sklearn HistGradientBoostingRegressor (fast, native NaN support)."""

    def __init__(self, hgb_max_depth, hgb_learning_rate, hgb_max_iter, hgb_random_state):
        self.hgb_max_depth     = hgb_max_depth
        self.hgb_learning_rate = hgb_learning_rate
        self.hgb_max_iter      = hgb_max_iter
        self.hgb_random_state  = hgb_random_state

    def build_steps(self):
        return [
            ('to_dense', ToDenseTransformer()),
            ('model',    HistGradientBoostingRegressor(
                max_depth     = self.hgb_max_depth,
                learning_rate = self.hgb_learning_rate,
                max_iter      = self.hgb_max_iter,
                random_state  = self.hgb_random_state,
            )),
        ]


class XGBoostStrategy(ModelStrategy):
    """Strategy for XGBoost regression."""

    def __init__(self, xgb_n_estimators, xgb_learning_rate, xgb_max_depth,
                 xgb_subsample, xgb_colsample_bytree, xgb_n_jobs, xgb_random_state):
        self.xgb_n_estimators    = xgb_n_estimators
        self.xgb_learning_rate   = xgb_learning_rate
        self.xgb_max_depth       = xgb_max_depth
        self.xgb_subsample       = xgb_subsample
        self.xgb_colsample_bytree = xgb_colsample_bytree
        self.xgb_n_jobs          = xgb_n_jobs
        self.xgb_random_state    = xgb_random_state

    def build_steps(self):
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError("xgboost is not installed. Run: pip install xgboost") from e

        return [
            ('to_dense', ToDenseTransformer()),
            ('model',    XGBRegressor(
                n_estimators      = self.xgb_n_estimators,
                learning_rate     = self.xgb_learning_rate,
                max_depth         = self.xgb_max_depth,
                subsample         = self.xgb_subsample,
                colsample_bytree  = self.xgb_colsample_bytree,
                n_jobs            = self.xgb_n_jobs,
                random_state      = self.xgb_random_state,
                tree_method       = 'hist',   # fast on CPU
                verbosity         = 0,
            )),
        ]