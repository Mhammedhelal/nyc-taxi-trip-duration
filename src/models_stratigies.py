
from abc import ABC, abstractmethod

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from transformers import ToDenseTransformer


class ModelStrategy(ABC):
    """Base class for model strategies."""

    @abstractmethod
    def build_steps(self):
        """Build and return pipeline steps for the model."""
        pass


class RidgeStrategy(ModelStrategy):
    """Strategy for Ridge regression with optional feature selection."""

    def __init__(self, poly_degree, ridge_alpha, feature_selection, lasso_alpha, lasso_max_iter):
        self.poly_degree       = poly_degree
        self.ridge_alpha       = ridge_alpha
        self.feature_selection = feature_selection
        self.lasso_alpha       = lasso_alpha
        self.lasso_max_iter    = lasso_max_iter

    def build_steps(self):
        """Build Ridge regression pipeline steps."""
        steps = []

        # Feature selection (optional)
        if self.feature_selection:
            print(f"Performing feature selection (Lasso alpha={self.lasso_alpha})...")
            lasso = lasso(alpha=self.lasso_alpha, max_iter=self.lasso_max_iter)
            steps.append(('selector', SelectFromModel(estimator=lasso, prefit=False)))

        # ColumnTransformer outputs a sparse matrix (OneHotEncoder default).
        # PolynomialFeatures does not accept sparse input, so densify first.
        steps.extend([
            ('to_dense',  ToDenseTransformer()),
            ('polynomial', PolynomialFeatures(degree=self.poly_degree)),
            ('regression', Ridge(alpha=self.ridge_alpha))
        ])

        return steps


class KNNStrategy(ModelStrategy):
    """Strategy for KNN regression."""

    def __init__(self, knn_neighbors, knn_weights):
        self.knn_neighbors = knn_neighbors
        self.knn_weights   = knn_weights

    def build_steps(self):
        """Build KNN regression pipeline steps."""
        return [
            ('to_dense', ToDenseTransformer()),
            ('scaler',   StandardScaler()),
            ('model',    KNeighborsRegressor(n_neighbors=self.knn_neighbors, weights=self.knn_weights))
        ]