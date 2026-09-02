import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from app.schemas import GaussianResult, TransactionInput


class GaussianAnomalyEngine:
    """
    Unsupervised Multivariate Gaussian Anomaly Detection Engine.

    Fits independent per-feature mean (mu) and variance (sigma^2) on normal transactions.
    Scores transactions by summing log-probabilities across features to avoid numerical underflow.
    Flags transactions as anomalies when log p(x) < epsilon.
    """

    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        sigma2: Optional[np.ndarray] = None,
        epsilon: Optional[float] = None,
        feature_cols: Optional[List[str]] = None,
        metrics_test: Optional[Dict[str, float]] = None,
        eps_numerical: float = 1e-10,
    ):
        self.mu = mu
        self.sigma2 = sigma2
        self.epsilon = epsilon
        self.feature_cols = feature_cols or [f"V{i}" for i in range(1, 29)] + ["LogAmount"]
        self.metrics_test = metrics_test or {}
        self.eps_numerical = eps_numerical

    def fit(self, X: Union[np.ndarray, pd.DataFrame], feature_cols: Optional[List[str]] = None) -> "GaussianAnomalyEngine":
        """
        Fit per-feature mean (mu) and variance (sigma2) on normal training data.
        """
        if feature_cols is not None:
            self.feature_cols = feature_cols

        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_cols].to_numpy()
        else:
            X_mat = np.asarray(X, dtype=float)

        self.mu = np.mean(X_mat, axis=0)
        # Using ddof=1 to match sample variance from pandas / EDA
        self.sigma2 = np.var(X_mat, axis=0, ddof=1)
        return self

    def log_density(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute summed log-Gaussian density across features for given input(s).

        log p(x) = sum_j [ -0.5 * ln(2 * pi * sigma_j^2 + eps) - (x_j - mu_j)^2 / (2 * sigma_j^2 + eps) ]
        """
        if self.mu is None or self.sigma2 is None:
            raise ValueError("GaussianAnomalyEngine has not been fitted or loaded with parameters.")

        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_cols].to_numpy()
        else:
            X_mat = np.asarray(X, dtype=float)

        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(1, -1)

        log_p = -0.5 * np.log(2.0 * np.pi * self.sigma2 + self.eps_numerical) - (
            (X_mat - self.mu) ** 2
        ) / (2.0 * self.sigma2 + self.eps_numerical)

        return np.sum(log_p, axis=1)

    def tune_threshold(
        self,
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series, List[int]],
        n_steps: int = 1000,
    ) -> float:
        """
        Tune epsilon threshold on a validation set to maximize F1 score.
        """
        val_log_p = self.log_density(X_val)
        y_true = np.asarray(y_val)

        thresholds = np.linspace(val_log_p.min(), val_log_p.max(), n_steps)
        best_f1 = -1.0
        best_eps = float(val_log_p.min())

        for eps in thresholds:
            preds = (val_log_p < eps).astype(int)
            f1 = float(f1_score(y_true, preds, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_eps = float(eps)

        self.epsilon = best_eps
        return best_eps

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict log-density and anomaly flag for input matrix.
        Returns (log_probabilities, is_anomaly_flags).
        """
        if self.epsilon is None:
            raise ValueError("Epsilon threshold has not been set or tuned.")

        log_probs = self.log_density(X)
        is_anomaly = log_probs < self.epsilon
        return log_probs, is_anomaly

    def predict_transaction(self, transaction: TransactionInput) -> GaussianResult:
        """
        Predict anomaly score and flag for a single TransactionInput.
        """
        feature_vector = transaction.to_feature_vector(self.feature_cols)
        log_prob = float(self.log_density(feature_vector)[0])
        is_anomaly = bool(log_prob < self.epsilon)

        return GaussianResult(
            log_probability=round(log_prob, 6),
            threshold=round(float(self.epsilon), 6),
            is_anomaly=is_anomaly,
        )

    def save_params(self, filepath: str) -> None:
        """
        Serialize parameters (mu, sigma2, epsilon, feature_cols) to JSON.
        """
        if self.mu is None or self.sigma2 is None or self.epsilon is None:
            raise ValueError("Cannot save uninitialized GaussianAnomalyEngine.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "feature_cols": self.feature_cols,
            "mu": {col: float(val) for col, val in zip(self.feature_cols, self.mu)},
            "sigma2": {col: float(val) for col, val in zip(self.feature_cols, self.sigma2)},
            "epsilon": float(self.epsilon),
            "metrics_test": self.metrics_test,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_params(cls, filepath: str) -> "GaussianAnomalyEngine":
        """
        Load parameters from JSON file into a new GaussianAnomalyEngine instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Gaussian params file not found at: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        feature_cols = data["feature_cols"]
        mu = np.array([data["mu"][col] for col in feature_cols], dtype=float)
        sigma2 = np.array([data["sigma2"][col] for col in feature_cols], dtype=float)
        epsilon = float(data["epsilon"])
        metrics_test = data.get("metrics_test", {})

        return cls(
            mu=mu,
            sigma2=sigma2,
            epsilon=epsilon,
            feature_cols=feature_cols,
            metrics_test=metrics_test,
        )
