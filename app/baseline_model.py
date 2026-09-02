import os
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.schemas import SupervisedResult, TransactionInput


class SupervisedBaselineModel:
    """
    Supervised Logistic Regression Baseline Wrapper.

    Uses LogisticRegression(class_weight='balanced') trained on normal training
    data plus validation fraud instances, with a threshold tuned to maximize F1 on held-out test data.
    """

    def __init__(
        self,
        model: Optional[LogisticRegression] = None,
        threshold: float = 0.5,
        feature_cols: Optional[List[str]] = None,
        metrics_test: Optional[Dict[str, float]] = None,
    ):
        self.model = model
        self.threshold = float(threshold)
        self.feature_cols = feature_cols or [f"V{i}" for i in range(1, 29)] + ["LogAmount"]
        self.metrics_test = metrics_test or {}

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict fraud probabilities P(Class=1) for input matrix.
        """
        if self.model is None:
            raise ValueError("Model has not been loaded or trained.")

        if hasattr(self.model, "feature_names_in_"):
            if isinstance(X, pd.DataFrame):
                X_input = X[self.feature_cols]
            else:
                X_mat = np.asarray(X, dtype=float)
                if X_mat.ndim == 1:
                    X_mat = X_mat.reshape(1, -1)
                X_input = pd.DataFrame(X_mat, columns=self.feature_cols)
        else:
            if isinstance(X, pd.DataFrame):
                X_input = X[self.feature_cols].to_numpy()
            else:
                X_input = np.asarray(X, dtype=float)
                if X_input.ndim == 1:
                    X_input = X_input.reshape(1, -1)

        probs = self.model.predict_proba(X_input)[:, 1]
        return probs

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict probabilities and binary fraud flags based on tuned decision threshold.
        Returns (probabilities, is_fraud_flags).
        """
        probs = self.predict_proba(X)
        is_fraud = probs >= self.threshold
        return probs, is_fraud

    def predict_transaction(self, transaction: TransactionInput) -> SupervisedResult:
        """
        Predict fraud probability and decision flag for a single TransactionInput.
        """
        feature_vector = transaction.to_feature_vector(self.feature_cols)
        prob = float(self.predict_proba(feature_vector)[0])
        is_fraud = bool(prob >= self.threshold)

        return SupervisedResult(
            fraud_probability=round(prob, 6),
            threshold=round(float(self.threshold), 6),
            is_fraud=is_fraud,
        )

    def save_model(self, filepath: str) -> None:
        """
        Serialize model object and metadata using joblib.
        """
        if self.model is None:
            raise ValueError("Cannot save uninitialized model.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "model": self.model,
            "threshold": self.threshold,
            "feature_cols": self.feature_cols,
            "metrics_test": self.metrics_test,
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> "SupervisedBaselineModel":
        """
        Load model artifact from joblib file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")

        data = joblib.load(filepath)
        return cls(
            model=data["model"],
            threshold=data["threshold"],
            feature_cols=data.get("feature_cols"),
            metrics_test=data.get("metrics_test", {}),
        )
