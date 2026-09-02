from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class TransactionInput(BaseModel):
    """Transaction input payload with 28 PCA features and transaction Amount."""

    V1: float = Field(..., description="Anonymized PCA component V1")
    V2: float = Field(..., description="Anonymized PCA component V2")
    V3: float = Field(..., description="Anonymized PCA component V3")
    V4: float = Field(..., description="Anonymized PCA component V4")
    V5: float = Field(..., description="Anonymized PCA component V5")
    V6: float = Field(..., description="Anonymized PCA component V6")
    V7: float = Field(..., description="Anonymized PCA component V7")
    V8: float = Field(..., description="Anonymized PCA component V8")
    V9: float = Field(..., description="Anonymized PCA component V9")
    V10: float = Field(..., description="Anonymized PCA component V10")
    V11: float = Field(..., description="Anonymized PCA component V11")
    V12: float = Field(..., description="Anonymized PCA component V12")
    V13: float = Field(..., description="Anonymized PCA component V13")
    V14: float = Field(..., description="Anonymized PCA component V14")
    V15: float = Field(..., description="Anonymized PCA component V15")
    V16: float = Field(..., description="Anonymized PCA component V16")
    V17: float = Field(..., description="Anonymized PCA component V17")
    V18: float = Field(..., description="Anonymized PCA component V18")
    V19: float = Field(..., description="Anonymized PCA component V19")
    V20: float = Field(..., description="Anonymized PCA component V20")
    V21: float = Field(..., description="Anonymized PCA component V21")
    V22: float = Field(..., description="Anonymized PCA component V22")
    V23: float = Field(..., description="Anonymized PCA component V23")
    V24: float = Field(..., description="Anonymized PCA component V24")
    V25: float = Field(..., description="Anonymized PCA component V25")
    V26: float = Field(..., description="Anonymized PCA component V26")
    V27: float = Field(..., description="Anonymized PCA component V27")
    V28: float = Field(..., description="Anonymized PCA component V28")
    Amount: float = Field(..., ge=0.0, description="Transaction amount (must be non-negative)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "V1": -29.200329,
                "V2": 16.155701,
                "V3": -30.013712,
                "V4": 6.476731,
                "V5": -21.225810,
                "V6": -4.902997,
                "V7": -19.791248,
                "V8": 19.168327,
                "V9": -3.617242,
                "V10": -7.870122,
                "V11": 4.066255,
                "V12": -5.661492,
                "V13": 1.292950,
                "V14": -5.079846,
                "V15": -0.126523,
                "V16": -5.244472,
                "V17": -11.274973,
                "V18": -4.678437,
                "V19": 0.650807,
                "V20": 1.715862,
                "V21": 1.809371,
                "V22": -2.175815,
                "V23": -1.365104,
                "V24": 0.174286,
                "V25": 2.103868,
                "V26": -0.209944,
                "V27": 1.278681,
                "V28": 0.372393,
                "Amount": 99.99,
            }
        }
    )

    def to_feature_vector(self, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """
        Convert transaction input into the 29-feature array (V1-V28 + LogAmount).
        LogAmount is computed internally as log1p(Amount).
        """
        if feature_cols is None:
            feature_cols = [f"V{i}" for i in range(1, 29)] + ["LogAmount"]

        log_amount = float(np.log1p(self.Amount))
        feature_dict: Dict[str, float] = {
            f"V{i}": getattr(self, f"V{i}") for i in range(1, 29)
        }
        feature_dict["LogAmount"] = log_amount

        return np.array([[feature_dict[col] for col in feature_cols]], dtype=float)


class GaussianResult(BaseModel):
    """Scoring result from the unsupervised Gaussian anomaly engine."""

    log_probability: float = Field(..., description="Summed log Gaussian density across 29 features")
    threshold: float = Field(..., description="Tuned decision boundary epsilon")
    is_anomaly: bool = Field(..., description="True if log_probability < threshold")
    model_type: str = Field(
        default="Unsupervised Gaussian (from-scratch density estimation)",
        description="Model architecture description",
    )


class SupervisedResult(BaseModel):
    """Scoring result from the supervised Logistic Regression baseline."""

    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted fraud probability P(Class=1)")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Tuned decision boundary probability threshold")
    is_fraud: bool = Field(..., description="True if fraud_probability >= threshold")
    model_type: str = Field(
        default="Supervised Logistic Regression (balanced class weights)",
        description="Model architecture description",
    )


class ComparisonSummary(BaseModel):
    """Diagnostic comparison between the unsupervised and supervised model predictions."""

    status: str = Field(..., description="Classification category: both_flagged, supervised_only, gaussian_only, or both_cleared")
    message: str = Field(..., description="Human-readable interpretation of model agreement or divergence")


class VerificationResponse(BaseModel):
    """Comprehensive transaction verification response containing both model evaluations."""

    transaction_amount: float = Field(..., description="Original transaction Amount")
    log_amount: float = Field(..., description="Computed log1p(Amount)")
    gaussian_model: GaussianResult = Field(..., description="Gaussian anomaly detection engine evaluation")
    supervised_model: SupervisedResult = Field(..., description="Supervised baseline logistic regression evaluation")
    comparison: ComparisonSummary = Field(..., description="Comparative analysis summary")


class HealthResponse(BaseModel):
    """API health status and loaded model metadata."""

    status: str
    gaussian_model_loaded: bool
    supervised_model_loaded: bool
    models_test_metrics: Dict[str, Dict[str, float]]
