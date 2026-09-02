from contextlib import asynccontextmanager
import os
from typing import Dict
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.anomaly_engine import GaussianAnomalyEngine
from app.baseline_model import SupervisedBaselineModel
from app.schemas import (
    ComparisonSummary,
    HealthResponse,
    TransactionInput,
    VerificationResponse,
)

# Global model state
gaussian_engine: GaussianAnomalyEngine = None
baseline_model: SupervisedBaselineModel = None


def load_models():
    """Load both trained model artifacts from the models/ directory."""
    global gaussian_engine, baseline_model

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gaussian_path = os.path.join(base_dir, "models", "gaussian_params.json")
    logistic_path = os.path.join(base_dir, "models", "logistic_model.pkl")

    if not os.path.exists(gaussian_path):
        raise RuntimeError(
            f"Gaussian parameters artifact not found at '{gaussian_path}'. "
            "Run scripts/train_and_export.py first."
        )
    if not os.path.exists(logistic_path):
        raise RuntimeError(
            f"Supervised model artifact not found at '{logistic_path}'. "
            "Run scripts/train_and_export.py first."
        )

    gaussian_engine = GaussianAnomalyEngine.load_params(gaussian_path)
    baseline_model = SupervisedBaselineModel.load_model(logistic_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to initialize models on startup."""
    load_models()
    yield


app = FastAPI(
    title="Credit Card Fraud Anomaly Detection & Supervised Baseline API",
    description=(
        "A dual-model inference service comparing from-scratch Unsupervised Gaussian "
        "Anomaly Detection against a Supervised Logistic Regression baseline on Kaggle credit card data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root_info():
    """Root metadata endpoint describing service capabilities and documentation."""
    return {
        "service": "Fraud Anomaly Detection & Supervised Baseline API",
        "version": "1.0.0",
        "description": (
            "Dual-model comparison API: Unsupervised Gaussian Anomaly Detection vs "
            "Supervised Logistic Regression Baseline."
        ),
        "docs_url": "/docs",
        "health_url": "/health",
        "verify_endpoint": "/v1/transactions/verify",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check API health and retrieve loaded models' test evaluation metrics."""
    global gaussian_engine, baseline_model

    if gaussian_engine is None or baseline_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not initialized.",
        )

    return HealthResponse(
        status="healthy",
        gaussian_model_loaded=gaussian_engine.mu is not None,
        supervised_model_loaded=baseline_model.model is not None,
        models_test_metrics={
            "gaussian_anomaly_model": gaussian_engine.metrics_test,
            "supervised_baseline_model": baseline_model.metrics_test,
        },
    )


@app.post(
    "/v1/transactions/verify",
    response_model=VerificationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
def verify_transaction(transaction: TransactionInput):
    """
    Score a transaction using BOTH the Unsupervised Gaussian Anomaly Engine
    and the Supervised Baseline Model.

    Accepts 28 PCA features (V1-V28) and raw transaction Amount (computing LogAmount internally).
    Returns individual model scores, decision flags, thresholds, and a comparative diagnostic summary.
    """
    global gaussian_engine, baseline_model

    if gaussian_engine is None or baseline_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not loaded.",
        )

    try:
        # Run Gaussian engine prediction
        gaussian_result = gaussian_engine.predict_transaction(transaction)

        # Run Supervised baseline prediction
        supervised_result = baseline_model.predict_transaction(transaction)

        # Comparative diagnostic logic
        if gaussian_result.is_anomaly and supervised_result.is_fraud:
            comp_status = "both_flagged"
            comp_message = (
                "High risk: Both the statistical Gaussian anomaly engine (unsupervised density) "
                "and the supervised baseline flagged this transaction."
            )
        elif supervised_result.is_fraud and not gaussian_result.is_anomaly:
            comp_status = "supervised_only"
            comp_message = (
                "Supervised model flagged fraud based on learned decision boundaries, "
                "while Gaussian density remained within nominal range."
            )
        elif gaussian_result.is_anomaly and not supervised_result.is_fraud:
            comp_status = "gaussian_only"
            comp_message = (
                "Gaussian anomaly engine detected extreme distribution tail behavior, "
                "though supervised classifier scored below fraud probability threshold."
            )
        else:
            comp_status = "both_cleared"
            comp_message = (
                "Low risk: Transaction aligns with normal Gaussian distributions "
                "and scored low fraud probability on the supervised model."
            )

        log_amount = float(np.log1p(transaction.Amount))

        return VerificationResponse(
            transaction_amount=round(float(transaction.Amount), 2),
            log_amount=round(log_amount, 6),
            gaussian_model=gaussian_result,
            supervised_model=supervised_result,
            comparison=ComparisonSummary(
                status=comp_status,
                message=comp_message,
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )
