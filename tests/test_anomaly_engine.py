import json
import os
import tempfile
import numpy as np
import pytest

from app.anomaly_engine import GaussianAnomalyEngine
from app.schemas import TransactionInput


@pytest.fixture
def normal_payload():
    return {
        "V1": 0.448012, "V2": 2.035533, "V3": -0.608395, "V4": 4.684705,
        "V5": 1.296935, "V6": 0.211782, "V7": 0.763525, "V8": -0.036693,
        "V9": -1.621939, "V10": 1.435061, "V11": 1.131147, "V12": -0.965795,
        "V13": -0.791461, "V14": -2.463358, "V15": 0.065032, "V16": 0.892818,
        "V17": 1.871823, "V18": 1.494008, "V19": 1.176817, "V20": 0.464369,
        "V21": -0.473552, "V22": -0.962575, "V23": 0.082715, "V24": -0.728430,
        "V25": -0.700013, "V26": 0.032017, "V27": 0.244196, "V28": -0.094077,
        "Amount": 3.09,
    }


@pytest.fixture
def fraud_payload():
    return {
        "V1": -18.018561, "V2": 10.558600, "V3": -24.667741, "V4": 11.786180,
        "V5": -10.564657, "V6": -2.645681, "V7": -18.023468, "V8": 2.693655,
        "V9": -6.219464, "V10": -12.744761, "V11": 9.318138, "V12": -13.377857,
        "V13": 1.021924, "V14": -12.784346, "V15": -0.451146, "V16": -7.551365,
        "V17": -11.325997, "V18": -4.701253, "V19": 0.516317, "V20": 1.007703,
        "V21": -2.319479, "V22": 0.908839, "V23": 1.352904, "V24": -1.059222,
        "V25": 0.185751, "V26": 0.687037, "V27": 2.078081, "V28": -1.429517,
        "Amount": 1.00,
    }


def test_gaussian_math_exact():
    """Verify hand-calculated log-Gaussian density on 2D vectors."""
    mu = np.array([0.0, 0.0])
    sigma2 = np.array([1.0, 1.0])
    engine = GaussianAnomalyEngine(mu=mu, sigma2=sigma2, epsilon=-5.0, feature_cols=["V1", "V2"])

    # At the mean [0, 0], log p(x) = 2 * (-0.5 * ln(2*pi*1) - 0) = -ln(2*pi) ≈ -1.837877
    log_p_mean = engine.log_density(np.array([[0.0, 0.0]]))
    expected_mean = -np.log(2.0 * np.pi)
    assert np.isclose(log_p_mean[0], expected_mean, atol=1e-5)

    # Point far in tail [10, 10] should have extremely negative log_density
    log_p_tail = engine.log_density(np.array([[10.0, 10.0]]))
    assert log_p_tail[0] < -50.0


def test_fit_and_threshold_tuning():
    """Verify fitting and threshold search on synthetic normal and anomaly clusters."""
    rng = np.random.default_rng(42)
    # Normal cluster around 0 with variance 1
    X_train = rng.normal(loc=0.0, scale=1.0, size=(1000, 3))
    
    engine = GaussianAnomalyEngine()
    engine.fit(X_train, feature_cols=["F1", "F2", "F3"])
    assert len(engine.mu) == 3
    assert len(engine.sigma2) == 3

    # Validation: 200 normal + 20 anomalies far out
    X_val_norm = rng.normal(loc=0.0, scale=1.0, size=(200, 3))
    X_val_anom = rng.normal(loc=10.0, scale=1.0, size=(20, 3))
    X_val = np.vstack([X_val_norm, X_val_anom])
    y_val = np.array([0] * 200 + [1] * 20)

    best_eps = engine.tune_threshold(X_val, y_val)
    assert engine.epsilon == best_eps

    _, is_anom = engine.predict(X_val)
    # Anomalies should be correctly identified
    assert sum(is_anom[200:]) >= 18


def test_predict_transaction_with_saved_params(normal_payload, fraud_payload):
    """Verify scoring and threshold classification using the actual trained models/gaussian_params.json."""
    params_path = os.path.join(os.path.dirname(__file__), "..", "models", "gaussian_params.json")
    if not os.path.exists(params_path):
        pytest.skip("models/gaussian_params.json not found.")

    engine = GaussianAnomalyEngine.load_params(params_path)

    normal_tx = TransactionInput(**normal_payload)
    normal_res = engine.predict_transaction(normal_tx)
    assert normal_res.log_probability > engine.epsilon
    assert not normal_res.is_anomaly

    fraud_tx = TransactionInput(**fraud_payload)
    fraud_res = engine.predict_transaction(fraud_tx)
    assert fraud_res.log_probability < engine.epsilon
    assert fraud_res.is_anomaly


def test_save_and_load_roundtrip():
    """Verify parameters serialize to and load from JSON without precision loss."""
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["LogAmount"]
    mu = np.linspace(-1.0, 1.0, 29)
    sigma2 = np.ones(29) * 1.5
    engine = GaussianAnomalyEngine(
        mu=mu,
        sigma2=sigma2,
        epsilon=-219.939,
        feature_cols=feature_cols,
        metrics_test={"f1": 0.3868},
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        engine.save_params(tmp_path)
        loaded = GaussianAnomalyEngine.load_params(tmp_path)

        assert loaded.epsilon == engine.epsilon
        assert np.allclose(loaded.mu, engine.mu)
        assert np.allclose(loaded.sigma2, engine.sigma2)
        assert loaded.feature_cols == engine.feature_cols
        assert loaded.metrics_test["f1"] == 0.3868
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
