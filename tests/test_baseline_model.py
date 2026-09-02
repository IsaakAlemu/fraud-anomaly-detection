import os
import tempfile
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from app.baseline_model import SupervisedBaselineModel
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


def test_baseline_predictions_range():
    """Verify Logistic Regression outputs valid probabilities in [0, 1]."""
    X_train = np.random.randn(100, 29)
    y_train = np.array([0] * 90 + [1] * 10)
    clf = LogisticRegression().fit(X_train, y_train)

    wrapper = SupervisedBaselineModel(model=clf, threshold=0.8)
    probs, flags = wrapper.predict(X_train)

    assert probs.shape == (100,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert ((probs >= 0.8) == flags).all()


def test_predict_transaction_with_saved_model(normal_payload, fraud_payload):
    """Verify supervised predictions on test vectors using models/logistic_model.pkl."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "logistic_model.pkl")
    if not os.path.exists(model_path):
        pytest.skip("models/logistic_model.pkl not found.")

    wrapper = SupervisedBaselineModel.load_model(model_path)

    normal_tx = TransactionInput(**normal_payload)
    normal_res = wrapper.predict_transaction(normal_tx)
    assert normal_res.fraud_probability < wrapper.threshold
    assert not normal_res.is_fraud

    fraud_tx = TransactionInput(**fraud_payload)
    fraud_res = wrapper.predict_transaction(fraud_tx)
    assert fraud_res.fraud_probability >= wrapper.threshold
    assert fraud_res.is_fraud


def test_model_save_and_load_roundtrip():
    """Verify model artifact joblib serialization and deserialization."""
    X = np.random.randn(50, 5)
    y = np.array([0] * 40 + [1] * 10)
    clf = LogisticRegression().fit(X, y)
    feature_cols = [f"F{i}" for i in range(5)]

    wrapper = SupervisedBaselineModel(
        model=clf,
        threshold=0.999,
        feature_cols=feature_cols,
        metrics_test={"f1": 0.8428},
    )

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        wrapper.save_model(tmp_path)
        loaded = SupervisedBaselineModel.load_model(tmp_path)

        assert loaded.threshold == 0.999
        assert loaded.feature_cols == feature_cols
        assert loaded.metrics_test["f1"] == 0.8428

        test_vec = np.random.randn(1, 5)
        np.testing.assert_allclose(
            wrapper.predict_proba(test_vec),
            loaded.predict_proba(test_vec),
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
