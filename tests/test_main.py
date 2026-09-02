import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


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


def test_root_endpoint(client):
    """Test root service information endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["docs_url"] == "/docs"


def test_health_endpoint(client):
    """Test health check and model loading status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["gaussian_model_loaded"] is True
    assert data["supervised_model_loaded"] is True
    assert "gaussian_anomaly_model" in data["models_test_metrics"]
    assert "supervised_baseline_model" in data["models_test_metrics"]


def test_verify_normal_transaction(client, normal_payload):
    """Verify normal transaction returns 200 and low risk flags."""
    response = client.post("/v1/transactions/verify", json=normal_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_amount"] == 3.09
    assert "gaussian_model" in data
    assert "supervised_model" in data
    assert "comparison" in data

    # Normal transaction should not trigger anomalies or high fraud probability
    assert data["gaussian_model"]["is_anomaly"] is False
    assert data["supervised_model"]["is_fraud"] is False
    assert data["comparison"]["status"] == "both_cleared"


def test_verify_fraud_transaction(client, fraud_payload):
    """Verify fraud transaction returns 200 and triggers both detection flags."""
    response = client.post("/v1/transactions/verify", json=fraud_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_amount"] == 1.0
    assert data["gaussian_model"]["is_anomaly"] is True
    assert data["supervised_model"]["is_fraud"] is True
    assert data["comparison"]["status"] == "both_flagged"


def test_verify_missing_required_field(client, normal_payload):
    """Verify missing fields result in 422 Unprocessable Entity."""
    incomplete = normal_payload.copy()
    del incomplete["Amount"]
    response = client.post("/v1/transactions/verify", json=incomplete)
    assert response.status_code == 422

    del normal_payload["V1"]
    response = client.post("/v1/transactions/verify", json=normal_payload)
    assert response.status_code == 422


def test_verify_negative_amount(client, normal_payload):
    """Verify negative transaction Amount is rejected with 422."""
    invalid = normal_payload.copy()
    invalid["Amount"] = -50.0
    response = client.post("/v1/transactions/verify", json=invalid)
    assert response.status_code == 422


def test_verify_non_numeric_field(client, normal_payload):
    """Verify non-numeric string values trigger validation errors."""
    invalid = normal_payload.copy()
    invalid["V5"] = "invalid_string_not_a_float"
    response = client.post("/v1/transactions/verify", json=invalid)
    assert response.status_code == 422
