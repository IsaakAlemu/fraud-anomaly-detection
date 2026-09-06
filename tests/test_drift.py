import numpy as np
import pandas as pd
import pytest
from scripts.monitor_drift import run_drift_report, create_drifted_data


def test_drift_report_on_synthetic_data():
    """Verify run_drift_report executes on synthetic DataFrames without exceptions and returns expected keys."""
    rng = np.random.default_rng(42)
    ref_df = pd.DataFrame({
        "V1": rng.normal(0, 1, 100),
        "V2": rng.normal(0, 1, 100),
        "LogAmount": rng.normal(3, 1, 100),
    })

    curr_df = pd.DataFrame({
        "V1": rng.normal(0, 1, 100),
        "V2": rng.normal(5, 1, 100),
        "LogAmount": rng.normal(3, 1, 100),
    })

    report, metrics = run_drift_report(ref_df, curr_df)

    assert isinstance(metrics, dict)
    assert "dataset_drift" in metrics
    assert "share_of_drifted_columns" in metrics
    assert isinstance(metrics["dataset_drift"], bool)
    assert isinstance(metrics["share_of_drifted_columns"], float)
    assert 0.0 <= metrics["share_of_drifted_columns"] <= 1.0


def test_create_drifted_data():
    """Verify create_drifted_data introduces shifts on targeted columns."""
    df = pd.DataFrame({
        "V4": np.zeros(50),
        "V11": np.zeros(50),
        "LogAmount": np.zeros(50),
        "V1": np.zeros(50),
    })

    drifted = create_drifted_data(df, random_state=42)
    assert not np.allclose(drifted["V4"], df["V4"])
    assert not np.allclose(drifted["V11"], df["V11"])
    assert not np.allclose(drifted["LogAmount"], df["LogAmount"])
    assert np.allclose(drifted["V1"], df["V1"])
