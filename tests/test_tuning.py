import numpy as np
import pytest
from scripts.tune_hyperparameters import create_objective, run_tuning


def test_tuning_objective_on_synthetic_data():
    """Verify tuning objective runs without error and returns float PR-AUC score on synthetic data."""
    rng = np.random.default_rng(42)
    # Synthetic dataset with normal and positive examples
    X_train = rng.normal(loc=0.0, scale=1.0, size=(100, 5))
    y_train = np.array([0] * 80 + [1] * 20)

    X_val = rng.normal(loc=0.0, scale=1.0, size=(50, 5))
    y_val = np.array([0] * 40 + [1] * 10)

    study = run_tuning(
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials=3,
        study_name="test-synthetic-tuning",
        seed=42,
    )

    assert len(study.trials) == 3
    assert study.best_value is not None
    assert 0.0 <= study.best_value <= 1.0


def test_best_params_contain_expected_keys():
    """Verify best_params contains 'C' and 'solver' with valid types and ranges."""
    rng = np.random.default_rng(123)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(80, 4))
    y_train = np.array([0] * 70 + [1] * 10)

    X_val = rng.normal(loc=0.0, scale=1.0, size=(40, 4))
    y_val = np.array([0] * 35 + [1] * 5)

    study = run_tuning(
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials=2,
        study_name="test-keys-tuning",
        seed=123,
    )

    best_params = study.best_params
    assert "C" in best_params
    assert "solver" in best_params
    assert isinstance(best_params["C"], float)
    assert 1e-4 <= best_params["C"] <= 1e2
    assert best_params["solver"] in ["lbfgs", "liblinear"]
