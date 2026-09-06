"""
Optuna Hyperparameter Tuning for Supervised Logistic Regression.

This script:
1. Loads Kaggle credit card fraud data and creates train/val/test splits.
2. Uses Optuna to tune regularization strength (C) and solver for LogisticRegression.
3. Optimizes for PR-AUC (Average Precision Score) on the validation split.
4. Logs the best trial parameters, validation metric, trials table, and visualization to MLflow.
"""

import os
import tempfile
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


def load_and_split_data(data_path: str):
    """Load dataset and apply train/validation/test split logic."""
    df = pd.read_csv(data_path)
    df["LogAmount"] = np.log1p(df["Amount"])

    normal = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]

    # Train: only normal transactions
    train, normal_rest = train_test_split(normal, test_size=0.4, random_state=42)

    # Split remaining normal + all fraud into val/test
    val_normal, test_normal = train_test_split(normal_rest, test_size=0.5, random_state=42)
    val_fraud, test_fraud = train_test_split(fraud, test_size=0.5, random_state=42)

    val = pd.concat([val_normal, val_fraud])
    test = pd.concat([test_normal, test_fraud])

    # Supervised training data includes normal train + val fraud
    sup_train = pd.concat([train, val[val["Class"] == 1]])
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["LogAmount"]

    return sup_train, val, test, feature_cols


def create_objective(X_train, y_train, X_val, y_val):
    """Create an Optuna objective function for Logistic Regression."""
    def objective(trial: optuna.Trial) -> float:
        c_param = trial.suggest_float("C", 1e-4, 1e2, log=True)
        solver_param = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])

        clf = LogisticRegression(
            C=c_param,
            solver=solver_param,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_train, y_train)

        val_probs = clf.predict_proba(X_val)[:, 1]
        score = float(average_precision_score(y_val, val_probs))
        return score

    return objective


def run_tuning(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = 20,
    study_name: str = "logistic-regression-tuning",
    seed: int = 42,
) -> optuna.Study:
    """Execute Optuna study optimization."""
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(study_name=study_name, direction="maximize", sampler=sampler)
    objective = create_objective(X_train, y_train, X_val, y_val)
    study.optimize(objective, n_trials=n_trials)
    return study


def plot_optimization_history(study: optuna.Study):
    """Plot optimization history of trials."""
    trials = [t for t in study.trials if t.value is not None]
    trial_numbers = [t.number for t in trials]
    values = [t.value for t in trials]
    best_values = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(trial_numbers, values, color="tab:blue", alpha=0.7, label="Trial PR-AUC")
    ax.plot(trial_numbers, best_values, color="tab:red", linestyle="--", label="Best PR-AUC so far")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation PR-AUC")
    ax.set_title("Optuna Hyperparameter Optimization History")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, "data", "creditcard.csv")

    print(f"Loading data from {data_path}...")
    sup_train, val, test, feature_cols = load_and_split_data(data_path)

    X_train = sup_train[feature_cols]
    y_train = sup_train["Class"]
    X_val = val[feature_cols]
    y_val = val["Class"]

    print(f"Train size: {X_train.shape}, Val size: {X_val.shape}")
    print("Starting Optuna study: 'logistic-regression-tuning' (20 trials)...")

    study_name = "logistic-regression-tuning"
    study = run_tuning(X_train, y_train, X_val, y_val, n_trials=20, study_name=study_name)

    print("\n--- Optuna Tuning Results ---")
    print(f"Best Trial #{study.best_trial.number}")
    print(f"Best Validation PR-AUC: {study.best_value:.6f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Log to MLflow
    experiment_name = "fraud-anomaly-detection"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Optuna_LogisticRegression_Tuning"):
        mlflow.log_param("study_name", study_name)
        mlflow.log_param("n_trials", 20)
        mlflow.log_param("metric_optimized", "PR-AUC")
        for param_name, param_val in study.best_params.items():
            mlflow.log_param(f"best_{param_name}", param_val)

        mlflow.log_metric("best_val_pr_auc", study.best_value)
        mlflow.log_metric("best_trial_number", study.best_trial.number)

        # Save trials summary dataframe
        with tempfile.TemporaryDirectory() as tmp_dir:
            trials_csv_path = os.path.join(tmp_dir, "optuna_trials.csv")
            study.trials_dataframe().to_csv(trials_csv_path, index=False)
            mlflow.log_artifact(trials_csv_path, artifact_path="optuna_study")

        # Plot and log optimization history
        fig = plot_optimization_history(study)
        mlflow.log_figure(fig, "optuna_study/optimization_history.png")
        plt.close(fig)

        print("Logged Optuna tuning study and artifacts to MLflow successfully.")


if __name__ == "__main__":
    main()
