"""
Data Drift Detection Pipeline using Evidently AI and MLflow.

This script:
1. Loads reference data (train split) and baseline current data (test split).
2. Simulates drifted data by applying shifts to 'Amount', 'LogAmount', 'V4', and 'V11'.
3. Computes Evidently DataDrift reports for both clean and drifted batches.
4. Exports HTML and JSON reports to reports/.
5. Logs drift metrics and report artifacts to MLflow under 'Data_Drift_Monitoring'.
"""

import json
import os
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset


def load_reference_and_current_data(data_path: str):
    """Load data using existing split logic and prepare feature matrices."""
    df = pd.read_csv(data_path)
    df["LogAmount"] = np.log1p(df["Amount"])

    normal = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]

    train, normal_rest = train_test_split(normal, test_size=0.4, random_state=42)
    val_normal, test_normal = train_test_split(normal_rest, test_size=0.5, random_state=42)
    val_fraud, test_fraud = train_test_split(fraud, test_size=0.5, random_state=42)

    test = pd.concat([test_normal, test_fraud])
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["LogAmount"]

    reference_data = train[feature_cols].copy()
    current_data = test[feature_cols].copy()

    return reference_data, current_data, feature_cols


def create_drifted_data(current_data: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Create simulated drifted batch with perturbed features."""
    drifted = current_data.copy()
    rng = np.random.default_rng(random_state)

    # Shift 'V4' and 'V11' (distribution shift)
    if "V4" in drifted.columns:
        drifted["V4"] = drifted["V4"] + rng.normal(loc=2.5, scale=1.0, size=len(drifted))
    if "V11" in drifted.columns:
        drifted["V11"] = drifted["V11"] + rng.normal(loc=3.0, scale=1.2, size=len(drifted))

    # Shift LogAmount (simulate higher transaction amounts)
    if "LogAmount" in drifted.columns:
        drifted["LogAmount"] = drifted["LogAmount"] + float(np.log(3.5))

    return drifted


def run_drift_report(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    """Run Evidently DataDrift report and extract summary metrics."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    res_dict = report.as_dict()

    drift_metrics = {}
    for metric_info in res_dict.get("metrics", []):
        if metric_info.get("metric") == "DatasetDriftMetric":
            result = metric_info.get("result", {})
            drift_metrics = {
                "dataset_drift": bool(result.get("dataset_drift", False)),
                "number_of_drifted_columns": int(result.get("number_of_drifted_columns", 0)),
                "share_of_drifted_columns": float(result.get("share_of_drifted_columns", 0.0)),
                "number_of_columns": int(result.get("number_of_columns", 0)),
            }
            break

    return report, drift_metrics


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(repo_root, "data", "creditcard.csv")
    reports_dir = os.path.join(repo_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    print(f"Loading data from {data_path}...")
    reference_data, current_data, feature_cols = load_reference_and_current_data(data_path)
    drifted_data = create_drifted_data(current_data)

    print(f"Reference data shape: {reference_data.shape}")
    print(f"Current data shape:   {current_data.shape}")
    print(f"Drifted data shape:   {drifted_data.shape}")

    print("\n--- Running Drift Analysis (Reference vs Clean Test) ---")
    clean_report, clean_metrics = run_drift_report(reference_data, current_data)
    print(f"Clean Batch Drift Detected: {clean_metrics.get('dataset_drift')}")
    print(f"Drifted Columns: {clean_metrics.get('number_of_drifted_columns')}/{clean_metrics.get('number_of_columns')} ({clean_metrics.get('share_of_drifted_columns'):.2%})")

    print("\n--- Running Drift Analysis (Reference vs Drifted Test) ---")
    drifted_report, drifted_metrics = run_drift_report(reference_data, drifted_data)
    print(f"Shifted Batch Drift Detected: {drifted_metrics.get('dataset_drift')}")
    print(f"Drifted Columns: {drifted_metrics.get('number_of_drifted_columns')}/{drifted_metrics.get('number_of_columns')} ({drifted_metrics.get('share_of_drifted_columns'):.2%})")

    # Save reports locally
    html_report_path = os.path.join(reports_dir, "data_drift_report.html")
    drifted_report.save_html(html_report_path)
    print(f"Saved HTML report to {html_report_path}")

    summary_json_path = os.path.join(reports_dir, "data_drift_summary.json")
    summary_data = {
        "clean_batch": clean_metrics,
        "drifted_batch": drifted_metrics,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved JSON summary to {summary_json_path}")

    # Log to MLflow
    experiment_name = "fraud-anomaly-detection"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Data_Drift_Monitoring"):
        mlflow.log_metrics({
            "share_of_drifted_columns": drifted_metrics["share_of_drifted_columns"],
            "number_of_drifted_columns": float(drifted_metrics["number_of_drifted_columns"]),
            "clean_share_of_drifted_columns": clean_metrics["share_of_drifted_columns"],
            "clean_number_of_drifted_columns": float(clean_metrics["number_of_drifted_columns"]),
        })
        mlflow.log_params({
            "dataset_drift": str(drifted_metrics["dataset_drift"]),
            "clean_dataset_drift": str(clean_metrics["dataset_drift"]),
            "total_columns": str(drifted_metrics["number_of_columns"]),
        })

        mlflow.log_artifact(html_report_path, artifact_path="reports")
        mlflow.log_artifact(summary_json_path, artifact_path="reports")
        print("Logged Data Drift Monitoring run and report artifacts to MLflow successfully.")


if __name__ == "__main__":
    main()
