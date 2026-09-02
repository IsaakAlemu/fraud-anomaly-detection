"""
Training and export pipeline for fraud anomaly detection and supervised baseline.

This script:
1. Loads Kaggle credit card fraud data.
2. Applies LogAmount = log1p(Amount).
3. Splits data into:
   - train: 170,589 normal transactions (for fitting Gaussian).
   - val: 57,109 transactions (56,863 normal, 246 fraud) (for threshold tuning).
   - test: 57,109 transactions (56,863 normal, 246 fraud) (held out).
4. Fits Gaussian parameters (mu, sigma2) on train and tunes epsilon on val.
5. Trains LogisticRegression(class_weight='balanced') on train + val fraud and tunes threshold on test.
6. Exports models/gaussian_params.json and models/logistic_model.pkl.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def log_gaussian_prob(X: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """Calculate summed log Gaussian density across features."""
    eps = 1e-10
    log_p = -0.5 * np.log(2 * np.pi * sigma2 + eps) - ((X - mu) ** 2) / (2 * sigma2 + eps)
    return np.sum(log_p, axis=1)


def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df["LogAmount"] = np.log1p(df["Amount"])

    # Separate normal and fraud
    normal = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]

    # Train: only normal transactions (used to fit the Gaussian)
    train, normal_rest = train_test_split(normal, test_size=0.4, random_state=42)

    # Split remaining normal + all fraud into val/test
    val_normal, test_normal = train_test_split(normal_rest, test_size=0.5, random_state=42)
    val_fraud, test_fraud = train_test_split(fraud, test_size=0.5, random_state=42)

    val = pd.concat([val_normal, val_fraud])
    test = pd.concat([test_normal, test_fraud])

    print(f"Data splits: Train={train.shape} (all normal), Val={val.shape} ({val['Class'].sum()} fraud), Test={test.shape} ({test['Class'].sum()} fraud)")

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["LogAmount"]

    # 1. Fit Gaussian model
    mu = train[feature_cols].mean().to_dict()
    sigma2 = train[feature_cols].var().to_dict()

    mu_vec = np.array([mu[col] for col in feature_cols])
    sigma2_vec = np.array([sigma2[col] for col in feature_cols])

    # Tune epsilon on validation set
    val_log_p = log_gaussian_prob(val[feature_cols].to_numpy(), mu_vec, sigma2_vec)
    thresholds = np.linspace(val_log_p.min(), val_log_p.max(), 1000)

    best_f1_gaussian = 0.0
    best_epsilon = float(val_log_p.min())

    for eps in thresholds:
        preds = (val_log_p < eps).astype(int)
        f1 = f1_score(val["Class"], preds)
        if f1 > best_f1_gaussian:
            best_f1_gaussian = float(f1)
            best_epsilon = float(eps)

    # Evaluate Gaussian on held-out test
    test_log_p = log_gaussian_prob(test[feature_cols].to_numpy(), mu_vec, sigma2_vec)
    test_preds_gaussian = (test_log_p < best_epsilon).astype(int)

    gauss_prec = precision_score(test["Class"], test_preds_gaussian)
    gauss_rec = recall_score(test["Class"], test_preds_gaussian)
    gauss_f1 = f1_score(test["Class"], test_preds_gaussian)

    print("\n--- Gaussian Anomaly Detection Results (Test Set) ---")
    print(f"Epsilon: {best_epsilon:.6f}")
    print(f"Precision: {gauss_prec:.4f}")
    print(f"Recall:    {gauss_rec:.4f}")
    print(f"F1-Score:  {gauss_f1:.4f}")

    gaussian_params = {
        "feature_cols": feature_cols,
        "mu": mu,
        "sigma2": sigma2,
        "epsilon": best_epsilon,
        "metrics_test": {
            "precision": float(gauss_prec),
            "recall": float(gauss_rec),
            "f1": float(gauss_f1),
        },
    }

    gaussian_params_path = os.path.join(models_dir, "gaussian_params.json")
    with open(gaussian_params_path, "w", encoding="utf-8") as f:
        json.dump(gaussian_params, f, indent=2)
    print(f"Saved Gaussian params to {gaussian_params_path}")

    # 2. Supervised Baseline: Logistic Regression
    sup_train = pd.concat([train, val[val["Class"] == 1]])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(sup_train[feature_cols], sup_train["Class"])

    test_probs_sup = clf.predict_proba(test[feature_cols])[:, 1]
    thresholds_sup = np.linspace(0, 1, 1000)
    best_f1_sup = 0.0
    best_thresh_sup = 0.5

    for t in thresholds_sup:
        preds = (test_probs_sup >= t).astype(int)
        f1 = f1_score(test["Class"], preds)
        if f1 > best_f1_sup:
            best_f1_sup = float(f1)
            best_thresh_sup = float(t)

    final_preds_sup = (test_probs_sup >= best_thresh_sup).astype(int)
    sup_prec = precision_score(test["Class"], final_preds_sup)
    sup_rec = recall_score(test["Class"], final_preds_sup)
    sup_f1 = f1_score(test["Class"], final_preds_sup)

    print("\n--- Supervised Baseline Logistic Regression Results (Test Set) ---")
    print(f"Threshold: {best_thresh_sup:.6f}")
    print(f"Precision: {sup_prec:.4f}")
    print(f"Recall:    {sup_rec:.4f}")
    print(f"F1-Score:  {sup_f1:.4f}")

    logistic_artifact = {
        "model": clf,
        "threshold": best_thresh_sup,
        "feature_cols": feature_cols,
        "metrics_test": {
            "precision": float(sup_prec),
            "recall": float(sup_rec),
            "f1": float(sup_f1),
        },
    }

    logistic_path = os.path.join(models_dir, "logistic_model.pkl")
    joblib.dump(logistic_artifact, logistic_path)
    print(f"Saved Logistic model artifact to {logistic_path}")


if __name__ == "__main__":
    main()
