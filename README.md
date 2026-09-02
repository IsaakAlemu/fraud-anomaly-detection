# Fraud Anomaly Detection & Supervised Baseline Comparison API

An educational machine learning project and REST API comparing a **from-scratch Unsupervised Gaussian Anomaly Detection engine** against a **Supervised Logistic Regression baseline** on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Built from concepts in the Coursera Machine Learning Specialization (Anomaly Detection lab), this repository focuses on an **honest empirical comparison**: evaluating how an unsupervised statistical density estimator performs when trained strictly on normal transactions versus a supervised classifier that learns from labeled fraud patterns.

---

## 📊 Honest Empirical Findings & Evaluation

> **Core Takeaway**: The from-scratch Gaussian anomaly model captures genuine tail-distribution anomaly signal ($F_1 = 0.3868$), but the supervised baseline with proper feature-label weighting is dramatically stronger ($F_1 = 0.8428$). The primary purpose of this project is to demonstrate and explain this exact contrast—not to claim a standalone statistical density model is sufficient for production fraud prevention.

### Test Set Performance Comparison (57,109 Held-Out Transactions)

| Model | Approach | Tuned Decision Boundary | Precision | Recall | $F_1$-Score |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Gaussian Anomaly Detection** | Unsupervised density estimation (from scratch) | $\epsilon = -219.939267$ | **33.84%** | **45.12%** | **0.3868** |
| **Supervised Baseline** | Logistic Regression (`class_weight='balanced'`) | Probability threshold = $0.998999$ | **87.01%** | **81.71%** | **0.8428** |

*Note: All reported metrics were evaluated on the exact same 57,109 held-out test transactions (containing 246 fraud cases) that were never seen during Gaussian fitting or threshold tuning.*

---

## 🔬 Methodology & Architecture

### 1. Dataset & Feature Preprocessing
- **Dataset**: 284,807 European credit card transactions with 492 fraud cases ($0.1727\%$ fraud rate).
- **Features**: 28 PCA-transformed numerical components (`V1` to `V28`), `Time`, and `Amount`.
- **Log Transformation**: Raw transaction amounts are heavily right-skewed with extreme outliers. We compute `LogAmount = np.log1p(Amount)` ($\ln(1 + \text{Amount})$) to ensure safe handling of zero-dollar authorizations and produce a distribution suitable for Gaussian modeling.
- **Feature Set**: 29 features (`V1`–`V28` + `LogAmount`).

### 2. Rigorous Data Splitting
To prevent data leakage and ensure fair threshold tuning:
1. **Train Set (170,589 samples)**: $100\%$ normal transactions ($0$ fraud). Used exclusively to fit Gaussian feature parameters ($\mu$ and $\sigma^2$).
2. **Validation Set (57,109 samples, 246 fraud)**: Mixed set used strictly to sweep 1,000 candidate $\epsilon$ values and select the threshold that maximizes validation $F_1$.
3. **Test Set (57,109 samples, 246 fraud)**: Strictly held out for final evaluation.

For the supervised baseline, training utilized the normal training transactions plus the validation set's fraud examples, ensuring the classifier learned fraud boundaries without ever touching the held-out test set.

### 3. From-Scratch Gaussian Anomaly Engine
The Gaussian engine assumes independent per-feature normal distributions:
$$\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)}, \quad \sigma_j^2 = \frac{1}{m-1} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2$$

To prevent catastrophic floating-point underflow when multiplying 29 small probability densities, calculations are performed entirely in **log-space**:
$$\ln p(x) = \sum_{j=1}^{29} \left[ -\frac{1}{2} \ln(2\pi\sigma_j^2 + \epsilon_0) - \frac{(x_j - \mu_j)^2}{2\sigma_j^2 + \epsilon_0} \right]$$
A transaction is flagged as anomalous if $\ln p(x) < \epsilon$.

---

## 🚫 What This Is Not (Production Reality Check)

This project is an **educational baseline and API prototype**, not an end-to-end industrial fraud detection engine. A production-grade fraud prevention pipeline in fintech / banking would require:

1. **Velocity & Temporal Feature Engineering**: Aggregations across sliding windows (e.g., *number of transactions in the last 10 minutes*, *ratio of current amount to 30-day historical mean*, *geographic displacement velocity between successive card swipes*).
2. **Gradient Boosting & Sequence Models**: Production systems rely on LightGBM/CatBoost or Graph Neural Networks (GNNs) on transaction graphs rather than univariate Gaussian assumptions.
3. **Cost-Sensitive Thresholds**: Business thresholds are tuned to minimize total financial loss (False Negative fraud cost vs. False Positive customer friction / review overhead), rather than raw $F_1$.
4. **Real-Time Concept Drift Monitoring**: Live distribution tracking (e.g., Population Stability Index, Wasserstein distance) to detect evolving fraud rings and adversarial behavioral shifts.

---

## 🚀 Repository Structure

```
fraud-anomaly-detection/
├── app/
│   ├── __init__.py
│   ├── schemas.py           # Pydantic models for request validation & dual response
│   ├── anomaly_engine.py    # Multivariate Gaussian density engine (fit/tune/predict/save)
│   ├── baseline_model.py    # Supervised Logistic Regression wrapper
│   └── main.py              # FastAPI endpoints (/health, /v1/transactions/verify)
├── data/
│   └── creditcard.csv       # Kaggle dataset (git-ignored)
├── models/
│   ├── gaussian_params.json # Serialized mu, sigma2, epsilon, and metrics
│   └── logistic_model.pkl   # Serialized Logistic Regression artifact & threshold
├── notebooks/
│   └── 01_eda.ipynb         # Original EDA and experimentation notebook
├── scripts/
│   └── train_and_export.py  # Standalone training & artifact generation script
├── tests/
│   ├── __init__.py
│   ├── test_anomaly_engine.py # Math verification, fit, save/load, vector scoring
│   ├── test_baseline_model.py # Supervised prediction and probability threshold tests
│   └── test_main.py           # FastAPI integration tests (200 OK, 422 validations)
├── .gitignore
├── Dockerfile               # Python 3.11-slim container definition
├── requirements.txt
└── README.md
```

---

## 🛠️ Quickstart & Local Setup

### 1. Prerequisites
- Python 3.11 or 3.12
- Kaggle `creditcard.csv` inside `data/` (optional if using pre-generated model artifacts in `models/`)

### 2. Installation & Virtual Environment

```bash
# Clone the repository
git clone <YOUR_REPO_URL>
cd fraud-anomaly-detection

# Create and activate virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. (Optional) Re-run Training & Artifact Export

```bash
python scripts/train_and_export.py
```

### 4. Run Test Suite

```bash
pytest -v tests
```

### 5. Launch FastAPI Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive Swagger documentation is available at: [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 📡 API Reference & Mock Requests

### `POST /v1/transactions/verify`
Scores an incoming transaction against **both** models simultaneously.

#### Example Request: Normal Transaction

```bash
curl -X POST "http://localhost:8000/v1/transactions/verify" \
     -H "Content-Type: application/json" \
     -d '{
       "V1": 0.448012, "V2": 2.035533, "V3": -0.608395, "V4": 4.684705,
       "V5": 1.296935, "V6": 0.211782, "V7": 0.763525, "V8": -0.036693,
       "V9": -1.621939, "V10": 1.435061, "V11": 1.131147, "V12": -0.965795,
       "V13": -0.791461, "V14": -2.463358, "V15": 0.065032, "V16": 0.892818,
       "V17": 1.871823, "V18": 1.494008, "V19": 1.176817, "V20": 0.464369,
       "V21": -0.473552, "V22": -0.962575, "V23": 0.082715, "V24": -0.728430,
       "V25": -0.700013, "V26": 0.032017, "V27": 0.244196, "V28": -0.094077,
       "Amount": 3.09
     }'
```

#### Example Response: Normal Transaction

```json
{
  "transaction_amount": 3.09,
  "log_amount": 1.408545,
  "gaussian_model": {
    "log_probability": -48.147945,
    "threshold": -219.939267,
    "is_anomaly": false,
    "model_type": "Unsupervised Gaussian (from-scratch density estimation)"
  },
  "supervised_model": {
    "fraud_probability": 0.708548,
    "threshold": 0.998999,
    "is_fraud": false,
    "model_type": "Supervised Logistic Regression (balanced class weights)"
  },
  "comparison": {
    "status": "both_cleared",
    "message": "Low risk: Transaction aligns with normal Gaussian distributions and scored low fraud probability on the supervised model."
  }
}
```

---

#### Example Request: Fraud Transaction

```bash
curl -X POST "http://localhost:8000/v1/transactions/verify" \
     -H "Content-Type: application/json" \
     -d '{
       "V1": -18.018561, "V2": 10.558600, "V3": -24.667741, "V4": 11.786180,
       "V5": -10.564657, "V6": -2.645681, "V7": -18.023468, "V8": 2.693655,
       "V9": -6.219464, "V10": -12.744761, "V11": 9.318138, "V12": -13.377857,
       "V13": 1.021924, "V14": -12.784346, "V15": -0.451146, "V16": -7.551365,
       "V17": -11.325997, "V18": -4.701253, "V19": 0.516317, "V20": 1.007703,
       "V21": -2.319479, "V22": 0.908839, "V23": 1.352904, "V24": -1.059222,
       "V25": 0.185751, "V26": 0.687037, "V27": 2.078081, "V28": -1.429517,
       "Amount": 1.00
     }'
```

#### Example Response: Fraud Transaction

```json
{
  "transaction_amount": 1.0,
  "log_amount": 0.693147,
  "gaussian_model": {
    "log_probability": -963.049166,
    "threshold": -219.939267,
    "is_anomaly": true,
    "model_type": "Unsupervised Gaussian (from-scratch density estimation)"
  },
  "supervised_model": {
    "fraud_probability": 1.0,
    "threshold": 0.998999,
    "is_fraud": true,
    "model_type": "Supervised Logistic Regression (balanced class weights)"
  },
  "comparison": {
    "status": "both_flagged",
    "message": "High risk: Both the statistical Gaussian anomaly engine (unsupervised density) and the supervised baseline flagged this transaction."
  }
}
```

---

## 🐳 Docker Deployment

To build and run the service within a minimal container environment:

```bash
# Build Docker image
docker build -t fraud-anomaly-api .

# Run container on port 8000
docker run -d -p 8000:8000 --name fraud-api fraud-anomaly-api

# Check health
curl http://localhost:8000/health
```

---

## 📜 License & Acknowledgments

- Built upon concepts from the **Coursera Machine Learning Specialization (Unsupervised Learning, Recommenders, Reinforcement Learning by Andrew Ng)**.
- Dataset provided by **Machine Learning Group - ULB** on Kaggle.
