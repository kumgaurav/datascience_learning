Got it 👍 Let’s structure this into **feature requirements** for your evaluation and snapshot framework. Since you already have weekly predictions (XGB + LSTM), the goal is to:

1. Take a **frozen snapshot** of inputs & predictions every week (Sunday).
2. Evaluate **daily** (Mon–Fri) how predictions perform relative to real prices.
3. Diagnose **feature importance and failure modes** when predictions miss.
4. Provide **recommendations for improvements** in feature engineering and model retraining.

---

## 📂 Feature Requirements Outline

### 1. **Snapshot Creation (Weekly – Sunday)**

* **Input Data**:

  * Last 25 weeks of returns (or your chosen horizon).
  * All engineered features (technical indicators, volume trends, macro features, etc.).
  * XGB prediction scores.
  * LSTM prediction scores.
  * Actual prices (up to snapshot day).
* **Output**:

  * Store in `/snapshot/{yyyy-mm-dd}/snapshot.parquet` (or CSV).
  * Metadata file with model versions, feature schema, and parameters.
* **Purpose**:

  * Freeze state for reproducibility and fair evaluation.

---

### 2. **Daily Evaluation (Mon–Fri)**

* **Data Inputs**:

  * Actual market data (closing prices, volumes).
  * Snapshot predictions (from Sunday).
* **Checks**:

  * **Prediction Accuracy**:

    * MAE / RMSE for XGB & LSTM predictions.
    * Directional Accuracy (% correct up/down predictions).
    * Precision @K (top stocks predicted to rise vs actual rises).
  * **Feature Drift**:

    * Compare feature values on daily data vs snapshot (KS-test, PSI).
    * Detect if a key feature distribution shifted unexpectedly.
  * **Error Attribution**:

    * For wrong predictions, compute SHAP values (XGB) or integrated gradients (LSTM).
    * Identify which features contributed most to the failure.

---

### 3. **Feature Validation**

* **Diagnostics**:

  * Track correlation of each feature with actual returns this week.
  * Highlight features that **lost predictive power** (e.g., RSI stopped being useful).
  * Identify **redundant or noisy features**.
* **Failure Attribution**:

  * If a stock dropped when predicted to rise:

    * Was prediction too dependent on 1–2 features?
    * Did macro/sector trend override local features?
    * Was volume signal misleading?
  * Log “possible feature selection issue” for each failure case.

---

### 4. **Recommendations & Alerts**

* **Recommendations Engine** (automated):

  * Suggest feature re-weighting (down-weight noisy features).
  * Suggest feature enrichment (new sentiment, macro indicators).
  * Suggest retraining when cumulative prediction error exceeds threshold.
  * Suggest removing features with consistently negative contribution.
* **Alerts**:

  * "Feature X lost 70% correlation with returns this week."
  * "Model predictions diverging: LSTM accuracy 55%, XGB accuracy 80%."
  * "Top 3 mispredicted stocks show volume spike → unseen market condition."

---

### 5. **System Architecture**

* **Modules Needed**:

  1. **Snapshot Generator** (weekly freeze).
  2. **Evaluator** (daily actual vs predicted).
  3. **Feature Drift Detector**.
  4. **Attribution Engine** (SHAP/IG analysis).
  5. **Report Generator** (dashboard, PDF, or logs).
* **Storage**:

  * Keep 12+ weeks of snapshots for trend analysis.
* **Automation**:

  * Cron schedule:

    * Sunday → create snapshot.
    * Mon–Fri → evaluate & log.

---

✅ **High-level workflow**:

1. **Sunday**: Freeze → Store predictions + features.
2. **Monday–Friday**: Compare daily closes vs predictions → run attribution → log mispredictions.
3. **End of Week**: Generate evaluation report → highlight failing features → recommend retraining/improvement.

---

👉 Would you like me to design a **sample folder structure + Python evaluation script outline** for this snapshot framework so you can plug it into your pipeline?

project_root/
│
├── snapshot/
│   ├── 2025-08-17/                # Sunday snapshot
│   │   ├── snapshot.csv           # All features + predictions
│   │   ├── metadata.json          # Model versions, config, feature list
│   │   └── eval/                  # Daily evaluations
│   │       ├── 2025-08-18_eval.csv
│   │       ├── 2025-08-19_eval.csv
│   │       └── ...
│   └── 2025-08-24/
│       └── ...
│
├── scripts/
│   ├── snapshot_generator.py      # Weekly snapshot creation
│   ├── daily_evaluator.py         # Daily prediction vs actual
│   ├── feature_drift.py           # Feature drift detection
│   └── attribution.py             # SHAP/IG analysis
│
├── reports/
│   ├── 2025-08-17_report.pdf      # Weekly evaluation summary
│   └── ...
│
└── config/
    └── model_config.yaml          # Model settings, thresholds
