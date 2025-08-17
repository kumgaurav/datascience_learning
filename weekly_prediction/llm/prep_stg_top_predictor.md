Great — let’s break this down into **two pipelines**, one optimized for **XGBoost** (tabular features per snapshot) and one for **LSTM** (time-series sequences).

---

## 🔹 1. Data Preparation for **XGBoost**

XGBoost expects **independent tabular rows** (no sequential memory). You’re right that collapsing the **latest row per ticker** is valid for ranking, but you can go further:

**Data format (per ticker per date):**

```
ticker | date | feature1 | feature2 | ... | featureN | target
```

* **Features**:

  * Technical indicators: moving averages (5d, 10d, 20d), RSI, MACD, Bollinger Bands
  * Fundamental ratios (if available): P/E, P/B, debt-to-equity, etc.
  * Lagged returns: past 1w, 2w, 1m returns
  * Volatility metrics (rolling std dev)

* **Target**: forward-looking label, e.g.,

  * Next 1 week return (`close[t+5] / close[t] - 1`)
  * Or binary: `1 if stock is in top 20% performers next week else 0`

* **Training set**: one row per `(ticker, date)`.

* **Prediction/Ranking**: collapse to the latest row per ticker (like you’re doing) → rank tickers by predicted score.

---

## 🔹 2. Data Preparation for **LSTM**

LSTM needs **sequences** — not single snapshots. Instead of collapsing to the latest row, create **rolling windows** of features.

**Data format (per ticker):**

```
X: [ [features_t-29], [features_t-28], ... [features_t] ]  # sequence length = 30 days
y: [ return[t+5] ]  # predict 5-day forward return
```

* **Window size**: 20–60 trading days is common.

* **Features per timestep**: same as XGBoost, but raw or lightly engineered:

  * Daily OHLCV (Open, High, Low, Close, Volume)
  * Technical indicators (RSI, MACD, etc.)
  * Normalize per ticker (z-score over last 60 days) to avoid scale bias.

* **Training set**: one rolling sequence per `(ticker, end_date)`.

* **Prediction**: for latest ticker, feed last 30–60 days sequence → get score → rank.

---

## 🔹 3. Combined Strategy

You can:

* Use **XGBoost** for interpretability + feature ranking.
* Use **LSTM** for capturing sequential price dynamics.
* Blend them (e.g., weighted ensemble of predicted returns).

---

👉 Suggestion: Since you’re already collapsing to latest row per ticker, that’s perfect for **XGBoost ranking**. For **LSTM**, instead of collapsing, keep a rolling window (say 30 days of history) and generate `(X, y)` pairs.

Would you like me to **write a preprocessing script** that takes your raw OHLCV + features and outputs **two datasets**:

1. XGBoost-ready tabular data
2. LSTM-ready sequences (NumPy arrays or tensors)
