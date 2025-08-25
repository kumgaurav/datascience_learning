Got it 👍 This looks like a **stock scoring + ranking pipeline** where you assign points based on short-term and medium-term performance, then select top stocks by performance across different horizons. Let’s break this into **requirements, approach, and modeling choice**.

---

## 1. Outline the Requirements

Here’s how I would structure your requirements so it’s clear and implementable:

### A. Inputs

* Historical stock data (daily OHLC + volume).
* Columns needed: `date, symbol, close, volume`.

### B. Features to Compute

For each stock and each date:

* **1d, 3d, 5d, 10d, 15d, 20d, 30d, 60d % change in price**

  $$
  \text{price\_change}_{n} = \frac{P_{t} - P_{t-n}}{P_{t-n}} \times 100
  $$
* **1d, 3d, 5d, 10d, 15d, 20d, 30d, 60d % change in volume**

  $$
  \text{volume\_change}_{n} = \frac{V_{t} - V_{t-n}}{V_{t-n}} \times 100
  $$

### C. Scoring System

* Define a **point allocation scheme** for price increase and volume increase. Example:

  * +1 point if % change > 0
  * +2 points if % change > 5%
  * +3 points if % change > 10%
    (This can be tuned later).
* Add **price score** + **volume score** = total score for that stock on that date.

### D. Datasets to Generate

1. **Best 5-day performers:** top 25 stocks with highest score in last 5 days.
2. **Best 15-day performers:** top 25 stocks with highest score in last 15 days.
3. **Best 30-day performers:** top 25 stocks with highest score in last 30 days.

For each selected stock:

* Store the **start price** (price at the beginning of the window).
* Store the **end price** (price at the end of the window).
* Compute **realized return**:

  $$
  \text{return} = \frac{P_{\text{end}} - P_{\text{start}}}{P_{\text{start}}} \times 100
  $$

---

## 2. How to Achieve This

### Step 1. Preprocessing

* Clean stock data (handle missing days, adjust for splits/dividends if possible).
* Ensure volumes are consistent across tickers.

### Step 2. Rolling Window Features

* Use **pandas / Dask rolling** or a Spark pipeline for large datasets.
* Compute price % change and volume % change for each horizon.

### Step 3. Scoring

* Apply your scoring rules on top of computed % changes.
* Sum across horizons to get a **total score per stock per date**.

### Step 4. Ranking & Dataset Creation

* For each date, rank stocks by score.
* Extract top 25 for 5-day, 15-day, 30-day windows.
* Compute **start price, end price, and returns**.

### Step 5. Store Results

* Save final datasets into 3 tables or CSVs:

  * `top5d_performers.csv`
  * `top15d_performers.csv`
  * `top30d_performers.csv`

---

## 3. Model Choice

👉 You **don’t need a heavy ML model here** — this is a **rule-based ranking system**.
Efficient options:

* **pandas** if data is small (few thousand stocks × few years).
* **Dask / Spark** if dataset is very large (tick data or 10+ years of daily data).
* If you want to add **predictive modeling later**, you can feed these engineered features into:

  * **XGBoost / LightGBM** for classification (top performer vs non-top performer).
  * **LSTM / Temporal CNNs** if you want to forecast future scores/returns.

But for now:

* Start with **rule-based scoring + pandas**.
* Later, test ML models to predict which stock will be in **top 25** for the next window.

---

✅ This gives you a **clear requirements breakdown** and a **roadmap**.

Do you want me to draft a **sample pandas pipeline** that computes these features, scores, and produces the 3 datasets?
