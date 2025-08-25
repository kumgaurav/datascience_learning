Here’s a clean summary of your **feature selection, modeling approach, and strategy**, with a few suggested additional features:

---

## 🔹 Feature Engineering (Planned)

1. **Upcoming Earnings** → Indicator if earnings are scheduled within the next 3 weeks.
2. **Quarterly Performance** → Stock has performed well in the last 2 quarters.
3. **Recent Trend** → Consistent upward movement in the last 3 weeks.
4. **Resistance Breakout** → Enhanced `is_upward_trending` feature where stock crosses historical resistance.
5. **Post-Earnings Rally** → Pattern where stock rises on strong earnings, dips, and then rallies again (bullish moment).

---

## 🔹 Model Objective

* Predict **price change in the next week** (not raw price level) to avoid trivial autocorrelation effects.
* Target (y) = **price\_change\_in\_5\_days** instead of absolute price.

---

## 🔹 Stock Selection Module (`stock_selector.py`)

1. Compute latest features for all stocks.
2. Use trained model to predict 5-day **price change**.
3. Convert predictions → **return %**.
4. Apply **confidence filters**:

   * `broke_resistance = True` OR `post_earnings_dip_rally = True`.
5. Rank by predicted return %.
6. Select **top 20 stocks**.

---

## 🔹 Additional Feature Ideas (to improve accuracy)

* **Relative Strength Index (RSI 14d)** → momentum indicator.
* **Moving Average Crossovers** (e.g., 50d vs. 200d).
* **Volume Spike** → unusual increase in trading volume.
* **Volatility Measures** (e.g., Bollinger Band breakouts, ATR).
* **Insider / Institutional Activity** → if available.
* **Sector/Index Comparison** → stock performance relative to sector or S\&P 500.
* **Gap Analysis** → price gaps post-earnings or news events.
* **Sentiment Score** (if using news/LLMs later).

---

✅ In short: You’re designing **event-driven + technical momentum features**, shifting prediction from *absolute price* → *relative price change*, and then ranking stocks by predicted return with bullish filters to pick the **top 20 candidates**.

Here’s the highest-ROI plan, in priority order:

1) Optimize for ranking (top-N), not MAE
- XGBoost: switch to objective='rank:pairwise', time-based split, early_stopping_rounds=100; track NDCG@20/precision@20.
- Weight ensemble by validation precision@20 (or IC), not MAE.

2) Better target for weekly picks
- Horizon = 10 trading days; label = rolling 5d sum of returns (smoothed).
- Winsorize labels at 1–99th pct (keep).

3) Add context features that move the needle
- Market/sector: SPY 1–5d returns (you have 1d), VIX 1–5d, sector index return.
- Risk model: 60d rolling beta to SPY and idiosyncratic (residual) returns.
- Regime flag: high/low vol (e.g., VIX or realized vol percentile).

4) LSTM upgrades (only if you keep it)
- lookback=60, epochs=30–50, dropout=0.2, EarlyStopping(patience=5) + ReduceLROnPlateau.
- Add per-date cross-sectional z-score (in addition to per-window).

5) Validation you can trust
- Rolling time-series CV; report precision@20 and IC by fold and by regime.
- Keep strict time splits and leakage checks.

Concrete defaults to try next:
- XGB: depth=4, lr=0.03, subsample=0.8, colsample=0.8, min_child_weight=3, reg_lambda=1.0, n_estimators=3000 with early stopping.
- Target: 10d horizon, 5d rolling-sum return, winsorize 1–99.
- New features: 60d beta, residual returns, VIX 1–5d, sector 1–5d, vol regime flag.

These changes most directly improve top-20 selection quality with minimal churn.
