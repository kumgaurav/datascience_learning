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
