# Stock Investment Data Analysis

This project downloads and analyzes stock market data using the Yahoo Finance API through the `yfinance` library.

## Files

- `stocks/stk_batch_download.py` - Original script for downloading real stock data
- `stocks/stk_batch_download_offline.py` - Sample data generator for demonstration/testing
- `requirements.txt` - Python dependencies
- `data/` - Folder containing all generated CSV files

## Setup

1. **Activate your conda environment:**
   ```bash
   conda activate uiapp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### For Real Data (requires proper network access):
```bash
python stocks/stk_batch_download.py
```

### For Sample Data (works offline):
```bash
python stocks/stk_batch_download_offline.py
```

## SSL Certificate Issues

If you encounter SSL certificate errors like:
```
curl: (60) SSL certificate problem: self signed certificate in certificate chain
```

This is common in corporate environments. Here are solutions:

### Solution 1: Configure Corporate Proxy
If you're behind a corporate firewall, configure your proxy settings:

```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### Solution 2: Update SSL Certificates
```bash
# Update certificates on macOS
sudo /usr/bin/security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain > /tmp/certs.pem
export SSL_CERT_FILE=/tmp/certs.pem
```

### Solution 3: Use Sample Data
For development and testing, use the offline version:
```bash
python stocks/stk_batch_download_offline.py
```

## Output Files

The scripts generate the following CSV files in the `data/` folder:
- `data/stock_prices_[date].csv` - Daily stock price data
- `data/stock_earnings_[date].csv` - Company earnings information
- `data/earnings_estimates_[date].csv` - Analyst earnings estimates
- `data/quarterly_revenue_[date].csv` - Quarterly revenue data
- `data/growth_estimates_[date].csv` - Growth projections
- `data/revenue_estimates_[date].csv` - Revenue forecasts
- `data/quarterly_income_[date].csv` - Quarterly income statements

## Stock Tickers

Currently configured for:
- AAPL (Apple)
- GOOGL (Google)
- MSFT (Microsoft)
- AI (C3.ai)
- OPFI (OppFi)

## Troubleshooting

1. **ModuleNotFoundError: No module named 'yfinance'**
   - Ensure you're in the correct conda environment
   - Run: `pip install yfinance`

2. **SSL Certificate Errors**
   - Use the offline version for testing
   - Contact your IT department for proxy configuration
   - Try running from a different network (home vs office)

3. **Pandas API Changes**
   - The script has been updated to handle pandas version changes
   - If you encounter API errors, update pandas: `pip install --upgrade pandas`

## Data Structure

Each CSV file contains structured financial data that can be used for:
- Investment analysis
- Portfolio management
- Financial modeling
- Data visualization

## Next Steps

python run_pipeline.py --date 2024-06-01

Consider building a Streamlit dashboard to visualize this data:
```bash
pip install streamlit
streamlit run your_dashboard.py
```

## Updated Architecture (Training + Weekly Prediction)

### End-to-end flow

1. Export and validate raw data
   - Export: `python stocks/execute_data_creator.py` (writes CSVs under `data/`)
   - Validation: `python -m data_validation.validate_data --only undated | cat`

2. Orchestrate ML pipeline and UI outputs
   - Run: `python run_pipeline_v2.py --prices data/stock_prices.csv --horizon 5 --lookback 30 --top_n 20`
   - This prepares features, trains models, and writes UI-compatible artifacts.

### What each component does

- Data export and shaping
  - `stocks/stock_data_creator.py`: Reads MySQL tables (e.g., `stocksinfp`) and writes CSVs. Applies per-table transforms, restricts to recent data, and for `stocksinfp` keeps only tickers whose row counts match AAPL (uniform histories).
  - `stocks/execute_data_creator.py`: Convenience entrypoint to export all required tables and map them to standard CSV names (e.g., `stocksinfp` → `data/stock_prices.csv`).

- Data validation
  - `data_validation/validate_data.py`: Schema and sanity checks for all CSVs. For prices, validates chronological order, min coverage, datatypes, and reports tickers whose history length differs from AAPL.

- Data loading and feature engineering
  - `data_loader.py`: Loads all CSVs and produces `(master_df, prices_df)` consistent with the rest of the stack.
  - `feature_engineering.py`: Builds rich, per-ticker technical/fundamental signals (e.g., RSI, MACD, breakout flags, earnings-based signals) and merges latest technicals into the master frame for analysis/ranking.

- Model-ready datasets
  - `utils/data_prep.py`:
    - `build_xgb_dataset(prices_df, horizon)`: Per `(ticker,date)` snapshot features for XGBoost with forward-return label (e.g., 5d ahead). Includes lag returns, rolling mean/std, realized volatility, RSI, MACD, Bollinger position, volume trends, optional SPY return, and momentum aliases.
    - `build_xgb_features(prices_df, horizon)`: Same features as training but does not drop the latest rows (targets can be NaN). Used for inference/ranking on the most recent date.
    - `build_lstm_sequences(prices_df, lookback, horizon)`: Sliding-window sequences per ticker with per-window z-score normalization for sequence models.

- Trainers
  - `models/base_trainer.py`: Common prep including label creation (`target = % return over next horizon`) and baseline momentum fields.
  - `models/xgb_trainer.py`: Auto-selects numeric/bool features (excludes leakage columns) and trains a tuned XGBoost regressor to predict forward return.
  - `models/lstm_trainer.py`: Builds fixed-length sliding windows per ticker (OHLCV + engineered features), normalizes per window, and trains a simple LSTM to predict forward return.

- Pipeline orchestrator
  - `run_pipeline_v2.py`:
    - Loads data via `load_all_data` and creates engineered signals via `create_all_features` (for diagnostics and UI flags).
    - Builds XGB training dataset and LSTM sequences (`utils/data_prep.py`).
    - Trains XGB and LSTM. Saves the XGB model to `models/stock_predictor_top.joblib`.
    - Builds XGB-style inference features for the latest date and merges bullish signals for filtering/ranking.
    - Produces the UI CSV `data/featured_stocks_top.csv` so the existing UI works unchanged.

- Selection and UI integration
  - `stock_selector_v2.py`: Ranks tickers by predicted return and confidence, with optional LSTM handling. Used inside the pipeline for diagnostics.
  - `stock_selector.py`: Legacy selector used by the UI; reads `data/featured_stocks_top.csv` and `models/stock_predictor_top.joblib`.

### Key outputs (consumed by the UI)

- `data/featured_stocks_top.csv`: Latest per-ticker snapshot including technical/fundamental signals and prices; regenerated by `run_pipeline_v2.py` for UI compatibility.
- `models/stock_predictor_top.joblib`: Trained XGBoost model used for weekly predictions in the UI.
- Momentum page features: `data/momentum/featured_stocks_momentum.csv`.

### Weekly artifacts (XGB, LSTM, Ensemble) under `data/top/`

- XGB
  - `xgb_dataset_input.csv`: model-ready training dataset (built from filtered prices)
  - `xgb_features_input.csv`: latest snapshot features for inference
  - `xgb_ranked_output.csv`: full ranked list with `predicted_return_pct`, `confidence_score`, `risk_score`, `composite_score`

- LSTM (weekly horizon)
  - `lstm_features_input.csv`: full-history technical features for all symbols (sequences source)
  - `lstm_weekly_predictions_output.csv`: per-ticker weekly return predictions

- Ensemble
  - Inputs: `xgb_ranked_output.csv` and `lstm_weekly_predictions_output.csv`
  - Output: `ensemble_scores_output.csv` with `ticker`, `confidence_score`, `lstm_predicted_return_pct`, `ensemble_score`

### Configuration and environments

- Database access for export: `stocks/conf/config.ini` ([mysql] section: `url`, `username`, `password`, `database`, optional `port`).
- Conda install hints:
  - Core: `conda install -c conda-forge sqlalchemy pymysql pandas numpy xgboost scikit-learn`
  - LSTM: `conda install -c conda-forge tensorflow`

### Weekly prediction setup

- By default, the weekly horizon is 5 trading days (`--horizon 5`). Change it in `run_pipeline_v2.py` CLI args.
- LSTM sequence length defaults to 30 (`--lookback 30`).

### One-command pipeline

```bash
python run_pipeline_v2.py --prices data/stock_prices.csv --horizon 5 --lookback 30 --top_n 25
```

This command regenerates features, trains models, writes UI artifacts, and prints the top-N ranked tickers.
python stocks/execute_data_creator.py 

python run_pipeline_v2.py --skip_xgb --xgb_ranked_csv data/top/xgb_ranked_output.csv --lookback 30

SELLOFF_STRONG_VRATIO=2 SELLOFF_STRONG_DROP_PCT=-0.03 \
SELLOFF_FACTOR_STRONG=0.6 SELLOFF_FACTOR_MILD=0.9 \
python run_pipeline_v2.py

python utils/generate_ensemble.py --xgb data/top/xgb_ranked_output.csv --lstm data/top/lstm_weekly_predictions_output.csv --alpha 0.6 --output data/top/ensemble_scores_output.csv

python utils/compare_symbols.py --a IAG --b LYSDY --out data/top/compare.md

python data_validation/debug_feature_compare.py \
  --prices data/input/stock_prices_with_clean_data.csv \
  --ticker_a APLD --ticker_b LYSDY \
  --end_date 2025-08-23 --lookback 5

# Full pipeline
python run_pipeline.py

# Skip DB fetch + features, just models and ensemble
python run_pipeline.py --skip_fetch --skip_features

# Run only ensemble (assumes XGB/LSTM outputs exist)
python run_pipeline.py --skip_fetch --skip_features --skip_xgb --skip_lstm

python utils/lstm_diagnostics.py --features data/features/stock_features_clean.csv --out data/lstm/diagnostics_report.json

cd /Users/gaurav/workspace/datascience/ds_uiv2/datascience_learning/weekly_prediction && APP_CONFIG=config.yaml conda run -n stocks3 python utils/ui_diagnose.py --ticker RGTI | cat

no, I want refactor the code . so it run in steps. 
It helps me understand what is not working well
in the run_pipeline_v2.py
Step 1 just pull the file from database - print file is read with location, make sure you are not removing the existing logic like you are creating 2 files 1. stock_prices.csv 2. stock_prices_input.csv
Step 2 create the feature file
Step 3 

### Configurable feature engineering (`feature_config`)

You can customize indicator windows and earnings pattern thresholds when building features. Pass a `feature_config` dict to `create_all_features`.

```python
from datetime import date
from feature_engineering import create_all_features

# master_df, prices_df prepared via data_loader.py
features = create_all_features(
    master_df,
    prices_df,
    today=date.today(),
    feature_config={
        'rsi_period': 14,
        'macd_spans': (12, 26, 9),
        'bb_window': 20,
        'bb_k': 2.0,
        'volume_ma_window': 20,
        'volatility_window': 30,
        'trend_periods': [5, 10, 20, 50],
        'momentum_periods': [5, 10, 20, 30, 60],
        'support_resistance_window': 20,
        'slope_windows': (15, 30),
        'up_day_ratio_window': 20,
        'atr_window': 14,
        'vol_spike_window': 20,
        # Post-earnings rally detection
        'post_earnings_lookback_days': 90,
        'post_earnings_min_days': 10,
        'post_earnings_initial_jump_pct': 0.03,
        'post_earnings_dip_from_peak_pct': 0.05,
        'post_earnings_rally_from_dip_pct': 0.03,
    }
)
```

#### Example presets

```python
# Conservative (stable/low-vol regime)
conservative_cfg = {
    'rsi_period': 21,
    'bb_k': 2.5,
    'volatility_window': 60,
    'volume_ma_window': 30,
    'slope_windows': (30, 60),
    'trend_periods': [10, 20, 50, 100],
    'momentum_periods': [10, 20, 30, 60, 120],
    'post_earnings_min_days': 15,
    'post_earnings_initial_jump_pct': 0.05,
    'post_earnings_dip_from_peak_pct': 0.06,
    'post_earnings_rally_from_dip_pct': 0.04,
}

# High-volatility (fast/short windows)
high_vol_cfg = {
    'rsi_period': 10,
    'bb_k': 1.8,
    'volatility_window': 20,
    'volume_ma_window': 10,
    'slope_windows': (10, 20),
    'trend_periods': [5, 10, 20, 40],
    'momentum_periods': [5, 10, 20, 30, 60],
    'post_earnings_min_days': 8,
    'post_earnings_initial_jump_pct': 0.03,
    'post_earnings_dip_from_peak_pct': 0.08,
    'post_earnings_rally_from_dip_pct': 0.05,
}
```