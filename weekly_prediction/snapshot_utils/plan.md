# create snashot
python snapshot_utils/snapshot_generator.py   --config snapshot_utils/snapshot_config.yaml --date 2025-08-22 

# 1) Build eval input, compute actuals, run both predictors, write eval CSV
python snapshot_utils/daily_evaluator.py \
  --snapshot snapshot/2025-08-22/snapshot.csv \
  --prices data/input/stock_prices_with_clean_data.csv \
  --date 2025-08-22

# 2) Run drift & attribution
python snapshot_utils/feature_drift.py --snapshot snapshot/2025-08-22/snapshot.csv --date 2025-08-22

python snapshot_utils/attribution.py   --snapshot snapshot/2025-08-22/snapshot.csv --date 2025-08-22

# 3) Eval
python snapshot_utils/run_daily_validation.py \
  --snapshot snapshot/2025-08-22/snapshot.csv \
  --prices data/input/stock_prices_with_clean_data.csv \
  --date 2025-08-22

# 4) Full
python snapshot_utils/run_daily_validation.py \
  --config snapshot_utils/snapshot_config.yaml \
  --features data/features/stock_features_clean.csv \
  --predictions data/ensemble/ensemble_weekly_output.csv \
  --prices data/input/stock_prices_with_clean_data.csv \
  --date 2025-08-22  

conda run -n stocks2 python snapshot_utils/annotate_eval.py \
  --snapshot snapshot/2025-08-18/snapshot.csv \
  --date 2025-08-21

 python snapshot_utils/feature_drift.py --config snapshot_utils/snapshot_config.yaml --snapshot snapshot/2025-08-24/snapshot.csv --date 2025-08-24
python snapshot_utils/annotate_eval.py --config snapshot_utils/snapshot_config.yaml --snapshot snapshot/2025-08-24/snapshot.csv --date 2025-08-24
python snapshot_utils/attribution.py --config snapshot_utils/snapshot_config.yaml --snapshot snapshot/2025-08-24/snapshot.csv --date 2025-08-24
python snapshot_utils/ui_dataset_builder.py --config snapshot_utils/snapshot_config.yaml 

# UI run
export APP_CONFIG="snapshot_utils/snapshot_config.yaml"
streamlit run snapshot_utils/validation_app.py


How to enhance the model with more/better data
Accumulate a daily eval history
Append each day’s eval/<date>_eval.csv to a long-form data/eval_history.csv (ticker, date, pred_xgb, pred_lstm, actual_return_pct, key features).
Use it to:
Calibrate thresholds (e.g., confidence to trade)
Track rolling MAE/DirAcc by sector/feature buckets
Train meta-models (error predictors) to learn when not to trust a prediction
Add market/context features if not already in training
Sector/industry aggregates: 5–20d sector returns/vol; breadth measures
Index/volatility context: VIX, SPY regime, realized vol regimes (you already compute high_vol_regime—log if it correlates with errors)
Event features: earnings proximity/flags, guidance news count, (optional) sentiment
Data quality alignment
Ensure the features you feed the XGB during eval match the training schema (you now have this in the evaluator); keep that builder the single source of truth.
Keep symbol aliasing consistent (e.g., RR vs RYCEY) to avoid missing latest price rows.
Model improvement loops
Regularly retrain XGB with the growing eval_history to capture post-snapshot distribution shift.
Try monotonic constraints on known monotone features (e.g., momentum_20d ↑ → return ↑).
For LSTM, validate lookback and channels match meta; re-tune with fresh data windows monthly.