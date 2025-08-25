# 🏗️ App Refactoring Summary

# Full pipeline
python run_pipeline.py

# Skip DB fetch + features, just models and ensemble
python run_pipeline.py --skip_fetch --skip_features

# Run only ensemble (assumes XGB/LSTM outputs exist)
python run_pipeline.py --skip_fetch --skip_features --skip_xgb --skip_lstm

# dignostic for lstm
python utils/lstm_diagnostics.py --features data/features/stock_features_clean.csv --out data/lstm/diagnostics_report.json

# dignostic for ui
cd /Users/gaurav/workspace/datascience/ds_uiv2/datascience_learning/weekly_prediction && APP_CONFIG=config.yaml conda run -n stocks3 python utils/ui_diagnose.py --ticker RGTI | cat

# Rum Momentum
APP_CONFIG=config.yaml python utils/train_momentum_weekly.py