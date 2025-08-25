import argparse
import time
import pandas as pd
import numpy as np
import os
import joblib
from models.xgb_trainer import XGBTrainer
from models.lstm_trainer import LSTMTrainer
from models.ensemble_trainer import EnsembleTrainer
from utils.build_xgb_weekly_output import build_xgb_weekly_output
from stock_selector_v2 import StockSelector
from data_loader import load_all_data
from feature_engineering import create_all_features, _calculate_momentum
from utils.data_prep import build_xgb_dataset, build_lstm_sequences, build_xgb_features
from utils.tracking import get_tracker
from utils.snapshots import write_latest_snapshot


def main():
    parser = argparse.ArgumentParser(description="Train top model and rank stocks (v2 from base prices)")
    parser.add_argument("--prices", default="data/stock_prices.csv", help="Path to base stock prices CSV (date,ticker,OHLCV) for TRAINING")
    parser.add_argument("--infer_prices", default="data/stock_prices_input.csv", help="Path to base stock prices CSV for INFERENCE (unfiltered/all symbols)")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon in trading days")
    parser.add_argument("--lookback", type=int, default=30, help="Sequence length for LSTM")
    parser.add_argument("--top_n", type=int, default=25, help="Number of top stocks to show")
    parser.add_argument("--save", default=None, help="Optional path to save ranked CSV")
    parser.add_argument("--skip_xgb", action="store_true", help="Skip XGB training/ranking and use existing artifacts")
    parser.add_argument("--xgb_ranked_csv", default=os.getenv('XGB_RANKED_CSV', 'data/top/xgb_ranked_output.csv'), help="Path to existing XGB ranked CSV when skipping XGB")
    parser.add_argument("--lstm_features_csv", default=os.getenv('ENSEMBLE_LSTM_FEATURES_CSV', 'data/top/lstm_features_input.csv'), help="Features CSV to build LSTM sequences from (full history, all symbols)")
    parser.add_argument("--tracking", choices=["none", "mlflow", "wandb"], default=os.environ.get('TRACKING_MODE', 'none'), help="Experiment tracking backend")
    parser.add_argument("--experiment", default=os.environ.get('EXPERIMENT_NAME', 'weekly_prediction'), help="Experiment name / project")
    parser.add_argument("--mlflow-uri", default=os.environ.get('MLFLOW_TRACKING_URI'), help="MLflow tracking URI")
    parser.add_argument("--wandb-entity", default=os.environ.get('WANDB_ENTITY'), help="W&B entity/org")
    # Ensemble configuration
    parser.add_argument("--ensemble-method", choices=["weighted", "rank"], default=os.environ.get('ENSEMBLE_METHOD', 'weighted'), help="Ensemble blend method: weighted (min-max) or rank")
    parser.add_argument("--ensemble-alpha", type=float, default=float(os.environ.get('ENSEMBLE_ALPHA', '0.6')), help="Weight on XGB component in ensemble (0..1)")
    parser.add_argument("--ensemble-stack-model", default=os.environ.get('ENSEMBLE_STACK_MODEL'), help="Optional path to joblib meta-learner for stacking")
    args = parser.parse_args()

    print("[PIPE] Starting pipeline v2...")
    t0 = time.perf_counter()
    # Load all data (master + prices), mirroring run_pipeline
    t_load_start = time.perf_counter()
    master_df, prices_df = load_all_data(file_date=None)
    t_load = time.perf_counter() - t_load_start
    print(f"[PIPE TIME] Data load: {t_load:.2f}s (master rows={len(master_df)}, prices rows={len(prices_df)})")
    if master_df.empty or prices_df.empty:
        print("[PIPE DIAG] ERROR: Failed to load base datasets via load_all_data().")
        return

    # Prepare feature set using the same logic as run_pipeline (but ensure date,ticker,close retained)
    try:
        latest_prices_date = prices_df['date'].max()
        today_date = latest_prices_date.date() if hasattr(latest_prices_date, 'date') else None
    except Exception:
        today_date = None
    print("[PIPE] Feature engineering started...")
    t_feat_start = time.perf_counter()
    # Programmatic feature configuration (can be tweaked here without CLI flags)
    feature_config = {
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
    features_df = create_all_features(master_df, prices_df, today=today_date, feature_config=feature_config)
    t_feat = time.perf_counter() - t_feat_start
    print(f"[PIPE TIME] Feature engineering: {t_feat:.2f}s (rows={len(features_df)})")
    # Save filtered (training) features for UI/diagnostics
    os.makedirs(os.path.join('data','top'), exist_ok=True)
    features_out_path = os.path.join('data','top','featured_stocks_top.csv')
    try:
        features_df.to_csv(features_out_path, index=False)
        print(f"[PIPE DIAG] Wrote features to: {features_out_path}")
    except Exception as e:
        print(f"[PIPE DIAG] Failed to write features CSV: {e}")

    # Ensure date, ticker, close columns are present
    latest_price_cols = prices_df.sort_values('date').groupby('ticker').tail(1)[['ticker', 'date', 'close']]
    if 'date' not in features_df.columns or 'close' not in features_df.columns:
        features_df = features_df.merge(latest_price_cols, on='ticker', how='left', suffixes=('', ''))
    # Diagnostics
    print(f"[PIPE DIAG] Features prepared with columns: {list(features_df.columns)}")

    # --- New: Build prepared datasets for XGB and LSTM ---
    try:
        t_xgbds_start = time.perf_counter()
        xgb_df = build_xgb_dataset(prices_df, horizon=args.horizon, feature_config=feature_config)
        os.makedirs(os.path.join('data', 'top'), exist_ok=True)
        xgb_dataset_path = os.path.join('data', 'top', 'xgb_dataset_input.csv')
        xgb_df.to_csv(xgb_dataset_path, index=False)
        t_xgbds = time.perf_counter() - t_xgbds_start
        print(f"[PIPE DIAG] Wrote XGB dataset to: {xgb_dataset_path} (rows={len(xgb_df)})")
        print(f"[PIPE TIME] Build XGB dataset: {t_xgbds:.2f}s")
    except Exception as e:
        print(f"[PIPE DIAG] Failed to build/write XGB dataset: {e}")
        xgb_dataset_path = args.prices

    try:
        print("[PIPE] Building LSTM sequences...")
        t_lstmseq_start = time.perf_counter()
        lstm_X, lstm_y, lstm_index = build_lstm_sequences(prices_df, lookback=args.lookback, horizon=args.horizon, feature_config=feature_config)
        os.makedirs(os.path.join('data', 'top'), exist_ok=True)
        np.save(os.path.join('data','top','lstm_X.npy'), lstm_X)
        np.save(os.path.join('data','top','lstm_y.npy'), lstm_y)
        if lstm_index:
            idx_df = pd.DataFrame(lstm_index, columns=["ticker", "window_end_date"]) 
            idx_df.to_csv(os.path.join('data','top','lstm_index.csv'), index=False)
        t_lstmseq = time.perf_counter() - t_lstmseq_start
        print(f"[PIPE DIAG] Wrote LSTM arrays to data/top/lstm_X.npy (shape={lstm_X.shape}) and data/top/lstm_y.npy (shape={lstm_y.shape})")
        print(f"[PIPE TIME] Build LSTM sequences: {t_lstmseq:.2f}s")
    except Exception as e:
        print(f"[PIPE DIAG] Failed to build/write LSTM sequences: {e}")

    # Train models from prepared datasets
    tracker = get_tracker(mode=args.tracking, experiment_name=args.experiment, mlflow_uri=args.mlflow_uri, wandb_project=args.experiment, wandb_entity=args.wandb_entity)

    if not args.skip_xgb:
        print("[PIPE] Training XGB model...")
        t_xgb_train_start = time.perf_counter()
        xgb_model, xgb_preds = XGBTrainer(xgb_dataset_path, tracker=tracker, run_name='xgb_ranker').train()
        t_xgb_train = time.perf_counter() - t_xgb_train_start
        print("[PIPE] XGB training done.")
        print(f"[PIPE TIME] XGB training: {t_xgb_train:.2f}s")
        # Persist XGB model where the UI expects it
        try:
            os.makedirs("models", exist_ok=True)
            joblib.dump(xgb_model, os.path.join("models", "stock_predictor_top.joblib"))
            print("[PIPE DIAG] Saved XGB model to models/stock_predictor_top.joblib")
        except Exception as e:
            print(f"[PIPE DIAG] Failed to save XGB model: {e}")
    print("[PIPE] Training LSTM model...")
    t_lstm_train_start = time.perf_counter()
    lstm_model, lstm_preds = LSTMTrainer(xgb_dataset_path, tracker=tracker, run_name='lstm').train(
        lookback=args.lookback, horizon=args.horizon
    )
    t_lstm_train = time.perf_counter() - t_lstm_train_start
    print("[PIPE] LSTM training done.")
    print(f"[PIPE TIME] LSTM training: {t_lstm_train:.2f}s")

    # Rank stocks using XGB model
    if not args.skip_xgb:
        try:
            feature_cols = xgb_model.get_booster().feature_names
            print(f"[PIPE DIAG] XGB expects features ({len(feature_cols)}): {feature_cols}")
        except Exception:
            feature_cols = None
            print("[PIPE DIAG] Could not read XGB feature names; falling back to selector inference.")
    else:
        feature_cols = None

    # Build raw (unfiltered) feature set for complete-universe diagnostics/validation
    try:
        try:
            infer_prices_df = pd.read_csv("data/stock_prices_input.csv")
            print(f"[PIPE DIAG] Loaded raw input prices for raw features: data/stock_prices_input.csv (rows={len(infer_prices_df)})")
        except Exception as e:
            infer_prices_df = prices_df
            print(f"[PIPE DIAG] Could not load data/stock_prices_input.csv ({e}); using filtered prices for raw features fallback")
        # Build full-history raw features (today=None) so LSTM has adequate sequences per ticker
        # Technical-only full-history features for all symbols (no master merge)
        tech_full_df = _calculate_momentum(infer_prices_df.copy())
        # Add pre_earning_rally flag per date using stock_earnings.csv (date within 21 days before next earnings)
        try:
            earn_df = pd.read_csv('data/stock_earnings.csv', parse_dates=['earnings_date'])
            earn_df['ticker'] = earn_df['ticker'].astype(str).str.upper()
            tech_full_df['ticker'] = tech_full_df['ticker'].astype(str).str.upper()
            tech_full_df['date'] = pd.to_datetime(tech_full_df['date'], errors='coerce')
            tech_full_df.sort_values(['ticker','date'], inplace=True)
            earn_df.sort_values(['ticker','earnings_date'], inplace=True)
            tech_full_df['pre_earning_rally'] = False
            for tkr, g in earn_df.groupby('ticker'):
                edates = g['earnings_date'].to_numpy()
                if edates.size == 0:
                    continue
                mask_t = tech_full_df['ticker'] == tkr
                idx = mask_t[mask_t].index if hasattr(mask_t, 'index') else None
                sub = tech_full_df.loc[mask_t, ['date']]
                if sub.empty:
                    continue
                dvals = sub['date'].to_numpy()
                # For each earnings date, flag dates in [ed-21, ed]
                from numpy import timedelta64
                flag = pd.Series(False, index=sub.index)
                for ed in edates:
                    start = ed - pd.Timedelta(days=21)
                    curr = (dvals >= start) & (dvals <= ed)
                    if curr.any():
                        flag |= pd.Series(curr, index=sub.index)
                tech_full_df.loc[flag.index, 'pre_earning_rally'] = tech_full_df.loc[flag.index, 'pre_earning_rally'] | flag
        except Exception as _e:
            print(f"[PIPE DIAG] Could not add pre_earning_rally to RAW features: {_e}")
        raw_features_out_path = os.path.join('data','top','features_raw_full.csv')
        tech_full_df.to_csv(raw_features_out_path, index=False)
        print(f"[PIPE DIAG] Wrote RAW technical features (full history) to: {raw_features_out_path} (rows={len(tech_full_df)})")
        # Also write a latest-per-ticker snapshot for quick UI/diagnostics using shared utility
        try:
            ui_cols = [
                "ticker", "date", "close", "open", "high", "low", "volume",
                "broke_resistance", "post_earnings_dip_rally", "strong_momentum", "breakout_confirmed", "pre_earning_rally",
                "golden_cross", "trend_slope_15d", "trend_slope_30d", "rsi_14d", "volume_ratio",
                "volatility_30d", "momentum_5d", "momentum_10d", "momentum_20d", "momentum_30d",
                "momentum_60d", "up_day_ratio_20d", "momentum_winner", "trend_persistence_20d",
                "earnings_in_3_weeks", "last_2q_positive_surprises", "profit_margin", "long_term_growth_rate",
                "sector", "industry", "next_earnings_date"
            ]
            # Dynamically include any new engineered numeric/bool features in RAW snapshot
            try:
                base_numeric = tech_full_df.select_dtypes(include=["number", "bool"]).columns.tolist()
                extra_exclude = set(["target", "price_target_5d", "price_change_5d_pct"]) | set(ui_cols)
                dynamic_cols = [c for c in base_numeric if c not in extra_exclude]
                ui_cols_dynamic = ui_cols + [c for c in dynamic_cols if c not in ui_cols]
            except Exception:
                ui_cols_dynamic = ui_cols
            raw_latest_out = os.path.join('data','top','featured_stocks_raw.csv')
            write_latest_snapshot(tech_full_df, features_df, ui_cols_dynamic, raw_latest_out)
            print(f"[PIPE DIAG] Wrote RAW latest snapshot to: {raw_latest_out}")
        except Exception as _e:
            print(f"[PIPE DIAG] Failed to write featured_stocks_raw.csv: {_e}")
    except Exception as e:
        print(f"[PIPE DIAG] Failed to write RAW features CSV: {e}")

    # Read back features (for explicit traceability)
    # Use INFERENCE prices for building features so all symbols are covered.
    # Merge in bullish signal flags from engineered features (filtered) for ranking filters.
    try:
        # Build inference features from unfiltered prices to include all symbols
        try:
            infer_prices_df = pd.read_csv(args.infer_prices)
            print(f"[PIPE DIAG] Loaded INFERENCE prices from {args.infer_prices} (rows={len(infer_prices_df)})")
        except Exception as e:
            print(f"[PIPE DIAG] Failed to load --infer_prices {args.infer_prices}: {e}; falling back to --prices")
            infer_prices_df = prices_df
        xgb_infer_df = build_xgb_features(infer_prices_df, horizon=args.horizon, feature_config=feature_config)
        signal_cols = ["broke_resistance", "post_earnings_dip_rally", "breakout_confirmed", "strong_momentum"]
        available_signal_cols = [c for c in signal_cols if c in features_df.columns]
        if available_signal_cols:
            merge_cols = ["ticker", "date"] + available_signal_cols
            signals_frame = features_df[merge_cols].copy()
            xgb_infer_df = xgb_infer_df.merge(signals_frame, on=["ticker", "date"], how="left")
        # Fill missing signals with False (avoid future downcasting warnings)
        for col in signal_cols:
            if col not in xgb_infer_df.columns:
                xgb_infer_df[col] = False
            # normalize dtype to boolean without silent downcasting
            xgb_infer_df[col] = (
                pd.Series(xgb_infer_df[col], copy=False)
                  .astype('boolean')
                  .fillna(False)
                  .astype(bool, copy=False)
            )
        # Ensure all expected model features exist in inference frame
        if feature_cols is not None:
            for col in feature_cols:
                if col not in xgb_infer_df.columns:
                    xgb_infer_df[col] = 0.0
        # Union with all tickers from engineered features (fill missing model cols with zeros)
        try:
            xgb_infer_df['ticker'] = xgb_infer_df['ticker'].astype(str).str.upper()
            infer_prices_df['ticker'] = infer_prices_df['ticker'].astype(str).str.upper()
        except Exception:
            pass
        ui_latest_all = infer_prices_df.copy()
        try:
            ui_latest_all['date'] = pd.to_datetime(ui_latest_all['date'])
        except Exception:
            pass
        ui_latest_all = ui_latest_all.sort_values('date').groupby('ticker').tail(1)[['ticker','date']]
        # Left join to preserve all tickers from engineered features
        read_df = ui_latest_all.merge(xgb_infer_df, on=['ticker','date'], how='left', suffixes=('',''))
        # Backfill any missing model columns with zeros
        model_cols = feature_cols if feature_cols is not None else [c for c in xgb_infer_df.columns if c not in {'ticker','date'}]
        for col in model_cols:
            if col not in read_df.columns:
                read_df[col] = 0.0
        read_df[model_cols] = read_df[model_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        # Persist inference features for UI/selector_v2 to ensure model-feature alignment
        try:
            os.makedirs(os.path.join('data', 'top'), exist_ok=True)
            infer_out_path = os.path.join('data', 'top', 'xgb_features_input.csv')
            read_df.to_csv(infer_out_path, index=False)
            print(f"[PIPE DIAG] Saved XGB inference features to: {infer_out_path}")
        except Exception as e:
            print(f"[PIPE DIAG] Failed to save XGB inference features: {e}")
        print(f"[PIPE DIAG] Using XGB inference features (rows={len(read_df)}, cols={len(read_df.columns)})")
    except Exception:
        read_df = features_df
        print("[PIPE DIAG] Fallback to engineered features for ranking; feature mismatch may occur.")

    top_stocks_full = None
    if not args.skip_xgb:
        selector = StockSelector(xgb_model, read_df, feature_cols=feature_cols)
        # Build full-scored list, then slice for display
        top_stocks_full = selector.rank_stocks(top_n=None)
        top_stocks = top_stocks_full.head(args.top_n)
    else:
        # Load precomputed ranked CSV
        try:
            top_stocks = pd.read_csv(args.xgb_ranked_csv)
            print(f"[PIPE DIAG] Loaded existing XGB ranked CSV: {args.xgb_ranked_csv} (rows={len(top_stocks)})")
        except Exception as e:
            print(f"[PIPE DIAG] Failed to load {args.xgb_ranked_csv}: {e}")
            top_stocks = pd.DataFrame(columns=['ticker'])
    # Explicit diagnostic print for tabular (XGB) ranking
    try:
        print("[PIPE DIAG] Top-N by XGB (tabular):")
        print(top_stocks.to_string(index=False))
        print(f"[PIPE DIAG] Shape: {top_stocks.shape}")
    except Exception:
        print(top_stocks)
    if args.save:
        top_stocks.to_csv(args.save, index=False)
        print(f"Saved ranked stocks to {args.save}")

    # Persist XGB rankings to data/top for step-1 artifact (full list if available)
    try:
        os.makedirs(os.path.join('data', 'top'), exist_ok=True)
        xgb_ranked_out = os.path.join('data', 'top', 'xgb_ranked_output.csv')
        _xgb_out = (top_stocks_full.copy() if top_stocks_full is not None else top_stocks.copy())
        if 'predicted_return_pct' not in _xgb_out.columns and 'pred_return_pct' in _xgb_out.columns:
            _xgb_out['predicted_return_pct'] = _xgb_out['pred_return_pct']
        # Include xgb_pred alias for convenience
        if 'xgb_pred' not in _xgb_out.columns and 'predicted_return_pct' in _xgb_out.columns:
            try:
                _xgb_out['xgb_pred'] = pd.to_numeric(_xgb_out['predicted_return_pct'], errors='coerce')
            except Exception:
                _xgb_out['xgb_pred'] = _xgb_out['predicted_return_pct']
        cols = [c for c in ['ticker', 'predicted_return_pct', 'xgb_pred', 'confidence_score', 'risk_score', 'composite_score'] if c in _xgb_out.columns]
        _xgb_out[cols].to_csv(xgb_ranked_out, index=False)
        print(f"[PIPE DIAG] Wrote XGB ranked CSV to: {xgb_ranked_out} (rows={len(_xgb_out)})")
    except Exception as e:
        print(f"[PIPE DIAG] Failed to write data/top/xgb_ranked.csv: {e}")

    # Build latest-per-ticker feature snapshot used by UI and weekly merge
    try:
        latest_feat = read_df.copy()
        latest_feat['ticker'] = latest_feat['ticker'].astype(str).str.upper()
        if 'date' in latest_feat.columns:
            try:
                latest_feat['date'] = pd.to_datetime(latest_feat['date'])
            except Exception:
                pass
            latest_feat = latest_feat.sort_values('date').groupby('ticker').tail(1)
        xgb_features_latest_path = os.path.join('data', 'top', 'xgb_features_latest.csv')
        latest_feat.to_csv(xgb_features_latest_path, index=False)
        print(f"[PIPE DIAG] Wrote latest XGB features CSV to: {xgb_features_latest_path} (rows={len(latest_feat)})")
    except Exception as e:
        print(f"[PIPE DIAG] Failed to write xgb_features_latest.csv: {e}")

    # Build merged weekly XGB output (features + ranked) for UI convenience
    try:
        weekly_out = os.path.join('data', 'top', 'xgb_weekly_output.csv')
        build_xgb_weekly_output(
            features_path=os.path.join('data', 'top', 'xgb_features_latest.csv'),
            ranked_path=os.path.join('data', 'top', 'xgb_ranked_output.csv'),
            output_path=weekly_out,
        )
    except Exception as e:
        print(f"[PIPE DIAG] Failed to build xgb_weekly_output.csv: {e}")

    # --- NEW: Write UI-compatible CSV to keep UI unchanged ---
    try:
        # Build a UI dataframe using XGB inference features merged with engineered signals
        ui_cols = [
            # identity / base
            "ticker", "date", "close", "open", "high", "low", "volume",
            # signals used by UI
            "broke_resistance", "post_earnings_dip_rally", "strong_momentum", "breakout_confirmed", "pre_earning_rally",
            "golden_cross", "trend_slope_15d", "trend_slope_30d", "rsi_14d", "volume_ratio",
            "volatility_30d", "momentum_5d", "momentum_10d", "momentum_20d", "momentum_30d",
            "momentum_60d", "up_day_ratio_20d", "momentum_winner", "trend_persistence_20d",
            # fundamentals/context if present
            "earnings_in_3_weeks", "last_2q_positive_surprises", "profit_margin", "long_term_growth_rate",
            "sector", "industry"
        ]

        # Dynamically include any new engineered numeric/bool features in UI snapshot
        try:
            base_numeric = read_df.select_dtypes(include=["number", "bool"]).columns.tolist()
            extra_exclude = set(["target", "price_target_5d", "price_change_5d_pct"]) | set(ui_cols)
            dynamic_cols = [c for c in base_numeric if c not in extra_exclude]
            ui_cols = ui_cols + [c for c in dynamic_cols if c not in ui_cols]
        except Exception:
            pass

        # Start from xgb_infer_df (same frame used for prediction features)
        ui_df = read_df.copy()

        # Merge additional engineered columns from features_df if missing
        need_from_features = [c for c in ui_cols if c not in ui_df.columns and c in features_df.columns]
        if need_from_features:
            ui_df = ui_df.merge(
                features_df[["ticker", "date"] + need_from_features],
                on=["ticker", "date"], how="left"
            )

        # Collapse to latest row per ticker
        ui_latest = ui_df.sort_values("date").groupby("ticker").tail(1)

        # Ensure only available columns are written; fill missing ui fields with sensible defaults
        final_cols = [c for c in ui_cols if c in ui_latest.columns]
        for c in ui_cols:
            if c not in final_cols:
                # Add defaults: booleans -> False, numerics -> 0
                if c in [
                    "broke_resistance", "post_earnings_dip_rally", "strong_momentum", "breakout_confirmed",
                    "golden_cross", "momentum_winner", "trend_persistence_20d", "earnings_in_3_weeks",
                    "last_2q_positive_surprises"
                ]:
                    ui_latest[c] = False
                else:
                    ui_latest[c] = 0
                final_cols.append(c)

        # Order columns for consistency
        ui_out = ui_latest[final_cols]

        # Write to top path for UI using shared utility to ensure consistent schema
        ui_out_path = os.path.join('data','top', 'featured_stocks_top.csv')
        write_latest_snapshot(ui_latest, features_df, final_cols, ui_out_path)
        print(f"[PIPE DIAG] Wrote UI features CSV to: {ui_out_path}")
    except Exception as e:
        print(f"[PIPE DIAG] Failed to write UI CSV: {e}")

    # --- Ensemble predictions (XGB + LSTM) diagnostics and optional CSV ---
    try:
        # Latest snapshot per ticker (for writing UI files); fallback if skipping XGB
        if 'date' in read_df.columns:
            latest_all = read_df.sort_values('date').groupby('ticker').tail(1)
        else:
            latest_all = features_df.sort_values('date').groupby('ticker').tail(1)

        # XGB component: either predict via model or load from ranked CSV
        if not args.skip_xgb and 'xgb_model' in locals():
            xgb_feats = feature_cols if feature_cols is not None else xgb_model.get_booster().feature_names
            xgb_feats = [c for c in xgb_feats if c in latest_all.columns]
            X_xgb_latest = latest_all[xgb_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
            preds_xgb_latest = xgb_model.predict(X_xgb_latest)
            df_xgb_latest = pd.DataFrame({'ticker': latest_all['ticker'].astype(str).values, 'xgb_pred': preds_xgb_latest})
        else:
            try:
                df_xgb = pd.read_csv(args.xgb_ranked_csv)
                # Use confidence_score if present; else predicted_return_pct
                score_col = 'confidence_score' if 'confidence_score' in df_xgb.columns else (
                    'predicted_return_pct' if 'predicted_return_pct' in df_xgb.columns else None)
                if score_col is None:
                    raise ValueError('No suitable score column in XGB ranked CSV')
                df_xgb_latest = df_xgb[['ticker', score_col]].rename(columns={score_col: 'xgb_pred'})
            except Exception as e:
                print(f"[PIPE DIAG] Failed to load XGB component from {args.xgb_ranked_csv}: {e}")
                df_xgb_latest = pd.DataFrame(columns=['ticker', 'xgb_pred'])

        # LSTM sequences for latest window per ticker (use provided features CSV for sufficient history)
        # If LSTM model is not built (no layers), skip LSTM in ensemble and fall back to XGB-only
        if not hasattr(lstm_model, 'layers') or len(getattr(lstm_model, 'layers', [])) == 0:
            print('[PIPE DIAG] LSTM model has no layers; using XGB-only ensemble output.')
            try:
                wx, wl = 1.0, 0.0
                df_combined = df_xgb_latest.copy()
                df_combined['ensemble_pred'] = df_combined['xgb_pred']
                top_ens = df_combined.sort_values('ensemble_pred', ascending=False).head(args.top_n)
                print('[PIPE DIAG] Top-N by Ensemble (XGB-only):')
                print(top_ens[['ticker', 'ensemble_pred']].to_string(index=False))
                ui_ens = latest_all.merge(df_combined[['ticker', 'ensemble_pred']], on='ticker', how='inner').copy()
                ui_ens['predicted_return_pct'] = ui_ens['ensemble_pred']
                ui_ens['w_xgb'] = float(wx)
                ui_ens['w_lstm'] = float(wl)
                os.makedirs('data', exist_ok=True)
                ens_out_path = os.path.join('data', 'featured_stocks_top_ensemble.csv')
                ui_ens[['ticker', 'date', 'close', 'predicted_return_pct', 'w_xgb', 'w_lstm']].to_csv(ens_out_path, index=False)
                print(f"[PIPE DIAG] Wrote ensemble CSV to: {ens_out_path} (rows={len(ui_ens)})")
            except Exception as e:
                print(f"[PIPE DIAG] Failed to write XGB-only ensemble CSV: {e}")
            # Skip the rest of LSTM ensemble block
            return
        # Use the same feature set and lookback as the trained LSTM model to avoid shape mismatches
        try:
            lstm_feat_candidates = list(getattr(lstm_model, 'feature_cols'))
            lookback = int(getattr(lstm_model, 'lookback'))
        except Exception:
            lstm_feat_candidates = [
                c for c in read_df.select_dtypes(include=['number', 'bool']).columns
                if c not in {'ticker', 'date', 'target'}
            ]
            lookback = args.lookback
        # Load sequence features CSV
        try:
            seq_df = pd.read_csv(args.lstm_features_csv)
            print(f"[PIPE DIAG] Loaded LSTM features CSV: {args.lstm_features_csv} (rows={len(seq_df)})")
        except Exception as e:
            print(f"[PIPE DIAG] Failed to load LSTM features CSV {args.lstm_features_csv}: {e}")
            seq_df = read_df.copy()
        # Ensure needed feature columns exist (compute _csz if only base exists)
        try:
            seq_df['date'] = pd.to_datetime(seq_df['date'])
        except Exception:
            pass
        seq_df['ticker'] = seq_df['ticker'].astype(str).str.upper()
        needed = list(lstm_feat_candidates)
        for fc in needed:
            if fc not in seq_df.columns and fc.endswith('_csz'):
                base = fc[:-4]
                if base in seq_df.columns and 'date' in seq_df.columns:
                    try:
                        seq_df[fc] = seq_df.groupby('date')[base].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
                    except Exception:
                        seq_df[fc] = 0.0
            if fc not in seq_df.columns:
                seq_df[fc] = 0.0
        eps = 1e-8
        seq_list, tickers_seq = [], []
        for tkr, grp in seq_df.groupby('ticker'):
            g = grp.sort_values('date').tail(lookback)
            if len(g) < lookback:
                continue
            use_cols = [c for c in lstm_feat_candidates if c in g.columns]
            if not use_cols:
                continue
            window = g[use_cols].to_numpy(dtype=float, copy=False)
            # Model expects _csz inputs already, no additional normalization here
            seq_list.append(window)
            tickers_seq.append(str(tkr))

        df_ens_out = None
        if seq_list:
            X_lstm_latest = np.asarray(seq_list)
            try:
                preds_lstm_latest = lstm_model.predict(X_lstm_latest).ravel()
            except Exception as e:
                print(f"[PIPE DIAG] LSTM prediction failed on latest sequences: {e}")
                preds_lstm_latest = None
            # Build LSTM output including all tickers (NaN when no sequence)
            all_tickers = seq_df['ticker'].astype(str).str.upper().unique().tolist()
            df_lstm_latest = pd.DataFrame({'ticker': all_tickers})
            pred_map = {}
            if preds_lstm_latest is not None:
                for tk, val in zip(tickers_seq, preds_lstm_latest):
                    pred_map[tk] = val
            df_lstm_latest['lstm_pred'] = df_lstm_latest['ticker'].map(pred_map).astype(float)

            # Persist LSTM predictions to data/top for step-2 artifact, including derived columns
            try:
                os.makedirs(os.path.join('data', 'top'), exist_ok=True)
                lstm_out = os.path.join('data', 'top', 'lstm_weekly_predictions_output.csv')
                out_lstm = df_lstm_latest.copy()
                # Standard columns
                out_lstm['lstm_predicted_return_pct'] = out_lstm['lstm_pred']
                out_lstm['predicted_return_pct'] = out_lstm['lstm_pred']
                # Confidence score via p95 scaling of absolute predictions
                try:
                    abs_vals = np.abs(out_lstm['lstm_pred'].astype(float))
                    p95 = float(np.nanpercentile(abs_vals, 95)) if np.isfinite(abs_vals).any() else 0.0
                    scale = p95 if p95 > 1e-8 else (float(np.nanmax(abs_vals)) if np.nanmax(abs_vals) > 1e-8 else 1.0)
                except Exception:
                    scale = 1.0
                out_lstm['confidence_score'] = (100.0 * (np.abs(out_lstm['lstm_pred'].astype(float)) / scale)).clip(0.0, 100.0)
                # Risk score: default 50 (can be refined using volatility features if desired)
                out_lstm['risk_score'] = 50.0
                # Composite score mirroring XGB weights
                out_lstm['composite_score'] = (
                    0.5 * out_lstm['predicted_return_pct'].fillna(0.0) +
                    0.3 * out_lstm['confidence_score'].fillna(0.0) +
                    0.2 * (100.0 - out_lstm['risk_score'].fillna(50.0))
                )
                # Column order
                cols = ['ticker', 'lstm_pred', 'lstm_predicted_return_pct', 'predicted_return_pct', 'confidence_score', 'risk_score', 'composite_score']
                keep_cols = [c for c in cols if c in out_lstm.columns]
                out_lstm[keep_cols].to_csv(lstm_out, index=False)
                print(f"[PIPE DIAG] Wrote LSTM predictions CSV to: {lstm_out} (rows={len(out_lstm)})")
            except Exception as e:
                print(f"[PIPE DIAG] Failed to write data/top/lstm_weekly_predictions_output.csv: {e}")

            # Align on union of tickers (outer merge)
            df_combined = df_xgb_latest.merge(df_lstm_latest, on='ticker', how='outer')
            # Add XGB regressor predicted return % if model available
            try:
                import joblib as _joblib
                reg_model = _joblib.load(os.path.join('models','stock_predictor_top.joblib'))
                try:
                    reg_cols = reg_model.get_booster().feature_names
                except Exception:
                    reg_cols = None
                if reg_cols is None:
                    # Fallback: use numeric/bool columns except identifiers
                    reg_cols = [c for c in read_df.select_dtypes(include=['number','bool']).columns if c not in {'ticker','date'}]
                rf = read_df.copy()
                rf['ticker'] = rf['ticker'].astype(str).str.upper()
                for c in reg_cols:
                    if c not in rf.columns:
                        rf[c] = 0.0
                Xr = rf[reg_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
                reg_preds = reg_model.predict(Xr)
                reg_df = pd.DataFrame({'ticker': rf['ticker'].values, 'xgb_predicted_return_pct': reg_preds})
                # Keep latest per ticker
                reg_df = reg_df.groupby('ticker', as_index=False).last()
                df_combined = df_combined.merge(reg_df, on='ticker', how='left')
            except Exception as _e:
                print(f"[PIPE DIAG] Skipped XGB regressor predictions: {_e}")
            # Compute masks but keep all tickers (even if both preds are NaN) so diagnostics cover full universe
            mask_x = np.isfinite(df_combined['xgb_pred'])
            mask_l = np.isfinite(df_combined['lstm_pred'])
            if not df_combined.empty:
                # Weight by validation MAE if available (lower MAE -> higher weight), unless overridden by CLI method
                xgb_mae = None
                if not args.skip_xgb and 'xgb_model' in locals():
                    try:
                        xgb_mae = getattr(xgb_model, '_validation_mae', None)
                    except Exception:
                        xgb_mae = None
                try:
                    lstm_mae = getattr(lstm_model, '_validation_mae', None)
                except Exception:
                    lstm_mae = None
                # Defaults
                method = str(args.ensemble_method).lower()
                alpha = float(args.ensemble_alpha)
                wx = wl = 0.5
                if method == 'weighted':
                    # Use CLI alpha directly
                    wx, wl = alpha, 1 - alpha
                elif method == 'rank':
                    # Rank normalize before blending
                    def _rank_score(s: pd.Series) -> pd.Series:
                        s = pd.to_numeric(s, errors='coerce')
                        try:
                            r = s.rank(method='min', ascending=False)
                            return (len(r) - r) / (len(r) - 1) if len(r) > 1 else pd.Series(0.0, index=s.index)
                        except Exception:
                            return pd.Series(0.0, index=s.index)
                    rx = _rank_score(df_combined['xgb_pred'])
                    rl = _rank_score(df_combined['lstm_pred'])
                    df_combined['ensemble_score'] = alpha * rx + (1 - alpha) * rl
                else:
                    # Fallback to MAE-based weighting
                    if isinstance(xgb_mae, (int, float)) and np.isfinite(xgb_mae) and xgb_mae > 0 and \
                       isinstance(lstm_mae, (int, float)) and np.isfinite(lstm_mae) and lstm_mae > 0:
                        wx = lstm_mae / (xgb_mae + lstm_mae)
                        wl = xgb_mae / (xgb_mae + lstm_mae)
                # If one model has no predictions or invalid weights, fallback to the other
                only_xgb = mask_x & ~mask_l
                only_lstm = mask_l & ~mask_x
                both = mask_x & mask_l
                ens_vals = np.zeros(len(df_combined), dtype=float)
                ens_vals[only_xgb.values] = df_combined.loc[only_xgb, 'xgb_pred'].values
                ens_vals[only_lstm.values] = df_combined.loc[only_lstm, 'lstm_pred'].values
                if both.any():
                    if method == 'rank':
                        # already set ensemble_score above; keep ens_vals for fallback
                        pass
                    else:
                        ens_vals[both.values] = (
                            wx * df_combined.loc[both, 'xgb_pred'].values +
                            wl * df_combined.loc[both, 'lstm_pred'].values
                        )
                # For weighted/mae methods, set ensemble_pred; for rank, ensure ensemble_score exists
                if method == 'rank':
                    if 'ensemble_score' not in df_combined.columns:
                        # Safety: compute simple average ranks if missing
                        def _rank_score2(s: pd.Series) -> pd.Series:
                            s = pd.to_numeric(s, errors='coerce')
                            try:
                                r = s.rank(method='min', ascending=False)
                                return (len(r) - r) / (len(r) - 1) if len(r) > 1 else pd.Series(0.0, index=s.index)
                            except Exception:
                                return pd.Series(0.0, index=s.index)
                        rx = _rank_score2(df_combined['xgb_pred'])
                        rl = _rank_score2(df_combined['lstm_pred'])
                        df_combined['ensemble_score'] = alpha * rx + (1 - alpha) * rl
                else:
                    df_combined['ensemble_pred'] = ens_vals
                # Merge selloff flag from latest features for penalty
                try:
                    if 'selloff_flag' in latest_all.columns:
                        so_map = dict(zip(latest_all['ticker'].astype(str).str.upper(), pd.to_numeric(latest_all['selloff_flag'], errors='coerce').fillna(0.0)))
                        df_combined['selloff_flag'] = df_combined['ticker'].astype(str).str.upper().map(lambda t: so_map.get(t, 0.0)).fillna(0.0)
                    else:
                        df_combined['selloff_flag'] = 0.0
                except Exception:
                    df_combined['selloff_flag'] = 0.0

                # Apply proportional selloff penalty to ensemble scores/preds (post-combination)
                try:
                    import os as _os
                    vr_min = float(_os.getenv('SELLOFF_VRATIO_MIN', '1.2'))
                    vr_ref = float(_os.getenv('SELLOFF_VRATIO_REF', '2.5'))
                    drop_ref = float(_os.getenv('SELLOFF_REF_DROP_PCT', '0.05'))
                    f_min = float(_os.getenv('SELLOFF_FACTOR_MIN', '0.60'))
                    f_max = float(_os.getenv('SELLOFF_FACTOR_MAX', '0.75'))
                    vr = pd.to_numeric(latest_all.get('volume_ratio', 0.0), errors='coerce').fillna(0.0)
                    pc = pd.to_numeric(latest_all.get('price_change_pct', latest_all.get('return_1d', 0.0)), errors='coerce').fillna(0.0)
                    so = pd.to_numeric(df_combined.get('selloff_flag', 0.0), errors='coerce').fillna(0.0) >= 1.0
                    # Align factors by ticker using latest_all
                    fac_map = {}
                    for tkr, v, p, s in zip(latest_all['ticker'].astype(str).str.upper(), vr, pc, so):
                        if bool(s) and (v > vr_min) and (p < 0.0):
                            sv = max(0.0, min(1.0, (float(v) - vr_min) / max(1e-6, (vr_ref - vr_min))))
                            sd = max(0.0, min(1.0, (-float(p)) / max(1e-6, drop_ref)))
                            sev = 0.5 * sv + 0.5 * sd
                            fac_map[tkr] = float(f_max - (f_max - f_min) * sev)
                        else:
                            fac_map[tkr] = 1.0
                    factor_series = df_combined['ticker'].astype(str).str.upper().map(lambda t: fac_map.get(t, 1.0))
                    if method == 'rank':
                        if 'ensemble_score' in df_combined.columns:
                            df_combined['ensemble_score'] = pd.to_numeric(df_combined['ensemble_score'], errors='coerce') * factor_series
                    else:
                        if 'ensemble_pred' in df_combined.columns:
                            df_combined['ensemble_pred'] = pd.to_numeric(df_combined['ensemble_pred'], errors='coerce') * factor_series
                except Exception:
                    pass

                # Optional continuous penalty weight (selloff + RSI) and utility-style final score
                try:
                    import os as _os
                    PEN_ALPHA = float(_os.getenv('PEN_ALPHA', '8.0'))
                    RSI_GAMMA = float(_os.getenv('RSI_GAMMA', '0.35'))
                    RSI_BONUS = float(_os.getenv('RSI_BONUS', '0.10'))
                    W_MIN = float(_os.getenv('PEN_W_MIN', '0.50'))
                    W_MAX = float(_os.getenv('PEN_W_MAX', '1.10'))
                    # Map features by ticker
                    feat_map = latest_all[['ticker','rsi_14d','volume_ratio','price_change_pct']].copy()
                    feat_map['ticker'] = feat_map['ticker'].astype(str).str.upper()
                    feat_map = feat_map.set_index('ticker')
                    def _selloff_penalty(vol_ratio: float, price_chg_pct: float, alpha: float) -> float:
                        vr_excess = max(0.0, float(vol_ratio) - 1.0)
                        drop_mag = max(0.0, -float(price_chg_pct))
                        try:
                            import numpy as _np
                            return float(_np.exp(-alpha * vr_excess * drop_mag))
                        except Exception:
                            return 1.0
                    def _overbought_penalty(rsi: float, gamma: float) -> float:
                        over = max(0.0, (float(rsi) - 70.0) / 30.0)
                        return float(1.0 - gamma * min(1.0, over))
                    def _healthy_low_rsi_bonus(rsi: float, price_chg_pct: float, vol_ratio: float, bonus: float) -> float:
                        if (float(price_chg_pct) > 0.0) and (float(rsi) < 45.0) and (float(vol_ratio) <= 1.2):
                            return float(1.0 + bonus * ((45.0 - float(rsi)) / 45.0))
                        return 1.0
                    def _compute_weight(row) -> float:
                        t = str(row['ticker']).upper()
                        try:
                            rsi = float(feat_map.at[t, 'rsi_14d']) if 'rsi_14d' in feat_map.columns else 50.0
                            vr = float(feat_map.at[t, 'volume_ratio']) if 'volume_ratio' in feat_map.columns else 1.0
                            pc = float(feat_map.at[t, 'price_change_pct']) if 'price_change_pct' in feat_map.columns else 0.0
                        except Exception:
                            rsi, vr, pc = 50.0, 1.0, 0.0
                        w = _selloff_penalty(vr, pc, PEN_ALPHA) * _overbought_penalty(rsi, RSI_GAMMA) * _healthy_low_rsi_bonus(rsi, pc, vr, RSI_BONUS)
                        try:
                            import numpy as _np
                            return float(_np.clip(w, W_MIN, W_MAX))
                        except Exception:
                            return max(W_MIN, min(W_MAX, float(w)))
                    df_combined['w_pen'] = df_combined.apply(_compute_weight, axis=1)
                    # Apply penalties to ensemble metric and confidence if present
                    if method == 'rank':
                        if 'ensemble_score' in df_combined.columns:
                            df_combined['ensemble_score'] = pd.to_numeric(df_combined['ensemble_score'], errors='coerce') * df_combined['w_pen']
                    else:
                        if 'ensemble_pred' in df_combined.columns:
                            df_combined['ensemble_pred'] = pd.to_numeric(df_combined['ensemble_pred'], errors='coerce') * df_combined['w_pen']
                    # Utility-style final score (optional)
                    use_util = _os.getenv('USE_UTILITY_SCORE', '0') == '1'
                    if use_util:
                        mu_col = 'ensemble_score' if (method == 'rank' and 'ensemble_score' in df_combined.columns) else ('ensemble_pred' if 'ensemble_pred' in df_combined.columns else None)
                        if mu_col is not None:
                            mu_adj = pd.to_numeric(df_combined[mu_col], errors='coerce')
                            conf_adj = pd.to_numeric(final_df['confidence_score'], errors='coerce') if 'confidence_score' in (locals().get('final_df', pd.DataFrame())).columns else pd.Series(1.0, index=df_combined.index)
                            # If no confidence available, fall back to absolute mu
                            if 'confidence_score' not in (locals().get('final_df', pd.DataFrame())).columns:
                                conf_adj = pd.Series(1.0, index=df_combined.index)
                            df_combined['final_score'] = mu_adj * conf_adj
                    if os.getenv('DEBUG_SELLOFF', '0') == '1':
                        try:
                            dbg_cols2 = ['ticker','w_pen']
                            sc = 'ensemble_score' if 'ensemble_score' in df_combined.columns else ('ensemble_pred' if 'ensemble_pred' in df_combined.columns else None)
                            if sc:
                                dbg_cols2.append(sc)
                            os.makedirs(os.path.join('data','top'), exist_ok=True)
                            df_combined[dbg_cols2].to_csv(os.path.join('data','top','debug_penalty_weights.csv'), index=False)
                            print('[DEBUG] Wrote penalty weights to data/top/debug_penalty_weights.csv')
                        except Exception:
                            pass
                except Exception:
                    pass

                # Optional meta-ranking (stacking) as mandatory fallback when no external model provided
                try:
                    import os as _os
                    force_stack = _os.getenv('FORCE_STACK', '0') == '1'
                    meta_model_path = args.ensemble_stack_model or os.environ.get('ENSEMBLE_STACK_MODEL')
                    use_meta = force_stack or (meta_model_path is not None)
                    if use_meta:
                        # Prepare meta features
                        meta = df_combined.copy()
                        # Attach latest features needed
                        meta = meta.merge(
                            latest_all[['ticker','rsi_14d','volume_ratio','price_change_pct','selloff_flag']] if 'selloff_flag' in latest_all.columns else 
                            latest_all[['ticker','rsi_14d','volume_ratio','price_change_pct']],
                            on='ticker', how='left'
                        )
                        # If external model provided, use it
                        used_external = False
                        if isinstance(meta_model_path, str) and os.path.isfile(meta_model_path):
                            try:
                                import joblib as _joblib
                                mdl = _joblib.load(meta_model_path)
                                feat_cols = []
                                for c in ['xgb_pred','lstm_pred','ensemble_score','ensemble_pred','w_pen','rsi_14d','volume_ratio','price_change_pct','selloff_flag']:
                                    if c in meta.columns:
                                        feat_cols.append(c)
                                Xs = meta[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
                                meta['meta_score'] = pd.to_numeric(mdl.predict(Xs), errors='coerce')
                                used_external = True
                                print(f"[PIPE DIAG] Applied external stacking model: {meta_model_path}")
                            except Exception as _e:
                                print(f"[PIPE WARN] Failed to apply stacking model {meta_model_path}: {_e}")
                        if not used_external:
                            # Deterministic meta: overweight LSTM and penalties
                            b_x = float(_os.getenv('META_BETA_XGB', '0.35'))
                            b_l = float(_os.getenv('META_BETA_LSTM', '0.65'))
                            b_w = float(_os.getenv('META_BETA_W', '0.50'))
                            b_pc = float(_os.getenv('META_BETA_PC', '0.20'))
                            b_vr = float(_os.getenv('META_BETA_VR', '-0.10'))
                            b_rsi = float(_os.getenv('META_BETA_RSI', '-0.05'))
                            b_so = float(_os.getenv('META_BETA_SO', '-0.30'))
                            x = pd.to_numeric(meta.get('xgb_pred', 0.0), errors='coerce').fillna(0.0)
                            l = pd.to_numeric(meta.get('lstm_pred', 0.0), errors='coerce').fillna(0.0)
                            w = pd.to_numeric(meta.get('w_pen', 1.0), errors='coerce').fillna(1.0)
                            rsi = pd.to_numeric(meta.get('rsi_14d', 50.0), errors='coerce').fillna(50.0)
                            vr  = pd.to_numeric(meta.get('volume_ratio', 1.0), errors='coerce').fillna(1.0)
                            pc  = pd.to_numeric(meta.get('price_change_pct', 0.0), errors='coerce').fillna(0.0)
                            so  = pd.to_numeric(meta.get('selloff_flag', 0.0), errors='coerce').fillna(0.0)
                            meta['meta_score'] = (
                                b_x * x + b_l * l + b_w * w + b_pc * pc + b_vr * vr + b_rsi * rsi + b_so * so
                            )
                        # Use meta score as primary sort column
                        df_combined = df_combined.merge(meta[['ticker','meta_score']], on='ticker', how='left')
                        if df_combined['meta_score'].notna().any():
                            df_combined['ensemble_score'] = pd.to_numeric(df_combined['meta_score'], errors='coerce')
                            method = 'rank'  # ensure we use ensemble_score downstream
                except Exception:
                    pass

                # Print diagnostic Top-N
                sort_col = 'ensemble_score' if method == 'rank' else 'ensemble_pred'
                top_ens = df_combined.sort_values(sort_col, ascending=False).head(args.top_n)
                print("[PIPE DIAG] Top-N by Ensemble (XGB+LSTM):")
                print(top_ens[['ticker', sort_col]].to_string(index=False))
                # Debug dump
                if os.getenv('DEBUG_SELLOFF', '0') == '1':
                    try:
                        dbg_cols = ['ticker','selloff_flag',sort_col]
                        for c in ['xgb_pred','lstm_pred','xgb_predicted_return_pct']:
                            if c in df_combined.columns:
                                dbg_cols.append(c)
                        dbg = df_combined[dbg_cols].copy().sort_values(sort_col, ascending=False)
                        os.makedirs(os.path.join('data','top'), exist_ok=True)
                        dbg.to_csv(os.path.join('data','top','debug_ensemble_penalties.csv'), index=False)
                        print('[DEBUG] Wrote ensemble penalty debug to data/top/debug_ensemble_penalties.csv')
                    except Exception:
                        pass

                # If skipping XGB, build ensemble via normalized blend
                if args.skip_xgb:
                    def _minmax(s: pd.Series) -> pd.Series:
                        s = pd.to_numeric(s, errors='coerce')
                        mn, mx = s.min(), s.max()
                        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                            return pd.Series(0.5, index=s.index)
                        return (s - mn) / (mx - mn)
                    alpha = float(args.ensemble_alpha)
                    rn = _minmax(df_combined['xgb_pred'])
                    ln = _minmax(df_combined['lstm_pred'])
                    df_combined['ensemble_pred'] = alpha * rn + (1 - alpha) * ln

                # Write final ensemble scores (step-3 artifact) to data/top
                t_ens_start = time.perf_counter()
                try:
                    os.makedirs(os.path.join('data', 'top'), exist_ok=True)
                    # Attach confidence if available from top_stocks
                    final_df = df_combined.copy()
                    if 'confidence_score' in top_stocks.columns:
                        final_df = final_df.merge(top_stocks[['ticker', 'confidence_score']], on='ticker', how='left')
                    # Standardize columns
                    final_df = final_df.rename(columns={'lstm_pred': 'lstm_predicted_return_pct', 'ensemble_pred': 'ensemble_score'})
                    # Keep names with positive signal in ANY available prediction (XGB/LSTM)
                    import numpy as _np
                    import pandas as _pd
                    def _num(s):
                        return _pd.to_numeric(s, errors='coerce')
                    candidates = []
                    for c in ['xgb_predicted_return_pct', 'xgb_pred', 'lstm_predicted_return_pct']:
                        if c in final_df.columns:
                            candidates.append(_num(final_df[c]))
                    if candidates:
                        try:
                            mat = _pd.concat(candidates, axis=1)
                            # Row-wise max across available predictions
                            ret_proxy = mat.max(axis=1, skipna=True)
                        except Exception:
                            ret_proxy = candidates[0]
                        before = len(final_df)
                        final_df = final_df[ret_proxy > 0].copy()
                        print(f"[PIPE DIAG] Filtered positive returns (any model): {before} -> {len(final_df)}")
                    # Optional policy: exclude selloff names from final top-N
                    try:
                        import os as _os
                        if _os.getenv('EXCLUDE_SELLOFF_FROM_TOP', '0') == '1' and 'selloff_flag' in final_df.columns:
                            bef = len(final_df)
                            m_ex = pd.to_numeric(final_df['selloff_flag'], errors='coerce').fillna(0.0) >= 1.0
                            final_df = final_df[~m_ex].copy()
                            print(f"[PIPE DIAG] Policy exclude selloff from top: {bef} -> {len(final_df)}")
                            # Backfill if needed with non-selloff from the broader combined set
                            if len(final_df) < args.top_n:
                                need = args.top_n - len(final_df)
                                sort_col2 = 'ensemble_score' if 'ensemble_score' in df_combined.columns else ('ensemble_pred' if 'ensemble_pred' in df_combined.columns else None)
                                if sort_col2 is not None:
                                    # Build pool: positive return proxy and non-selloff, excluding already selected
                                    pool = df_combined.copy()
                                    pool = pool[~pool['ticker'].isin(final_df['ticker'])]
                                    if 'selloff_flag' in pool.columns:
                                        pool = pool[pd.to_numeric(pool['selloff_flag'], errors='coerce').fillna(0.0) < 1.0]
                                    # Positive signal proxy reuse
                                    cand_cols = [c for c in ['xgb_predicted_return_pct', 'xgb_pred', 'lstm_predicted_return_pct'] if c in pool.columns]
                                    if cand_cols:
                                        rp = pd.concat([pd.to_numeric(pool[c], errors='coerce') for c in cand_cols], axis=1).max(axis=1, skipna=True)
                                        pool = pool[rp > 0]
                                    pool = pool.sort_values(sort_col2, ascending=False).head(need)
                                    if not pool.empty:
                                        pool = pool.rename(columns={'lstm_pred': 'lstm_predicted_return_pct'})
                                        if 'confidence_score' in top_stocks.columns:
                                            pool = pool.merge(top_stocks[['ticker', 'confidence_score']], on='ticker', how='left')
                                        final_df = pd.concat([final_df, pool], ignore_index=True, sort=False).drop_duplicates('ticker').head(args.top_n)
                    except Exception:
                        pass

                    # Sort by ensemble score or pred and take top-N (log fallback)
                    if 'ensemble_score' in final_df.columns:
                        final_df = final_df.sort_values('ensemble_score', ascending=False)
                    elif 'ensemble_pred' in final_df.columns:
                        print('[PIPE DIAG] ensemble_score not present; falling back to ensemble_pred for Top-N sort.')
                        final_df = final_df.sort_values('ensemble_pred', ascending=False)
                    else:
                        print('[PIPE DIAG] Neither ensemble_score nor ensemble_pred present; Top-N sort may be unstable.')
                    final_df = final_df.head(args.top_n)
                    # Persist selloff diagnostics for transparency
                    if 'selloff_flag' not in final_df.columns and 'selloff_flag' in latest_all.columns:
                        final_df = final_df.merge(latest_all[['ticker','selloff_flag','volume_ratio','price_change_pct']], on='ticker', how='left')
                    out_cols = [c for c in ['ticker', 'xgb_pred', 'xgb_predicted_return_pct', 'lstm_predicted_return_pct', 'confidence_score', 'ensemble_score', 'selloff_flag', 'volume_ratio', 'price_change_pct'] if c in final_df.columns]
                    final_df[out_cols].to_csv(os.path.join('data', 'top', 'ensemble_scores_output.csv'), index=False)
                    print(f"[PIPE DIAG] Wrote final ensemble scores CSV to: data/top/ensemble_scores_output.csv (rows={len(final_df)})")
                    print(f"[PIPE TIME] Ensemble write: {time.perf_counter() - t_ens_start:.2f}s")
                except Exception as e:
                    print(f"[PIPE DIAG] Failed to write data/top/final_ensemble_scores.csv: {e}")
        else:
            print('[PIPE DIAG] Ensemble skipped: no valid LSTM sequences built for latest window.')
    except Exception as e:
        print(f"[PIPE DIAG] Failed to compute ensemble predictions: {e}")

    print(f"[PIPE TIME] Total pipeline: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
