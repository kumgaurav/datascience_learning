import argparse
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
    args = parser.parse_args()

    # Load all data (master + prices), mirroring run_pipeline
    master_df, prices_df = load_all_data(file_date=None)
    if master_df.empty or prices_df.empty:
        print("[PIPE DIAG] ERROR: Failed to load base datasets via load_all_data().")
        return

    # Prepare feature set using the same logic as run_pipeline (but ensure date,ticker,close retained)
    try:
        latest_prices_date = prices_df['date'].max()
        today_date = latest_prices_date.date() if hasattr(latest_prices_date, 'date') else None
    except Exception:
        today_date = None
    features_df = create_all_features(master_df, prices_df, today=today_date)
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
        xgb_df = build_xgb_dataset(prices_df, horizon=args.horizon)
        os.makedirs(os.path.join('data', 'top'), exist_ok=True)
        xgb_dataset_path = os.path.join('data', 'top', 'xgb_dataset_input.csv')
        xgb_df.to_csv(xgb_dataset_path, index=False)
        print(f"[PIPE DIAG] Wrote XGB dataset to: {xgb_dataset_path} (rows={len(xgb_df)})")
    except Exception as e:
        print(f"[PIPE DIAG] Failed to build/write XGB dataset: {e}")
        xgb_dataset_path = args.prices

    try:
        lstm_X, lstm_y, lstm_index = build_lstm_sequences(prices_df, lookback=args.lookback, horizon=args.horizon)
        os.makedirs(os.path.join('data', 'top'), exist_ok=True)
        np.save(os.path.join('data','top','lstm_X.npy'), lstm_X)
        np.save(os.path.join('data','top','lstm_y.npy'), lstm_y)
        if lstm_index:
            idx_df = pd.DataFrame(lstm_index, columns=["ticker", "window_end_date"]) 
            idx_df.to_csv(os.path.join('data','top','lstm_index.csv'), index=False)
        print(f"[PIPE DIAG] Wrote LSTM arrays to data/top/lstm_X.npy (shape={lstm_X.shape}) and data/top/lstm_y.npy (shape={lstm_y.shape})")
    except Exception as e:
        print(f"[PIPE DIAG] Failed to build/write LSTM sequences: {e}")

    # Train models from prepared datasets
    tracker = get_tracker(mode=args.tracking, experiment_name=args.experiment, mlflow_uri=args.mlflow_uri, wandb_project=args.experiment, wandb_entity=args.wandb_entity)

    if not args.skip_xgb:
        print("[PIPE] Training XGB model...")
        xgb_model, xgb_preds = XGBTrainer(xgb_dataset_path, tracker=tracker, run_name='xgb_ranker').train()
        print("[PIPE] XGB training done.")
        # Persist XGB model where the UI expects it
        try:
            os.makedirs("models", exist_ok=True)
            joblib.dump(xgb_model, os.path.join("models", "stock_predictor_top.joblib"))
            print("[PIPE DIAG] Saved XGB model to models/stock_predictor_top.joblib")
        except Exception as e:
            print(f"[PIPE DIAG] Failed to save XGB model: {e}")
    print("[PIPE] Training LSTM model...")
    lstm_model, lstm_preds = LSTMTrainer(xgb_dataset_path, tracker=tracker, run_name='lstm').train(
        lookback=args.lookback, horizon=args.horizon
    )
    print("[PIPE] LSTM training done.")

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
            raw_latest_out = os.path.join('data','top','featured_stocks_raw.csv')
            write_latest_snapshot(tech_full_df, features_df, ui_cols, raw_latest_out)
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
        xgb_infer_df = build_xgb_features(infer_prices_df, horizon=args.horizon)
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
        cols = [c for c in ['ticker', 'predicted_return_pct', 'confidence_score', 'risk_score', 'composite_score'] if c in _xgb_out.columns]
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
        write_latest_snapshot(ui_latest, features_df, ui_cols + [c for c in ui_out.columns if c not in ui_cols], ui_out_path)
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

            # Persist LSTM predictions to data/top for step-2 artifact
            try:
                os.makedirs(os.path.join('data', 'top'), exist_ok=True)
                lstm_out = os.path.join('data', 'top', 'lstm_weekly_predictions_output.csv')
                df_lstm_latest.rename(columns={'lstm_pred': 'lstm_predicted_return_pct'}).to_csv(lstm_out, index=False)
                print(f"[PIPE DIAG] Wrote LSTM predictions CSV to: {lstm_out} (rows={len(df_lstm_latest)})")
            except Exception as e:
                print(f"[PIPE DIAG] Failed to write data/top/lstm_weekly_predictions.csv: {e}")

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
                # Weight by validation MAE if available (lower MAE -> higher weight)
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
                wx = wl = 0.5
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
                    ens_vals[both.values] = (
                        wx * df_combined.loc[both, 'xgb_pred'].values +
                        wl * df_combined.loc[both, 'lstm_pred'].values
                    )
                df_combined['ensemble_pred'] = ens_vals
                # Print diagnostic Top-N
                top_ens = df_combined.sort_values('ensemble_pred', ascending=False).head(args.top_n)
                print("[PIPE DIAG] Top-N by Ensemble (XGB+LSTM):")
                print(top_ens[['ticker', 'ensemble_pred']].to_string(index=False))

                # If skipping XGB, build ensemble via normalized blend
                if args.skip_xgb:
                    def _minmax(s: pd.Series) -> pd.Series:
                        s = pd.to_numeric(s, errors='coerce')
                        mn, mx = s.min(), s.max()
                        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                            return pd.Series(0.5, index=s.index)
                        return (s - mn) / (mx - mn)
                    alpha = float(os.getenv('ENSEMBLE_ALPHA', '0.6'))
                    rn = _minmax(df_combined['xgb_pred'])
                    ln = _minmax(df_combined['lstm_pred'])
                    df_combined['ensemble_pred'] = alpha * rn + (1 - alpha) * ln

                # Write final ensemble scores (step-3 artifact) to data/top
                try:
                    os.makedirs(os.path.join('data', 'top'), exist_ok=True)
                    # Attach confidence if available from top_stocks
                    final_df = df_combined.copy()
                    if 'confidence_score' in top_stocks.columns:
                        final_df = final_df.merge(top_stocks[['ticker', 'confidence_score']], on='ticker', how='left')
                    final_df = final_df.rename(columns={'lstm_pred': 'lstm_predicted_return_pct', 'ensemble_pred': 'ensemble_score'})
                    final_df[['ticker', 'xgb_pred', 'xgb_predicted_return_pct', 'lstm_predicted_return_pct', 'confidence_score', 'ensemble_score']].to_csv(
                        os.path.join('data', 'top', 'ensemble_scores_output.csv'), index=False
                    )
                    print(f"[PIPE DIAG] Wrote final ensemble scores CSV to: data/top/ensemble_scores_output.csv (rows={len(final_df)})")
                except Exception as e:
                    print(f"[PIPE DIAG] Failed to write data/top/final_ensemble_scores.csv: {e}")
        else:
            print('[PIPE DIAG] Ensemble skipped: no valid LSTM sequences built for latest window.')
    except Exception as e:
        print(f"[PIPE DIAG] Failed to compute ensemble predictions: {e}")


if __name__ == "__main__":
    main()
