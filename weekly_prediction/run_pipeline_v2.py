import argparse
import pandas as pd
import numpy as np
import os
import joblib
from models.xgb_trainer import XGBTrainer
from models.lstm_trainer import LSTMTrainer
from models.ensemble_trainer import EnsembleTrainer
from stock_selector_v2 import StockSelector
from data_loader import load_all_data
from feature_engineering import create_all_features
from utils.data_prep import build_xgb_dataset, build_lstm_sequences, build_xgb_features
from utils.tracking import get_tracker


def main():
    parser = argparse.ArgumentParser(description="Train top model and rank stocks (v2 from base prices)")
    parser.add_argument("--prices", default="data/stock_prices.csv", help="Path to base stock prices CSV (date,ticker,OHLCV)")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon in trading days")
    parser.add_argument("--lookback", type=int, default=30, help="Sequence length for LSTM")
    parser.add_argument("--top_n", type=int, default=20, help="Number of top stocks to show")
    parser.add_argument("--save", default=None, help="Optional path to save ranked CSV")
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
    features_out_path = "data/featured_stocks_top.csv"
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
        xgb_dataset_path = "data/xgb_dataset.csv"
        xgb_df.to_csv(xgb_dataset_path, index=False)
        print(f"[PIPE DIAG] Wrote XGB dataset to: {xgb_dataset_path} (rows={len(xgb_df)})")
    except Exception as e:
        print(f"[PIPE DIAG] Failed to build/write XGB dataset: {e}")
        xgb_dataset_path = args.prices

    try:
        lstm_X, lstm_y, lstm_index = build_lstm_sequences(prices_df, lookback=args.lookback, horizon=args.horizon)
        np.save("data/lstm_X.npy", lstm_X)
        np.save("data/lstm_y.npy", lstm_y)
        if lstm_index:
            idx_df = pd.DataFrame(lstm_index, columns=["ticker", "window_end_date"]) 
            idx_df.to_csv("data/lstm_index.csv", index=False)
        print(f"[PIPE DIAG] Wrote LSTM arrays to data/lstm_X.npy (shape={lstm_X.shape}) and data/lstm_y.npy (shape={lstm_y.shape})")
    except Exception as e:
        print(f"[PIPE DIAG] Failed to build/write LSTM sequences: {e}")

    # Train models from prepared datasets
    tracker = get_tracker(mode=args.tracking, experiment_name=args.experiment, mlflow_uri=args.mlflow_uri, wandb_project=args.experiment, wandb_entity=args.wandb_entity)

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
    lstm_model, lstm_preds = LSTMTrainer(xgb_dataset_path, tracker=tracker, run_name='lstm').train()
    print("[PIPE] LSTM training done.")

    # Rank stocks using XGB model
    try:
        feature_cols = xgb_model.get_booster().feature_names
        print(f"[PIPE DIAG] XGB expects features ({len(feature_cols)}): {feature_cols}")
    except Exception:
        feature_cols = None
        print("[PIPE DIAG] Could not read XGB feature names; falling back to selector inference.")

    # Read back features (for explicit traceability)
    # Use XGB training dataset for inference to ensure feature alignment.
    # Merge in bullish signal flags from engineered features for ranking filters.
    try:
        # Build inference features from prices_df to include the latest rows (targets may be NaN)
        xgb_infer_df = build_xgb_features(prices_df, horizon=args.horizon)
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
        read_df = xgb_infer_df
        # Persist inference features for UI/selector_v2 to ensure model-feature alignment
        try:
            os.makedirs("data", exist_ok=True)
            infer_out_path = os.path.join("data", "xgb_features_latest.csv")
            read_df.to_csv(infer_out_path, index=False)
            print(f"[PIPE DIAG] Saved XGB inference features to: {infer_out_path}")
        except Exception as e:
            print(f"[PIPE DIAG] Failed to save XGB inference features: {e}")
        print(f"[PIPE DIAG] Using XGB inference features (rows={len(read_df)}, cols={len(read_df.columns)})")
    except Exception:
        read_df = features_df
        print("[PIPE DIAG] Fallback to engineered features for ranking; feature mismatch may occur.")

    selector = StockSelector(xgb_model, read_df, feature_cols=feature_cols)
    top_stocks = selector.rank_stocks(top_n=args.top_n)
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

    # --- NEW: Write UI-compatible CSV to keep UI unchanged ---
    try:
        # Build a UI dataframe using XGB inference features merged with engineered signals
        ui_cols = [
            # identity / base
            "ticker", "date", "close", "open", "high", "low", "volume",
            # signals used by UI
            "broke_resistance", "post_earnings_dip_rally", "strong_momentum", "breakout_confirmed",
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

        # Write to legacy path used by UI
        os.makedirs("data", exist_ok=True)
        ui_out_path = os.path.join("data", "featured_stocks_top.csv")
        ui_out.to_csv(ui_out_path, index=False)
        print(f"[PIPE DIAG] Wrote UI features CSV to: {ui_out_path} (rows={len(ui_out)})")
    except Exception as e:
        print(f"[PIPE DIAG] Failed to write UI CSV: {e}")

    # --- Ensemble predictions (XGB + LSTM) diagnostics and optional CSV ---
    try:
        # Latest snapshot per ticker
        latest_all = read_df.sort_values('date').groupby('ticker').tail(1)

        # XGB predictions on latest snapshot
        xgb_feats = feature_cols if feature_cols is not None else xgb_model.get_booster().feature_names
        xgb_feats = [c for c in xgb_feats if c in latest_all.columns]
        X_xgb_latest = latest_all[xgb_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        preds_xgb_latest = xgb_model.predict(X_xgb_latest)
        df_xgb_latest = pd.DataFrame({
            'ticker': latest_all['ticker'].astype(str).values,
            'xgb_pred': preds_xgb_latest
        })

        # LSTM sequences for latest window per ticker
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
        eps = 1e-8
        seq_list, tickers_seq = [], []
        for tkr, grp in read_df.groupby('ticker'):
            g = grp.sort_values('date').tail(lookback)
            if len(g) < lookback:
                continue
            use_cols = [c for c in lstm_feat_candidates if c in g.columns]
            if not use_cols:
                continue
            window = g[use_cols].to_numpy(dtype=float, copy=False)
            mean = window.mean(axis=0)
            std = window.std(axis=0)
            std_safe = np.where(std < eps, 1.0, std)
            window_norm = (window - mean) / std_safe
            seq_list.append(window_norm)
            tickers_seq.append(str(tkr))

        df_ens_out = None
        if seq_list:
            X_lstm_latest = np.asarray(seq_list)
            try:
                preds_lstm_latest = lstm_model.predict(X_lstm_latest).ravel()
            except Exception as e:
                print(f"[PIPE DIAG] LSTM prediction failed on latest sequences: {e}")
                preds_lstm_latest = None
            df_lstm_latest = pd.DataFrame({
                'ticker': tickers_seq,
                'lstm_pred': preds_lstm_latest if preds_lstm_latest is not None else np.nan
            })

            # Align on intersection of tickers and filter non-finite preds
            df_combined = df_xgb_latest.merge(df_lstm_latest, on='ticker', how='inner')
            # Keep rows where at least one model produced a finite prediction
            mask_x = np.isfinite(df_combined['xgb_pred'])
            mask_l = np.isfinite(df_combined['lstm_pred'])
            df_combined = df_combined[mask_x | mask_l]
            if not df_combined.empty:
                # Weight by validation MAE if available (lower MAE -> higher weight)
                xgb_mae = getattr(xgb_model, '_validation_mae', None)
                lstm_mae = getattr(lstm_model, '_validation_mae', None)
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

                # Write ensemble CSV for UI/debug
                try:
                    ui_ens = latest_all.merge(df_combined[['ticker', 'ensemble_pred']], on='ticker', how='inner').copy()
                    ui_ens['predicted_return_pct'] = ui_ens['ensemble_pred']
                    ui_ens['w_xgb'] = float(wx)
                    ui_ens['w_lstm'] = float(wl)
                    os.makedirs('data', exist_ok=True)
                    ens_out_path = os.path.join('data', 'featured_stocks_top_ensemble.csv')
                    ui_ens[['ticker', 'date', 'close', 'predicted_return_pct', 'w_xgb', 'w_lstm']].to_csv(ens_out_path, index=False)
                    print(f"[PIPE DIAG] Wrote ensemble CSV to: {ens_out_path} (rows={len(ui_ens)})")
                except Exception as e:
                    print(f"[PIPE DIAG] Failed to write ensemble CSV: {e}")

                # Optional: write ensemble CSV akin to UI file
                ui_ens = latest_all.merge(df_combined[['ticker', 'ensemble_pred']], on='ticker', how='inner').copy()
                ui_ens['predicted_return_pct'] = ui_ens['ensemble_pred']
                # Add weights used for transparency
                ui_ens['w_xgb'] = float(wx)
                ui_ens['w_lstm'] = float(wl)
                os.makedirs('data', exist_ok=True)
                ens_out_path = os.path.join('data', 'featured_stocks_top_ensemble.csv')
                ui_ens[['ticker', 'date', 'close', 'predicted_return_pct', 'w_xgb', 'w_lstm']].to_csv(ens_out_path, index=False)
                print(f"[PIPE DIAG] Wrote ensemble CSV to: {ens_out_path} (rows={len(ui_ens)})")
        else:
            print('[PIPE DIAG] Ensemble skipped: no valid LSTM sequences built for latest window.')
    except Exception as e:
        print(f"[PIPE DIAG] Failed to compute ensemble predictions: {e}")


if __name__ == "__main__":
    main()
