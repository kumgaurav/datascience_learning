"""Daily evaluator.

Computes daily actual returns for the snapshot tickers and evaluates prediction quality.
Assumes snapshot predictions are return percentages (not absolute prices).
Outputs an eval CSV in snapshot/<date>/eval/<eval_date>_eval.csv
"""

import argparse
import glob
import os
from datetime import date, datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
try:
    import joblib
except Exception:
    joblib = None

try:
    # Utilities to generate LSTM sequences consistent with training
    from utils.generate_lstm_predictions import load_lstm as _load_lstm_model, build_sequences_for_latest as _build_lstm_sequences
except Exception:
    _load_lstm_model = None
    _build_lstm_sequences = None

# New: feature computation per user request
try:
    import feature_engineering as fe
except Exception:
    fe = None


def _refresh_prices_from_db(config_path: str | None = None) -> str:
    """Export latest prices from DB using the mapping stocksinfp -> stock_prices.

    Mirrors stocks/execute_data_creator.py logic for the prices table only.
    Returns the path to the generated CSV.
    """
    try:
        # Import here to avoid hard dependency unless requested
        from pathlib import Path
        import sys
        # Ensure project root is on sys.path (same pattern as execute_data_creator.py)
        _this_file = Path(__file__).resolve()
        _project_root = _this_file.parents[1]
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))

        from stocks.stock_data_creator import StockDataCreator

        creator = StockDataCreator(
            config_path=Path(config_path) if config_path else None,
            data_dir=Path("data"),
        )
        outputs = creator.export_tables([
            "stocksinfp",
        ], name_mapping={"stocksinfp": "stock_prices"})
        out_path = outputs.get("stocksinfp")
        if out_path is None:
            raise RuntimeError("Export returned no path for stocksinfp")
        return str(out_path)
    except Exception as e:
        # Surface a clear hint but allow fallback to existing CSVs
        print(f"[DAILY_EVAL] DB refresh failed: {e}. Falling back to existing CSVs if present.")
        return ""


def _find_latest_snapshot_file(base_dir: str = "snapshot") -> str:
    subdirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    if not subdirs:
        raise FileNotFoundError("No snapshot directories found")
    subdirs_sorted = sorted(subdirs)
    latest_dir = subdirs_sorted[-1]
    fp = os.path.join(latest_dir, "snapshot.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Snapshot CSV not found in {latest_dir}")
    return fp


def _find_prices_csv(default_path: str = "data/stock_prices.csv") -> str:
    """Resolve prices CSV path.

    Always prefer data/stock_prices.csv unless an explicit --prices is provided.
    As a fallback, use the most recent data/stock_prices_*.csv.
    """
    if os.path.exists(default_path):
        return default_path
    # Fallback: pick the newest stock_prices_*.csv
    cands = sorted(glob.glob("data/stock_prices_*.csv"))
    if not cands:
        raise FileNotFoundError("No prices CSV found. Expected data/stock_prices.csv or data/stock_prices_*.csv")
    return cands[-1]


def _prepare_prices_for_eval(prices_df: pd.DataFrame, tickers: list[str], eval_date: str) -> pd.DataFrame:
    df = prices_df.copy()
    # Normalize columns
    cols = {c: c.lower() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    if "ticker" not in df.columns:
        if "symbol" in df.columns:
            df.rename(columns={"symbol": "ticker"}, inplace=True)
        else:
            raise ValueError("Prices CSV must have 'ticker' or 'symbol' column")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    if "date" not in df.columns:
        raise ValueError("Prices CSV must have 'date' column")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Choose a price column for close
    price_col = "close" if "close" in df.columns else ("adj close" if "adj close" in df.columns else None)
    if price_col is None:
        raise ValueError("Prices CSV must include 'close' or 'adj close' column")
    # Optional OHLCV columns
    open_col = "open" if "open" in df.columns else None
    high_col = "high" if "high" in df.columns else None
    low_col = "low" if "low" in df.columns else None
    vol_col = "volume" if "volume" in df.columns else None

    # Restrict to tickers of interest
    df = df[df["ticker"].isin([str(t).upper() for t in tickers])]
    if df.empty:
        return pd.DataFrame(columns=["ticker", "actual_price", "prev_close", "actual_return_pct"])  # empty

    # Parse eval date and compute for each ticker the last close <= eval_date and the prior close
    eval_dt = pd.to_datetime(eval_date)
    df = df[df["date"] <= eval_dt]
    # Latest per ticker as of eval_dt
    latest = df.sort_values(["ticker", "date"]).groupby("ticker").tail(1)
    # Prior close: last date < latest.date per ticker
    merged = df.merge(latest[["ticker", "date"]].rename(columns={"date": "latest_date"}), on="ticker", how="inner")
    prior = merged[merged["date"] < merged["latest_date"]].sort_values(["ticker", "date"]).groupby("ticker").tail(1)

    # Build output with returns and carry dates used (plus latest OHLCV if available)
    latest_price = latest.set_index("ticker")[price_col].rename("actual_price")
    latest_date = latest.set_index("ticker")["date"].rename("latest_date")
    prior_price = prior.set_index("ticker")[price_col].rename("prev_close")
    prev_date = prior.set_index("ticker")["date"].rename("prev_date")
    pieces = [latest_price, latest_date, prior_price, prev_date]
    if open_col:
        pieces.append(latest.set_index("ticker")[open_col].rename("latest_open"))
    if high_col:
        pieces.append(latest.set_index("ticker")[high_col].rename("latest_high"))
    if low_col:
        pieces.append(latest.set_index("ticker")[low_col].rename("latest_low"))
    if vol_col:
        pieces.append(latest.set_index("ticker")[vol_col].rename("latest_volume"))
    out = pd.concat(pieces, axis=1)
    out.index.name = "ticker"
    out = out.reset_index()
    out["actual_return_pct"] = (out["actual_price"] / out["prev_close"] - 1.0) * 100.0
    return out.dropna(subset=["prev_close"])  # drop where no prior


def _precision_at_k_cross_section(df: pd.DataFrame, pred_col: str, k: int = 20) -> float:
    if pred_col not in df.columns or "actual_return_pct" not in df.columns:
        return 0.0
    # Coerce to numeric and drop NaN/Inf rows for fair comparison
    numeric_pred = pd.to_numeric(df[pred_col], errors="coerce")
    numeric_actual = pd.to_numeric(df["actual_return_pct"], errors="coerce")
    mask = (
        numeric_pred.notna()
        & numeric_actual.notna()
        & ~np.isinf(numeric_pred)
        & ~np.isinf(numeric_actual)
    )
    cleaned = df.loc[mask].copy()
    if cleaned.empty:
        return 0.0
    k = min(k, len(cleaned))
    if k <= 0:
        return 0.0
    ranked = cleaned.assign(**{pred_col: numeric_pred[mask]}).sort_values(pred_col, ascending=False).head(k)
    hits = (pd.to_numeric(ranked["actual_return_pct"], errors="coerce") > 0).sum()
    return float(hits) / float(k)


def _load_symbol_aliases(path: Optional[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return mapping
    try:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        # try common column names
        ui_col = cols.get('ui') or cols.get('ui_ticker') or cols.get('ticker')
        price_col = cols.get('price') or cols.get('price_ticker') or cols.get('alias')
        if ui_col and price_col:
            for _, r in df.iterrows():
                ui = str(r.get(ui_col, '')).upper()
                pr = str(r.get(price_col, '')).upper()
                if ui and pr:
                    mapping[ui] = pr
    except Exception:
        return {}
    return mapping


def evaluate_daily(
    snapshot_path: str,
    prices_df: pd.DataFrame,
    eval_date: str | None = None,
    features_backfill_df: Optional[pd.DataFrame] = None,
    symbol_aliases_csv: Optional[str] = None,
) -> pd.DataFrame:
    if eval_date is None:
        eval_date = date.today().isoformat()
    
    snapshot = pd.read_csv(snapshot_path)
    # Ensure key exists
    if "ticker" not in snapshot.columns:
        if "symbol" in snapshot.columns:
            snapshot["ticker"] = snapshot["symbol"].astype(str)
        else:
            raise ValueError("Snapshot must contain 'ticker' or 'symbol' column")
    snapshot["ticker"] = snapshot["ticker"].astype(str).str.upper()
    # Apply symbol alias mapping if provided (map UI tickers to price tickers)
    alias_map = _load_symbol_aliases(symbol_aliases_csv)
    if alias_map:
        snapshot["price_ticker"] = snapshot["ticker"].map(lambda t: alias_map.get(t, t))
        tickers_for_prices = snapshot["price_ticker"].tolist()
    else:
        tickers_for_prices = snapshot["ticker"].tolist()

    # Build actuals (close and prev_close) per snapshot ticker
    actuals = _prepare_prices_for_eval(prices_df, tickers_for_prices, eval_date)
    # If alias mapping used, merge on price_ticker then restore UI ticker
    if alias_map and not actuals.empty:
        actuals = actuals.rename(columns={"ticker": "price_ticker"})
        eval_df = snapshot.merge(actuals, on="price_ticker", how="left")
        # Fill missing by trying direct ticker join as fallback
        missing_mask = eval_df["actual_price"].isna()
        if missing_mask.any():
            direct = snapshot[missing_mask].merge(
                _prepare_prices_for_eval(prices_df, snapshot.loc[missing_mask, "ticker"].tolist(), eval_date),
                on="ticker",
                how="left",
            )
            # Columns to fill if present
            fill_cols = [c for c in ["actual_price", "prev_close", "actual_return_pct", "latest_date", "prev_date"] if c in direct.columns]
            eval_df.loc[missing_mask, fill_cols] = direct[fill_cols].values
        eval_df["ticker"] = eval_df["ticker"].astype(str).str.upper()
    else:
        eval_df = snapshot.merge(actuals, on="ticker", how="left")

    # Compute prediction errors in return percentage space
    if "pred_xgb" in eval_df.columns:
        eval_df["error_xgb_pct"] = eval_df["actual_return_pct"] - pd.to_numeric(eval_df["pred_xgb"], errors="coerce")
    if "pred_lstm" in eval_df.columns:
        eval_df["error_lstm_pct"] = eval_df["actual_return_pct"] - pd.to_numeric(eval_df["pred_lstm"], errors="coerce")

    # Metrics (returns-based MAE)
    mae_xgb = None
    mae_lstm = None
    # Clean actuals: coerce to numeric and drop infs
    eval_df["actual_return_pct"] = pd.to_numeric(eval_df["actual_return_pct"], errors="coerce")
    eval_df["actual_return_pct"] = eval_df["actual_return_pct"].replace([np.inf, -np.inf], np.nan)
    if "pred_xgb" in eval_df.columns:
        pred_xgb_num = pd.to_numeric(eval_df["pred_xgb"], errors="coerce")
        pred_xgb_num = pred_xgb_num.replace([np.inf, -np.inf], np.nan)
        mask_xgb = eval_df["actual_return_pct"].notna() & pred_xgb_num.notna()
        if mask_xgb.any():
            mae_xgb = mean_absolute_error(eval_df.loc[mask_xgb, "actual_return_pct"], pred_xgb_num.loc[mask_xgb])  # type: ignore[arg-type]
    if "pred_lstm" in eval_df.columns:
        pred_lstm_num = pd.to_numeric(eval_df["pred_lstm"], errors="coerce")
        pred_lstm_num = pred_lstm_num.replace([np.inf, -np.inf], np.nan)
        mask_lstm = eval_df["actual_return_pct"].notna() & pred_lstm_num.notna()
        if mask_lstm.any():
            mae_lstm = mean_absolute_error(eval_df.loc[mask_lstm, "actual_return_pct"], pred_lstm_num.loc[mask_lstm])  # type: ignore[arg-type]

    # Directional accuracy using returns
    acc_xgb = None
    if "pred_xgb" in eval_df.columns:
        pred_xgb_num = pd.to_numeric(eval_df["pred_xgb"], errors="coerce")
        pred_xgb_num = pred_xgb_num.replace([np.inf, -np.inf], np.nan)
        mask_xgb = eval_df["actual_return_pct"].notna() & pred_xgb_num.notna()
        if mask_xgb.any():
            acc_xgb = accuracy_score((eval_df.loc[mask_xgb, "actual_return_pct"] > 0), (pred_xgb_num.loc[mask_xgb] > 0))

    # Precision@K on cross-section
    p20_xgb = _precision_at_k_cross_section(eval_df, "pred_xgb", k=min(20, len(eval_df))) if "pred_xgb" in eval_df.columns else 0.0
    p20_lstm = _precision_at_k_cross_section(eval_df, "pred_lstm", k=min(20, len(eval_df))) if "pred_lstm" in eval_df.columns else 0.0

    # Console output
    parts = [f"📊 {eval_date}"]
    if mae_xgb is not None:
        parts.append(f"XGB MAE%: {mae_xgb:.2f}")
    if mae_lstm is not None:
        parts.append(f"LSTM MAE%: {mae_lstm:.2f}")
    if acc_xgb is not None:
        parts.append(f"XGB DirAcc: {acc_xgb:.2%}")
    parts.append(f"XGB P@20: {p20_xgb:.2%}")
    parts.append(f"LSTM P@20: {p20_lstm:.2%}")
    print(" - ".join(parts))

    # Save eval CSV with metrics and compatibility columns
    eval_dir = os.path.join(os.path.dirname(snapshot_path), "eval")
    os.makedirs(eval_dir, exist_ok=True)
    # For downstream compatibility
    eval_df["symbol"] = eval_df.get("symbol", eval_df["ticker"])  # ensure symbol exists
    # Make the eval 'date' reflect the latest pricing date used (fallback to eval_date)
    if "latest_date" in eval_df.columns:
        # Convert to string ISO date for CSV consistency
        latest_date_str = pd.to_datetime(eval_df["latest_date"], errors="coerce").dt.date.astype(str)
        eval_df["date"] = latest_date_str.fillna(eval_date)
    else:
        eval_df["date"] = eval_date
    # Overwrite OHLCV columns with latest OHLCV from prices if available to avoid confusion with snapshot features
    for base_col, latest_col in [("open", "latest_open"), ("high", "latest_high"), ("low", "latest_low"), ("volume", "latest_volume")]:
        if latest_col in eval_df.columns:
            eval_df[base_col] = eval_df[latest_col]
    # Close should reflect the latest close; use actual_price
    if "actual_price" in eval_df.columns:
        eval_df["close"] = eval_df["actual_price"]
    eval_df["actual_price"] = eval_df["actual_price"]  # alias for weekly_report
    eval_df["p20_xgb"] = p20_xgb
    eval_df["p20_lstm"] = p20_lstm
    out_path = os.path.join(eval_dir, f"{eval_date}_eval.csv")
    eval_df.to_csv(out_path, index=False)
    return eval_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily evaluator: build eval input, optionally predict with XGB/LSTM, and write eval CSV")
    parser.add_argument("--snapshot", default=None, help="Path to snapshot CSV. Defaults to latest snapshot")
    parser.add_argument("--prices", default=None, help="Path to prices CSV. Defaults to data/stock_prices.csv or latest stock_prices_*.csv")
    parser.add_argument("--date", dest="eval_date", default=date.today().isoformat(), help="Evaluation date YYYY-MM-DD (defaults to today)")
    parser.add_argument("--refresh_prices", action="store_true", help="Refresh prices from DB before evaluation")
    parser.add_argument("--db_config", default=os.getenv("DB_CONFIG_PATH", ""), help="Optional path to DB config.ini for refresh")
    parser.add_argument("--predict_xgb", action="store_true", help="Generate XGB predictions on latest features")
    parser.add_argument("--predict_lstm", action="store_true", help="Generate LSTM predictions on latest sequences")
    parser.add_argument("--xgb_model", default=os.getenv("XGB_MODEL_PATH", "models/stock_predictor_top.joblib"), help="Path to XGB model .joblib")
    parser.add_argument("--lstm_model", default=os.getenv("LSTM_MODEL_PATH", "models/lib/keras/lstm_model.keras"), help="Path to Keras LSTM model")
    parser.add_argument("--lstm_meta", default=os.getenv("LSTM_META_PATH", "models/top/json/lstm_model_meta.json"), help="Path to LSTM meta JSON")
    args = parser.parse_args()
    # Run both predictors by default; allow disabling via env toggles
    try:
        args.predict_xgb = os.getenv("EVAL_DISABLE_XGB", "0") != "1"
    except Exception:
        args.predict_xgb = True
    try:
        args.predict_lstm = os.getenv("EVAL_DISABLE_LSTM", "0") != "1"
    except Exception:
        args.predict_lstm = True

    # 1) Snapshot: default to latest if not provided
    snapshot_path = args.snapshot or _find_latest_snapshot_file()
    snapshot_dir = os.path.dirname(snapshot_path)

    # 2) Prices: optionally refresh, else use provided or resolved default
    prices_path = args.prices
    if args.refresh_prices:
        refreshed = _refresh_prices_from_db(args.db_config or None)
        # After refresh, prefer an explicitly provided --prices if given,
        # otherwise prefer the unfiltered input if it exists, else use refreshed path
        pref_input = os.path.join("data", "stock_prices_input.csv")
        if prices_path and os.path.exists(prices_path):
            pass  # keep user-provided
        elif os.path.exists(pref_input):
            prices_path = pref_input
        elif refreshed:
            prices_path = refreshed
    prices_path = prices_path or _find_prices_csv(os.getenv("STOCK_PRICES_CSV", "data/stock_prices.csv"))
    print(f"[DAILY_EVAL] Using snapshot: {snapshot_path}")
    print(f"[DAILY_EVAL] Using prices CSV: {prices_path}")

    # 3) Eval date
    eval_date = pd.to_datetime(args.eval_date)

    # Ensure project root on sys.path to import feature_engineering
    try:
        import sys
        from pathlib import Path
        _this_file = Path(__file__).resolve()
        _project_root = _this_file.parents[1]
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))
        # Import after path fix
        import feature_engineering as fe  # type: ignore
    except Exception as e:
        print(f"[DAILY_EVAL] Failed to import feature_engineering: {e}")
        fe = None  # type: ignore
    # Try import XGB feature builder
    try:
        from utils.data_prep import build_xgb_features as _build_xgb_features  # type: ignore
    except Exception as e:
        try:
            from snapshot_utils.data_prep import build_xgb_features as _build_xgb_features  # type: ignore
        except Exception as e2:
            print(f"[DAILY_EVAL] XGB feature builder unavailable: {e}; fallback failed: {e2}. Will skip model-based XGB features.")
            _build_xgb_features = None  # type: ignore

    # 4) Build snapshot tickers-only frame
    snap = pd.read_csv(snapshot_path)
    if "ticker" not in snap.columns:
        if "symbol" in snap.columns:
            snap["ticker"] = snap["symbol"]
        else:
            raise ValueError("Snapshot must contain 'ticker' or 'symbol' column")
    snap["ticker"] = snap["ticker"].astype(str).str.upper()
    tickers = sorted(snap["ticker"].dropna().unique().tolist())
    snap_symbols = pd.DataFrame({"ticker": tickers})

    # 5) Join with prices on ticker, filter date >= eval_date
    prices = pd.read_csv(prices_path)
    prices.columns = [c.lower() for c in prices.columns]
    if "symbol" in prices.columns and "ticker" not in prices.columns:
        prices = prices.rename(columns={"symbol": "ticker"})
    if "ticker" not in prices.columns or "date" not in prices.columns:
        raise ValueError("Prices CSV must have 'ticker' (or 'symbol') and 'date' columns")
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    joined = prices.merge(snap_symbols, on="ticker", how="inner")
    # Use sufficient lookback history up to eval_date so rolling features are computable
    lookback_days = int(os.getenv("EVAL_LOOKBACK_DAYS", "250"))
    start_date = eval_date - pd.Timedelta(days=lookback_days)
    joined = joined[(joined["date"] <= eval_date) & (joined["date"] >= start_date)]

    # 6) Compute features using available feature engineering implementation
    try:
        from utils.feature_engineering import _compute_features as _compute_features_fe  # type: ignore
    except Exception:
        _compute_features_fe = None  # type: ignore
    if fe is not None and hasattr(fe, "_calculate_momentum"):
        features_hist = fe._calculate_momentum(joined)
    elif _compute_features_fe is not None:
        features_hist = _compute_features_fe(joined)
    else:
        raise RuntimeError("No feature engineering available. Install or expose utils.feature_engineering._compute_features")
    # Keep only the latest row per ticker as of eval_date
    features_df = features_hist.sort_values(["ticker", "date"]).groupby("ticker").tail(1).reset_index(drop=True)
    # Ensure ALL snapshot tickers are present in eval_input even if prices missing (include placeholders)
    present = set(features_df["ticker"].astype(str))
    missing_tickers = [t for t in tickers if t not in present]
    if missing_tickers:
        # Create placeholder rows with just ticker and date; other columns will be NaN
        placeholder = pd.DataFrame({
            "ticker": missing_tickers,
            "date": [eval_date] * len(missing_tickers),
        })
        # Align columns
        for c in features_df.columns:
            if c not in placeholder.columns:
                placeholder[c] = np.nan
        # Order columns like features_df
        placeholder = placeholder[features_df.columns]
        features_df = pd.concat([features_df, placeholder], ignore_index=True)
        # Keep only one row per ticker
        features_df = features_df.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker"], keep="last")

    # Enrich eval_input OHLCV using latest prices as of eval_date (no prior needed)
    try:
        latest_ohlcv = (
            prices[(prices["ticker"].isin(tickers)) & (prices["date"] <= eval_date)]
            .sort_values(["ticker", "date"]).groupby("ticker").tail(1)
            [["ticker", "open", "high", "low", "close", "volume", "date"]]
            .rename(columns={"date": "latest_date"})
        )
        features_df = features_df.merge(latest_ohlcv, on="ticker", how="left", suffixes=("", "_px"))
        for col in ["open", "high", "low", "close", "volume"]:
            if col in features_df.columns and (col + "_px") in features_df.columns:
                features_df[col] = features_df[col].where(features_df[col].notna(), features_df[col + "_px"])
        drop_cols = [c for c in ["open_px", "high_px", "low_px", "close_px", "volume_px"] if c in features_df.columns]
        if drop_cols:
            features_df.drop(columns=drop_cols, inplace=True)
    except Exception:
        pass

    # 7) Save as data_eval_input.csv in snapshot directory
    # Write under eval/ subdirectory of the snapshot
    eval_dir = os.path.join(snapshot_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    out_name = f"{eval_date.date().isoformat()}_eval_input.csv"
    out_csv = os.path.join(eval_dir, out_name)
    features_df.to_csv(out_csv, index=False)
    print(f"[DAILY_EVAL] Wrote features input: {out_csv} (rows={len(features_df)})")

    # --- Optional: Generate predictions (XGB / LSTM) on latest features ---
    pred_cols = []
    preds_df = pd.DataFrame({"ticker": features_df["ticker"].astype(str)})
    if args.predict_xgb and joblib is not None:
        try:
            xgb_model = joblib.load(args.xgb_model)
            # Build XGB-style features from the same joined price history to match training schema
            if _build_xgb_features is None:
                raise RuntimeError("XGB feature builder unavailable (utils.data_prep.build_xgb_features)")
            xgb_hist = _build_xgb_features(joined)
            # Add engineered features expected by the trained model if missing
            try:
                # Per-date cross-sectional z-scores helper
                def _z_by_date(df_in: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
                    if col not in df_in.columns:
                        df_in[out_col] = np.nan
                        return df_in
                    grp = df_in.groupby("date")[col]
                    mean = grp.transform("mean")
                    std = grp.transform("std").replace(0, np.nan)
                    df_in[out_col] = (df_in[col] - mean) / std
                    return df_in

                # volume_z (per-date z of raw volume)
                xgb_hist = _z_by_date(xgb_hist, "volume", "volume_z")
                # volatility_20d_z (per-date z of volatility_20d)
                xgb_hist = _z_by_date(xgb_hist, "volatility_20d", "volatility_20d_z")
                # close_std_20d_z (per-date z of close_std_20d)
                xgb_hist = _z_by_date(xgb_hist, "close_std_20d", "close_std_20d_z")
                # volume_ratio_z (per-date z of volume_ratio)
                xgb_hist = _z_by_date(xgb_hist, "volume_ratio", "volume_ratio_z")
                # mom20_x_volratio interaction
                if "momentum_20d" in xgb_hist.columns and "volume_ratio" in xgb_hist.columns:
                    xgb_hist["mom20_x_volratio"] = xgb_hist["momentum_20d"] * xgb_hist["volume_ratio"]
                else:
                    xgb_hist["mom20_x_volratio"] = np.nan
            except Exception as _:
                pass
            latest_xgb = xgb_hist.sort_values(["ticker", "date"]).groupby("ticker").tail(1)
            # Determine feature columns expected by the model and present in latest_xgb
            xgb_feats = None
            try:
                booster = getattr(xgb_model, "get_booster", None)
                if booster is not None:
                    xgb_feats = booster().feature_names  # type: ignore
            except Exception:
                xgb_feats = None
            if not xgb_feats:
                # Fallback: all numeric cols except identifiers
                exclude = {"ticker", "date", "target"}
                xgb_feats = [c for c in latest_xgb.columns if c not in exclude and pd.api.types.is_numeric_dtype(latest_xgb[c])]
            # Build a unified frame that includes both xgb and engineered features to satisfy model expectations
            # Use engineered features_df already computed for the same eval_date (latest per ticker)
            try:
                eng_cols = [c for c in xgb_feats if c in features_df.columns]
                base = latest_xgb.merge(features_df[["ticker", *eng_cols]], on="ticker", how="left", suffixes=("", "_eng"))
            except Exception:
                base = latest_xgb.copy()
            # For any expected feature missing, add with 0.0; if both *_eng and base exist, prefer base then fill from *_eng
            for feat in xgb_feats:
                if feat not in base.columns:
                    alt = f"{feat}_eng"
                    if alt in base.columns:
                        base[feat] = base[alt]
                    else:
                        base[feat] = 0.0
            # Drop any helper *_eng columns
            drop_eng = [c for c in base.columns if c.endswith("_eng")]
            if drop_eng:
                base = base.drop(columns=drop_eng)
            # Ensure final matrix matches booster feature order
            X_latest = base[xgb_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
            preds = xgb_model.predict(X_latest)
            preds_df["pred_xgb"] = preds
            pred_cols.append("pred_xgb")
        except Exception as e:
            print(f"[DAILY_EVAL] XGB prediction skipped: {e}")

    if args.predict_lstm and (_load_lstm_model is not None) and (_build_lstm_sequences is not None):
        try:
            lstm_model, lstm_feature_cols, lstm_lookback = _load_lstm_model(args.lstm_model, args.lstm_meta)
            # Ensure required LSTM feature columns present
            missing = [c for c in lstm_feature_cols if c not in features_hist.columns]
            if missing:
                raise RuntimeError(f"Missing LSTM feature columns: {missing[:5]}...")
            seq_map = _build_lstm_sequences(features_hist, lstm_feature_cols, lstm_lookback)
            lstm_preds = {}
            for tkr, X_seq in seq_map.items():
                try:
                    y = float(lstm_model.predict(X_seq, verbose=0).ravel()[0])
                    lstm_preds[tkr] = y
                except Exception:
                    continue
            preds_df["pred_lstm"] = preds_df["ticker"].map(lstm_preds)
            pred_cols.append("pred_lstm")
        except Exception as e:
            print(f"[DAILY_EVAL] LSTM prediction skipped: {e}")

    # --- Build final eval CSV using the input features we just created ---
    # Merge predictions from snapshot onto features
    snap_pred_cols = [c for c in ["pred_xgb", "pred_lstm", "confidence_score", "symbol"] if c in snap.columns]
    snap_preds = snap[["ticker", *snap_pred_cols]].drop_duplicates("ticker") if snap_pred_cols else snap[["ticker"]]
    base = features_df.merge(snap_preds, on="ticker", how="left")
    # Overlay model predictions if generated
    if not preds_df.empty and pred_cols:
        base = base.merge(preds_df[["ticker", *pred_cols]], on="ticker", how="left", suffixes=("", "_model"))
        # Prefer model predictions over snapshot when available without triggering FutureWarning
        for c in ["pred_xgb", "pred_lstm"]:
            cm = c + "_model"
            if cm in base.columns:
                if c in base.columns:
                    base[c] = base[c].where(base[cm].isna(), base[cm])
                else:
                    base[c] = base[cm]
                base.drop(columns=[cm], inplace=True)

    # Compute actuals from prices as of eval_date
    actuals = _prepare_prices_for_eval(prices, base["ticker"].tolist(), eval_date.date().isoformat())
    eval_df = base.merge(actuals, on="ticker", how="left")

    # Set eval date column to latest pricing date
    if "latest_date" in eval_df.columns:
        eval_df["date"] = pd.to_datetime(eval_df["latest_date"], errors="coerce").dt.date.astype(str)
    else:
        eval_df["date"] = eval_date.date().isoformat()

    # Overwrite OHLCV to reflect latest prices (close from actual_price)
    for base_col, latest_col in [("open", "latest_open"), ("high", "latest_high"), ("low", "latest_low"), ("volume", "latest_volume")]:
        if latest_col in eval_df.columns:
            eval_df[base_col] = eval_df[latest_col]
    if "actual_price" in eval_df.columns:
        eval_df["close"] = eval_df["actual_price"]

    # Errors and metrics
    mae_xgb = mae_lstm = None
    acc_xgb = None
    if "pred_xgb" in eval_df.columns:
        px = pd.to_numeric(eval_df["pred_xgb"], errors="coerce")
        ar = pd.to_numeric(eval_df["actual_return_pct"], errors="coerce")
        m = ar.notna() & px.notna()
        if m.any():
            mae_xgb = mean_absolute_error(ar[m], px[m])
            acc_xgb = accuracy_score((ar[m] > 0), (px[m] > 0))
    if "pred_lstm" in eval_df.columns:
        pl = pd.to_numeric(eval_df["pred_lstm"], errors="coerce")
        ar = pd.to_numeric(eval_df["actual_return_pct"], errors="coerce")
        m = ar.notna() & pl.notna()
        if m.any():
            mae_lstm = mean_absolute_error(ar[m], pl[m])

    p20_xgb = _precision_at_k_cross_section(eval_df, "pred_xgb", k=min(20, len(eval_df))) if "pred_xgb" in eval_df.columns else 0.0
    p20_lstm = _precision_at_k_cross_section(eval_df, "pred_lstm", k=min(20, len(eval_df))) if "pred_lstm" in eval_df.columns else 0.0
    eval_df["p20_xgb"] = p20_xgb
    eval_df["p20_lstm"] = p20_lstm

    # Save final eval CSV
    out_eval = os.path.join(eval_dir, f"{eval_date.date().isoformat()}_eval.csv")
    eval_df.to_csv(out_eval, index=False)
    parts = [f"📊 {eval_date.date().isoformat()} (from eval_input)"]
    if mae_xgb is not None:
        parts.append(f"XGB MAE%: {mae_xgb:.2f}")
    if mae_lstm is not None:
        parts.append(f"LSTM MAE%: {mae_lstm:.2f}")
    if acc_xgb is not None:
        parts.append(f"XGB DirAcc: {acc_xgb:.2%}")
    parts.append(f"XGB P@20: {p20_xgb:.2%}")
    parts.append(f"LSTM P@20: {p20_lstm:.2%}")
    print(" - ".join(parts))


if __name__ == "__main__":
    main()
