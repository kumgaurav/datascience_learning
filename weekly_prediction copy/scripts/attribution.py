"""Generate SHAP-based attribution for XGB predictions on today's eval tickers.

Outputs under snapshot/<date>/attr:
 - <date>_feature_importance.csv (global top features)
 - <date>_attribution.csv (per-ticker top features)
"""

import argparse
import os
import glob
from datetime import date
from typing import List

import numpy as np
import pandas as pd
import shap

try:
    import joblib
except Exception:
    joblib = None


def _find_latest_snapshot_file(base_dir: str = "snapshot") -> str:
    subdirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    if not subdirs:
        raise FileNotFoundError("No snapshot directories found")
    latest_dir = sorted(subdirs)[-1]
    fp = os.path.join(latest_dir, "snapshot.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Snapshot CSV not found in {latest_dir}")
    return fp


def _default_prices_path() -> str:
    pref = os.path.join("data", "stock_prices_input.csv")
    if os.path.exists(pref):
        return pref
    alt = os.path.join("data", "stock_prices.csv")
    if os.path.exists(alt):
        return alt
    return pref  # will fail later with clear error


def _build_latest_xgb_features_for_tickers(prices_csv: str, tickers: List[str], eval_date: str) -> pd.DataFrame:
    # Ensure project root on sys.path
    import sys
    from pathlib import Path
    _this = Path(__file__).resolve()
    _root = _this.parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from utils.data_prep import build_xgb_features as _build_xgb_features  # type: ignore

    prices = pd.read_csv(prices_csv)
    prices.columns = [c.lower() for c in prices.columns]
    if "symbol" in prices.columns and "ticker" not in prices.columns:
        prices = prices.rename(columns={"symbol": "ticker"})
    req = {"ticker", "date"}
    if not req.issubset(set(prices.columns)):
        raise ValueError(f"Prices missing columns: {req - set(prices.columns)}")
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    eval_dt = pd.to_datetime(eval_date)
    lookback_days = int(os.getenv("EVAL_LOOKBACK_DAYS", "250"))
    start_date = eval_dt - pd.Timedelta(days=lookback_days)
    joined = prices[prices["ticker"].isin([str(t).upper() for t in tickers])]
    joined = joined[(joined["date"] <= eval_dt) & (joined["date"] >= start_date)]
    xgb_hist = _build_xgb_features(joined)
    latest = xgb_hist.sort_values(["ticker", "date"]).groupby("ticker").tail(1)

    # Add engineered features sometimes expected by model
    def _z_by_date(df_in: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
        if col not in df_in.columns:
            df_in[out_col] = np.nan
            return df_in
        grp = df_in.groupby("date")[col]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        df_in[out_col] = (df_in[col] - mean) / std
        return df_in

    latest = _z_by_date(latest, "volume", "volume_z")
    latest = _z_by_date(latest, "volatility_20d", "volatility_20d_z")
    latest = _z_by_date(latest, "close_std_20d", "close_std_20d_z")
    latest = _z_by_date(latest, "volume_ratio", "volume_ratio_z")
    if "momentum_20d" in latest.columns and "volume_ratio" in latest.columns:
        latest["mom20_x_volratio"] = latest["momentum_20d"] * latest["volume_ratio"]
    else:
        latest["mom20_x_volratio"] = np.nan
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP attribution for today's eval tickers")
    parser.add_argument("--snapshot", default=None, help="Path to snapshot CSV. Defaults to latest snapshot")
    parser.add_argument("--date", dest="eval_date", default=date.today().isoformat(), help="Evaluation date YYYY-MM-DD")
    parser.add_argument("--xgb_model", default=os.getenv("XGB_MODEL_PATH", "models/stock_predictor_top.joblib"), help="Path to XGB model joblib")
    parser.add_argument("--prices", default=_default_prices_path(), help="Path to prices CSV (for feature rebuild)")
    parser.add_argument("--topk", type=int, default=5, help="Top features per ticker")
    args = parser.parse_args()

    snap_csv = args.snapshot or _find_latest_snapshot_file()
    snap_dir = os.path.dirname(snap_csv)
    eval_dir = os.path.join(snap_dir, "eval")
    os.makedirs(os.path.join(snap_dir, "attr"), exist_ok=True)
    attr_dir = os.path.join(snap_dir, "attr")

    # Use eval CSV to source tickers
    eval_csv = os.path.join(eval_dir, f"{args.eval_date}_eval.csv")
    if not os.path.exists(eval_csv):
        raise FileNotFoundError(f"Eval CSV not found: {eval_csv}. Run evaluator first.")
    edf = pd.read_csv(eval_csv)
    tickers = edf.get("ticker", edf.get("symbol")).astype(str).str.upper().tolist()
    tickers = sorted(list(dict.fromkeys(tickers)))

    if joblib is None:
        raise RuntimeError("joblib not available to load XGB model")
    xgb_model = joblib.load(args.xgb_model)

    latest = _build_latest_xgb_features_for_tickers(args.prices, tickers, args.eval_date)
    # Determine feature columns expected by model in both booster and frame
    xgb_feats = None
    try:
        booster = getattr(xgb_model, "get_booster", None)
        if booster is not None:
            xgb_feats = booster().feature_names  # type: ignore
    except Exception:
        xgb_feats = None
    if not xgb_feats:
        # fallback: numeric columns except IDs
        exclude = {"ticker", "date", "target"}
        xgb_feats = [c for c in latest.columns if c not in exclude and pd.api.types.is_numeric_dtype(latest[c])]
    xgb_feats = [c for c in xgb_feats if c in latest.columns]

    X = latest[xgb_feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X)
    # shap_values may be (n_samples, n_features)
    if isinstance(shap_vals, list):  # multiclass
        shap_arr = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
    else:
        shap_arr = np.abs(np.array(shap_vals))

    # Global feature importance
    global_imp = np.mean(shap_arr, axis=0)
    gdf = pd.DataFrame({"feature": xgb_feats, "importance": global_imp}).sort_values("importance", ascending=False)
    gdf.to_csv(os.path.join(attr_dir, f"{args.eval_date}_feature_importance.csv"), index=False)

    # Per-ticker top-k
    rows = []
    for i, tkr in enumerate(latest["ticker"].astype(str)):
        contrib = shap_arr[i]
        order = np.argsort(-np.abs(contrib))[: args.topk]
        for r, j in enumerate(order, start=1):
            rows.append({
                "ticker": tkr,
                "rank": r,
                "feature": xgb_feats[j],
                "shap_abs": float(abs(contrib[j]))
            })
    pd.DataFrame(rows).to_csv(os.path.join(attr_dir, f"{args.eval_date}_attribution.csv"), index=False)


if __name__ == "__main__":
    main()
