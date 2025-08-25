"""Generate SHAP-based attribution for XGB predictions on today's eval tickers.

Outputs under snapshot/<date>/attr:
 - <date>_feature_importance.csv (global top features)
 - <date>_attribution.csv (per-ticker top features)
"""

import argparse
import os
import sys
import glob
import logging
from datetime import date
from typing import List

import numpy as np
import pandas as pd

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
    parser.add_argument("--config", default=os.getenv("APP_CONFIG", os.path.join(os.path.dirname(__file__), "snapshot_config.yaml")), help="Path to YAML/JSON config for paths and logging")
    parser.add_argument("--snapshot", default=None, help="Path to snapshot CSV. Defaults to latest snapshot")
    parser.add_argument("--date", dest="eval_date", default=date.today().isoformat(), help="Evaluation date YYYY-MM-DD")
    parser.add_argument("--xgb_model", default=None, help="Path to XGB model joblib")
    parser.add_argument("--prices", default=None, help="Path to prices CSV (for feature rebuild)")
    parser.add_argument("--topk", type=int, default=5, help="Top features per ticker")
    args = parser.parse_args()

    # Ensure project root and load config/logging
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from utils.config import load_config, apply_env_from_config  # type: ignore
    from utils.logging_utils import configure_logging  # type: ignore
    cfg = load_config(args.config)
    apply_env_from_config(cfg)
    configure_logging(level=os.getenv('LOG_LEVEL'), log_file=os.getenv('LOG_FILE'))
    logger = logging.getLogger(__name__)

    # Optional SHAP import; skip gracefully if not installed
    try:
        import shap  # type: ignore
    except Exception as e:
        logger.warning("SHAP not installed (%s). Skipping attribution.", e)
        return

    snap_root = os.getenv('SNAPSHOT_ROOT_DIR', 'snapshot')
    snap_csv = args.snapshot or _find_latest_snapshot_file(snap_root)
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
    # Resolve model path: CLI > YAML xgboost.model_path > env XGB_MODEL_PATH
    xgb_model_path = args.xgb_model
    if not xgb_model_path:
        try:
            xgb_model_path = (cfg.get('xgboost') or {}).get('model_path') if isinstance(cfg, dict) else None
        except Exception:
            xgb_model_path = None
    if not xgb_model_path:
        xgb_model_path = os.getenv('XGB_MODEL_PATH')
    if not xgb_model_path or not os.path.exists(xgb_model_path):
        raise FileNotFoundError(f"XGB model not found. Provide --xgb_model or set xgboost.model_path in {args.config} or XGB_MODEL_PATH env.")
    xgb_model = joblib.load(xgb_model_path)

    # Resolve prices: CLI > YAML paths.prices_csv/clean_prices > env STOCK_PRICES_CSV
    prices_path = args.prices
    if not prices_path:
        try:
            paths = cfg.get('paths') if isinstance(cfg, dict) else {}
            if isinstance(paths, dict):
                prices_path = paths.get('prices_csv') or paths.get('clean_prices') or paths.get('unclean_prices')
        except Exception:
            prices_path = None
    if not prices_path:
        prices_path = os.getenv('STOCK_PRICES_CSV') or _default_prices_path()
    if not os.path.exists(prices_path):
        raise FileNotFoundError(f"Prices CSV not found: {prices_path}")
    # Ensure builder import path works whether data_prep is under utils/ or snapshot_utils/
    try:
        latest = _build_latest_xgb_features_for_tickers(prices_path, tickers, args.eval_date)
    except Exception:
        # Rebind builder to use fallback import path
        def _build_latest_xgb_features_for_tickers(prices_csv: str, tickers: List[str], eval_date: str) -> pd.DataFrame:
            # Ensure project root on sys.path
            from pathlib import Path
            _this = Path(__file__).resolve()
            _root = _this.parents[1]
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from snapshot_utils.data_prep import build_xgb_features as _build_xgb_features  # type: ignore

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
            return latest

        latest = _build_latest_xgb_features_for_tickers(prices_path, tickers, args.eval_date)
    # Optionally enrich with engineered eval features if available
    try:
        eval_input_csv = os.path.join(eval_dir, f"{args.eval_date}_eval_input.csv")
        if os.path.exists(eval_input_csv):
            eng = pd.read_csv(eval_input_csv)
            eng["ticker"] = eng.get("ticker", eng.get("symbol")).astype(str).str.upper()
            num_cols = [c for c in eng.columns if c != "ticker" and pd.api.types.is_numeric_dtype(eng[c])]
            latest = latest.merge(eng[["ticker", *num_cols]], on="ticker", how="left", suffixes=("", "_eng"))
    except Exception:
        pass

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

    # Align matrix strictly to model features; add missing as zeros and order by booster feature names
    expected = list(xgb_feats)
    present = [c for c in expected if c in latest.columns]
    missing = [c for c in expected if c not in latest.columns]
    if missing:
        logger.warning("Attribution: %d expected features missing; filling with zeros. Examples: %s", len(missing), ", ".join(missing[:5]))
    X = latest[present].copy()
    for m in missing:
        X[m] = 0.0
    # Reorder columns to exactly expected order
    X = X[expected].replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')

    explainer = shap.TreeExplainer(xgb_model)
    try:
        shap_vals = explainer.shap_values(X)
    except Exception as e:
        # Fallback: use XGBoost's pred_contribs to approximate SHAP values
        logger.warning("SHAP explainer failed (%s). Falling back to xgboost pred_contribs.", e)
        try:
            import xgboost as xgb  # type: ignore
            # Provide expected feature names to match the booster
            dmat = xgb.DMatrix(X, feature_names=expected)
            shap_contribs = xgb_model.get_booster().predict(dmat, pred_contribs=True)
            shap_vals = shap_contribs
        except Exception as e2:
            logger.error("Attribution fallback failed: %s", e2)
            return
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
