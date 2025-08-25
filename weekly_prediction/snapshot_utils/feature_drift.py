"""Compute feature drift between snapshot features and today's eval features.

Writes snapshot/<date>/drift/<date>_drift.csv with columns:
  feature, ks_stat, p_value, status, n_snapshot, n_eval
"""

import argparse
import os
import sys
import logging
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _load_eval_features(snapshot_dir: str, eval_date: str) -> pd.DataFrame:
    eval_dir = os.path.join(snapshot_dir, "eval")
    eval_input = os.path.join(eval_dir, f"{eval_date}_eval_input.csv")
    eval_csv = os.path.join(eval_dir, f"{eval_date}_eval.csv")
    if os.path.exists(eval_input):
        df = pd.read_csv(eval_input)
    elif os.path.exists(eval_csv):
        df = pd.read_csv(eval_csv)
    else:
        raise FileNotFoundError(f"Eval files not found for {eval_date} under {eval_dir}")
    # Normalize key
    if "ticker" not in df.columns and "symbol" in df.columns:
        df["ticker"] = df["symbol"]
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def compute_drift(snapshot_df: pd.DataFrame, eval_df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    # Align numeric features present in both
    snap_num = set(_numeric_columns(snapshot_df))
    eval_num = set(_numeric_columns(eval_df))
    ignore = {"pred_xgb", "pred_lstm"}
    common = [c for c in sorted(snap_num.intersection(eval_num)) if c not in ignore]
    rows = []
    for col in common:
        a = snapshot_df[col].dropna().to_numpy()
        b = eval_df[col].dropna().to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        try:
            stat, pval = ks_2samp(a, b)
        except Exception:
            continue
        rows.append({
            "feature": col,
            "ks_stat": float(stat),
            "p_value": float(pval),
            "status": "Drift" if pval < threshold else "Stable",
            "n_snapshot": int(len(a)),
            "n_eval": int(len(b)),
        })
    return pd.DataFrame(rows).sort_values(["status", "p_value", "feature"]) if rows else pd.DataFrame(columns=["feature","ks_stat","p_value","status","n_snapshot","n_eval"]) 


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute feature drift vs snapshot for a given date")
    parser.add_argument("--config", default=os.getenv("APP_CONFIG", os.path.join(os.path.dirname(__file__), "snapshot_config.yaml")), help="Path to YAML/JSON config for paths and logging")
    parser.add_argument("--snapshot", default=None, help="Path to snapshot CSV (snapshot/<date>/snapshot.csv)")
    parser.add_argument("--date", dest="eval_date", default=date.today().isoformat(), help="Evaluation date YYYY-MM-DD")
    parser.add_argument("--threshold", type=float, default=0.05, help="Significance threshold for KS test")
    args = parser.parse_args()

    # Ensure project root import and configure logging
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from utils.config import load_config, apply_env_from_config  # type: ignore
    from utils.logging_utils import configure_logging  # type: ignore
    cfg = load_config(args.config)
    apply_env_from_config(cfg)
    configure_logging(level=os.getenv('LOG_LEVEL'), log_file=os.getenv('LOG_FILE'))
    logger = logging.getLogger(__name__)

    if not args.snapshot or not os.path.exists(args.snapshot):
        raise FileNotFoundError("--snapshot must point to an existing snapshot CSV")
    snapshot_dir = os.path.dirname(args.snapshot)

    snap = pd.read_csv(args.snapshot)
    # Normalize key
    if "ticker" not in snap.columns and "symbol" in snap.columns:
        snap["ticker"] = snap["symbol"]
    snap["ticker"] = snap["ticker"].astype(str).str.upper()

    today_df = _load_eval_features(snapshot_dir, args.eval_date)

    drift_df = compute_drift(snap, today_df, threshold=args.threshold)

    out_dir = os.path.join(snapshot_dir, "drift")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{args.eval_date}_drift.csv")
    drift_df.to_csv(out_csv, index=False)
    logger.info("[DRIFT] Wrote %s (rows=%d)", out_csv, len(drift_df))


if __name__ == "__main__":
    main()
