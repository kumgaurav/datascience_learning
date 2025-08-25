# scripts/snapshot_generator.py
import argparse
import json
import os
from datetime import date
from typing import Optional

import pandas as pd
import logging
import sys

# Ensure project root is importable when executing from snapshot_utils/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.logging_utils import configure_logging
from utils.config import load_config, apply_env_from_config


def _standardize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize ticker key
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()
    elif "symbol" in df.columns:
        df.rename(columns={"symbol": "ticker"}, inplace=True)
        df["ticker"] = df["ticker"].astype(str).str.upper()
    # Parse date if present
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            pass
    return df


def _latest_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if "ticker" not in df.columns:
        return df
    if "date" in df.columns:
        try:
            return (
                df.sort_values(["ticker", "date"])  # type: ignore[arg-type]
                  .drop_duplicates(subset=["ticker"], keep="last")
            )
        except Exception:
            return df.drop_duplicates(subset=["ticker"], keep="last")
    return df.drop_duplicates(subset=["ticker"], keep="last")


def _pick_top25_tickers(features_latest: pd.DataFrame, predictions_small: pd.DataFrame) -> pd.Index:
    """Return the tickers for the UI's top-25 selection.

    Tries to use stock_selector_v2.get_top_stocks for consistency with the UI.
    Falls back to ensemble_score or xgb_pred sorting if selector is unavailable.
    """
    # Attempt to use the enhanced selector for parity with UI
    try:
        from stock_selector_v2 import get_top_stocks  # type: ignore
        top_df = get_top_stocks(n=25)
        if top_df is not None and not top_df.empty and 'ticker' in top_df.columns:
            return top_df['ticker'].astype(str).str.upper().drop_duplicates().tolist()
    except Exception:
        pass

    # Fallback: use ensemble_score if present, else xgb_pred, else xgb_predicted_return_pct
    df_rank = predictions_small.copy()
    for score_col in ["ensemble_score", "xgb_pred", "xgb_predicted_return_pct", "pred_xgb"]:
        if score_col in df_rank.columns:
            try:
                df_rank = df_rank.sort_values(score_col, ascending=False)
                return df_rank['ticker'].astype(str).str.upper().drop_duplicates().head(25).tolist()
            except Exception:
                continue
    # Last resort: take any 25 tickers present
    return predictions_small['ticker'].astype(str).str.upper().drop_duplicates().head(25).tolist()


def create_snapshot(
    features_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    metadata: dict,
    snapshot_date: Optional[str] = None,
) -> str:
    logger = logging.getLogger(__name__)
    use_date = snapshot_date or date.today().isoformat()
    # Allow overriding root dir via env (set by YAML)
    snapshot_root = os.getenv('SNAPSHOT_ROOT_DIR', 'snapshot')
    snapshot_dir = f"{snapshot_root}/{use_date}/"
    os.makedirs(snapshot_dir, exist_ok=True)

    # Standardize keys and reduce to latest-per-ticker features
    features_df = _standardize_keys(features_df)
    features_latest = _latest_per_ticker(features_df)

    # Normalize predictions columns and create alias columns while preserving originals
    predictions_df = _standardize_keys(predictions_df).copy()
    # XGB aliases
    if "xgb_predicted_return_pct" in predictions_df.columns and "pred_xgb" not in predictions_df.columns:
        predictions_df["pred_xgb"] = predictions_df["xgb_predicted_return_pct"]
    if "xgb_pred" in predictions_df.columns and "pred_xgb" not in predictions_df.columns:
        predictions_df["pred_xgb"] = predictions_df["xgb_pred"]
    # LSTM aliases
    if "lstm_predicted_return_pct" in predictions_df.columns and "pred_lstm" not in predictions_df.columns:
        predictions_df["pred_lstm"] = predictions_df["lstm_predicted_return_pct"]
    if "lstm_pred" in predictions_df.columns and "pred_lstm" not in predictions_df.columns:
        predictions_df["pred_lstm"] = predictions_df["lstm_pred"]
    # Keep ensemble_score and any additional columns from ensemble file

    # Select all prediction columns that do not collide with feature columns to avoid overwriting
    feature_cols_set = set(features_latest.columns)
    keep_pred_cols = ["ticker"] + [
        c for c in predictions_df.columns
        if c != "ticker" and c not in feature_cols_set
    ]
    predictions_small = predictions_df[keep_pred_cols].drop_duplicates(subset=["ticker"], keep="last")

    # Ranked-first construction: if ranked CSV exists, build snapshot from it, then enrich with features
    ranked_path_env = os.getenv('XGB_RANKED_CSV')
    weekly_path_env = os.getenv('XGB_WEEKLY_OUTPUT')
    if isinstance(ranked_path_env, str) and ranked_path_env.strip() and os.path.exists(ranked_path_env):
        rk = _standardize_keys(pd.read_csv(ranked_path_env))
        if 'ticker' in rk.columns:
            # Preserve UI order: by confidence_score desc if present, else file order
            if 'confidence_score' in rk.columns:
                rk = rk.sort_values('confidence_score', ascending=False)
            rk['__ui_order__'] = range(1, len(rk) + 1)
            # Base snapshot from ranked list (ensures all 25 tickers appear)
            base_cols = ['ticker']
            if 'confidence_score' in rk.columns:
                base_cols.append('confidence_score')
            if 'predicted_return_pct' in rk.columns:
                base_cols.append('predicted_return_pct')
            base = rk[base_cols + ['__ui_order__']].drop_duplicates(subset=['ticker'])
            # Enrich with features: prefer provided features_latest; if weekly exists, left-join as an extra source
            snap = base.merge(features_latest, on='ticker', how='left', suffixes=('', '_feat'))
            if isinstance(weekly_path_env, str) and weekly_path_env.strip() and os.path.exists(weekly_path_env):
                try:
                    wk = _standardize_keys(pd.read_csv(weekly_path_env))
                    wk_latest = _latest_per_ticker(wk)
                    snap = snap.merge(wk_latest, on='ticker', how='left', suffixes=('', '_wk'))
                except Exception:
                    pass
            # Attach prediction columns: pred_xgb from ranked predicted_return_pct if not already present
            if 'pred_xgb' not in snap.columns and 'predicted_return_pct' in base.columns:
                snap['pred_xgb'] = base['predicted_return_pct'].values
            # Merge any additional predictions_small (e.g., ensemble_score) without losing order
            if not predictions_small.empty:
                snap = snap.merge(predictions_small, on='ticker', how='left', suffixes=('', '_pred'))
            # Add symbol alias
            snap['symbol'] = snap.get('symbol', snap['ticker'])
            # Order by UI order and select a clean set of columns (avoid duplicate names)
            snap = snap.sort_values(['__ui_order__', 'ticker']).drop(columns=[c for c in snap.columns if c.endswith('_feat') or c.endswith('_wk') or c.endswith('_pred') or c == '__ui_order__'], errors='ignore')
            # Write and return
            csv_path = f"{snapshot_dir}snapshot.csv"
            snap.to_csv(csv_path, index=False)
            meta = {
                "snapshot_date": use_date,
                **metadata,
                "row_count": int(len(snap)),
                "selected_top_n": int(len(snap)),
                "source_ranked": os.path.abspath(ranked_path_env),
                "source_weekly": os.path.abspath(weekly_path_env) if os.path.exists(weekly_path_env) else None,
                "order_by": 'confidence_score' if 'confidence_score' in rk.columns else 'file_order',
            }
            with open(f"{snapshot_dir}/metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"Snapshot created (ranked-first): %s", csv_path)
            return csv_path

    # Determine UI-consistent top-25 tickers (fallback)
    top_tickers = _pick_top25_tickers(features_latest, predictions_small)
    # Filter to the top 25 before merging to keep snapshot focused
    features_top = features_latest[features_latest['ticker'].isin(top_tickers)] if 'ticker' in features_latest.columns else features_latest
    preds_top = predictions_small[predictions_small['ticker'].isin(top_tickers)] if 'ticker' in predictions_small.columns else predictions_small

    # Merge features + predictions
    snapshot = features_top.merge(preds_top, on="ticker", how="left")
    # Add compatibility alias for downstream evaluators expecting 'symbol'
    if "symbol" not in snapshot.columns and "ticker" in snapshot.columns:
        snapshot["symbol"] = snapshot["ticker"]

    # Save snapshot CSV
    csv_path = f"{snapshot_dir}snapshot.csv"
    snapshot.to_csv(csv_path, index=False)

    # Save metadata JSON
    meta = {
        "snapshot_date": use_date,
        **metadata,
        "row_count": int(len(snapshot)),
        "selected_top_n": 25,
        "feature_columns": [c for c in snapshot.columns if c not in {"ticker", "date", "pred_xgb", "pred_lstm", "ensemble_score"}],
        "prediction_columns": [c for c in ["pred_xgb", "pred_lstm", "ensemble_score"] if c in snapshot.columns],
    }
    with open(f"{snapshot_dir}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Snapshot created: %s", csv_path)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create weekly snapshot of features + predictions")
    parser.add_argument(
        "--config",
        default=os.getenv("APP_CONFIG", os.path.join(os.path.dirname(__file__), "snapshot_config.yaml")),
        help="Path to YAML/JSON config to read paths and logging settings",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Path to features CSV containing at least ticker,date and engineered features",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Path to predictions CSV (should include ticker and pred columns)",
    )
    parser.add_argument(
        "--meta_model_xgb",
        default=None,
        help="Path or identifier for XGB model used (optional, for metadata)",
    )
    parser.add_argument(
        "--meta_model_lstm",
        default=None,
        help="Path or identifier for LSTM model used (optional, for metadata)",
    )
    parser.add_argument("--date", dest="snapshot_date", default=None, help="Snapshot date (YYYY-MM-DD). Default: today")
    args = parser.parse_args()

    # Load config and apply env overrides
    cfg = load_config(args.config)
    apply_env_from_config(cfg)
    # Configure logging
    log_level = os.getenv('LOG_LEVEL')
    log_file = os.getenv('LOG_FILE')
    configure_logging(level=log_level, log_file=log_file)
    logger = logging.getLogger(__name__)
    logger.info("Using config file: %s", args.config)

    # Determine input files from CLI > config > env
    features_path = args.features
    predictions_path = args.predictions
    paths = {}
    if isinstance(cfg, dict):
        paths = cfg.get('paths') or {}
    if not features_path:
        cfg_features = paths.get('features_csv') if isinstance(paths.get('features_csv'), str) else None
        features_path = cfg_features or os.getenv('SNAPSHOT_FEATURES_CSV')
    if not predictions_path:
        cfg_predictions = paths.get('predictions_csv') if isinstance(paths.get('predictions_csv'), str) else None
        predictions_path = cfg_predictions or os.getenv('SNAPSHOT_PREDICTIONS_CSV')
    # Ranked-first optional inputs and snapshot root directory
    ranked_csv = paths.get('xgb_ranked_csv') or os.getenv('XGB_RANKED_CSV')
    weekly_out = paths.get('xgb_weekly_output_csv') or os.getenv('XGB_WEEKLY_OUTPUT')
    snapshot_root_dir = paths.get('snapshot_root_dir')
    if isinstance(snapshot_root_dir, str) and snapshot_root_dir.strip():
        os.environ['SNAPSHOT_ROOT_DIR'] = snapshot_root_dir
    if isinstance(ranked_csv, str) and ranked_csv.strip():
        os.environ['XGB_RANKED_CSV'] = ranked_csv
    if isinstance(weekly_out, str) and weekly_out.strip():
        os.environ['XGB_WEEKLY_OUTPUT'] = weekly_out

    # Validate required paths are provided
    if not features_path:
        raise ValueError("Missing features CSV path. Provide --features or set paths.features_csv in config or SNAPSHOT_FEATURES_CSV env.")
    if not predictions_path:
        raise ValueError("Missing predictions CSV path. Provide --predictions or set paths.predictions_csv in config or SNAPSHOT_PREDICTIONS_CSV env.")

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features CSV not found: {features_path}")
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_path}")

    logger.info("Reading features from: %s", os.path.abspath(features_path))
    logger.info("Reading predictions from: %s", os.path.abspath(predictions_path))
    features_df = pd.read_csv(features_path)
    predictions_df = pd.read_csv(predictions_path)

    # Resolve optional model identifiers for metadata: CLI > env > config
    meta_xgb = args.meta_model_xgb or os.getenv("XGB_MODEL_PATH")
    meta_lstm = args.meta_model_lstm or os.getenv("LSTM_MODEL_PATH")
    if not meta_lstm and isinstance(cfg, dict):
        try:
            lstm_cfg = (cfg.get('lstm') or {})
            if isinstance(lstm_cfg.get('model_path'), str):
                meta_lstm = lstm_cfg.get('model_path')
        except Exception:
            pass
    if not meta_xgb and isinstance(cfg, dict):
        try:
            xgb_cfg = (cfg.get('xgboost') or {})
            if isinstance(xgb_cfg.get('model_path'), str):
                meta_xgb = xgb_cfg.get('model_path')
        except Exception:
            pass

    metadata = {
        "features_source": os.path.abspath(features_path),
        "predictions_source": os.path.abspath(predictions_path),
    }
    if meta_xgb:
        metadata["xgb_model"] = meta_xgb
    if meta_lstm:
        metadata["lstm_model"] = meta_lstm

    create_snapshot(features_df, predictions_df, metadata, snapshot_date=args.snapshot_date)


if __name__ == "__main__":
    main()
