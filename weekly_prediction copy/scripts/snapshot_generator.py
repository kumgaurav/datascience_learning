# scripts/snapshot_generator.py
import argparse
import json
import os
from datetime import date
from typing import Optional

import pandas as pd


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
    use_date = snapshot_date or date.today().isoformat()
    snapshot_dir = f"snapshot/{use_date}/"
    os.makedirs(snapshot_dir, exist_ok=True)

    # Standardize keys and reduce to latest-per-ticker features
    features_df = _standardize_keys(features_df)
    features_latest = _latest_per_ticker(features_df)

    # Normalize predictions columns
    predictions_df = _standardize_keys(predictions_df)
    rename_map = {}
    if "xgb_predicted_return_pct" in predictions_df.columns:
        rename_map["xgb_predicted_return_pct"] = "pred_xgb"
    if "xgb_pred" in predictions_df.columns:
        rename_map["xgb_pred"] = "pred_xgb"
    if "lstm_predicted_return_pct" in predictions_df.columns:
        rename_map["lstm_predicted_return_pct"] = "pred_lstm"
    # Keep ensemble score if present
    if "ensemble_score" in predictions_df.columns:
        rename_map["ensemble_score"] = "ensemble_score"
    predictions_norm = predictions_df.rename(columns=rename_map)

    # Select only the needed pred columns to avoid duplicate col names on merge
    pred_cols = [c for c in ["ticker", "pred_xgb", "pred_lstm", "ensemble_score"] if c in predictions_norm.columns]
    predictions_small = predictions_norm[pred_cols].drop_duplicates(subset=["ticker"], keep="last")

    # Ranked-first construction: if xgb_ranked.csv exists, build the snapshot from it, then enrich with features
    ranked_path_env = os.getenv('XGB_RANKED_CSV', 'data/top/xgb_ranked.csv')
    weekly_path_env = os.getenv('XGB_WEEKLY_OUTPUT', 'data/top/xgb_weekly_output.csv')
    if os.path.exists(ranked_path_env):
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
            if os.path.exists(weekly_path_env):
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
            print(f"✅ Snapshot created (ranked-first): {csv_path}")
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

    print(f"✅ Snapshot created: {csv_path}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create weekly snapshot of features + predictions")
    parser.add_argument(
        "--features",
        default=os.getenv("SNAPSHOT_FEATURES_CSV", "data/top/features_raw_full.csv"),
        help="Path to features CSV containing at least ticker,date and engineered features",
    )
    parser.add_argument(
        "--predictions",
        default=os.getenv("SNAPSHOT_PREDICTIONS_CSV", "data/top/ensemble_scores_output.csv"),
        help="Path to predictions CSV (should include ticker and pred columns)",
    )
    parser.add_argument(
        "--meta_model_xgb",
        default=os.getenv("XGB_MODEL_PATH", "models/stock_predictor_top.joblib"),
        help="Path or identifier for XGB model used",
    )
    parser.add_argument(
        "--meta_model_lstm",
        default=os.getenv("LSTM_MODEL_PATH", "models/lib/keras/lstm_model.keras"),
        help="Path or identifier for LSTM model used",
    )
    parser.add_argument("--date", dest="snapshot_date", default=None, help="Snapshot date (YYYY-MM-DD). Default: today")
    args = parser.parse_args()

    if not os.path.exists(args.features):
        raise FileNotFoundError(f"Features CSV not found: {args.features}")
    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"Predictions CSV not found: {args.predictions}")

    features_df = pd.read_csv(args.features)
    predictions_df = pd.read_csv(args.predictions)

    metadata = {
        "features_source": os.path.abspath(args.features),
        "predictions_source": os.path.abspath(args.predictions),
        "xgb_model": args.meta_model_xgb,
        "lstm_model": args.meta_model_lstm,
    }

    create_snapshot(features_df, predictions_df, metadata, snapshot_date=args.snapshot_date)


if __name__ == "__main__":
    main()
