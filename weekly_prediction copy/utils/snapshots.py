import os
import pandas as pd
from typing import List, Optional


def build_latest_snapshot(
    base_df: pd.DataFrame,
    engineered_df: Optional[pd.DataFrame],
    required_cols: List[str],
) -> pd.DataFrame:
    """
    Create a consistent latest-per-ticker snapshot with required columns.

    - base_df: time-series features (must include ticker,date and some signals)
    - engineered_df: optional engineered features to backfill missing columns
    - required_cols: column list to include/order in the output
    """
    if base_df is None or base_df.empty:
        return pd.DataFrame(columns=required_cols)

    df = base_df.copy()
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].astype(str).str.upper()
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        except Exception:
            pass
        df = df.sort_values('date').groupby('ticker', as_index=False).tail(1)

    # Merge in any missing required columns from engineered_df on ticker/date
    if engineered_df is not None and not engineered_df.empty:
        fe = engineered_df.copy()
        if 'ticker' in fe.columns:
            fe['ticker'] = fe['ticker'].astype(str).str.upper()
        if 'date' in fe.columns:
            try:
                fe['date'] = pd.to_datetime(fe['date'], errors='coerce')
            except Exception:
                pass
        need = [c for c in required_cols if c not in df.columns and c in fe.columns]
        if need:
            key_cols = ['ticker'] + (['date'] if 'date' in df.columns and 'date' in fe.columns else [])
            df = df.merge(fe[key_cols + need], on=key_cols, how='left')

    # Ensure all required columns exist and are ordered
    out = df.copy()
    for c in required_cols:
        if c not in out.columns:
            if c == 'next_earnings_date':
                out[c] = pd.NaT
            elif c in [
                'broke_resistance', 'post_earnings_dip_rally', 'strong_momentum', 'breakout_confirmed',
                'golden_cross', 'momentum_winner', 'trend_persistence_20d', 'earnings_in_3_weeks',
                'last_2q_positive_surprises', 'pre_earning_rally'
            ]:
                out[c] = False
            else:
                out[c] = 0

    return out[required_cols]


def write_latest_snapshot(
    base_df: pd.DataFrame,
    engineered_df: Optional[pd.DataFrame],
    required_cols: List[str],
    out_path: str,
) -> str:
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    snap = build_latest_snapshot(base_df, engineered_df, required_cols)
    snap.to_csv(out_path, index=False)
    return out_path


