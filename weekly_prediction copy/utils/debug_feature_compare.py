#!/usr/bin/env python3
"""
Debug utility: step-by-step feature comparison for two tickers.

Shows side-by-side OHLCV for the selected lookback window, highlights
the most recent 3 days and the earliest 3 days, and computes key features
exactly as in feature_engineering:
 - RSI(14): rolling mean of gains/losses (not Wilder's smoothing)
 - Volume Ratio: volume / rolling mean(volume, 20)

Usage:
  python utils/debug_feature_compare.py \
    --prices data/stock_prices_input.csv \
    --ticker_a REAL --ticker_b APLD \
    --end_date 2025-08-21 --lookback 60
"""

import argparse
import os
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd


def _normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols)
    if 'ticker' not in df.columns:
        if 'symbol' in df.columns:
            df = df.rename(columns={'symbol': 'ticker'})
        else:
            raise ValueError("Prices must have ticker/symbol column")
    df['ticker'] = df['ticker'].astype(str).str.upper()
    if 'date' not in df.columns:
        raise ValueError("Prices must have date column")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


def _rsi_simple(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _compute_features(pr: pd.DataFrame) -> pd.DataFrame:
    out = pr.sort_values('date').copy()
    out['rsi_14d'] = _rsi_simple(out['close'], period=14)
    out['volume_ma_20'] = out['volume'].rolling(20).mean()
    out['volume_ratio'] = out['volume'] / out['volume_ma_20']
    return out


def _side_by_side(a: pd.DataFrame, b: pd.DataFrame, cols=('open','high','low','close','volume')) -> pd.DataFrame:
    aa = a[['date', *cols]].copy()
    bb = b[['date', *cols]].copy()
    aa.columns = ['date'] + [f"{c}_A" for c in cols]
    bb.columns = ['date'] + [f"{c}_B" for c in cols]
    merged = aa.merge(bb, on='date', how='outer').sort_values('date')
    return merged


def main() -> None:
    p = argparse.ArgumentParser(description='Step-by-step feature comparison for two tickers')
    p.add_argument('--prices', required=True, help='Path to prices CSV (e.g., data/stock_prices_input.csv)')
    p.add_argument('--ticker_a', required=True)
    p.add_argument('--ticker_b', required=True)
    p.add_argument('--end_date', required=True, help='YYYY-MM-DD')
    p.add_argument('--lookback', type=int, default=60, help='Days before end_date to include')
    args = p.parse_args()

    if not os.path.exists(args.prices):
        raise FileNotFoundError(args.prices)

    prices = pd.read_csv(args.prices)
    prices = _normalize_prices(prices)
    end_dt = pd.to_datetime(args.end_date)
    start_dt = end_dt - pd.Timedelta(days=args.lookback)

    a = prices[(prices['ticker'] == args.ticker_a.upper()) & (prices['date'].between(start_dt, end_dt))].copy()
    b = prices[(prices['ticker'] == args.ticker_b.upper()) & (prices['date'].between(start_dt, end_dt))].copy()

    if a.empty or b.empty:
        raise RuntimeError('No rows for one or both tickers in the selected window')

    a_feat = _compute_features(a)
    b_feat = _compute_features(b)

    # Print last 3 rows (most recent)
    print(f"\n=== Most recent 3 days (ending {end_dt.date()}) ===")
    recent = _side_by_side(a_feat.tail(3), b_feat.tail(3), cols=('open','high','low','close','volume'))
    print(recent.to_string(index=False))

    # Print earliest 3 rows (start of window)
    print(f"\n=== Earliest 3 days in window (starting {start_dt.date()}) ===")
    earliest = _side_by_side(a_feat.head(3), b_feat.head(3), cols=('open','high','low','close','volume'))
    print(earliest.to_string(index=False))

    # Show RSI 14d and Volume Ratio series tails
    print("\n=== RSI(14) tails ===")
    print(pd.DataFrame({
        'date': a_feat['date'].tail(5).dt.date,
        f'RSI_14_{args.ticker_a.upper()}': a_feat['rsi_14d'].tail(5).round(3).values,
        f'RSI_14_{args.ticker_b.upper()}': b_feat['rsi_14d'].tail(5).round(3).values,
    }).to_string(index=False))

    print("\n=== Volume Ratio tails (vol/vol_ma_20) ===")
    print(pd.DataFrame({
        'date': a_feat['date'].tail(5).dt.date,
        f'VolRatio_{args.ticker_a.upper()}': a_feat['volume_ratio'].tail(5).round(3).values,
        f'VolRatio_{args.ticker_b.upper()}': b_feat['volume_ratio'].tail(5).round(3).values,
    }).to_string(index=False))

    # Current values and the windows used
    def _window_info(feat_df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
        # For RSI(14): last 15 points include today + 14 history
        w_rsi = feat_df['close'].dropna().tail(15).index
        # For vol ratio: last 20 points volume series
        w_vol = feat_df['volume'].dropna().tail(20).index
        return (feat_df.loc[w_rsi, 'date'].min() if len(w_rsi) else pd.NaT,
                feat_df.loc[w_vol, 'date'].min() if len(w_vol) else pd.NaT)

    a_rsi_start, a_vol_start = _window_info(a_feat)
    b_rsi_start, b_vol_start = _window_info(b_feat)
    print("\n=== Current feature values ===")
    print(f"RSI_14 {args.ticker_a.upper()} = {float(a_feat['rsi_14d'].iloc[-1]):.3f} (window start {str(getattr(a_rsi_start,'date',a_rsi_start))})")
    print(f"RSI_14 {args.ticker_b.upper()} = {float(b_feat['rsi_14d'].iloc[-1]):.3f} (window start {str(getattr(b_rsi_start,'date',b_rsi_start))})")
    print(f"VolRatio {args.ticker_a.upper()} = {float(a_feat['volume_ratio'].iloc[-1]):.3f} (vol_ma_20 from last 20 days starting {str(getattr(a_vol_start,'date',a_vol_start))})")
    print(f"VolRatio {args.ticker_b.upper()} = {float(b_feat['volume_ratio'].iloc[-1]):.3f} (vol_ma_20 from last 20 days starting {str(getattr(b_vol_start,'date',b_vol_start))})")


if __name__ == '__main__':
    main()


