import os
import sys
import argparse
from datetime import datetime

import pandas as pd


def _apply_config_env():
    try:
        from utils.config import load_config, apply_env_from_config
        cfg_path = os.getenv('APP_CONFIG', 'config.yaml')
        cfg = load_config(cfg_path)
        apply_env_from_config(cfg)
    except Exception:
        pass


def _exists(path: str) -> bool:
    return bool(path) and os.path.isfile(path)


def _num(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def diagnose_ticker(ticker: str) -> int:
    _apply_config_env()
    ticker = str(ticker).upper().strip()

    paths = {
        'features_clean': os.getenv('PATH_FEATURES_CLEAN', 'data/features/stock_features_clean.csv'),
        'prices_clean': 'data/input/stock_prices_with_clean_data.csv',
        'ensemble_out': os.getenv('PATH_ENSEMBLE_OUT', 'data/ensemble/ensemble_weekly_output.csv'),
        'xgb_out': os.getenv('PATH_XGB_OUT', 'data/xgboost/xgboost_weekly_output.csv'),
        'lstm_out': os.getenv('PATH_LSTM_OUT', 'data/lstm/lstm_weekly_output.csv'),
        'earnings_history': os.getenv('EARNINGS_HISTORY_CSV', 'data/input/earnings_history.csv'),
    }

    print(f"=== UI Data Diagnosis for {ticker} ===")
    missing_files = []
    for name, path in paths.items():
        ok = _exists(path)
        print(f"[{name}] {path} -> {'OK' if ok else 'MISSING'}")
        if not ok:
            missing_files.append(name)

    if missing_files:
        print("\nMissing files detected; UI may not render some sections.")

    # Prices / support-resistance
    if _exists(paths['prices_clean']):
        dfp = pd.read_csv(paths['prices_clean'])
        if 'ticker' in dfp.columns:
            dfp['ticker'] = dfp['ticker'].astype(str).str.upper()
        tp = dfp[dfp['ticker'] == ticker].copy()
        if tp.empty:
            print(f"\n[prices_clean] No rows for {ticker}")
        else:
            tp['date'] = pd.to_datetime(tp['date'], errors='coerce')
            tp = tp.sort_values('date')
            last_dt = tp['date'].max()
            last20 = tp.tail(20)
            sup = pd.to_numeric(last20['close'], errors='coerce').min()
            res = pd.to_numeric(last20['close'], errors='coerce').max()
            print(f"\n[prices_clean] rows={len(tp)} last_date={last_dt.date() if isinstance(last_dt, pd.Timestamp) else last_dt}")
            print(f"  last20_count={len(last20)} support_20d(min close)={sup:.2f} resistance_20d(max close)={res:.2f}")

    # Features
    if _exists(paths['features_clean']):
        dff = pd.read_csv(paths['features_clean'])
        if 'ticker' in dff.columns:
            dff['ticker'] = dff['ticker'].astype(str).str.upper()
        tf = dff[dff['ticker'] == ticker].copy()
        have_cols = [c for c in ['support_20d','resistance_20d','rsi_14d','momentum_5d','momentum_10d'] if c in dff.columns]
        print(f"\n[features_clean] present_cols={have_cols}")
        if tf.empty:
            print(f"  No rows for {ticker}")
        else:
            # Show most recent values
            tf['date'] = pd.to_datetime(tf['date'], errors='coerce') if 'date' in tf.columns else None
            tf = tf.sort_values('date') if 'date' in tf.columns else tf
            tail = tf.tail(1)
            vals = {c: _num(tail[c].iloc[0]) for c in have_cols if c in tail.columns}
            print(f"  latest: {vals}")

    # Weekly outputs
    for key in ['ensemble_out','xgb_out','lstm_out']:
        p = paths[key]
        if not _exists(p):
            continue
        df = pd.read_csv(p)
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper()
        row = df[df['ticker'] == ticker].head(1)
        cols = [c for c in df.columns if 'pred' in c or 'score' in c or 'confidence' in c]
        print(f"\n[{key}] columns_hint={cols[:8]}")
        if row.empty:
            print(f"  No row for {ticker}")
        else:
            show = {c: row.iloc[0][c] for c in cols if c in row.columns}
            print(f"  values: {show}")

    # Earnings
    if _exists(paths['earnings_history']):
        de = pd.read_csv(paths['earnings_history'])
        lower = {c.lower(): c for c in de.columns}
        needed = ['ticker','earnings_date','reported_eps','estimate_eps','surprise_percentage']
        missing = [c for c in needed if c not in lower]
        if missing:
            print(f"\n[earnings_history] missing columns: {missing}")
        # show ticker rows
        tcol = lower.get('ticker','ticker')
        if tcol in de.columns:
            de[tcol] = de[tcol].astype(str).str.upper()
            te = de[de[tcol] == ticker]
            print(f"\n[earnings_history] rows for {ticker}: {len(te)}")
            if len(te) > 0:
                print(te.head(3).to_string(index=False))

    print("\nDiagnosis complete.")
    return 0


def main():
    p = argparse.ArgumentParser(description='Diagnose presence of UI-required files and columns for a ticker')
    p.add_argument('--ticker', required=True, help='Ticker symbol (e.g., AAPL)')
    args = p.parse_args()
    sys.exit(diagnose_ticker(args.ticker))


if __name__ == '__main__':
    main()


