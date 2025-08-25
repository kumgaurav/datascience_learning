import os
import sys
import logging
from pathlib import Path
import argparse
import pandas as pd

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config, apply_env_from_config
from utils.logging_utils import configure_logging
from models.momentum.prediction_model_momentum import MomentumModelTrainer


def export_weekly(features_path: str, model_path: str, out_path: str) -> str:
    # Validate inputs strictly (no fallbacks)
    if not features_path:
        raise FileNotFoundError("Momentum features path not provided. Set momentum.features_path in config.yaml.")
    if not os.path.isfile(features_path):
        raise FileNotFoundError(f"Momentum features file not found at {features_path}")
    if not model_path:
        raise FileNotFoundError("Momentum model path not provided. Set momentum.model_path in config.yaml.")

    # Ensure env matches provided paths for trainer initialization
    os.environ['MOMENTUM_FEATURES_PATH'] = features_path
    os.environ['MOMENTUM_MODEL_PATH'] = model_path

    trainer = MomentumModelTrainer()
    trainer.train()

    # Build weekly output from latest available rows per ticker
    df = pd.read_csv(features_path)
    if 'ticker' not in df.columns:
        raise RuntimeError("features must include 'ticker'")
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        except Exception:
            pass
        df = df.dropna(subset=['date']).sort_values('date')
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    latest = df.groupby('ticker').tail(1).copy()

    # Use simple momentum-based heuristic for now: 5d and 20d momentum
    # If columns missing, compute minimal requirement from close
    price_col = None
    for c in ['close','price','Close']:
        if c in df.columns:
            price_col = c
            break
    if 'momentum_5d' not in latest.columns or 'momentum_20d' not in latest.columns:
        if price_col is None:
            raise RuntimeError("features must include close/price to compute momentum columns")
        tmp = df[['ticker','date',price_col]].copy()
        tmp = tmp.sort_values(['ticker','date'])
        tmp['momentum_5d'] = tmp.groupby('ticker')[price_col].pct_change(periods=5)
        tmp['momentum_20d'] = tmp.groupby('ticker')[price_col].pct_change(periods=20)
        latest = latest.merge(
            tmp.groupby('ticker').tail(1)[['ticker','momentum_5d','momentum_20d']],
            on='ticker', how='left'
        )

    # Predicted return proxy: weighted combo of short/medium momentum
    latest['momentum_pred'] = (
        0.6 * pd.to_numeric(latest.get('momentum_5d', 0.0), errors='coerce').fillna(0.0) +
        0.4 * pd.to_numeric(latest.get('momentum_20d', 0.0), errors='coerce').fillna(0.0)
    ) * 100.0

    # Optional confidence: scaled absolute momentum
    try:
        abs_vals = (latest['momentum_pred'].abs())
        p95 = float(abs_vals.quantile(0.95)) if len(abs_vals) > 0 else 1.0
        scale = p95 if p95 > 1e-8 else (float(abs_vals.max()) if float(abs_vals.max()) > 1e-8 else 1.0)
    except Exception:
        scale = 1.0
    latest['confidence_score'] = (100.0 * (latest['momentum_pred'].abs() / scale)).clip(0.0, 100.0)

    keep = [c for c in ['ticker','date','close','momentum_5d','momentum_20d','momentum_pred','confidence_score'] if c in latest.columns]
    latest = latest[keep].sort_values('momentum_pred', ascending=False)
    if not out_path:
        raise FileNotFoundError("Momentum weekly output path not provided. Set paths.momentum_out in config.yaml.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    latest.to_csv(out_path, index=False)
    logging.getLogger(__name__).info(f"[MOMENTUM WEEKLY] Wrote weekly output to {out_path} (rows={len(latest)})")
    return out_path


def main():
    # Load config first so parser defaults can come from env set by config
    cfg_path = os.getenv('APP_CONFIG', 'config.yaml')
    cfg = load_config(cfg_path)
    apply_env_from_config(cfg)
    configure_logging()

    p = argparse.ArgumentParser(description='Train momentum model and export weekly momentum rankings')
    p.add_argument('--features', default=os.getenv('MOMENTUM_FEATURES_PATH'), help='Features CSV path (required via config/env)')
    p.add_argument('--model', default=os.getenv('MOMENTUM_MODEL_PATH'), help='Model output path (required via config/env)')
    p.add_argument('--out', default=os.getenv('PATH_MOMENTUM_OUT'), help='Weekly output CSV (required via config/env)')
    args = p.parse_args()

    export_weekly(features_path=args.features, model_path=args.model, out_path=args.out)


if __name__ == '__main__':
    main()


