import os
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

# Ensure project root on path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config, apply_env_from_config
from utils.logging_utils import configure_logging
from utils.train_lstm_weekly import _build_latest_sequences


EXPECTED_FEATURES = [
    'ma_20','ma_50','volatility_30d','rsi_14d',
    'macd','macd_signal','macd_histogram',
    'bb_upper','bb_middle','bb_lower','bb_position',
    'volume_ma_20','volume_ratio','pvt','obv',
    'drawdown_30d','var_95_30d','sharpe_ratio','trend_strength',
    'momentum_5d','momentum_10d','momentum_20d','momentum_30d','momentum_60d',
    'support_20d','resistance_20d','distance_from_support','distance_from_resistance','breakout_strength',
    'trend_slope_15d','trend_slope_30d','up_day_ratio_20d',
    'price_change_pct','vol_change_pct','rsi_price_interaction','volratio_price_interaction',
    'selloff_flag','overbought_spike','prev50d_high','broke_resistance','days_since_last_broke_resistance',
    'volume_zscore_20','volume_spike','gap_pct','gap_up_2pct','gap_down_2pct',
    'relative_strength_20d','vol_adj_momentum_20d'
]


@dataclass
class LSTMDiagnosticReport:
    features_path: str
    model_path: Optional[str]
    meta_path: Optional[str]
    lookback: int
    meta_feature_cols: Optional[List[str]]
    meta_lookback: Optional[int]
    feature_cols_match_meta: Optional[bool]
    feature_cols_order_match_meta: Optional[bool]
    feature_cols_used: List[str]
    missing_feature_cols: List[str]
    constant_or_zero_features: List[str]
    dropped_tickers: int
    built_sequences: int
    unique_sequences: int
    identical_sequences: bool
    preds_constant: Optional[bool]
    confidence_saturated: Optional[bool]
    suggestions: List[str]


def _load_meta_feature_cols(meta_path: Optional[str]) -> Optional[List[str]]:
    if not meta_path or not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        cols = meta.get('feature_cols')
        if isinstance(cols, list) and cols:
            return cols
    except Exception:
        pass
    return None


def _load_meta_lookback(meta_path: Optional[str]) -> Optional[int]:
    if not meta_path or not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        lb = meta.get('lookback')
        return int(lb) if lb is not None else None
    except Exception:
        return None


def _infer_feature_cols(df: pd.DataFrame, meta_path: Optional[str]) -> List[str]:
    # Priority 1: meta
    meta_cols = _load_meta_feature_cols(meta_path)
    if meta_cols:
        return [c for c in meta_cols if c in df.columns]
    # Priority 2: env
    env_cols = os.getenv('LSTM_FEATURE_COLS')
    if env_cols:
        requested = [c.strip() for c in env_cols.split(',') if c.strip()]
        cols = [c for c in requested if c in df.columns]
        if cols:
            return cols
    # Priority 3: expected list
    cols = [c for c in EXPECTED_FEATURES if c in df.columns]
    if cols:
        return cols
    # Fallback: numeric except identity/label
    exclude = {'ticker','date','target'}
    return [c for c in df.select_dtypes(include=['number','bool']).columns if c not in exclude]


def _sequence_hash(arr: np.ndarray) -> int:
    # Simple hash based on rounded values to detect identical sequences
    try:
        return hash(np.round(arr, 6).tobytes())
    except Exception:
        return hash(str(arr.shape))


def run_lstm_diagnostics(features_path: str,
                         model_path: Optional[str] = None,
                         meta_path: Optional[str] = None,
                         lookback: Optional[int] = None) -> LSTMDiagnosticReport:
    log = logging.getLogger(__name__)
    df = pd.read_csv(features_path)
    if 'ticker' not in df.columns or 'date' not in df.columns:
        raise ValueError("Features must include 'ticker' and 'date'")
    df['ticker'] = df['ticker'].astype(str).str.upper()
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception:
        pass

    # Determine feature columns and lookback
    feature_cols = _infer_feature_cols(df, meta_path)
    lb_env = os.getenv('LSTM_LOOKBACK')
    lb = int(lb_env) if lb_env else int(lookback or 60)

    # Meta comparisons
    meta_cols = _load_meta_feature_cols(meta_path)
    meta_lb = _load_meta_lookback(meta_path)
    match_meta = None
    match_order = None
    extra_vs_meta: List[str] = []
    missing_vs_meta: List[str] = []
    if meta_cols:
        set_used = set(feature_cols)
        set_meta = set(meta_cols)
        match_meta = (set_used == set_meta)
        match_order = (feature_cols == meta_cols)
        missing_vs_meta = [c for c in meta_cols if c not in set_used]
        extra_vs_meta = [c for c in feature_cols if c not in set_meta]

    # Column presence and constancy checks
    missing = [c for c in feature_cols if c not in df.columns]
    const_cols = []
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().sum() == 0:
            const_cols.append(c)
        else:
            v = float(np.nanvar(s.values))
            if not np.isfinite(v) or v == 0.0:
                const_cols.append(c)

    # Build latest sequences
    tickers, X_batch, latest_df = _build_latest_sequences(df, feature_cols, lb)
    dropped = int(df['ticker'].nunique() - len(tickers))
    built = int(len(tickers))
    # Identify identical sequences
    unique_hashes = set()
    if X_batch.size > 0:
        for i in range(X_batch.shape[0]):
            unique_hashes.add(_sequence_hash(X_batch[i]))
    identical = (len(unique_hashes) == 1 and built > 1)

    preds_constant = None
    confidence_saturated = None
    # If a model and recent outputs exist, check prediction distribution and confidence
    out_csv = os.getenv('PATH_LSTM_OUT', 'data/lstm/lstm_weekly_output.csv')
    try:
        if os.path.isfile(out_csv):
            out = pd.read_csv(out_csv)
            if 'lstm_pred' in out.columns:
                ap = np.abs(pd.to_numeric(out['lstm_pred'], errors='coerce'))
                if ap.size > 0:
                    preds_constant = bool(np.nanmax(ap) == np.nanmin(ap))
                    try:
                        conf = pd.to_numeric(out.get('confidence_score', pd.Series([])), errors='coerce')
                        if len(conf):
                            u = np.unique(np.round(conf, 6))
                            confidence_saturated = bool(len(u) == 1 and float(u[0]) == 100.0)
                    except Exception:
                        pass
    except Exception:
        pass

    suggestions: List[str] = []
    if missing:
        suggestions.append(f"Missing expected features: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if const_cols:
        suggestions.append(f"Constant/zero-variance features detected: {const_cols[:10]}{'...' if len(const_cols)>10 else ''}")
    if dropped > 0:
        suggestions.append(f"{dropped} tickers have < lookback={lb} rows; increase history or enable LSTM_PAD_SHORT_SEQS")
    if meta_cols is not None:
        if not match_meta:
            suggestions.append(f"Feature set differs from training meta: missing={missing_vs_meta[:10]} extra={extra_vs_meta[:10]}")
        elif match_meta and not match_order:
            suggestions.append("Feature set matches meta but order differs; reorder to match training meta.")
        if meta_lb is not None and meta_lb != lb:
            suggestions.append(f"Lookback mismatch: meta={meta_lb}, used={lb}; set LSTM_LOOKBACK to {meta_lb}.")
    if identical:
        suggestions.append("All sequences identical across tickers; check feature selection, fill strategy, and normalization.")
    if preds_constant:
        suggestions.append("Model predictions are constant across tickers; verify sequences differ and feature_cols match training meta.")
    if confidence_saturated:
        suggestions.append("Confidence scores saturated at 100. Consider alternative scaling when predictions are flat.")

    return LSTMDiagnosticReport(
        features_path=features_path,
        model_path=model_path,
        meta_path=meta_path,
        lookback=lb,
        meta_feature_cols=meta_cols,
        meta_lookback=meta_lb,
        feature_cols_match_meta=match_meta,
        feature_cols_order_match_meta=match_order,
        feature_cols_used=feature_cols,
        missing_feature_cols=missing,
        constant_or_zero_features=const_cols,
        dropped_tickers=dropped,
        built_sequences=built,
        unique_sequences=len(unique_hashes),
        identical_sequences=identical,
        preds_constant=preds_constant,
        confidence_saturated=confidence_saturated,
        suggestions=suggestions,
    )


def main():
    import argparse
    p = argparse.ArgumentParser(description='Run diagnostics on LSTM inputs and outputs')
    p.add_argument('--features', default=os.getenv('PATH_FEATURES_CLEAN', 'data/features/stock_features_clean.csv'))
    p.add_argument('--model', default=os.getenv('LSTM_MODEL_PATH'))
    p.add_argument('--meta', default=os.getenv('LSTM_META_PATH'))
    p.add_argument('--lookback', type=int, default=None)
    p.add_argument('--out', default='data/lstm/diagnostics_report.json')
    args = p.parse_args()

    # Config + logging
    cfg_path = os.getenv('APP_CONFIG', 'config.yaml')
    cfg = load_config(cfg_path)
    apply_env_from_config(cfg)
    configure_logging()
    log = logging.getLogger(__name__)

    try:
        rep = run_lstm_diagnostics(args.features, args.model, args.meta, args.lookback)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(asdict(rep), f, indent=2, default=str)
        log.info(f"[LSTM DIAG] Wrote diagnostic report to {args.out}")
        # Also print concise summary
        log.info(f"[LSTM DIAG] built_sequences={rep.built_sequences}, unique_sequences={rep.unique_sequences}, dropped={rep.dropped_tickers}")
        if rep.suggestions:
            for s in rep.suggestions:
                log.warning(f"[LSTM DIAG] Suggestion: {s}")
    except Exception as e:
        log.error(f"[LSTM DIAG] Failed diagnostics: {e}")


if __name__ == '__main__':
    main()


