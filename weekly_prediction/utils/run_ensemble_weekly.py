import os
import logging
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ensemble.ensemble_trainer import EnsembleTrainer
from utils.penalty import compute_penalty_weight
from utils.config import load_config, apply_env_from_config
from utils.logging_utils import configure_logging


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')


class StaticPredictor:
    def __init__(self, preds: np.ndarray):
        self._y = np.asarray(preds, dtype=float).ravel()
    def predict(self, X: np.ndarray):
        n = len(X) if hasattr(X, '__len__') else self._y.size
        y = self._y
        if y.size != n:
            m = min(y.size, n)
            return y[:m]
        return y


def _read_xgb(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    # Prefer adjusted score if present
    cand = ['xgb_score_adj','xgb_score','xgb_pred','xgb_predicted_return_pct','predicted_return_pct']
    for c in cand:
        if c in df.columns:
            df['xgb_pred'] = _num(df[c])
            break
    # Optional confidence
    for c in ['xgb_confidence_score_adj','xgb_confidence_score','confidence_score']:
        if c in df.columns:
            df['confidence_score_xgb'] = _num(df[c])
            break
    return df[['ticker','xgb_pred','confidence_score_xgb']].copy() if 'confidence_score_xgb' in df.columns else df[['ticker','xgb_pred']].copy()


def _read_lstm(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    cand = ['lstm_pred_adj','lstm_pred','lstm_predicted_return_pct']
    for c in cand:
        if c in df.columns:
            df['lstm_pred'] = _num(df[c])
            break
    for c in ['confidence_score_adj','confidence_score']:
        if c in df.columns:
            df['confidence_score_lstm'] = _num(df[c])
            break
    return df[['ticker','lstm_pred','confidence_score_lstm']].copy() if 'confidence_score_lstm' in df.columns else df[['ticker','lstm_pred']].copy()


def _latest_features(features_path: str) -> pd.DataFrame:
    """Read features CSV and return the latest snapshot per ticker.

    Returns an empty DataFrame on failure.
    """
    if not isinstance(features_path, str) or not os.path.isfile(features_path):
        return pd.DataFrame()
    try:
        feats = pd.read_csv(features_path)
        feats['ticker'] = feats['ticker'].astype(str).str.upper()
        if 'date' in feats.columns:
            feats['date'] = pd.to_datetime(feats['date'], errors='coerce')
            feats = feats.sort_values('date').groupby('ticker').tail(1)
        return feats
    except Exception:
        return pd.DataFrame()


def _penalty_weight_map(feats_latest: pd.DataFrame) -> dict:
    """Build a ticker->penalty weight map from the latest features snapshot."""
    if feats_latest is None or feats_latest.empty:
        return {}
    out = {}
    for _, row in feats_latest.iterrows():
        t = str(row.get('ticker')).upper()
        rsi = float(pd.to_numeric(row.get('rsi_14d', 50.0), errors='coerce'))
        vr = float(pd.to_numeric(row.get('volume_ratio', 1.0), errors='coerce'))
        pc = float(pd.to_numeric(row.get('price_change_pct', row.get('return_1d', 0.0)), errors='coerce'))
        out[t] = compute_penalty_weight(rsi, vr, pc)
    return out


def run_ensemble(xgb_path: str,
                 lstm_path: str,
                 features_path: str,
                 method: str,
                 alpha: float,
                 stack_model: str | None,
                 out_path: str) -> str:
    xgb = _read_xgb(xgb_path)
    lstm = _read_lstm(lstm_path)

    # Build per-model tickers and preds for alignment inside EnsembleTrainer
    models = []
    Xs = []
    tickers_lists = []
    frames_for_align: list[pd.DataFrame] = []

    if 'xgb_pred' in xgb.columns and len(xgb) > 0:
        xgb_pred = _num(xgb['xgb_pred']).to_numpy()
        models.append(StaticPredictor(xgb_pred))
        Xs.append(np.zeros((len(xgb_pred), 1), dtype=float))
        tickers_lists.append(list(xgb['ticker'].astype(str)))
        frames_for_align.append(xgb[['ticker','xgb_pred']].copy())
        if 'confidence_score_xgb' in xgb.columns:
            frames_for_align[-1]['confidence_score_xgb'] = _num(xgb['confidence_score_xgb'])

    if 'lstm_pred' in lstm.columns and len(lstm) > 0:
        lstm_pred = _num(lstm['lstm_pred']).to_numpy()
        models.append(StaticPredictor(lstm_pred))
        Xs.append(np.zeros((len(lstm_pred), 1), dtype=float))
        tickers_lists.append(list(lstm['ticker'].astype(str)))
        frames_for_align.append(lstm[['ticker','lstm_pred']].copy())
        if 'confidence_score_lstm' in lstm.columns:
            frames_for_align[-1]['confidence_score_lstm'] = _num(lstm['confidence_score_lstm'])

    if not models:
        raise RuntimeError('No valid predictions to ensemble')

    weights = [alpha, 1 - alpha] if len(models) == 2 else None
    ens = EnsembleTrainer(models=models, weights=weights, method=method)
    # Align by ticker using intersection
    ens_scores = ens.predict(Xs, tickers=tickers_lists, fill_missing_value=None)

    # Construct aligned dataframe via inner-join (intersection) and log dropped
    df = frames_for_align[0]
    for f in frames_for_align[1:]:
        before_tks = set(df['ticker'].astype(str))
        next_tks = set(f['ticker'].astype(str))
        merged = df.merge(f, on='ticker', how='inner')
        dropped = len(before_tks | next_tks) - len(before_tks & next_tks)
        if dropped > 0:
            logging.getLogger(__name__).warning(f"[ENSEMBLE WARN] Dropped {dropped} tickers due to missing predictions in one model")
        df = merged
    # ret proxy: max of available base preds
    ret_proxy = None
    for c in ['xgb_pred','lstm_pred']:
        if c in df.columns:
            series = _num(df[c])
            ret_proxy = series if ret_proxy is None else np.maximum(ret_proxy, series)
    if ret_proxy is None:
        ret_proxy = pd.Series(0.0, index=df.index)
    df['_ret_proxy_'] = ret_proxy

    # Assign ensemble score aligned to df rows
    df['ensemble_score'] = pd.Series(ens_scores, index=df.index)

    # Stacking (optional)
    stack_path = stack_model or os.getenv('ENSEMBLE_STACK_MODEL')
    if isinstance(stack_path, str) and os.path.isfile(stack_path):
        try:
            import joblib
            feat_cols = [c for c in ['xgb_pred','lstm_pred','ensemble_score'] if c in df.columns]
            # include confidences if present
            if 'confidence_score_xgb' in df.columns:
                df['_xgb_conf'] = _num(df['confidence_score_xgb'])
                feat_cols.append('_xgb_conf')
            if 'confidence_score_lstm' in df.columns:
                df['_lstm_conf'] = _num(df['confidence_score_lstm'])
                feat_cols.append('_lstm_conf')
            Xstack = df[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
            mdl = joblib.load(stack_path)
            df['ensemble_score'] = _num(pd.Series(mdl.predict(Xstack)))
            logging.getLogger(__name__).info(f"[ENSEMBLE] Applied stacking model from {stack_path}")
        except Exception as e:
            logging.getLogger(__name__).error(f"[ENSEMBLE ERROR] Failed stacking: {e}; falling back to ensemble score")

    # Post-ensemble continuous penalty
    feats_latest = _latest_features(features_path)
    wmap = _penalty_weight_map(feats_latest)
    if wmap:
        w = df['ticker'].astype(str).str.upper().map(lambda t: wmap.get(t, 1.0)).astype(float)
        df['w_pen'] = w
        df['ensemble_score'] = _num(df['ensemble_score']) * w
    else:
        logging.getLogger(__name__).warning("[ENSEMBLE WARN] No valid features found; skipping penalty weights")

    # Filter positive-return candidates and take Top-N
    try:
        df = df[_num(df['_ret_proxy_']) > 0].copy()
    except Exception as e:
        logging.getLogger(__name__).warning(f"[ENSEMBLE WARN] Failed to filter positive returns: {e}; keeping all rows")
        df = df.copy()
    df = df.sort_values('ensemble_score', ascending=False)
    top_n = int(os.getenv('ENSEMBLE_TOP_N', '25'))
    df = df.head(top_n)

    # Select columns for output
    keep = [c for c in [
        'ticker',
        'xgb_pred', 'confidence_score_xgb',
        'lstm_pred', 'confidence_score_lstm',
        'ensemble_score','w_pen'
    ] if c in df.columns]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df[keep].to_csv(out_path, index=False)
    logging.getLogger(__name__).info(f"[ENSEMBLE] Wrote ensemble weekly output to {out_path} (rows={len(df)})")
    return out_path


def main():
    p = argparse.ArgumentParser(description='Ensemble XGB and LSTM weekly outputs with penalties and optional stacking')
    p.add_argument('--xgb', default=os.getenv('PATH_XGB_OUT', 'data/xgboost/xgboost_weekly_output.csv'), help='Path to XGB weekly output CSV')
    p.add_argument('--lstm', default=os.getenv('PATH_LSTM_OUT', 'data/lstm/lstm_weekly_output.csv'), help='Path to LSTM weekly output CSV')
    p.add_argument('--features', default=os.getenv('PATH_FEATURES_CLEAN', 'data/features/stock_features_clean.csv'), help='Features CSV for penalty computation (latest snapshot)')
    p.add_argument('--method', choices=['weighted','rank','voting','prob'], default=os.getenv('ENSEMBLE_METHOD','weighted'), help='Blending method')
    p.add_argument('--alpha', type=float, default=float(os.getenv('ENSEMBLE_ALPHA','0.6')), help='Weight on XGB component (0..1)')
    p.add_argument('--stack-model', default=os.getenv('ENSEMBLE_STACK_MODEL'), help='Optional joblib stacker')
    p.add_argument('--out', default=os.getenv('PATH_ENSEMBLE_OUT', 'data/ensemble/ensemble_weekly_output.csv'), help='Output CSV path')
    args = p.parse_args()
    # Optional config + logging setup when running standalone
    cfg_path = os.getenv('APP_CONFIG', 'config.yaml')
    cfg = load_config(cfg_path)
    apply_env_from_config(cfg)
    configure_logging()

    run_ensemble(
        xgb_path=args.xgb,
        lstm_path=args.lstm,
        features_path=args.features,
        method=str(args.method).lower(),
        alpha=float(args.alpha),
        stack_model=args.stack_model,
        out_path=args.out,
    )


if __name__ == '__main__':
    main()


