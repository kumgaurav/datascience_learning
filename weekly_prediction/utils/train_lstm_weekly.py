import os
import logging
import sys
from pathlib import Path
import argparse
import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `models` can be imported when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lstm.lstm_trainer import LSTMTrainer
from utils.penalty import compute_penalty_weight
from utils.config import load_config, apply_env_from_config
from utils.logging_utils import configure_logging


def _resolve_model_paths(model_path: Optional[str], meta_path: Optional[str]) -> Tuple[str, str]:
    # Read strictly from provided args or env set via config; no silent fallbacks
    mp = model_path or os.getenv('LSTM_MODEL_PATH')
    jp = meta_path or os.getenv('LSTM_META_PATH')
    if not mp:
        raise FileNotFoundError("LSTM model path not provided. Set lstm.model_path in config.yaml or LSTM_MODEL_PATH env.")
    if not jp:
        raise FileNotFoundError("LSTM meta path not provided. Set lstm.meta_path in config.yaml or LSTM_META_PATH env.")
    if not os.path.isfile(mp):
        raise FileNotFoundError(f"LSTM model not found at {mp}")
    if not os.path.isfile(jp):
        raise FileNotFoundError(f"LSTM meta not found at {jp}")
    logging.getLogger(__name__).info(f"[LSTM WEEKLY] Using model={mp} meta={jp}")
    return mp, jp


# penalty weight logic centralized in utils.penalty.compute_penalty_weight


def _build_latest_sequences(df: pd.DataFrame, feature_cols: list[str], lookback: int) -> Tuple[list[str], np.ndarray, pd.DataFrame]:
    """Build normalized latest sequences per ticker.

    Prepares the most recent `lookback` rows for each ticker, normalizes features per-window,
    and returns (tickers, X_batch, latest_df).
    Sequences shorter than `lookback` are optionally padded if LSTM_PAD_SHORT_SEQS env is true.
    """
    df = df.copy()
    df['ticker'] = df['ticker'].astype(str).str.upper()
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception:
        pass
    # Basic input validations
    if df.duplicated(subset=['ticker', 'date']).any():
        dup_cnt = int(df.duplicated(subset=['ticker', 'date']).sum())
        raise ValueError(f"[LSTM WEEKLY] Found {dup_cnt} duplicate (ticker,date) rows; please deduplicate inputs.")

    latest = df.sort_values('date').groupby('ticker').tail(lookback)
    seq_map = {}
    tickers, X_list = [], []
    eps = 1e-8
    dropped = 0
    pad_enabled = os.getenv('LSTM_PAD_SHORT_SEQS', '0') in ('1', 'true', 'True')
    for tkr, grp in latest.groupby('ticker'):
        g = grp.sort_values('date')
        if len(g) < lookback:
            if pad_enabled and len(g) > 0:
                need = lookback - len(g)
                last_row = g.tail(1).copy()
                pad_block = pd.concat([last_row] * need, ignore_index=True)
                g = pd.concat([g, pad_block], ignore_index=True)
            else:
                dropped += 1
                continue
        for c in feature_cols:
            if c not in g.columns:
                g[c] = 0.0
        window = g[feature_cols].tail(lookback).to_numpy(dtype=float)
        if window.shape[0] != lookback:
            # Should be unreachable given earlier checks; guard anyway
            dropped += 1
            continue
        # Fail-fast: zero variance across all features in window (degenerate input)
        if np.allclose(np.nanstd(window, axis=0), 0.0):
            raise ValueError(f"[LSTM WEEKLY] Zero variance across all features for ticker {tkr} in last {lookback} rows; check inputs.")
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.nanmean(window, axis=0)
            std = np.nanstd(window, axis=0, ddof=0)
            std_safe = np.where((std < eps) | ~np.isfinite(std), 1.0, std)
            normed = np.divide(window - mean, std_safe, out=np.zeros_like(window), where=std_safe != 0)
            normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)
        tickers.append(str(tkr))
        X_list.append(normed.reshape((1, normed.shape[0], normed.shape[1])))
        seq_map[str(tkr)] = normed
    if dropped > 0:
        logging.getLogger(__name__).warning(f"[LSTM WEEKLY] Dropped {dropped} tickers with insufficient data (< {lookback} rows)")
    if not X_list:
        return [], np.empty((0, lookback, len(feature_cols))), df
    X_batch = np.concatenate(X_list, axis=0)
    return tickers, X_batch, df


def train_and_export(features_path: str,
                     out_path: str = None,
                     lookback: Optional[int] = None,
                     horizon: Optional[int] = None,
                     skip_train: bool = False,
                     model_path: Optional[str] = None,
                     meta_path: Optional[str] = None) -> Optional[str]:
    out_path = out_path or os.getenv('PATH_LSTM_OUT', 'data/lstm/lstm_weekly_output.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Optionally configure GPU memory growth for TensorFlow (best-effort)
    def _maybe_configure_gpu():
        try:
            if os.getenv('TF_ENABLE_MEMORY_GROWTH', '1') in ('1','true','True'):
                import tensorflow as tf  # type: ignore
                gpus = getattr(tf.config, 'list_physical_devices', lambda *_: [])('GPU')
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception:
                        pass
        except Exception:
            pass

    _maybe_configure_gpu()

    # Train or load model
    if not skip_train:
        trainer = LSTMTrainer(features_path)
        # Apply YAML-configured hyperparameters if provided via env
        try:
            mc = trainer.model_config
            if os.getenv('LSTM_LOOKBACK'): mc['lookback'] = int(os.getenv('LSTM_LOOKBACK'))
            if os.getenv('LSTM_HORIZON'): mc['horizon'] = int(os.getenv('LSTM_HORIZON'))
            if os.getenv('LSTM_EPOCHS'): mc['epochs'] = int(os.getenv('LSTM_EPOCHS'))
            if os.getenv('LSTM_BATCH_SIZE'): mc['batch_size'] = int(os.getenv('LSTM_BATCH_SIZE'))
            if os.getenv('LSTM_VALIDATION_SPLIT'): mc['validation_split'] = float(os.getenv('LSTM_VALIDATION_SPLIT'))
            if os.getenv('LSTM_EARLY_STOPPING_PATIENCE'): mc['early_stopping_patience'] = int(os.getenv('LSTM_EARLY_STOPPING_PATIENCE'))
            if os.getenv('LSTM_REDUCE_LR_PATIENCE'): mc['reduce_lr_patience'] = int(os.getenv('LSTM_REDUCE_LR_PATIENCE'))
            if os.getenv('LSTM_USE_SEQUENCE_WEIGHTS'): mc['use_sequence_weights'] = os.getenv('LSTM_USE_SEQUENCE_WEIGHTS') in ('1','true','True')
            if os.getenv('LSTM_USE_PENALTY_WEIGHTS'): mc['use_penalty_weights'] = os.getenv('LSTM_USE_PENALTY_WEIGHTS') in ('1','true','True')
            if os.getenv('LSTM_PENALTY_WEIGHT_MODE'): mc['penalty_weight_mode'] = str(os.getenv('LSTM_PENALTY_WEIGHT_MODE')).lower()
            if os.getenv('LSTM_USE_CURRICULUM'): mc['use_curriculum'] = os.getenv('LSTM_USE_CURRICULUM') in ('1','true','True')
            if os.getenv('LSTM_CURRICULUM_WARMUP_EPOCHS'): mc['curriculum_warmup_epochs'] = int(os.getenv('LSTM_CURRICULUM_WARMUP_EPOCHS'))
            if os.getenv('LSTM_BIDIRECTIONAL'): mc['bidirectional'] = os.getenv('LSTM_BIDIRECTIONAL') in ('1','true','True')
            trainer.model_config = mc
        except Exception:
            pass
        model, _ = trainer.train(lookback=lookback, horizon=horizon)
        # Derive feature_cols and lookback from trained model when possible
        try:
            feature_cols = list(getattr(model, 'feature_cols'))
            lb = int(getattr(model, 'lookback'))
        except Exception:
            feature_cols = None
            lb = int(lookback or 60)
    else:
        # Load existing model + meta
        mp, jp = _resolve_model_paths(model_path, meta_path)
        # Best-effort: force CPU to avoid macOS/Metal segfaults during inference
        try:
            import tensorflow as _tf  # type: ignore
            _tf.config.set_visible_devices([], 'GPU')
            try:
                _tf.config.threading.set_intra_op_parallelism_threads(1)
                _tf.config.threading.set_inter_op_parallelism_threads(1)
            except Exception:
                pass
        except Exception:
            pass
        try:
            from tensorflow import keras
            model = keras.models.load_model(mp)
        except Exception as e:
            logging.getLogger(__name__).error(f"[LSTM WEEKLY] Failed to load model at {mp}: {e}")
            raise RuntimeError(f"Unable to load LSTM model at {mp}")
        try:
            with open(jp, 'r') as f:
                meta = json.load(f)
            feature_cols = meta.get('feature_cols')
            lb = int(meta.get('lookback', lookback or 60))
        except Exception as e:
            logging.getLogger(__name__).error(f"[LSTM WEEKLY] Failed to load meta at {jp}: {e}")
            raise RuntimeError(f"Unable to load LSTM meta at {jp}")

    # Load features dataframe (must contain history per ticker)
    df = pd.read_csv(features_path)
    if 'ticker' not in df.columns or 'date' not in df.columns:
        logging.getLogger(__name__).error("[LSTM WEEKLY] Features must include 'ticker' and 'date'.")
        return None
    # Infer feature columns if missing
    if not feature_cols:
        # Allow explicit list via env var (comma-separated)
        env_cols = os.getenv('LSTM_FEATURE_COLS')
        if env_cols:
            requested = [c.strip() for c in env_cols.split(',') if c.strip()]
            feature_cols = [c for c in requested if c in df.columns]
        else:
            # Use expected feature set from feature_engineering output
            expected = [
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
            feature_cols = [c for c in expected if c in df.columns]
        if not feature_cols:
            raise ValueError("No valid feature columns found in DataFrame")

    # If model expects cross-sectional z-scores ("*_csz"), generate them if missing
    try:
        if feature_cols:
            need_csz = [c for c in feature_cols if c.endswith('_csz') and c not in df.columns]
            if need_csz:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                for c in need_csz:
                    base = c[:-4]
                    if base in df.columns:
                        df[c] = df.groupby('date')[base].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
                # Any still-missing *_csz after attempt will be filled later in window builder
    except Exception:
        pass

    # Build latest sequences and predict
    tickers, X_batch, latest_df = _build_latest_sequences(df, feature_cols, lb)
    if len(tickers) == 0:
        logging.getLogger(__name__).warning('[LSTM WEEKLY] No valid sequences built; check lookback and feature coverage.')
        return None
    try:
        y_pred = model.predict(X_batch, verbose=0).ravel()
    except Exception as e:
        logging.getLogger(__name__).error(f"[LSTM WEEKLY] Batch prediction failed: {e}")
        raise RuntimeError("Prediction failed; cannot generate output")

    out = pd.DataFrame({'ticker': tickers, 'lstm_pred': y_pred})
    out['lstm_predicted_return_pct'] = out['lstm_pred']

    # Confidence via p95 scaling of absolute predictions
    try:
        abs_vals = np.abs(pd.to_numeric(out['lstm_pred'], errors='coerce'))
        p95 = float(np.nanpercentile(abs_vals, 95)) if np.isfinite(abs_vals).any() else 0.0
        scale = p95 if p95 > 1e-8 else (float(np.nanmax(abs_vals)) if np.nanmax(abs_vals) > 1e-8 else 1.0)
    except Exception:
        scale = 1.0
    out['confidence_score'] = (100.0 * (np.abs(pd.to_numeric(out['lstm_pred'], errors='coerce')) / scale)).clip(0.0, 100.0)
    # Guard: if predictions are constant, confidence is not informative
    try:
        ap = np.abs(pd.to_numeric(out['lstm_pred'], errors='coerce')).to_numpy()
        if ap.size > 0 and float(np.nanmax(ap)) == float(np.nanmin(ap)):
            logging.getLogger(__name__).warning('[LSTM WEEKLY] Predictions are constant; setting confidence_score to 0 and logging warning.')
            out['confidence_score'] = 0.0
    except Exception:
        pass

    # Attach latest per-ticker features required for penalties
    try:
        latest_snap = latest_df.sort_values('date').groupby('ticker').tail(1)
        latest_snap['ticker'] = latest_snap['ticker'].astype(str).str.upper()
        cols_needed = [c for c in ['date','close','rsi_14d','volume_ratio','price_change_pct','selloff_flag'] if c in latest_snap.columns]
        out = out.merge(latest_snap[['ticker'] + cols_needed], on='ticker', how='left')
    except Exception:
        pass

    # Compute continuous penalty weight and adjust scores
    out['w_pen'] = out.apply(lambda r: compute_penalty_weight(
        float(pd.to_numeric(r.get('rsi_14d', 50.0), errors='coerce')),
        float(pd.to_numeric(r.get('volume_ratio', 1.0), errors='coerce')),
        float(pd.to_numeric(r.get('price_change_pct', 0.0), errors='coerce'))
    ), axis=1)
    out['lstm_pred_adj'] = pd.to_numeric(out['lstm_pred'], errors='coerce') * out['w_pen']
    out['confidence_score_adj'] = pd.to_numeric(out['confidence_score'], errors='coerce') * out['w_pen']

    # Debug penalties
    if os.getenv('DEBUG_SELLOFF', '0') == '1':
        try:
            dbg_cols = [c for c in ['ticker','date','lstm_pred','lstm_pred_adj','confidence_score','confidence_score_adj','w_pen','rsi_14d','volume_ratio','price_change_pct','selloff_flag'] if c in out.columns]
            os.makedirs('data/lstm', exist_ok=True)
            out[dbg_cols].sort_values('lstm_pred_adj', ascending=False).to_csv('data/lstm/debug_lstm_weekly_penalties.csv', index=False)
            logging.getLogger(__name__).info('[DEBUG] Wrote LSTM weekly penalty debug to data/lstm/debug_lstm_weekly_penalties.csv')
        except Exception:
            pass

    # Order and write
    keep = [c for c in ['ticker','date','close','lstm_pred','lstm_predicted_return_pct','confidence_score','w_pen','lstm_pred_adj','confidence_score_adj'] if c in out.columns]
    out = out[keep].sort_values('lstm_pred_adj' if 'lstm_pred_adj' in out.columns else 'lstm_pred', ascending=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    logging.getLogger(__name__).info(f"[LSTM WEEKLY] Wrote weekly LSTM output to {out_path} (rows={len(out)})")
    return out_path


def main():
    p = argparse.ArgumentParser(description='Train LSTM (optional) and export weekly predictions')
    p.add_argument('--features', default=os.getenv('PATH_FEATURES_CLEAN', 'data/features/stock_features_clean.csv'), help='Path to historical features CSV')
    p.add_argument('--out', default=os.getenv('PATH_LSTM_OUT', 'data/lstm/lstm_weekly_output.csv'), help='Path to write weekly LSTM output CSV')
    p.add_argument('--lookback', type=int, default=None, help='Sequence length (overrides model default)')
    p.add_argument('--horizon', type=int, default=None, help='Forward return horizon (overrides model default)')
    p.add_argument('--skip-train', action='store_true', help='Skip training and use existing saved model/meta')
    p.add_argument('--model', default=None, help='Path to saved Keras model (.keras) if skipping training')
    p.add_argument('--meta', default=None, help='Path to LSTM meta JSON if skipping training')
    args = p.parse_args()
    # Optional config + logging setup when running standalone
    cfg_path = os.getenv('APP_CONFIG', 'config.yaml')
    cfg = load_config(cfg_path)
    apply_env_from_config(cfg)
    configure_logging()
    train_and_export(
        features_path=args.features,
        out_path=args.out,
        lookback=args.lookback,
        horizon=args.horizon,
        skip_train=bool(args.skip_train),
        model_path=args.model,
        meta_path=args.meta,
    )


if __name__ == '__main__':
    main()


