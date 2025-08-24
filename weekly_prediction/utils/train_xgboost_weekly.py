import os
import logging
import sys
from pathlib import Path
import argparse
import pandas as pd
from typing import Optional

# Ensure project root is on sys.path so `models` can be imported when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.xgboost.xgb_trainer import XGBTrainer
from utils.config import load_config, apply_env_from_config
from utils.logging_utils import configure_logging
from utils.penalty import compute_penalty_weight


def train_and_export(features_path: str, out_path: str = None, horizon: int = 5) -> Optional[str]:
    # Ensure output directory exists
    out_path = out_path or os.getenv('PATH_XGB_OUT', 'data/xgboost/xgboost_weekly_output.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Train ranker using provided features
    trainer = XGBTrainer(features_path)
    model, preds = trainer.train()

    # Build weekly output: latest per-date scored snapshot from test_df
    try:
        test_df = trainer.test_df.copy()
    except Exception:
        logging.getLogger(__name__).error('[XGB WEEKLY] Trainer missing test_df; cannot export.')
        return None

    # Basic validation of training outputs
    if model is None or preds is None:
        logging.getLogger(__name__).error('[XGB WEEKLY] Training returned no model or predictions; aborting.')
        return None
    if len(preds) != len(test_df):
        logging.getLogger(__name__).error(f"[XGB WEEKLY] Pred length {len(preds)} != test_df length {len(test_df)}; aborting.")
        return None

    test_df = test_df.sort_values(['date'])
    test_df['_pred'] = preds
    # Keep latest row per ticker (handles irregular dates)
    weekly = test_df.groupby('ticker').tail(1).copy()

    # Compute signals and confidence similar to selector logic
    weekly = weekly.copy()
    # Positive signals
    pos_signals = [
        'broke_resistance', 'breakout_confirmed', 'strong_momentum', 'post_earnings_dip_rally'
    ]
    present_signals = [c for c in pos_signals if c in weekly.columns]
    if not present_signals:
        logging.getLogger(__name__).warning('[XGB WEEKLY WARN] No positive signal columns found; setting signal_pos_count to 0')
        weekly['signal_pos_count'] = 0
    else:
        weekly['signal_pos_count'] = weekly[present_signals].astype(int).sum(axis=1)
    # Momentum counts from available columns
    def _pick_first_present_row(row: pd.Series, candidates: list[str], priority: list[str] | None = None) -> float:
        """Pick the first present, non-NaN value in priority order.

        Priority defaults to the candidates list order for explicit, stable preference.
        """
        order = priority or candidates
        for c in order:
            if c in row.index and pd.notna(row.get(c)):
                return float(row.get(c))
        return 0.0
    # Prefer momentum_*d → return_*d → price_change_*d_pct (explicit priority lists)
    m5 = weekly.apply(lambda r: _pick_first_present_row(r, ['momentum_5d','return_5d','price_change_5d_pct']), axis=1)
    m10 = weekly.apply(lambda r: _pick_first_present_row(r, ['momentum_10d','return_10d','price_change_10d_pct']), axis=1)
    m20 = weekly.apply(lambda r: _pick_first_present_row(r, ['momentum_20d','return_20d','price_change_20d_pct']), axis=1)
    weekly['momentum_pos_count'] = (pd.to_numeric(m5, errors='coerce') > 0).astype(int) \
                                 + (pd.to_numeric(m10, errors='coerce') > 0).astype(int) \
                                 + (pd.to_numeric(m20, errors='coerce') > 0).astype(int)
    # Recent return score (rank-normalized across last-date universe)
    def _pick_series(df: pd.DataFrame, cols: list[str]) -> pd.Series:
        """Build a single series by choosing the first available column in order.

        For each row, keeps the first column's value that exists and is numeric.
        """
        out = pd.Series(0.0, index=df.index)
        for c in cols:
            if c in df.columns:
                s = pd.to_numeric(df[c], errors='coerce')
                out = out.where(~out.eq(0.0), s)
        return out.fillna(0.0)
    s5_raw = _pick_series(weekly, ['price_change_5d_pct','return_5d','momentum_5d'])
    s15_raw = _pick_series(weekly, ['price_change_15d_pct','return_15d','momentum_15d'])
    s30_raw = _pick_series(weekly, ['price_change_30d_pct','return_30d','momentum_30d'])
    N = max(len(weekly), 1)
    score_5 = 1.0 - (s5_raw.rank(method='min', ascending=False) - 1.0) / float(N)
    score_15 = 1.0 - (s15_raw.rank(method='min', ascending=False) - 1.0) / float(N)
    score_30 = 1.0 - (s30_raw.rank(method='min', ascending=False) - 1.0) / float(N)
    weekly['recent_return_score'] = (0.4 * score_5 + 0.3 * score_15 + 0.3 * score_30).astype(float)

    # Base XGB score
    weekly = weekly.rename(columns={'_pred': 'xgb_score'})
    # Confidence mirroring selector formula
    weekly['xgb_confidence_score'] = (
        pd.to_numeric(weekly['xgb_score'], errors='coerce').fillna(0.0)
        * (1 + 0.5 * weekly['signal_pos_count'] + 0.5 * weekly['momentum_pos_count'])
        * (1 + weekly['recent_return_score'])
    )

    # Ensure required feature columns (fallbacks if missing)
    if 'price_change_pct' not in weekly.columns and 'return_1d' in weekly.columns:
        weekly['price_change_pct'] = pd.to_numeric(weekly['return_1d'], errors='coerce')
    if 'rsi_14d' not in weekly.columns and 'rsi' in weekly.columns:
        weekly['rsi_14d'] = pd.to_numeric(weekly['rsi'], errors='coerce')
    if 'volume_ratio' not in weekly.columns and 'volratio' in weekly.columns:
        weekly['volume_ratio'] = pd.to_numeric(weekly['volratio'], errors='coerce')
    # Continuous penalty weight using centralized function
    weekly['w_pen'] = weekly.apply(lambda r: compute_penalty_weight(
        float(pd.to_numeric(r.get('rsi_14d', 50.0), errors='coerce')),
        float(pd.to_numeric(r.get('volume_ratio', 1.0), errors='coerce')),
        float(pd.to_numeric(r.get('price_change_pct', 0.0), errors='coerce'))
    ), axis=1)
    # Apply penalty to scores
    weekly['xgb_score_adj'] = pd.to_numeric(weekly['xgb_score'], errors='coerce') * weekly['w_pen']
    weekly['xgb_confidence_score_adj'] = pd.to_numeric(weekly['xgb_confidence_score'], errors='coerce') * weekly['w_pen']

    # Optional debug dump
    if os.getenv('DEBUG_SELLOFF', '0') == '1':
        try:
            dbg_cols = [c for c in ['ticker','xgb_score','xgb_score_adj','xgb_confidence_score','xgb_confidence_score_adj','w_pen','rsi_14d','volume_ratio','price_change_pct','selloff_flag'] if c in weekly.columns]
            os.makedirs('data/xgboost', exist_ok=True)
            weekly[dbg_cols].sort_values('xgb_score_adj', ascending=False).to_csv('data/xgboost/debug_xgb_weekly_penalties.csv', index=False)
            logging.getLogger(__name__).info('[DEBUG] Wrote XGB weekly penalty debug to data/xgboost/debug_xgb_weekly_penalties.csv')
        except Exception as e:
            logging.getLogger(__name__).warning(f"[DEBUG] Failed to write penalty debug: {e}")

    # Select useful columns and sort by adjusted score
    keep_cols = [c for c in ['ticker', 'date', 'close', 'volume', 'xgb_score', 'xgb_score_adj', 'xgb_confidence_score', 'xgb_confidence_score_adj', 'w_pen'] if c in weekly.columns]
    weekly = weekly[keep_cols].sort_values('xgb_score_adj' if 'xgb_score_adj' in weekly.columns else 'xgb_score', ascending=False)
    weekly.to_csv(out_path, index=False)
    logging.getLogger(__name__).info(f"[XGB WEEKLY] Wrote weekly output to {out_path} (rows={len(weekly)})")
    return out_path


def main():
    p = argparse.ArgumentParser(description='Train XGBoost ranker and export weekly output')
    p.add_argument('--features', default=os.getenv('PATH_FEATURES_CLEAN', 'data/features/stock_features_clean.csv'), help='Path to features CSV (clean)')
    p.add_argument('--out', default=os.getenv('PATH_XGB_OUT', 'data/xgboost/xgboost_weekly_output.csv'), help='Path to write weekly output CSV')
    p.add_argument('--horizon', type=int, default=5, help='Horizon days for target')
    args = p.parse_args()
    # Optional config + logging setup when running standalone
    cfg_path = os.getenv('APP_CONFIG', 'config.yaml')
    cfg = load_config(cfg_path)
    apply_env_from_config(cfg)
    configure_logging()
    train_and_export(args.features, args.out, args.horizon)


if __name__ == '__main__':
    main()


