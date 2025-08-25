import argparse
import os
import sys
import logging
from typing import Optional, List

import pandas as pd


def read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path) if os.path.exists(path) else None
    except Exception as e:
        logging.getLogger(__name__).warning("[READ] Failed to read %s: %s", path, e)
        return None


def normalize_tickers(df: pd.DataFrame) -> pd.DataFrame:
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    return df


def pick_feature_source(candidates: List[str]) -> Optional[pd.DataFrame]:
    for path in candidates:
        df = read_csv_safe(path)
        if df is not None and 'ticker' in df.columns:
            logging.getLogger(__name__).info("[FEATURES] Using feature source: %s (rows=%d)", path, len(df))
            df = normalize_tickers(df)
            # Collapse to latest per ticker if date exists
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.sort_values(['ticker', 'date']).drop_duplicates(subset=['ticker'], keep='last')
                except Exception:
                    pass
            return df
    logging.getLogger(__name__).warning('[FEATURES] No valid feature source found among candidates')
    return None


def build_ui_dataset(
    ensemble_path: str,
    feature_candidates: List[str],
    lstm_path: Optional[str],
    xgb_ranked_path: Optional[str],
    top_n: int,
    out_path: str,
    strict: bool,
) -> int:
    # 1) Read ensemble (source of truth, already scored)
    ens = read_csv_safe(ensemble_path)
    if ens is None or 'ticker' not in ens.columns or 'ensemble_score' not in ens.columns:
        logging.getLogger(__name__).error("[ERROR] Ensemble file missing or invalid: %s", ensemble_path)
        return 2
    ens = normalize_tickers(ens)
    ens['ensemble_score'] = pd.to_numeric(ens['ensemble_score'], errors='coerce')
    ens = ens.dropna(subset=['ensemble_score'])
    ens_top = ens.sort_values('ensemble_score', ascending=False).head(top_n).reset_index(drop=True)
    ens_top['ensemble_rank'] = ens_top.index + 1
    logging.getLogger(__name__).info('[STEP] Ensemble Top N (ticker, score):\n%s', ens_top[['ensemble_rank', 'ticker', 'ensemble_score']].to_string(index=False))

    # 2) Pick feature-rich source
    features = pick_feature_source(feature_candidates)
    if features is None:
        print('[ERROR] Could not find any suitable feature source')
        return 3

    # 3) Optional: attach XGB ranked columns for predicted_return_pct / confidence_score
    if xgb_ranked_path:
        xgb = read_csv_safe(xgb_ranked_path)
        if xgb is not None and 'ticker' in xgb.columns:
            xgb = normalize_tickers(xgb)
            # Disambiguate common columns to avoid overwriting feature fields
            ren = {}
            if 'confidence_score' in xgb.columns:
                ren['confidence_score'] = 'xgb_confidence_score'
            if 'predicted_return_pct' in xgb.columns:
                ren['predicted_return_pct'] = 'xgb_predicted_return_pct'
            xgb = xgb.rename(columns=ren)
            features = features.merge(xgb[['ticker'] + [c for c in ['xgb_confidence_score', 'xgb_predicted_return_pct'] if c in xgb.columns]], on='ticker', how='left')
            logging.getLogger(__name__).info("[STEP] Merged XGB ranked columns from %s", xgb_ranked_path)

    # 4) Optional: attach LSTM predictions
    if lstm_path:
        lstm = read_csv_safe(lstm_path)
        if lstm is not None and 'ticker' in lstm.columns:
            lstm = normalize_tickers(lstm)
            keep = [c for c in ['ticker', 'lstm_predicted_return_pct'] if c in lstm.columns]
            features = features.merge(lstm[keep], on='ticker', how='left')
            logging.getLogger(__name__).info("[STEP] Merged LSTM predictions from %s", lstm_path)
            # Alias for convenience
            if 'lstm_predicted_return_pct' in features.columns:
                features['lstm_pred'] = features['lstm_predicted_return_pct']
            # Derive LSTM confidence/risk/composite like XGB
            try:
                # Ensure numeric
                features['lstm_predicted_return_pct'] = pd.to_numeric(features['lstm_predicted_return_pct'], errors='coerce')
                abs_vals = features['lstm_predicted_return_pct'].abs().dropna()
                if not abs_vals.empty:
                    p95 = float(abs_vals.quantile(0.95))
                    scale = p95 if p95 > 1e-8 else float(abs_vals.max()) if abs_vals.max() > 1e-8 else 1.0
                else:
                    scale = 1.0
                # Confidence 0..100 based on relative magnitude
                features['lstm_confidence_score'] = (100.0 * (features['lstm_predicted_return_pct'].abs() / scale)).clip(lower=0.0, upper=100.0)
            except Exception:
                features['lstm_confidence_score'] = 50.0
            try:
                # Risk: reuse existing risk_score if present, else derive from volatility
                if 'risk_score' in features.columns:
                    features['lstm_risk_score'] = features['risk_score']
                elif 'volatility_20d_z' in features.columns:
                    v = pd.to_numeric(features['volatility_20d_z'], errors='coerce')
                    features['lstm_risk_score'] = (50.0 + 15.0 * v).clip(lower=0.0, upper=100.0)
                elif 'volatility_20d' in features.columns:
                    v = pd.to_numeric(features['volatility_20d'], errors='coerce')
                    # Scale by percentile
                    p90 = float(v.dropna().quantile(0.90)) if v.notna().any() else 1.0
                    features['lstm_risk_score'] = (100.0 * (v / (p90 if p90 > 1e-8 else 1.0))).clip(lower=0.0, upper=100.0)
                else:
                    features['lstm_risk_score'] = 50.0
            except Exception:
                features['lstm_risk_score'] = 50.0
            try:
                # Composite mirroring XGB weights
                features['lstm_composite_score'] = (
                    0.5 * features['lstm_predicted_return_pct'].fillna(0.0) +
                    0.3 * features['lstm_confidence_score'].fillna(0.0) +
                    0.2 * (100.0 - features['lstm_risk_score'].fillna(50.0))
                )
            except Exception:
                features['lstm_composite_score'] = 0.0

    # 5) Enforce close > $3 if available in features
    try:
        if 'close' in features.columns:
            features['close'] = pd.to_numeric(features['close'], errors='coerce')
            before = len(features)
            features = features[features['close'] > 3.0].copy()
            logging.getLogger(__name__).info("[STEP] Filtered features by close > $3: %d -> %d", before, len(features))
    except Exception:
        pass

    # 6) Strict INNER JOIN to preserve ensemble membership and ordering
    merged = ens_top[['ticker', 'ensemble_score', 'ensemble_rank']].merge(
        features.drop(columns=['ensemble_score'], errors='ignore'),
        on='ticker', how='inner'
    ).sort_values('ensemble_rank')

    # 7) Validate order
    ens_list = ens_top['ticker'].tolist()
    merged_list = merged['ticker'].tolist()
    order_match = ens_list == merged_list
    logging.getLogger(__name__).info('[CHECK] Order matches ensemble? %s', str(order_match))
    if not order_match:
        for i, (a, b) in enumerate(zip(ens_list, merged_list), 1):
            if a != b:
                logging.getLogger(__name__).warning("[DIFF] First mismatch at position %d: ensemble=%s vs merged=%s", i, a, b)
                break
        if strict:
            logging.getLogger(__name__).error('[ERROR] Strict mode enabled and order mismatch detected. Aborting.')
            return 4

    # 8) Write output
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    merged.to_csv(out_path, index=False)
    logging.getLogger(__name__).info("[DONE] Wrote UI dataset to: %s (rows=%d, cols=%d)", out_path, len(merged), len(merged.columns))
    return 0


def main() -> None:
    # Ensure project root; load config and configure logging
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from utils.config import load_config, apply_env_from_config  # type: ignore
    from utils.logging_utils import configure_logging  # type: ignore

    p = argparse.ArgumentParser(description='Build a UI-ready dataset strictly ordered by ensemble scores')
    p.add_argument('--config', default=os.getenv('APP_CONFIG', os.path.join(os.path.dirname(__file__), 'snapshot_config.yaml')), help='Path to YAML/JSON config for paths and logging')
    p.add_argument('--ensemble', default=None, help='Path to ensemble scores CSV')
    p.add_argument('--features', nargs='*', default=None, help='Candidate feature sources (first valid will be used)')
    p.add_argument('--lstm', default=None, help='Optional LSTM predictions CSV')
    p.add_argument('--xgb_ranked', default=None, help='Optional XGB ranked CSV')
    p.add_argument('--top_n', type=int, default=None, help='Top-N from ensemble to keep')
    p.add_argument('--out', default=None, help='Output CSV for UI')
    p.add_argument('--strict', action='store_true', help='Fail if merged order differs from ensemble')
    args = p.parse_args()

    cfg = load_config(args.config)
    apply_env_from_config(cfg)
    configure_logging(level=os.getenv('LOG_LEVEL'), log_file=os.getenv('LOG_FILE'))
    logger = logging.getLogger(__name__)

    # Resolve inputs from CLI > YAML paths
    paths = cfg.get('paths', {}) if isinstance(cfg, dict) else {}
    ensemble_path = args.ensemble or paths.get('ensemble_out') or 'data/ensemble/ensemble_weekly_output.csv'
    feature_candidates = args.features or [
        paths.get('features_clean'),
        'data/featured_stocks_top.csv',
        paths.get('xgb_weekly_output_csv'),
    ]
    feature_candidates = [p for p in feature_candidates if isinstance(p, str) and p]
    lstm_path = args.lstm or paths.get('lstm_out')
    xgb_ranked_path = args.xgb_ranked or paths.get('xgb_ranked_csv')
    top_n = args.top_n or int((cfg.get('ensemble') or {}).get('top_n', 25) if isinstance(cfg, dict) else 25)
    out_path = args.out or 'data/top/ui_dataset.csv'
    logger.info('Using ensemble=%s, lstm=%s, xgb_ranked=%s', ensemble_path, lstm_path, xgb_ranked_path)

    code = build_ui_dataset(
        ensemble_path=ensemble_path,
        feature_candidates=feature_candidates,
        lstm_path=lstm_path,
        xgb_ranked_path=xgb_ranked_path,
        top_n=top_n,
        out_path=out_path,
        strict=args.strict,
    )
    sys.exit(code)

    code = build_ui_dataset(
        ensemble_path=args.ensemble,
        feature_candidates=args.features,
        lstm_path=args.lstm,
        xgb_ranked_path=args.xgb_ranked,
        top_n=args.top_n,
        out_path=args.out,
        strict=args.strict,
    )
    sys.exit(code)


if __name__ == '__main__':
    main()


