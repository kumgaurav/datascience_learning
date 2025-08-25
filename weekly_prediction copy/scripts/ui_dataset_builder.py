import argparse
import os
import sys
from typing import Optional, List

import pandas as pd


def read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path) if os.path.exists(path) else None
    except Exception as e:
        print(f"[READ] Failed to read {path}: {e}")
        return None


def normalize_tickers(df: pd.DataFrame) -> pd.DataFrame:
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    return df


def pick_feature_source(candidates: List[str]) -> Optional[pd.DataFrame]:
    for path in candidates:
        df = read_csv_safe(path)
        if df is not None and 'ticker' in df.columns:
            print(f"[FEATURES] Using feature source: {path} (rows={len(df)})")
            df = normalize_tickers(df)
            # Collapse to latest per ticker if date exists
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.sort_values(['ticker', 'date']).drop_duplicates(subset=['ticker'], keep='last')
                except Exception:
                    pass
            return df
    print('[FEATURES] No valid feature source found among candidates')
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
        print(f"[ERROR] Ensemble file missing or invalid: {ensemble_path}")
        return 2
    ens = normalize_tickers(ens)
    ens['ensemble_score'] = pd.to_numeric(ens['ensemble_score'], errors='coerce')
    ens = ens.dropna(subset=['ensemble_score'])
    ens_top = ens.sort_values('ensemble_score', ascending=False).head(top_n).reset_index(drop=True)
    ens_top['ensemble_rank'] = ens_top.index + 1
    print('[STEP] Ensemble Top N (ticker, score):')
    print(ens_top[['ensemble_rank', 'ticker', 'ensemble_score']].to_string(index=False))

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
            print(f"[STEP] Merged XGB ranked columns from {xgb_ranked_path}")

    # 4) Optional: attach LSTM predictions
    if lstm_path:
        lstm = read_csv_safe(lstm_path)
        if lstm is not None and 'ticker' in lstm.columns:
            lstm = normalize_tickers(lstm)
            keep = [c for c in ['ticker', 'lstm_predicted_return_pct'] if c in lstm.columns]
            features = features.merge(lstm[keep], on='ticker', how='left')
            print(f"[STEP] Merged LSTM predictions from {lstm_path}")
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
            print(f"[STEP] Filtered features by close > $3: {before} -> {len(features)}")
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
    print('[CHECK] Order matches ensemble?', order_match)
    if not order_match:
        for i, (a, b) in enumerate(zip(ens_list, merged_list), 1):
            if a != b:
                print(f"[DIFF] First mismatch at position {i}: ensemble={a} vs merged={b}")
                break
        if strict:
            print('[ERROR] Strict mode enabled and order mismatch detected. Aborting.')
            return 4

    # 8) Write output
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[DONE] Wrote UI dataset to: {out_path} (rows={len(merged)}, cols={len(merged.columns)})")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description='Build a UI-ready dataset strictly ordered by ensemble scores')
    p.add_argument('--ensemble', default='data/top/ensemble_scores_output.csv', help='Path to ensemble scores CSV')
    p.add_argument('--features', nargs='*', default=[
        'data/top/xgb_features_latest.csv',
        'data/featured_stocks_top.csv',
        'data/top/xgb_weekly_output.csv',
    ], help='Candidate feature sources (first valid will be used)')
    p.add_argument('--lstm', default='data/top/lstm_weekly_predictions_output.csv', help='Optional LSTM predictions CSV')
    p.add_argument('--xgb_ranked', default='data/top/xgb_ranked_output.csv', help='Optional XGB ranked CSV')
    p.add_argument('--top_n', type=int, default=25, help='Top-N from ensemble to keep')
    p.add_argument('--out', default='data/top/ui_dataset.csv', help='Output CSV for UI')
    p.add_argument('--strict', action='store_true', help='Fail if merged order differs from ensemble')
    args = p.parse_args()

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


