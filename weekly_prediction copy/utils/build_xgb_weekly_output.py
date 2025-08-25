import argparse
import os
import pandas as pd


def read_csv_if_exists(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path) if os.path.exists(path) else None
    except Exception:
        return None


def build_xgb_weekly_output(
    features_path: str,
    ranked_path: str,
    output_path: str,
) -> str:
    features = read_csv_if_exists(features_path)
    if features is None:
        raise FileNotFoundError(f"Features CSV not found: {features_path}")

    ranked = read_csv_if_exists(ranked_path)
    if ranked is None:
        raise FileNotFoundError(f"Ranked CSV not found: {ranked_path}")

    # Normalize keys
    features = features.copy()
    features['ticker'] = features['ticker'].astype(str).str.upper()
    if 'date' in features.columns:
        try:
            features['date'] = pd.to_datetime(features['date'], errors='coerce')
            features = features.sort_values(['ticker', 'date']).drop_duplicates(subset=['ticker'], keep='last')
        except Exception:
            pass

    ranked = ranked.copy()
    ranked['ticker'] = ranked['ticker'].astype(str).str.upper()

    # Disambiguate ranked columns
    rename_map = {}
    if 'confidence_score' in ranked.columns:
        rename_map['confidence_score'] = 'xgb_confidence_score'
    if 'predicted_return_pct' in ranked.columns:
        rename_map['predicted_return_pct'] = 'xgb_predicted_return_pct'
    ranked = ranked.rename(columns=rename_map)

    # Inner join to keep only ranked tickers; bring all feature columns forward
    merged = features.merge(ranked, on='ticker', how='inner')

    # Apply proportional selloff penalty to XGB scores for transparency/consistency in the weekly file
    try:
        import os as _os
        import pandas as _pd
        # Proportional parameters
        vr_min = float(_os.getenv('SELLOFF_VRATIO_MIN', '1.2'))
        vr_ref = float(_os.getenv('SELLOFF_VRATIO_REF', '2.5'))
        drop_ref = float(_os.getenv('SELLOFF_REF_DROP_PCT', '0.05'))
        f_min = float(_os.getenv('SELLOFF_FACTOR_MIN', '0.60'))
        f_max = float(_os.getenv('SELLOFF_FACTOR_MAX', '0.75'))
        f_min, f_max = (min(f_min, f_max), max(f_min, f_max))
        def _factor(vr: float, pc: float, so: int) -> float:
            if so and (vr > vr_min) and (pc < 0.0):
                sv = max(0.0, min(1.0, (vr - vr_min) / max(1e-6, (vr_ref - vr_min))))
                sd = max(0.0, min(1.0, (-pc) / max(1e-6, drop_ref)))
                sev = 0.5 * sv + 0.5 * sd
                return float(f_max - (f_max - f_min) * sev)
            return 1.0
        # Normalize selloff column
        if 'selloff_flag' not in merged.columns and 'selloff_flag' in features.columns:
            merged['selloff_flag'] = features.set_index('ticker').loc[merged['ticker']]['selloff_flag'].values
        vr = _pd.to_numeric(merged.get('volume_ratio', 0.0), errors='coerce').fillna(0.0)
        pc = _pd.to_numeric(merged.get('price_change_pct', merged.get('return_1d', 0.0)), errors='coerce').fillna(0.0)
        so = _pd.to_numeric(merged.get('selloff_flag', 0), errors='coerce').fillna(0.0).astype(int)
        factor = _pd.Series([_factor(float(v), float(p), int(s)) for v, p, s in zip(vr, pc, so)], index=merged.index, dtype='float64')
        # Apply to confidence and predicted returns when present; also persist a flag for transparency
        for col in ['xgb_confidence_score', 'confidence_score', 'xgb_predicted_return_pct', 'predicted_return_pct']:
            if col in merged.columns:
                merged[col] = _pd.to_numeric(merged[col], errors='coerce') * factor
        merged['selloff_penalty_factor'] = factor
        if _os.getenv('DEBUG_SELLOFF', '0') == '1':
            try:
                dbg_cols = ['ticker','selloff_flag','volume_ratio','price_change_pct','selloff_penalty_factor','xgb_confidence_score','xgb_predicted_return_pct']
                dbg = merged[[c for c in dbg_cols if c in merged.columns]].copy()
                dbg_path = _os.path.join(_os.path.dirname(output_path), 'debug_xgb_weekly_penalties.csv')
                dbg.to_csv(dbg_path, index=False)
                print(f"[DEBUG] Wrote XGB weekly penalty debug to {dbg_path}")
            except Exception:
                pass
    except Exception:
        pass

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"[XGB WEEKLY] Features: {len(features)} | Ranked: {len(ranked)} | Output: {len(merged)} → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Build merged XGB weekly output (features + ranked).')
    parser.add_argument('--features', default='data/top/xgb_features_latest.csv', help='Path to features CSV')
    parser.add_argument('--ranked', default='data/top/xgb_ranked_output.csv', help='Path to ranked output CSV')
    parser.add_argument('--fallback_ranked', default='data/top/xgb_ranked.csv', help='Fallback ranked CSV if main not found')
    parser.add_argument('--out', default='data/top/xgb_weekly_output.csv', help='Output CSV path')
    args = parser.parse_args()

    ranked_path = args.ranked if os.path.exists(args.ranked) else args.fallback_ranked
    build_xgb_weekly_output(args.features, ranked_path, args.out)


if __name__ == '__main__':
    main()


