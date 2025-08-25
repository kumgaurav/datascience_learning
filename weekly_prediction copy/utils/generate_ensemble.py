import os
import argparse
import numpy as np
import pandas as pd


def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    mn, mx = s.min(), s.max()
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


def main():
    parser = argparse.ArgumentParser(description="Blend XGB ranker outputs with LSTM predictions and save ensemble scores")
    parser.add_argument('--xgb', type=str, default='data/top/xgb_ranked_output.csv', help='Path to XGB-ranked CSV')
    parser.add_argument('--lstm', type=str, default='data/top/lstm_weekly_predictions_output.csv', help='Path to LSTM predictions CSV')
    parser.add_argument('--alpha', type=float, default=float(os.getenv('ENSEMBLE_ALPHA', '0.6')), help='Weight on XGB component (0..1)')
    parser.add_argument('--features', type=str, default=os.getenv('ENSEMBLE_FEATURES_CSV', 'data/top/xgb_features_latest.csv'), help='Optional features CSV to compute dynamic penalties (volume_ratio, price_change_pct)')
    parser.add_argument('--output', type=str, default='data/top/ensemble_scores_output.csv', help='Output CSV path')
    args = parser.parse_args()

    xgb_df = pd.read_csv(args.xgb)
    lstm_df = pd.read_csv(args.lstm)

    # Merge on ticker
    df = xgb_df.merge(lstm_df, on='ticker', how='left', suffixes=('_xgb', '_lstm'))
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    # Derive unified prediction columns with robust column detection
    import pandas as _pd
    import numpy as _np
    def _num(s):
        return _pd.to_numeric(s, errors='coerce')
    def _pick_numeric(df_in: pd.DataFrame, candidates: list[str]):
        for c in candidates:
            if c in df_in.columns:
                return _num(df_in[c])
        return _pd.Series(_np.nan, index=df_in.index)
    # Candidates consider common names and suffixes
    xgb_pred_series = _pick_numeric(df, [
        'xgb_pred',
        'xgb_predicted_return_pct',
        'predicted_return_pct_xgb',
        'predicted_return_pct_x',
        'predicted_return_pct',
    ])
    df['xgb_pred'] = xgb_pred_series
    lstm_pred_series = _pick_numeric(df, [
        'lstm_pred',
        'lstm_predicted_return_pct',
        'predicted_return_pct_lstm',
        'predicted_return_pct_y',
    ])
    df['lstm_pred'] = lstm_pred_series
    # Warnings if predictions missing
    if not _np.isfinite(df['xgb_pred']).any():
        print("[ENSEMBLE WARN] No valid XGB prediction column found in", list(xgb_df.columns))
    if not _np.isfinite(df['lstm_pred']).any():
        print("[ENSEMBLE WARN] No valid LSTM prediction column found in", list(lstm_df.columns))
    # Ensemble method
    method = os.getenv('ENSEMBLE_METHOD', 'weighted').lower()
    alpha = float(os.getenv('ENSEMBLE_ALPHA', str(args.alpha)))
    print(f"[ENSEMBLE] Method={method}, alpha={alpha:.2f}")
    if method == 'rank':
        def _rank_score(s):
            s = _num(s)
            try:
                r = s.rank(method='min', ascending=False)
                return (len(r) - r) / (len(r) - 1) if len(r) > 1 else _pd.Series(0.0, index=s.index)
            except Exception:
                return _pd.Series(0.0, index=s.index)
        score_xgb = _rank_score(df['xgb_pred'])
        score_lstm = _rank_score(df['lstm_pred'])
        df['ensemble_score'] = alpha * score_xgb + (1 - alpha) * score_lstm
    else:
        score_xgb = minmax(df['xgb_pred'])
        score_lstm = minmax(df['lstm_pred'])
        df['ensemble_score'] = alpha * score_xgb + (1 - alpha) * score_lstm
    # Confidence/Risk propagation (weighted average when available)
    def _get_pair(base: str):
        # Return tuple (xgb_col, lstm_col) if present
        cx = None
        cy = None
        for c in [f"{base}_xgb", f"{base}_x", base]:
            if c in df.columns:
                cx = _num(df[c]); break
        for c in [f"{base}_lstm", f"{base}_y", base]:
            if c in df.columns:
                cy = _num(df[c]); break
        return cx, cy
    conf_x, conf_l = _get_pair('confidence_score')
    risk_x, risk_l = _get_pair('risk_score')
    comp_x, comp_l = _get_pair('composite_score')
    def _blend(a, b):
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return alpha * a + (1 - alpha) * b
    ens_conf = _blend(conf_x, conf_l)
    ens_risk = _blend(risk_x, risk_l)
    ens_comp = _blend(comp_x, comp_l)
    if ens_conf is not None:
        df['ensemble_confidence_score'] = ens_conf
    if ens_risk is not None:
        df['ensemble_risk_score'] = ens_risk
    if ens_comp is not None:
        df['ensemble_composite_score'] = ens_comp
    # Optional stacking with meta-learner (joblib path via ENSEMBLE_STACK_MODEL)
    stack_model_path = os.getenv('ENSEMBLE_STACK_MODEL')
    if isinstance(stack_model_path, str) and os.path.isfile(stack_model_path):
        try:
            import joblib as _joblib
            mdl = _joblib.load(stack_model_path)
            # Minimal feature set for stacking
            feat_cols = []
            for c in ['xgb_pred', 'lstm_pred', 'ensemble_score']:
                if c in df.columns:
                    feat_cols.append(c)
            if conf_x is not None:
                df['_xgb_conf'] = conf_x
                feat_cols.append('_xgb_conf')
            if conf_l is not None:
                df['_lstm_conf'] = conf_l
                feat_cols.append('_lstm_conf')
            Xs = df[feat_cols].replace([_np.inf, -_np.inf], _np.nan).fillna(0.0)
            stack_pred = _num(pd.Series(mdl.predict(Xs)))
            df['ensemble_score'] = stack_pred
            print(f"[ENSEMBLE] Applied stacking model from {stack_model_path}")
        except Exception as _e:
            print(f"[ENSEMBLE WARN] Failed to apply stacking model {stack_model_path}: {_e}")
    # Ensure/alias standard columns exist for both models
    if 'lstm_pred' not in df.columns and 'lstm_predicted_return_pct' in df.columns:
        df['lstm_pred'] = df['lstm_predicted_return_pct']
    # XGB standard aliases
    if 'predicted_return_pct' not in df.columns and 'xgb_predicted_return_pct' in df.columns:
        df['predicted_return_pct'] = df['xgb_predicted_return_pct']

    # --- Proportional selloff penalty after ensemble (post-combination) ---
    try:
        feats = None
        if isinstance(args.features, str) and os.path.isfile(args.features):
            try:
                feats = pd.read_csv(args.features)
                feats = feats.sort_values('date').groupby('ticker').tail(1)
                feats['ticker'] = feats['ticker'].astype(str).str.upper()
            except Exception:
                feats = None
        vr_min = float(os.getenv('SELLOFF_VRATIO_MIN', '1.2'))
        vr_ref = float(os.getenv('SELLOFF_VRATIO_REF', '2.5'))
        drop_ref = float(os.getenv('SELLOFF_REF_DROP_PCT', '0.05'))
        f_min = float(os.getenv('SELLOFF_FACTOR_MIN', '0.60'))
        f_max = float(os.getenv('SELLOFF_FACTOR_MAX', '0.75'))
        def _factor_row(tkr: str) -> float:
            try:
                if feats is not None:
                    row = feats[feats['ticker'] == tkr]
                    if not row.empty:
                        vr = float(pd.to_numeric(row.iloc[0].get('volume_ratio', 0.0), errors='coerce'))
                        pc = float(pd.to_numeric(row.iloc[0].get('price_change_pct', row.iloc[0].get('return_1d', 0.0)), errors='coerce'))
                        so = int(pd.to_numeric(row.iloc[0].get('selloff_flag', 0), errors='coerce')) == 1
                        if so and (vr > vr_min) and (pc < 0.0):
                            sv = max(0.0, min(1.0, (vr - vr_min) / max(1e-6, (vr_ref - vr_min))))
                            sd = max(0.0, min(1.0, (-pc) / max(1e-6, drop_ref)))
                            sev = 0.5 * sv + 0.5 * sd
                            return float(f_max - (f_max - f_min) * sev)
                        return 1.0
            except Exception:
                return 1.0
            return 1.0
        weight_series = df['ticker'].astype(str).str.upper().map(_factor_row)
        if 'ensemble_score' in df.columns:
            df['ensemble_score'] = pd.to_numeric(df['ensemble_score'], errors='coerce') * weight_series
        elif 'ensemble_pred' in df.columns:
            df['ensemble_pred'] = pd.to_numeric(df['ensemble_pred'], errors='coerce') * weight_series
        # Optional utility-style final score for ranking/presentation
        use_util = os.getenv('USE_UTILITY_SCORE', '0') == '1'
        if use_util:
            mu_col = 'ensemble_score' if 'ensemble_score' in df.columns else ('ensemble_pred' if 'ensemble_pred' in df.columns else None)
            if mu_col is not None:
                # confidence proxies if available
                conf_x = pd.to_numeric(df.get('confidence_score_xgb', df.get('confidence_score', 1.0)), errors='coerce')
                conf_l = pd.to_numeric(df.get('confidence_score_lstm', 1.0), errors='coerce')
                conf_raw = alpha * conf_x + (1 - alpha) * conf_l
                df['final_score'] = pd.to_numeric(df[mu_col], errors='coerce') * conf_raw
                df = df.sort_values('final_score', ascending=False)
    except Exception:
        pass

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Build a return proxy for filtering (prefer XGB; fallback to LSTM)
    ret_proxy = None
    for c in ['xgb_pred', 'xgb_predicted_return_pct', 'predicted_return_pct_xgb', 'lstm_predicted_return_pct', 'lstm_pred']:
        if c in df.columns:
            ret_proxy = pd.to_numeric(df[c], errors='coerce')
            break
    if ret_proxy is None:
        print('[ENSEMBLE WARN] No return proxy columns found; defaulting to zeros (no positive-return filtering).')
        ret_proxy = pd.Series(0.0, index=df.index)
    df['_return_proxy_'] = ret_proxy

    # Keep only positive-return rows, then take top-25 by ensemble_score
    try:
        df = df[pd.to_numeric(df['_return_proxy_'], errors='coerce') > 0].copy()
    except Exception:
        df = df.copy()
    try:
        df = df.sort_values('ensemble_score', ascending=False)
    except Exception:
        pass
    top_n = int(os.getenv('ENSEMBLE_TOP_N', '25'))
    df = df.head(top_n)

    # Columns to include for UI convenience (both XGB and LSTM)
    keep_cols = [c for c in [
        'ticker',
        # XGB
        'xgb_pred', 'xgb_predicted_return_pct', 'predicted_return_pct_xgb', 'confidence_score_xgb', 'risk_score_xgb', 'composite_score_xgb',
        # LSTM
        'lstm_pred', 'lstm_predicted_return_pct', 'confidence_score_lstm', 'risk_score_lstm', 'composite_score_lstm',
        # Ensemble
        'ensemble_score', 'ensemble_confidence_score', 'ensemble_risk_score', 'ensemble_composite_score'
    ] if c in df.columns]
    # Write final
    df[keep_cols].to_csv(args.output, index=False)
    print(f"[ENSEMBLE STEP] Wrote ensemble scores to {args.output} (rows={len(df)})")


if __name__ == '__main__':
    main()


