#!/usr/bin/env python3
"""
Annotate eval CSV with avoid_reason based on:
 - drifted_driver: any of top-3 SHAP features drifted (p<0.05) that day
 - low_conf: |pred| < 0.5%
 - model_disagree: XGB and LSTM signs disagree
 - illiquid: volume_ratio < 0.7
 - stale_data: eval latest_date != requested date
 - large_gap: gap against signal > 2%

Writes updated eval CSV in place (adds avoid_reason and avoid_flag).
"""

import argparse
import os
from datetime import date
import numpy as np
import pandas as pd


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate eval with avoid reasons")
    parser.add_argument("--snapshot", required=True, help="Path to snapshot CSV")
    parser.add_argument("--date", dest="eval_date", default=date.today().isoformat(), help="Evaluation date YYYY-MM-DD")
    args = parser.parse_args()

    snap_dir = os.path.dirname(args.snapshot)
    eval_csv = os.path.join(snap_dir, "eval", f"{args.eval_date}_eval.csv")
    drift_csv = os.path.join(snap_dir, "drift", f"{args.eval_date}_drift.csv")
    attr_csv = os.path.join(snap_dir, "attr", f"{args.eval_date}_attribution.csv")

    if not os.path.exists(eval_csv):
        raise FileNotFoundError(f"Eval CSV not found: {eval_csv}")
    edf = pd.read_csv(eval_csv)
    edf['ticker'] = edf.get('ticker', edf.get('symbol')).astype(str).str.upper()

    # Attribution: top features per ticker (rank<=3)
    top_feats = {}
    if os.path.exists(attr_csv):
        a = pd.read_csv(attr_csv)
        if 'ticker' in a.columns and 'feature' in a.columns:
            a['ticker'] = a['ticker'].astype(str).str.upper()
            if 'rank' in a.columns:
                a3 = a[a['rank'] <= 3]
            else:
                a3 = a.sort_values(['ticker']).groupby('ticker').head(3)
            top_feats = a3.groupby('ticker')['feature'].apply(lambda s: set(map(str, s))).to_dict()

    # Drift: p-values by feature
    drifted = set()
    if os.path.exists(drift_csv):
        d = pd.read_csv(drift_csv)
        # Find p_value column
        pcol = None
        for c in d.columns:
            if c.lower() in ['p_value','p-value','pvalue']:
                pcol = c
                break
        if 'feature' in d.columns and pcol:
            drifted = set(d.loc[pd.to_numeric(d[pcol], errors='coerce') < 0.05, 'feature'].astype(str))

    reasons = []
    for _, r in edf.iterrows():
        t = str(r['ticker']).upper()
        rlist = []
        # drifted_driver
        feats = top_feats.get(t, set())
        if feats and (len(feats.intersection(drifted)) > 0):
            rlist.append('drifted_driver')
        # low_conf
        pred = None
        if 'pred_xgb' in edf.columns and pd.notna(r.get('pred_xgb')):
            pred = float(r.get('pred_xgb'))
        elif 'pred_lstm' in edf.columns and pd.notna(r.get('pred_lstm')):
            pred = float(r.get('pred_lstm'))
        if pred is not None and abs(pred) < 0.5:
            rlist.append('low_conf')
        # model_disagree
        if 'pred_xgb' in edf.columns and 'pred_lstm' in edf.columns:
            try:
                sx = _sign(float(r.get('pred_xgb')))
                sl = _sign(float(r.get('pred_lstm')))
                if sx != 0 and sl != 0 and sx != sl:
                    rlist.append('model_disagree')
            except Exception:
                pass
        # illiquid
        try:
            if 'volume_ratio' in edf.columns and pd.notna(r.get('volume_ratio')) and float(r.get('volume_ratio')) < 0.7:
                rlist.append('illiquid')
        except Exception:
            pass
        # stale_data
        try:
            ld = pd.to_datetime(r.get('latest_date')) if 'latest_date' in edf.columns else None
            if ld is not None and not pd.isna(ld):
                if str(ld.date()) != str(args.eval_date):
                    rlist.append('stale_data')
        except Exception:
            pass
        # large_gap
        try:
            gp = float(r.get('gap_pct')) if 'gap_pct' in edf.columns and pd.notna(r.get('gap_pct')) else None
            if gp is not None and pred is not None:
                if pred > 0 and gp < -2:
                    rlist.append('large_gap')
                if pred < 0 and gp > 2:
                    rlist.append('large_gap')
        except Exception:
            pass
        reasons.append(';'.join(sorted(set(rlist))))

    edf['avoid_reason'] = reasons
    edf['avoid_flag'] = edf['avoid_reason'].astype(str).str.len() > 0
    edf.to_csv(eval_csv, index=False)
    print(f"[ANNOTATE] Updated {eval_csv} with avoid_reason (avoid_count={int(edf['avoid_flag'].sum())})")


if __name__ == '__main__':
    main()


