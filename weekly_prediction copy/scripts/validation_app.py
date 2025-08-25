import os
import glob
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import subprocess
import sys
import re
from datetime import date
try:
    import feature_engineering as fe
except Exception:
    fe = None  # Fallback if module not available in runtime
import numpy as np


# -----------------------------
# Data loading utilities
# -----------------------------

def find_latest_snapshot_dir(base_dir: str = "snapshot") -> Optional[str]:
    try:
        subdirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
        return sorted(subdirs)[-1] if subdirs else None
    except Exception:
        return None


def load_ui_order(rank_paths: List[str]) -> List[str]:
    for path in rank_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if 'ticker' not in df.columns:
                    continue
                df['ticker'] = df['ticker'].astype(str).str.upper()
                if 'confidence_score' in df.columns:
                    df = df.sort_values('confidence_score', ascending=False)
                return df['ticker'].drop_duplicates().tolist()
        except Exception:
            continue
    return []


def load_daily_evals(snapshot_dir: str) -> Tuple[List[str], List[pd.DataFrame]]:
    eval_dir = os.path.join(snapshot_dir, 'eval')
    files = sorted(glob.glob(os.path.join(eval_dir, '*_eval.csv')))
    day_labels: List[str] = [os.path.basename(f).replace('_eval.csv', '') for f in files]
    daily_dfs: List[pd.DataFrame] = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if 'ticker' not in d.columns and 'symbol' in d.columns:
                d['ticker'] = d['symbol']
            d['ticker'] = d['ticker'].astype(str).str.upper()
            # Coerce numeric columns used below
            for col in ['pred_xgb', 'pred_lstm', 'actual_return_pct']:
                if col in d.columns:
                    d[col] = pd.to_numeric(d[col], errors='coerce')
            daily_dfs.append(d)
        except Exception:
            continue
    return day_labels, daily_dfs


# -----------------------------
# Table builders & styling
# -----------------------------

def build_outcomes_table(
    day_labels: List[str],
    daily_dfs: List[pd.DataFrame],
    ui_order: List[str],
    pred_col: str,
) -> pd.DataFrame:
    per_day = []
    for d in daily_dfs:
        if pred_col not in d.columns:
            d[pred_col] = np.nan
        if 'actual_return_pct' not in d.columns:
            d['actual_return_pct'] = np.nan
        if 'avoid_reason' not in d.columns:
            d['avoid_reason'] = ''
        per_day.append(d.set_index('ticker')[[pred_col, 'actual_return_pct', 'avoid_reason']])

    rows = []
    for tkr in ui_order:
        row = {'Ticker': tkr}
        for lbl, day_df in zip(day_labels, per_day):
            if tkr in day_df.index:
                pred = day_df.loc[tkr, pred_col]
                act = day_df.loc[tkr, 'actual_return_pct']
                ar = day_df.loc[tkr, 'avoid_reason']
                if pd.notna(pred) and pd.notna(act):
                    row[lbl] = f"{pred:+.2f}% / {act:+.2f}% | {ar}" if isinstance(ar, str) and ar else f"{pred:+.2f}% / {act:+.2f}%"
                else:
                    row[lbl] = "-"
            else:
                row[lbl] = "-"
        rows.append(row)
    return pd.DataFrame(rows)


def style_outcomes_table(df: pd.DataFrame):
    def _color_cell(val: str) -> str:
        try:
            if val and isinstance(val, str) and '%' in val and '/' in val:
                core = val.split('|')[0].strip() if '|' in val else val
                reasons = val.split('|')[1].strip() if '|' in val and len(val.split('|'))>1 else ''
                p_txt, a_txt = [s.strip().replace('%', '') for s in core.split('/')]
                a = float(a_txt)
                # Base color by actual sign
                base = 'color: green;' if a > 0 else ('color: red;' if a < 0 else '')
                # Overlay background based on avoid_reason severity
                if reasons:
                    if any(k in reasons for k in ['drifted_driver','model_disagree']):
                        return base + ' background-color: #ffebee;'  # light red
                    if any(k in reasons for k in ['low_conf','large_gap']):
                        return base + ' background-color: #fff8e1;'  # light amber
                    if any(k in reasons for k in ['stale_data','illiquid']):
                        return base + ' background-color: #f5f5f5;'  # light gray
                return base
        except Exception:
            pass
        return ''

    subset_cols = df.columns[1:]
    # applymap is deprecated; use map instead
    return df.style.map(_color_cell, subset=subset_cols)


def render_outcomes_tab(snapshot_dir: str) -> None:
    # Controls: one-click validation pipeline
    with st.container():
        cols = st.columns([1, 1, 6])
        with cols[0]:
            do_validate = st.button('▶️ Validate week', help='Run Evaluator → Drift → Attribution for this snapshot')
        with cols[1]:
            refresh_prices = st.checkbox('Refresh prices', value=False, help='Refresh prices from DB before evaluating')
        db_conf = None
        if refresh_prices:
            db_conf = st.text_input('DB config (stocks/conf/config.ini)', value='stocks/conf/config.ini')

    if do_validate:
        run_validation_pipeline(snapshot_dir, refresh_prices=refresh_prices, db_config=db_conf)

    # Establish order via ranked CSV the UI uses
    ui_order = load_ui_order([
        'data/top/xgb_ranked.csv',
        'data/top/xgb_ranked_output.csv',
    ])
    if not ui_order:
        st.warning('Could not determine UI order. Ensure data/top/xgb_ranked.csv exists.')
        return

    day_labels, daily_dfs = load_daily_evals(snapshot_dir)
    if not daily_dfs:
        st.info('No daily evaluation files found under snapshot/eval')
        return

    # Legend for cell color coding
    st.markdown(
        """
        <div style='font-size:13px;margin-bottom:8px;'>
          <b>Legend:</b>
          <span style='background:#ffebee;padding:2px 6px;border-radius:6px;margin-left:8px;'>Red background</span> drifted_driver or model_disagree
          <span style='background:#fff8e1;padding:2px 6px;border-radius:6px;margin-left:8px;'>Amber</span> low_conf or large_gap
          <span style='background:#f5f5f5;padding:2px 6px;border-radius:6px;margin-left:8px;'>Gray</span> stale_data or illiquid
        </div>
        """,
        unsafe_allow_html=True,
    )

    # XGB table
    st.markdown('**XGB: Predicted% / Actual% by Day**')
    xgb_df = build_outcomes_table(day_labels, daily_dfs, ui_order, pred_col='pred_xgb')
    st.dataframe(style_outcomes_table(xgb_df), use_container_width=True)

    # LSTM table
    st.markdown('**LSTM: Predicted% / Actual% by Day**')
    lstm_df = build_outcomes_table(day_labels, daily_dfs, ui_order, pred_col='pred_lstm')
    st.dataframe(style_outcomes_table(lstm_df), use_container_width=True)

    # Optional: Matplotlib heatmap (actual returns sign)
    try:
        sign_matrix = []
        for tkr in ui_order:
            signs = []
            for d in daily_dfs:
                if 'ticker' in d.columns and tkr in set(d['ticker']):
                    a = d.set_index('ticker').loc[tkr, 'actual_return_pct']
                    if pd.isna(a):
                        signs.append(0)
                    else:
                        signs.append(1 if a > 0 else -1)
                else:
                    signs.append(0)
            sign_matrix.append(signs)

        fig_width = max(6, min(14, 0.6 * max(1, len(day_labels))))
        fig_height = max(4, min(12, 0.3 * max(1, len(ui_order))))
        fig, ax = plt.subplots(constrained_layout=True, figsize=(fig_width, fig_height))
        cmap = plt.get_cmap('RdYlGn')
        # Map -1 -> red, 0 -> gray, 1 -> green
        data = np.array(sign_matrix)
        im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        ax.set_yticks(range(len(ui_order)))
        ax.set_yticklabels(ui_order, fontsize=8)
        ax.set_xticks(range(len(day_labels)))
        ax.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=8)
        ax.set_title('Actual Returns Sign by Day (Green=Up, Red=Down)')
        st.pyplot(fig)
    except Exception:
        pass


# -----------------------------
# Drift Tab
# -----------------------------

def render_drift_tab(snapshot_dir: str) -> None:
    mode = st.radio('View', options=['Global (distribution vs snapshot)', 'Per-Ticker'], horizontal=True)
    if mode.startswith('Global'):
        drift_dir = os.path.join(snapshot_dir, 'drift')
        drift_files = sorted(glob.glob(os.path.join(drift_dir, '*_drift.csv')))
        # Allow choosing a specific drift date (defaults to latest)
        drift_dates = [os.path.basename(p).replace('_drift.csv', '') for p in drift_files]
        # Also offer eval dates as candidates to generate if missing
        eval_dir = os.path.join(snapshot_dir, 'eval')
        eval_dates = [os.path.basename(f).replace('_eval.csv', '') for f in sorted(glob.glob(os.path.join(eval_dir, '*_eval.csv')))]
        sel_date = None
        if drift_dates:
            sel_date = st.selectbox('Drift date', options=drift_dates, index=len(drift_dates)-1)
        else:
            st.info('No drift CSVs found yet under drift/. Select an eval date and click Generate.')
        # Generation controls
        gen_cols = st.columns([1,2])
        with gen_cols[0]:
            gen_date = st.selectbox('Eval date to generate drift', options=eval_dates or [''], index=(len(eval_dates)-1 if eval_dates else 0))
        with gen_cols[1]:
            if st.button('Generate drift for selected date') and gen_date:
                snap_csv = os.path.join(snapshot_dir, 'snapshot.csv')
                cmd = [sys.executable, 'scripts/feature_drift.py', '--snapshot', snap_csv, '--date', gen_date]
                ok, out = _run_cmd(cmd, 'Feature drift')
                st.code(out or '', language='bash')
                if ok:
                    st.success(f'Drift generated for {gen_date}. Refresh to view.')
                else:
                    st.error('Drift generation failed.')
        # If we still don't have a drift file selected, stop
        if not drift_files:
            return
        # Load selected drift CSV
        path = os.path.join(drift_dir, f"{sel_date}_drift.csv") if sel_date else drift_files[-1]
        if not os.path.exists(path):
            st.warning(f'Drift file not found: {path}')
            return
        df = pd.read_csv(path)
        # Normalize columns for display
        cols = df.columns.str.lower().tolist()
        rename = {}
        if 'feature' not in cols:
            for c in df.columns:
                if c.lower() == 'feature':
                    rename[c] = 'feature'
        if 'status' not in cols:
            for c in df.columns:
                if c.lower() in ['status', 'drift status', 'drift_status']:
                    rename[c] = 'status'
        if 'p_value' not in cols:
            for c in df.columns:
                if c.lower() in ['p_value', 'p-value', 'pvalue']:
                    rename[c] = 'p_value'
        if rename:
            df = df.rename(columns=rename)
        keep = [c for c in ['feature', 'status', 'p_value', 'ks_stat'] if c in df.columns]
        if keep:
            df = df[keep]
        st.subheader('Feature Drift Report (Global)')
        st.dataframe(df, use_container_width=True)
        # Bar chart: count drift vs stable
        if 'status' in df.columns:
            counts = df['status'].value_counts().rename_axis('Status').reset_index(name='Count')
            st.bar_chart(counts.set_index('Status'))
        # Highlight: misses intersecting drifted features
        eval_dir = os.path.join(snapshot_dir, 'eval')
        eval_files = sorted(glob.glob(os.path.join(eval_dir, '*_eval.csv')))
        attr_dir = os.path.join(snapshot_dir, 'attr')
        attr_files = sorted(glob.glob(os.path.join(attr_dir, '*_attribution.csv')))
        if eval_files and attr_files and 'feature' in df.columns and 'status' in df.columns:
            st.subheader('Misses with drifting features (today)')
            edf = pd.read_csv(eval_files[-1])
            edf['ticker'] = edf.get('ticker', edf.get('symbol')).astype(str).str.upper()
            edf['pred_xgb'] = pd.to_numeric(edf.get('pred_xgb'), errors='coerce')
            edf['actual_return_pct'] = pd.to_numeric(edf.get('actual_return_pct'), errors='coerce')
            miss = edf[(edf['pred_xgb'].notna()) & (edf['actual_return_pct'].notna()) & ((edf['pred_xgb'] > 0) != (edf['actual_return_pct'] > 0))]
            rows = pd.read_csv(attr_files[-1])
            rows['ticker'] = rows['ticker'].astype(str).str.upper()
            drifted_feats = set(df[df['status'].astype(str).str.lower().eq('drift')]['feature'])
            rows = rows[(rows['ticker'].isin(set(miss['ticker']))) & (rows['feature'].isin(drifted_feats))]
            if not rows.empty:
                st.dataframe(rows.sort_values(['ticker','rank']).head(100), use_container_width=True)
            else:
                st.caption('No overlap between misses and drifted features today.')


def render_drift_timeline(snapshot_dir: str) -> None:
    drift_dir = os.path.join(snapshot_dir, 'drift')
    drift_files = sorted(glob.glob(os.path.join(drift_dir, '*_drift.csv')))
    if not drift_files:
        st.info('No drift files found under drift/. Generate daily drift to populate this view.')
        return
    frames = []
    for p in drift_files:
        try:
            d = pd.read_csv(p)
            d['date'] = os.path.basename(p).replace('_drift.csv', '')
            frames.append(d)
        except Exception:
            continue
    if not frames:
        st.info('No parsable drift files.')
        return
    df = pd.concat(frames, ignore_index=True)
    # Normalize
    if 'feature' not in df.columns:
        for c in df.columns:
            if c.lower() == 'feature':
                df = df.rename(columns={c: 'feature'})
                break
    if 'p_value' not in df.columns:
        for c in df.columns:
            if c.lower() in ['p_value','p-value','pvalue']:
                df = df.rename(columns={c: 'p_value'})
                break
    if 'feature' not in df.columns or 'p_value' not in df.columns:
        st.warning('Drift files missing required columns feature/p_value.')
        return
    df['p_value'] = pd.to_numeric(df['p_value'], errors='coerce')
    df['intensity'] = (-np.log10(df['p_value'].clip(lower=1e-12))).replace([np.inf, -np.inf], np.nan)
    # Filters and pivot
    feat_search = st.text_input('Filter features (substring)', value='')
    days = sorted(df['date'].dropna().unique().tolist())
    sel_days = st.multiselect('Select dates', options=days, default=days)
    dff = df[df['date'].isin(sel_days)].copy()
    if feat_search:
        dff = dff[dff['feature'].str.contains(feat_search, case=False, na=False)]
    piv = dff.pivot_table(index='feature', columns='date', values='intensity', aggfunc='mean')
    order = piv.max(axis=1).sort_values(ascending=False).index if not piv.empty else []
    piv = piv.loc[order]
    st.subheader('Drift Heatmap (−log10 p-value)')
    if not piv.empty:
        st.dataframe(piv.fillna(0).style.background_gradient(cmap='YlOrRd'), use_container_width=True)
    else:
        st.info('No matching features for current filters/dates.')
    st.subheader('Per-Feature Timeline')
    if not piv.empty:
        sel_feat = st.selectbox('Feature', options=piv.index.tolist()[:1000])
        s = piv.loc[sel_feat].fillna(0)
        chart_df = pd.DataFrame({'date': s.index, 'intensity': s.values}).sort_values('date')
        st.line_chart(chart_df.set_index('date'))
        # Drilldown: which tickers contributed for a specific day
        st.caption('Note: Drift is computed across all snapshot tickers. Use drilldown to see top tickers driving drift on a selected day for this feature.')
        detail_dates = chart_df['date'].tolist()
        if detail_dates:
            det_date = st.selectbox('Drilldown date', options=detail_dates, index=len(detail_dates)-1)
            try:
                snap_csv = os.path.join(snapshot_dir, 'snapshot.csv')
                eval_csv = os.path.join(snapshot_dir, 'eval', f"{det_date}_eval_input.csv")
                if os.path.exists(snap_csv) and os.path.exists(eval_csv):
                    s_df = pd.read_csv(snap_csv)
                    e_df = pd.read_csv(eval_csv)
                    # Normalize keys
                    sk = 'ticker' if 'ticker' in s_df.columns else ('symbol' if 'symbol' in s_df.columns else None)
                    ek = 'ticker' if 'ticker' in e_df.columns else ('symbol' if 'symbol' in e_df.columns else None)
                    if sk and ek and sel_feat in s_df.columns and sel_feat in e_df.columns:
                        s_df[sk] = s_df[sk].astype(str).str.upper()
                        e_df[ek] = e_df[ek].astype(str).str.upper()
                        # Keep only columns needed
                        s_keep = s_df[[sk, sel_feat]].rename(columns={sk: 'ticker', sel_feat: 'snapshot_val'})
                        e_keep = e_df[[ek, sel_feat]].rename(columns={ek: 'ticker', sel_feat: 'eval_val'})
                        merged = s_keep.merge(e_keep, on='ticker', how='inner')
                        merged['delta'] = merged['eval_val'] - merged['snapshot_val']
                        top = merged.reindex(columns=['ticker','snapshot_val','eval_val','delta']).copy()
                        # Show top movers by absolute delta
                        top['abs_delta'] = top['delta'].abs()
                        top = top.sort_values('abs_delta', ascending=False).drop(columns=['abs_delta']).head(20)
                        st.markdown('**Top tickers by absolute change**')
                        st.dataframe(top, use_container_width=True)
                    else:
                        st.info('Selected feature not found in snapshot/eval input for drilldown.')
                else:
                    st.info('Snapshot/eval input not found for drilldown.')
            except Exception:
                st.info('Drilldown unavailable due to data format differences.')
    else:
        st.subheader('Per-Ticker Feature Changes')
        # Load snapshot and let user pick ticker
        snap = pd.read_csv(os.path.join(snapshot_dir, 'snapshot.csv'))
        col_tk = 'ticker' if 'ticker' in snap.columns else ('symbol' if 'symbol' in snap.columns else None)
        if not col_tk:
            st.warning('Snapshot missing ticker/symbol column.')
            return
        snap[col_tk] = snap[col_tk].astype(str).str.upper()
        tickers = snap[col_tk].dropna().unique().tolist()
        tkr = st.selectbox('Ticker', options=sorted(tickers))
        eval_date = os.path.basename(os.path.normpath(snapshot_dir))
        eval_csv_path = os.path.join(snapshot_dir, 'eval', f"{eval_date}_eval.csv")
        eval_input_path = os.path.join(snapshot_dir, 'eval', f"{eval_date}_eval_input.csv")
        latest_source = st.radio(
            'Latest features source',
            options=['Eval CSV (recommended)', 'Eval input', 'Features CSV path'],
            index=0,
            horizontal=True
        )
        features_path = st.text_input(
            'Features CSV (full history)',
            value='data/top/featured_stocks_raw.csv',
            disabled=(latest_source != 'Features CSV path')
        )
        if tkr and (
            (latest_source == 'Eval CSV (recommended)' and os.path.exists(eval_csv_path)) or
            (latest_source == 'Eval input' and os.path.exists(eval_input_path)) or
            (latest_source == 'Features CSV path' and features_path and os.path.exists(features_path))
        ):
            try:
                if latest_source == 'Eval CSV (recommended)' and os.path.exists(eval_csv_path):
                    feat = pd.read_csv(eval_csv_path)
                    ctk = 'ticker' if 'ticker' in feat.columns else ('symbol' if 'symbol' in feat.columns else None)
                    if not ctk:
                        st.warning('Eval CSV missing ticker/symbol column.')
                        return
                    feat[ctk] = feat[ctk].astype(str).str.upper()
                    f_row = feat[feat[ctk] == tkr]
                elif latest_source == 'Eval input' and os.path.exists(eval_input_path):
                    feat = pd.read_csv(eval_input_path)
                    ctk = 'ticker' if 'ticker' in feat.columns else ('symbol' if 'symbol' in feat.columns else None)
                    if not ctk:
                        st.warning('Eval input missing ticker/symbol column.')
                        return
                    feat[ctk] = feat[ctk].astype(str).str.upper()
                    # Eval input already one row per ticker at eval_date
                    f_row = feat[feat[ctk] == tkr]
                else:
                    feat = pd.read_csv(features_path)
                    # Normalize
                    ctk = 'ticker' if 'ticker' in feat.columns else ('symbol' if 'symbol' in feat.columns else None)
                    if not ctk:
                        st.warning('Features CSV missing ticker/symbol column.')
                        return
                    feat[ctk] = feat[ctk].astype(str).str.upper()
                    if 'date' not in feat.columns:
                        st.warning('Features CSV missing date column.')
                        return
                    feat['date'] = pd.to_datetime(feat['date'], errors='coerce')
                    # Latest features for ticker up to eval_date
                    f_row = feat[(feat[ctk] == tkr) & (feat['date'] <= pd.to_datetime(eval_date))].sort_values('date').tail(1)
                s_row = snap[snap[col_tk] == tkr]
                if f_row.empty or s_row.empty:
                    st.warning('Could not find both snapshot and latest feature rows for this ticker.')
                    return
                # If using features CSV path (not eval/eval_input), override latest OHLCV and recompute features to align with evaluator
                if latest_source == 'Features CSV path':
                    try:
                        price_csv = 'data/stock_prices_input.csv'
                        if os.path.exists(price_csv):
                            p = pd.read_csv(price_csv)
                            p.columns = [c.lower() for c in p.columns]
                            if 'symbol' in p.columns and 'ticker' not in p.columns:
                                p = p.rename(columns={'symbol': 'ticker'})
                            if 'ticker' in p.columns and 'date' in p.columns:
                                p['ticker'] = p['ticker'].astype(str).str.upper()
                                p['date'] = pd.to_datetime(p['date'], errors='coerce')
                                px_all = p[(p['ticker'] == tkr) & (p['date'] <= pd.to_datetime(eval_date))].sort_values('date')
                                if px_all.empty and tkr == 'RR':
                                    px_all = p[(p['ticker'] == 'RYCEY') & (p['date'] <= pd.to_datetime(eval_date))].sort_values('date')
                                if not px_all.empty:
                                    # Latest OHLCV override
                                    px_last = px_all.tail(1)
                                    idx = f_row.index[0]
                                    for col_csv, col_feat in [('open', 'open'), ('high', 'high'), ('low', 'low'), ('close', 'close'), ('volume', 'volume')]:
                                        if col_csv in px_last.columns:
                                            try:
                                                f_row.loc[idx, col_feat] = float(px_last.iloc[0][col_csv])
                                            except Exception:
                                                pass

                                    # Recompute using feature_engineering
                                    try:
                                        if fe is not None and hasattr(fe, '_calculate_momentum'):
                                            px_all2 = px_all.copy()
                                            px_all2['ticker'] = tkr
                                            tech = fe._calculate_momentum(px_all2)
                                            tech_last = tech.sort_values('date').tail(1)
                                            if not tech_last.empty:
                                                last = tech_last.iloc[0]
                                                for col in last.index:
                                                    if col in f_row.columns and pd.notna(last[col]):
                                                        try:
                                                            f_row.loc[idx, col] = float(last[col]) if isinstance(last[col], (int, float, np.floating)) else last[col]
                                                        except Exception:
                                                            pass
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                # Work with single-row Series for snapshot/features
                s = s_row.iloc[0]
                f = f_row.iloc[0]
                # Determine numeric columns present in both (use DataFrame type detection)
                s_num_cols = s_row.select_dtypes(include=['number', 'bool']).columns.tolist()
                f_num_cols = f_row.select_dtypes(include=['number', 'bool']).columns.tolist()
                common = [c for c in s_num_cols if c in f_num_cols]
                deltas = []
                for c in common:
                    s_val = s.get(c, np.nan)
                    f_val = f.get(c, np.nan)
                    if pd.notna(s_val) and pd.notna(f_val):
                        try:
                            s_num = float(s_val)
                            f_num = float(f_val)
                            d = f_num - s_num
                        except Exception:
                            continue
                        deltas.append((c, s_num, f_num, d, abs(d)))
                if not deltas:
                    st.info('No comparable numeric features found for delta.')
                    return
                delta_df = pd.DataFrame(deltas, columns=['feature','snapshot','latest','delta','abs_delta']).sort_values('abs_delta', ascending=False).head(25)
                st.dataframe(delta_df.drop(columns=['abs_delta']), use_container_width=True)
            except Exception as e:
                st.error(f'Per-ticker drift failed: {e}')


# -----------------------------
# Attribution Tab
# -----------------------------

def render_attribution_tab(snapshot_dir: str, ui_order: List[str]) -> None:
    attr_dir = os.path.join(snapshot_dir, 'attr')
    feat_imp_files = sorted(glob.glob(os.path.join(attr_dir, '*_feature_importance.csv')))
    rows_files = sorted(glob.glob(os.path.join(attr_dir, '*_attribution.csv')))
    if not feat_imp_files and not rows_files:
        st.info('No attribution files found. Generate with scripts/attribution.py (requires xgboost/shap).')
        return
    if feat_imp_files:
        st.subheader('Global Feature Importance (Top 20)')
        fi = pd.read_csv(feat_imp_files[-1])
        fi = fi.sort_values(fi.columns[1], ascending=False).head(20)
        st.dataframe(fi, use_container_width=True)
    if rows_files:
        st.subheader('Per-Ticker Top Feature Contributions')
        rows = pd.read_csv(rows_files[-1])
        if 'ticker' in rows.columns:
            # Respect UI order in display
            order_map = {t: i for i, t in enumerate(ui_order)}
            rows['__ord__'] = rows['ticker'].map(order_map)
            # Optional: filter to red tickers only
            filt_col1, filt_col2 = st.columns(2)
            with filt_col1:
                only_red = st.checkbox('Show red tickers only', value=False)
            with filt_col2:
                topk = st.slider('Top features per ticker', min_value=3, max_value=10, value=5)
            # Load latest eval to identify red tickers
            eval_dir = os.path.join(snapshot_dir, 'eval')
            eval_files = sorted(glob.glob(os.path.join(eval_dir, '*_eval.csv')))
            red_set = set()
            if eval_files:
                edf = pd.read_csv(eval_files[-1])
                edf['ticker'] = edf.get('ticker', edf.get('symbol')).astype(str).str.upper()
                # Pick available prediction column
                pred_col = 'pred_xgb' if 'pred_xgb' in edf.columns else ('pred_lstm' if 'pred_lstm' in edf.columns else None)
                if pred_col:
                    edf[pred_col] = pd.to_numeric(edf.get(pred_col), errors='coerce')
                edf['actual_return_pct'] = pd.to_numeric(edf.get('actual_return_pct'), errors='coerce')
                if pred_col:
                    miss = edf[(edf[pred_col].notna()) & (edf['actual_return_pct'].notna()) & ((edf[pred_col] > 0) != (edf['actual_return_pct'] > 0))]
                    red_set = set(miss['ticker'])
                    st.caption(f"Eval: total={len(edf)}, misses={len(miss)} using {pred_col}")
                else:
                    st.caption('Eval file lacks prediction columns (pred_xgb/pred_lstm).')
            if only_red and red_set:
                rows = rows[rows['ticker'].isin(red_set)]
            # Rank inside each ticker and keep topk
            rows['rank'] = rows.groupby('ticker')['contribution'].rank(ascending=False, method='first') if 'contribution' in rows.columns else rows.get('rank', 0)
            if 'rank' in rows.columns:
                rows = rows[rows['rank'] <= topk]
            rows = rows.sort_values(['__ord__', 'ticker', 'rank'])
            rows = rows.drop(columns=['__ord__'])
        st.dataframe(rows, use_container_width=True)


# -----------------------------
# Red tickers tab (per-ticker latest chart & deltas)
# -----------------------------

def render_red_tickers_tab(snapshot_dir: str) -> None:
    # Load latest eval for this snapshot
    eval_dir = os.path.join(snapshot_dir, 'eval')
    files = sorted(glob.glob(os.path.join(eval_dir, '*_eval.csv')))
    if not files:
        st.info('No eval files found. Run evaluator first.')
        return
    df = pd.read_csv(files[-1])
    # Normalize
    if 'ticker' not in df.columns and 'symbol' in df.columns:
        df['ticker'] = df['symbol']
    df['ticker'] = df['ticker'].astype(str).str.upper()
    # Compute pass/fail by XGB sign match (pred_xgb vs actual)
    df['pred_xgb'] = pd.to_numeric(df.get('pred_xgb'), errors='coerce')
    if 'actual_return_pct' not in df.columns and 'actual_price' in df.columns:
        # cannot compute change without prior; outcomes tab covers returns
        st.info('Eval missing actual_return_pct. Re-run evaluator to compute returns-based metrics.')
        return
    df['actual_return_pct'] = pd.to_numeric(df['actual_return_pct'], errors='coerce')
    df['pass'] = ((df['pred_xgb'] > 0) & (df['actual_return_pct'] > 0)) | ((df['pred_xgb'] < 0) & (df['actual_return_pct'] < 0))
    reds = df[df['pass'] == False].copy()
    if reds.empty:
        st.success('No red tickers today. Great job!')
        return
    # Allow user to select a red ticker
    tickers = reds['ticker'].dropna().unique().tolist()
    sel = st.selectbox('Select red ticker', options=tickers)
    if not sel:
        return
    # Load latest prices for this ticker from unfiltered input
    price_path = 'data/stock_prices_input.csv'
    if not os.path.exists(price_path):
        st.warning('data/stock_prices_input.csv not found.')
        return
    p = pd.read_csv(price_path)
    col_tk = 'ticker' if 'ticker' in p.columns else ('symbol' if 'symbol' in p.columns else None)
    if not col_tk:
        st.warning('Prices CSV missing ticker/symbol columns.')
        return
    p[col_tk] = p[col_tk].astype(str).str.upper()
    # Try exact match; if none, show alias suggestion (e.g., RR vs RYCEY)
    px = p[p[col_tk] == sel].copy()
    if px.empty:
        # heuristic: if RYCEY present for RR, suggest it
        alt = 'RYCEY' if sel == 'RR' else None
        if alt:
            px = p[p[col_tk] == alt].copy()
            st.caption(f"Using alias {alt} for {sel}")
    if px.empty:
        st.warning(f'No price rows found for {sel}.')
        return
    # Prepare series and show small chart
    px['date'] = pd.to_datetime(px['date'], errors='coerce')
    px = px.sort_values('date').tail(120)
    st.line_chart(px.set_index('date')['close'], height=200)
    # Show predicted vs actual on the selected day
    row = reds[reds['ticker'] == sel].iloc[0]
    st.write({
        'Ticker': sel,
        'Pred_xgb_%': (row.get('pred_xgb') if pd.notna(row.get('pred_xgb')) else None),
        'Actual_%': (row.get('actual_return_pct') if pd.notna(row.get('actual_return_pct')) else None),
        'Error_%': (row.get('actual_return_pct') - row.get('pred_xgb')) if pd.notna(row.get('pred_xgb')) and pd.notna(row.get('actual_return_pct')) else None,
    })


# -----------------------------
# Validation pipeline runner
# -----------------------------

def _infer_eval_date_from_path(path: str) -> str:
    base = os.path.basename(os.path.normpath(path))
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", base):
        return base
    return date.today().isoformat()


def _run_cmd(cmd: List[str], title: str) -> Tuple[bool, str]:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ok = res.returncode == 0
        return ok, res.stdout
    except Exception as e:
        return False, f"{title} failed: {e}"


def run_validation_pipeline(snapshot_dir: str, refresh_prices: bool = False, db_config: Optional[str] = None) -> None:
    eval_date = _infer_eval_date_from_path(snapshot_dir)
    snap_csv = os.path.join(snapshot_dir, 'snapshot.csv')
    logs: List[Tuple[str, bool, str]] = []

    # 1) Daily evaluator
    cmd_eval = [sys.executable, 'scripts/daily_evaluator.py', '--snapshot', snap_csv, '--date', eval_date]
    if refresh_prices:
        cmd_eval += ['--refresh_prices']
        if db_config:
            cmd_eval += ['--db_config', db_config]
    ok, out = _run_cmd(cmd_eval, 'Daily evaluator')
    logs.append(('Evaluator', ok, out))

    # 2) Drift
    cmd_drift = [sys.executable, 'scripts/feature_drift.py', '--snapshot', snap_csv, '--date', eval_date]
    ok2, out2 = _run_cmd(cmd_drift, 'Feature drift')
    logs.append(('Drift', ok2, out2))

    # 3) Attribution
    cmd_attr = [sys.executable, 'scripts/attribution.py', '--snapshot', snap_csv, '--date', eval_date]
    ok3, out3 = _run_cmd(cmd_attr, 'Attribution')
    logs.append(('Attribution', ok3, out3))

    # Render logs
    for name, okflag, text in logs:
        st.markdown(f"**{name}: {'✅ Success' if okflag else '❌ Failed'}**")
        st.code(text or '', language='bash')
    if all(x[1] for x in logs):
        st.success('Validation pipeline completed. Refresh tabs to see new outputs.')
    else:
        st.warning('Validation pipeline had errors. See logs above.')


# -----------------------------
# Data Ops tab (Snapshot/Eval/Drift/Attribution controls)
# -----------------------------

def _abs(p: str) -> str:
    return os.path.abspath(p)


def render_data_ops_tab(current_snapshot_dir: str) -> None:
    st.subheader('Snapshot')
    c1, c2 = st.columns([2, 1])
    with c1:
        snap_date = st.text_input('Snapshot date (YYYY-MM-DD)', value=os.path.basename(os.path.normpath(current_snapshot_dir)) if os.path.isdir(current_snapshot_dir) else '')
    with c2:
        use_rank_join = st.checkbox('Use ranked join', value=True, help='Inner-join xgb_weekly_output with xgb_ranked to fix 25 UI tickers')
    wk = st.text_input('xgb_weekly_output.csv path', value='data/top/xgb_weekly_output.csv')
    rk = st.text_input('xgb_ranked.csv path', value='data/top/xgb_ranked.csv')
    if st.button('🧊 Create/Replace Snapshot'):
        env = {}
        if use_rank_join:
            env.update({'XGB_WEEKLY_OUTPUT': _abs(wk), 'XGB_RANKED_CSV': _abs(rk)})
        cmd = [sys.executable, 'scripts/snapshot_generator.py']
        if snap_date:
            cmd += ['--date', snap_date]
        ok, out = _run_cmd(cmd, 'Snapshot generator', env=env)
        st.code(out or '', language='bash')
        if ok:
            st.success('Snapshot generated.')
        else:
            st.error('Snapshot failed.')

    st.markdown('---')
    st.subheader('Evaluator / Drift / Attribution')
    eval_date = st.text_input('Eval date (YYYY-MM-DD)', value=os.path.basename(os.path.normpath(current_snapshot_dir)))
    refresh_prices = st.checkbox('Refresh prices (DB)', value=False)
    db_conf = st.text_input('DB config path', value='stocks/conf/config.ini') if refresh_prices else ''
    prices_path = st.text_input('Prices CSV path', value='data/stock_prices_input.csv')
    features_actuals = st.text_input(
        'Features actuals CSV (optional for backfill)',
        value='/Users/gaurav/workspace/datascience/ds_uiv2/datascience_learning/weekly_prediction/data/top/featured_stocks_raw.csv'
    )

    c3, c4, c5, c6, c7 = st.columns(5)
    with c3:
        if st.button('▶️ Run Evaluator'):
            snap_csv = os.path.join(current_snapshot_dir, 'snapshot.csv')
            cmd = [sys.executable, 'scripts/daily_evaluator.py', '--snapshot', snap_csv, '--date', eval_date]
            if prices_path:
                cmd += ['--prices', os.path.abspath(prices_path)]
            if refresh_prices:
                cmd += ['--refresh_prices', '--db_config', db_conf]
            if features_actuals:
                cmd += ['--actuals_features', _abs(features_actuals)]
            ok, out = _run_cmd(cmd, 'Evaluator')
            st.code(out or '', language='bash')
            st.success('Evaluator completed.' if ok else 'Evaluator failed.')
    with c4:
        if st.button('🌀 Run Drift'):
            snap_csv = os.path.join(current_snapshot_dir, 'snapshot.csv')
            cmd = [sys.executable, 'scripts/feature_drift.py', '--snapshot', snap_csv, '--date', eval_date]
            ok, out = _run_cmd(cmd, 'Drift')
            st.code(out or '', language='bash')
            st.success('Drift completed.' if ok else 'Drift failed.')
    with c5:
        if st.button('🔎 Run Attribution'):
            snap_csv = os.path.join(current_snapshot_dir, 'snapshot.csv')
            cmd = [sys.executable, 'scripts/attribution.py', '--snapshot', snap_csv, '--date', eval_date]
            ok, out = _run_cmd(cmd, 'Attribution')
            st.code(out or '', language='bash')
            st.success('Attribution completed.' if ok else 'Attribution failed.')
    with c6:
        if st.button('✅ Run All'):
            run_validation_pipeline(current_snapshot_dir, refresh_prices=refresh_prices, db_config=db_conf if refresh_prices else None)

    with c7:
        if st.button('🚀 Run All (Unified Utility)'):
            snap_csv = os.path.join(current_snapshot_dir, 'snapshot.csv')
            cmd = [sys.executable, 'scripts/run_daily_validation.py', '--snapshot', snap_csv, '--date', eval_date]
            if prices_path:
                cmd += ['--prices', _abs(prices_path)]
            if refresh_prices:
                cmd += ['--refresh_prices']
                if db_conf:
                    cmd += ['--db_config', db_conf]
            ok, out = _run_cmd(cmd, 'Daily validation utility')
            st.code(out or '', language='bash')
            st.success('Daily validation completed.' if ok else 'Daily validation failed.')

# -----------------------------
# Streamlit App
# -----------------------------

def main() -> None:
    st.set_page_config(page_title='Weekly Validation', layout='wide')
    st.title('📅 Weekly Validation Dashboard')

    # Quick guide with color highlights
    st.markdown(
        """
        <div style='padding:10px;border:1px solid #e0e0e0;border-radius:8px;background:#fafafa'>
          <b>Guided Workflow</b> — daily model validation
          <ol style='margin-top:6px'>
            <li><span style='background:#ffe5e5;color:#b00020;padding:2px 6px;border-radius:6px'>Outcomes</span> Identify <b style='color:#b00020'>red tickers</b> (prediction sign ≠ actual)</li>
            <li><span style='background:#e3f2fd;color:#0d47a1;padding:2px 6px;border-radius:6px'>Attribution</span> See <b>top contributing features</b> for those misses</li>
            <li><span style='background:#fff8e1;color:#8d6e63;padding:2px 6px;border-radius:6px'>Drift</span> Check if those features <b style='color:#8d6e63'>drifted</b> today; adjust strategy or retrain</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander('How to use this dashboard', expanded=False):
        st.markdown(
            """
            - Pick a snapshot (week). Click ▶️ Validate week to (re)run Evaluator → Drift → Attribution.
            - In Outcomes, scan colored cells and the Red Tickers list to find misses.
            - In Attribution, toggle "Show red tickers only" to focus on failures and their top features.
            - In Drift, use "Misses with drifting features" to see if failed predictions align with feature drift.
            - Use Data Ops to regenerate snapshot/eval or tweak inputs.
            """
        )

    # Snapshot picker
    latest = find_latest_snapshot_dir()
    snap_dirs = sorted([d for d in glob.glob(os.path.join('snapshot', '*')) if os.path.isdir(d)])
    snap_choice = st.selectbox('Select snapshot:', options=snap_dirs, index=(snap_dirs.index(latest) if latest in snap_dirs else 0)) if snap_dirs else None

    if not snap_choice:
        st.info('No snapshots available. Generate one first.')
        return

    tabs = st.tabs([
        '✅ Outcomes',
        '📈 Red Tickers (charts)',
        '⚠️ Drift',
        '📊 Drift Timeline',
        '🔎 Attribution',
        '⚙️ Data Ops'
    ])

    with tabs[0]:
        render_outcomes_tab(snap_choice)

    with tabs[1]:
        render_red_tickers_tab(snap_choice)

    with tabs[2]:
        render_drift_tab(snap_choice)

    with tabs[3]:
        render_drift_timeline(snap_choice)

    with tabs[4]:
        # Reuse UI order so views align with the Outcomes tables
        ui_order = load_ui_order([
            'data/top/xgb_ranked.csv',
            'data/top/xgb_ranked_output.csv',
        ])
        render_attribution_tab(snap_choice, ui_order)

    with tabs[5]:
        render_data_ops_tab(snap_choice)


if __name__ == '__main__':
    main()


