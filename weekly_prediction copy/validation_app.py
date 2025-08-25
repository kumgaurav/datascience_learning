import os
import glob
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Data loading utilities
# -----------------------------

def find_latest_snapshot_dir(base_dir: str = "snapshot") -> Optional[str]:
    try:
        subdirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
        return sorted(subdirs)[-1] if subdirs else None
    except Exception:
        return None


def load_ui_order(rank_paths: List[str], rank_by: str = 'confidence') -> List[str]:
    for path in rank_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if 'ticker' not in df.columns:
                    continue
                df['ticker'] = df['ticker'].astype(str).str.upper()
                # Choose ordering metric if available
                if rank_by == 'ensemble' and 'ensemble_score' in df.columns:
                    df = df.sort_values('ensemble_score', ascending=False)
                elif 'confidence_score' in df.columns:
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
        per_day.append(d.set_index('ticker')[[pred_col, 'actual_return_pct']])

    rows = []
    for tkr in ui_order:
        row = {'Ticker': tkr}
        for lbl, day_df in zip(day_labels, per_day):
            if tkr in day_df.index:
                pred = day_df.loc[tkr, pred_col]
                act = day_df.loc[tkr, 'actual_return_pct']
                if pd.notna(pred) and pd.notna(act):
                    row[lbl] = f"{pred:+.2f}% / {act:+.2f}%"
                else:
                    row[lbl] = "-"
            else:
                row[lbl] = "-"
        rows.append(row)
    return pd.DataFrame(rows)


def style_outcomes_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def _color_cell(val: str) -> str:
        try:
            if val and isinstance(val, str) and '%' in val and '/' in val:
                p_txt, a_txt = [s.strip().replace('%', '') for s in val.split('/')]
                a = float(a_txt)
                # Color by actual return sign (green positive, red negative)
                if a > 0:
                    return 'color: green;'
                elif a < 0:
                    return 'color: red;'
        except Exception:
            pass
        return ''

    subset_cols = df.columns[1:]
    return df.style.applymap(_color_cell, subset=subset_cols)


def render_outcomes_tab(snapshot_dir: str) -> None:
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

        fig, ax = plt.subplots(figsize=(min(12, 0.6 * len(day_labels)), min(10, 0.3 * len(ui_order))))
        cmap = plt.get_cmap('RdYlGn')
        # Map -1 -> red, 0 -> gray, 1 -> green
        data = np.array(sign_matrix)
        im = ax.imshow(data, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        ax.set_yticks(range(len(ui_order)))
        ax.set_yticklabels(ui_order, fontsize=8)
        ax.set_xticks(range(len(day_labels)))
        ax.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=8)
        ax.set_title('Actual Returns Sign by Day (Green=Up, Red=Down)')
        plt.tight_layout()
        st.pyplot(fig)
    except Exception:
        pass


# -----------------------------
# Streamlit App
# -----------------------------

def main() -> None:
    st.set_page_config(page_title='Weekly Validation', layout='wide')
    st.title('📅 Weekly Validation Dashboard')

    # Snapshot picker
    latest = find_latest_snapshot_dir()
    snap_dirs = sorted([d for d in glob.glob(os.path.join('snapshot', '*')) if os.path.isdir(d)])
    snap_choice = st.selectbox('Select snapshot:', options=snap_dirs, index=(snap_dirs.index(latest) if latest in snap_dirs else 0)) if snap_dirs else None

    if not snap_choice:
        st.info('No snapshots available. Generate one first.')
        return

    tabs = st.tabs([
        '✅ Outcomes',
        '📈 Charts (optional)',
        '⚠️ Drift (future)',
        '🔎 Attribution (future)'
    ])

    with tabs[0]:
        render_outcomes_tab(snap_choice)

    with tabs[1]:
        st.info('Additional charts and analyses can be added here without affecting the outcomes tab.')

    with tabs[2]:
        st.info('Feature drift visualizations can be integrated here later.')

    with tabs[3]:
        st.info('Attribution details (e.g., SHAP) can be added here later.')


if __name__ == '__main__':
    main()


