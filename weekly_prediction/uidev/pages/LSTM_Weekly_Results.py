import streamlit as st
import pandas as pd

# Robust import for local package when running a page directly
try:
    from uidev.data_loader import load_lstm_weekly, default_sort, load_xgboost_weekly
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from uidev.data_loader import load_lstm_weekly, default_sort, load_xgboost_weekly


st.set_page_config(page_title="LSTM Weekly Results", page_icon="🔮", layout="wide")

st.title("🔮 LSTM Weekly Results")
st.caption("Raw LSTM outputs with confidence and adjusted signal columns")

df = load_lstm_weekly()

if df.empty:
    st.warning("No LSTM data found. Ensure the CSV exists under data/lstm/lstm_weekly_output.csv")
    st.stop()

# Filters
TOP_N = 25
col_filters = st.columns([1, 1, 2])
with col_filters[0]:
    min_conf = st.slider(
        "Min Confidence (adj if available)",
        float(pd.to_numeric(df.get("lstm_confidence_score_adj", pd.Series([0])), errors='coerce').fillna(0).min()),
        float(pd.to_numeric(df.get("lstm_confidence_score_adj", pd.Series([100])), errors='coerce').fillna(100).max()),
        0.0,
        key="lstm_min_conf",
    )
with col_filters[1]:
    min_signal = st.slider(
        "Min LSTM Signal (adj if available)",
        float(pd.to_numeric(df.get("lstm_predicted_return_pct_adj", pd.Series([0])), errors='coerce').fillna(0).min()),
        float(pd.to_numeric(df.get("lstm_predicted_return_pct_adj", pd.Series([1])), errors='coerce').fillna(1).max()),
        0.0,
        key="lstm_min_signal",
    )
with col_filters[2]:
    search = st.text_input("Search ticker contains", key="lstm_search")

fdf = df.copy()
conf_candidates = ["lstm_confidence_score_adj", "lstm_confidence_score", "confidence_score_adj", "confidence_score"]
for col in conf_candidates:
    if col in fdf.columns:
        fdf = fdf[pd.to_numeric(fdf[col], errors='coerce').fillna(0) >= min_conf]
        break

sig_candidates = ["lstm_predicted_return_pct_adj", "lstm_predicted_return_pct", "lstm_pred", "lstm_pred_adj"]
for col in sig_candidates:
    if col in fdf.columns:
        fdf = fdf[pd.to_numeric(fdf[col], errors='coerce').fillna(0) >= min_signal]
        break

if search:
    fdf = fdf[fdf["ticker"].astype(str).str.contains(search, case=False, na=False)]

fdf = default_sort(fdf)
fdf = fdf.head(TOP_N)

# Mark overlap with XGBoost (only if ticker is in both models' TOP_N)
try:
    xgb_df = load_xgboost_weekly()
    other_df = default_sort(xgb_df).head(TOP_N)
    if 'ticker' in other_df.columns:
        other_tickers = other_df['ticker'].astype(str).str.upper()
    elif 'symbol' in other_df.columns:
        other_tickers = other_df['symbol'].astype(str).str.upper()
    else:
        other_tickers = pd.Series([], dtype=str)
    other_set = set(other_tickers.tolist())
    this_tickers = fdf['ticker'].astype(str).str.upper() if 'ticker' in fdf.columns else (
        fdf['symbol'].astype(str).str.upper() if 'symbol' in fdf.columns else pd.Series([], dtype=str))
    fdf['in_both'] = this_tickers.isin(other_set)
except Exception:
    fdf['in_both'] = False

# Style: make overlapping tickers green
try:
    import pandas as _pd
    rows = fdf.index[fdf['in_both'] == True]
    styler = fdf.style
    if len(rows) > 0 and 'ticker' in fdf.columns:
        styler = styler.apply(lambda s: ['color: green'] * len(s), axis=1, subset=_pd.IndexSlice[rows, ['ticker']])
    st.dataframe(styler, use_container_width=True, hide_index=True)
except Exception:
    st.dataframe(fdf, use_container_width=True, hide_index=True)

st.subheader("Bar Chart: LSTM Predictions")
try:
    import plotly.express as px
    # Prefer adjusted predicted return, then predicted return, then raw pred
    y_col = 'lstm_predicted_return_pct_adj' if 'lstm_predicted_return_pct_adj' in fdf.columns else (
        'lstm_predicted_return_pct' if 'lstm_predicted_return_pct' in fdf.columns else (
            'lstm_pred' if 'lstm_pred' in fdf.columns else None))
    # Color bars green if also in XGBoost, gray otherwise
    fdf['in_both_label'] = fdf['in_both'].map({True: 'In Both', False: 'Other'})
    if y_col is not None:
        fig = px.bar(
            fdf.head(TOP_N),
            x='ticker' if 'ticker' in fdf.columns else 'symbol',
            y=y_col,
            color='in_both_label',
            color_discrete_map={'In Both': 'green', 'Other': '#A0AEC0'},
            title='Top LSTM Predictions',
            labels={y_col: 'Prediction', 'in_both_label': 'Overlap'}
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No suitable prediction column to plot.')
except Exception as _e:
    st.warning(f"Plot not available: {_e}")

