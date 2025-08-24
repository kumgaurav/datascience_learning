import streamlit as st
import pandas as pd

# Robust import for local package when running a page directly
try:
    from uidev.data_loader import load_xgboost_weekly, default_sort, load_lstm_weekly
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from uidev.data_loader import load_xgboost_weekly, default_sort, load_lstm_weekly


st.set_page_config(page_title="XGBoost Weekly Results", page_icon="📈", layout="wide")

st.title("📈 XGBoost Weekly Results")
st.caption("Raw XGBoost outputs with confidence and adjustment columns")

df = load_xgboost_weekly()

if df.empty:
    st.warning("No XGBoost data found. Ensure the CSV exists under data/xgboost/xgboost_weekly_output.csv")
    st.stop()

# Filters
TOP_N = 25
col_filters = st.columns([1, 1, 2])
with col_filters[0]:
    min_conf = st.slider("Min Confidence", float(max(0.0, pd.to_numeric(df.get("xgb_confidence_score_adj", pd.Series([0])), errors='coerce').fillna(0).min())),
                         float(pd.to_numeric(df.get("xgb_confidence_score_adj", pd.Series([100])), errors='coerce').fillna(100).max()), 0.0, key="xgb_min_conf")
with col_filters[1]:
    min_score = st.slider("Min XGB Score (adj)", float(pd.to_numeric(df.get("xgb_score_adj", pd.Series([0])), errors='coerce').fillna(0).min()),
                          float(pd.to_numeric(df.get("xgb_score_adj", pd.Series([1])), errors='coerce').fillna(1).max()), 0.0, key="xgb_min_score")
with col_filters[2]:
    search = st.text_input("Search ticker contains", key="xgb_search")

fdf = df.copy()
if "xgb_confidence_score_adj" in fdf.columns:
    fdf = fdf[pd.to_numeric(fdf["xgb_confidence_score_adj"], errors='coerce').fillna(0) >= min_conf]
elif "xgb_confidence_score" in fdf.columns:
    fdf = fdf[pd.to_numeric(fdf["xgb_confidence_score"], errors='coerce').fillna(0) >= min_conf]

if "xgb_score_adj" in fdf.columns:
    fdf = fdf[pd.to_numeric(fdf["xgb_score_adj"], errors='coerce').fillna(0) >= min_score]
elif "xgb_score" in fdf.columns:
    fdf = fdf[pd.to_numeric(fdf["xgb_score"], errors='coerce').fillna(0) >= min_score]

if search:
    fdf = fdf[fdf["ticker"].astype(str).str.contains(search, case=False, na=False)]

fdf = default_sort(fdf)
fdf = fdf.head(TOP_N)

# Mark overlap with LSTM (only if ticker is in both models' TOP_N)
try:
    lstm_df = load_lstm_weekly()
    other_df = default_sort(lstm_df).head(TOP_N)
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

st.subheader("Bar Chart: XGBoost Predictions")
try:
    import plotly.express as px
    y_col = 'xgb_score_adj' if 'xgb_score_adj' in fdf.columns else (
        'xgb_score' if 'xgb_score' in fdf.columns else (
            'xgb_predicted_return_pct' if 'xgb_predicted_return_pct' in fdf.columns else None))
    # Color bars green if also in LSTM, gray otherwise
    fdf['in_both_label'] = fdf['in_both'].map({True: 'In Both', False: 'Other'})
    if y_col is not None:
        fig = px.bar(
            fdf.head(TOP_N),
            x='ticker' if 'ticker' in fdf.columns else 'symbol',
            y=y_col,
            color='in_both_label',
            color_discrete_map={'In Both': 'green', 'Other': '#A0AEC0'},
            title='Top XGBoost Predictions',
            labels={y_col: 'Prediction', 'in_both_label': 'Overlap'}
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No suitable prediction column to plot.')
except Exception as _e:
    st.warning(f"Plot not available: {_e}")

with st.expander("Columns description"):
    st.markdown(
        "- ticker, date, close, volume: base data\n"
        "- xgb_score/xgb_score_adj: raw vs adjusted model signals\n"
        "- xgb_confidence_score/xgb_confidence_score_adj: model confidence metrics\n"
        "- w_pen: weighting/penalty factor"
    )


