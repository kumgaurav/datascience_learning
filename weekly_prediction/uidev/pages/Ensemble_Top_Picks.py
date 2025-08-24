import streamlit as st
import pandas as pd

# Robust import for local package when running a page directly
try:
    from uidev.data_loader import load_ensemble_weekly, default_sort
except ModuleNotFoundError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from uidev.data_loader import load_ensemble_weekly, default_sort


st.set_page_config(page_title="Ensemble Top Picks", page_icon="🧠", layout="wide")

st.title("🧠 Ensemble Top Picks (Weekly)")
st.caption("Shows combined ensemble scores with XGBoost and LSTM context")

df = load_ensemble_weekly()

if df.empty:
    st.warning("No ensemble data found. Ensure the CSV exists under data/ensemble/ensemble_weekly_output.csv")
    st.stop()

# Filters
col_filters = st.columns([1, 1, 2, 2])
with col_filters[0]:
    min_conf = st.slider("Min Confidence (any model)", 0.0, 100.0, 0.0, key="ens_min_conf")
with col_filters[1]:
    min_ensemble = st.slider("Min Ensemble Score", float(df["ensemble_score"].min()) if "ensemble_score" in df else 0.0,
                             float(df["ensemble_score"].max()) if "ensemble_score" in df else 1.0, 0.0, key="ens_min_ens")
with col_filters[2]:
    search = st.text_input("Search ticker contains", key="ens_search")
with col_filters[3]:
    top_k = st.number_input("Top K", min_value=5, max_value=200, value=50, step=5, key="ens_topk")

fdf = df.copy()

# Apply ANY-model confidence filter
conf_cols = [c for c in ["xgb_confidence_score", "lstm_confidence_score", "confidence_score_xgb", "confidence_score_lstm"] if c in fdf.columns]
if conf_cols:
    conf_any = pd.concat([pd.to_numeric(fdf[c], errors='coerce').fillna(0) for c in conf_cols], axis=1).max(axis=1)
    fdf = fdf[conf_any >= min_conf]

if "ensemble_score" in fdf.columns:
    fdf = fdf[pd.to_numeric(fdf["ensemble_score"], errors='coerce').fillna(0) >= min_ensemble]

if search:
    fdf = fdf[fdf["ticker"].astype(str).str.contains(search, case=False, na=False)]

fdf = default_sort(fdf)
fdf = fdf.head(int(top_k))

st.dataframe(fdf, use_container_width=True, hide_index=True)

with st.expander("Columns description"):
    st.markdown(
        "- ticker: stock symbol\n"
        "- xgb_predicted_return_pct/lstm_predicted_return_pct: model signals\n"
        "- ensemble_score: combined ranking score\n"
        "- xgb_confidence_score/lstm_confidence_score: per-model confidence"
    )


