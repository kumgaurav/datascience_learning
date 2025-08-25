import streamlit as st
import pandas as pd
import os

# Robust import when running page directly or after hot-reload
try:
    from uidev.data_loader import load_momentum_weekly, load_xgboost_weekly, load_lstm_weekly
except (ModuleNotFoundError, ImportError):
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from uidev.data_loader import load_momentum_weekly, load_xgboost_weekly, load_lstm_weekly


st.set_page_config(page_title="Weekly Prediction Dashboard", page_icon="🏠", layout="wide")

st.title("🏠 Weekly Prediction Dashboard")
st.caption("Navigate to the new modular views to explore the latest results.")

st.markdown("### Quick Links")

# When running a page file directly, Streamlit may not register other pages,
# so avoid page_link/switch_page here to prevent KeyError. Show guidance instead.
st.write("Use the left sidebar Pages to open:")
st.markdown("- 🧠 Ensemble Top Picks")
st.markdown("- 📈 XGBoost Weekly Results")
st.markdown("- 🔮 LSTM Weekly Results")

st.divider()
st.markdown("### 🚀 Top 25 Momentum Stocks (Weekly)")

mom = load_momentum_weekly()
if mom.empty:
    st.info("Momentum weekly output not found at data/momentum/momentum_weekly_output.csv")
else:
    # Pick score column and sort desc
    score_col = None
    for c in ["momentum_pred", "momentum_5d", "momentum_20d"]:
        if c in mom.columns:
            score_col = c
            break
    if score_col is None:
        st.warning("Momentum score columns missing; cannot rank.")
    else:
        mdf = mom.copy()
        mdf[score_col] = pd.to_numeric(mdf[score_col], errors='coerce')
        mdf = mdf.sort_values(score_col, ascending=False).head(25)
        # Keep a compact set of columns if present
        keep = [c for c in ["ticker", "date", "close", "momentum_5d", "momentum_20d", "momentum_pred", "confidence_score"] if c in mdf.columns]
        mdf = mdf[keep]

        # Highlight tickers present in either LSTM or XGBoost weekly outputs
        try:
            xdf = load_xgboost_weekly()
        except Exception:
            xdf = pd.DataFrame()
        try:
            ldf = load_lstm_weekly()
        except Exception:
            ldf = pd.DataFrame()

        # Build sets of tickers that are in the TOP 25 of each model
        in_any = set()
        if 'ticker' in xdf.columns and not xdf.empty:
            x_score_col = None
            for c in ["xgb_score_adj", "xgb_score", "xgb_pred", "xgb_predicted_return_pct"]:
                if c in xdf.columns:
                    x_score_col = c
                    break
            if x_score_col is not None:
                xdf['_score_tmp_'] = pd.to_numeric(xdf[x_score_col], errors='coerce')
                x_top = xdf.sort_values('_score_tmp_', ascending=False).head(25)
                in_any.update(x_top['ticker'].astype(str).str.upper().tolist())
                xdf.drop(columns=['_score_tmp_'], inplace=True, errors='ignore')
        if 'ticker' in ldf.columns and not ldf.empty:
            l_score_col = None
            for c in ["lstm_predicted_return_pct_adj", "lstm_pred_adj", "lstm_pred", "lstm_predicted_return_pct"]:
                if c in ldf.columns:
                    l_score_col = c
                    break
            if l_score_col is not None:
                ldf['_score_tmp_'] = pd.to_numeric(ldf[l_score_col], errors='coerce')
                l_top = ldf.sort_values('_score_tmp_', ascending=False).head(25)
                in_any.update(l_top['ticker'].astype(str).str.upper().tolist())
                ldf.drop(columns=['_score_tmp_'], inplace=True, errors='ignore')

        if 'ticker' in mdf.columns and len(in_any) > 0:
            def _style_ticker(col: pd.Series) -> list[str]:
                if col.name != 'ticker':
                    return [''] * len(col)
                return ['color: green' if str(v).upper() in in_any else '' for v in col]
            st.dataframe(mdf.style.apply(_style_ticker, axis=0), use_container_width=True, hide_index=True)
        else:
            st.dataframe(mdf, use_container_width=True, hide_index=True)



