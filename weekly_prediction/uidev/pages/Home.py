import streamlit as st


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
st.markdown(
    """
    - The app now uses modular pages under `uidev/pages`.
    - Data is loaded from CSVs under `data/ensemble` and `data/xgboost`.
    - You can continue to use legacy features from the main app page if needed.
    """
)


