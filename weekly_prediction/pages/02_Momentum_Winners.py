import os
from datetime import timedelta

import pandas as pd
import joblib
import streamlit as st

from charts.stock_charts import create_stock_price_chart


st.set_page_config(page_title="Momentum Winners", page_icon="🚀", layout="wide")


def load_featured() -> pd.DataFrame:
    featured_path = os.getenv('FEATURED_STOCKS_MOMENTUM_CSV', 'data/momentum/featured_stocks_momentum.csv')
    return pd.read_csv(featured_path)


def load_prices() -> pd.DataFrame:
    prices_path = os.getenv('STOCK_PRICES_CSV', 'data/stock_prices.csv')
    return pd.read_csv(prices_path)


def compute_momentum_winners(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required columns exist
    for col in [
        'momentum_winner', 'momentum_30d', 'momentum_60d', 'trend_slope_15d',
        'trend_slope_30d', 'ma_20', 'macd', 'macd_signal', 'close'
    ]:
        if col not in df.columns:
            df[col] = 0

    if 'momentum_winner' in df.columns and df['momentum_winner'].notna().any():
        mask = df['momentum_winner'] == True
    else:
        # Fallback rules if flag not provided
        mask = (
            ((df['momentum_60d'] > 0.5) | (df['momentum_30d'] > 0.2)) &
            (df['trend_slope_15d'] > 0) &
            (df['close'] > df['ma_20']) &
            (df['macd'] > df['macd_signal'])
        )

    winners = df[mask].copy()

    def mw_score(row: pd.Series) -> float:
        m60 = row.get('momentum_60d', 0) if pd.notna(row.get('momentum_60d', 0)) else 0
        m30 = row.get('momentum_30d', 0) if pd.notna(row.get('momentum_30d', 0)) else 0
        slope30 = row.get('trend_slope_30d', 0)
        slope30 = slope30 if pd.notna(slope30) else 0
        return 0.6 * m60 + 0.3 * m30 + 0.1 * slope30

    if not winners.empty:
        winners['mw_score'] = winners.apply(mw_score, axis=1)
        winners = winners.sort_values('mw_score', ascending=False)

    return winners


def predict_weekly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add predicted 5-trading-day (weekly) return percentage using the trained model.
    Returns the same DataFrame with column 'predicted_weekly_return_pct' if model is available.
    """
    try:
        model = joblib.load('models/stock_predictor_momentum.joblib')
    except Exception as e:
        st.info(f"Model not available for weekly predictions: {str(e)}")
        return df

    # Determine training columns expected by the model
    try:
        training_cols = model.get_booster().feature_names
    except Exception:
        training_cols = None

    if not training_cols:
        st.info("Model feature names unavailable; skipping weekly predictions.")
        return df

    # Ensure all training columns exist in df
    df_pred = df.copy()
    for col in training_cols:
        if col not in df_pred.columns:
            df_pred[col] = 0

    X = df_pred[training_cols]
    try:
        preds_pct = model.predict(X)
        # Model is trained to output percentage change over 5 trading days
        df_pred['predicted_weekly_return_pct'] = preds_pct
    except Exception as e:
        st.info(f"Could not compute weekly predictions: {str(e)}")
        df_pred['predicted_weekly_return_pct'] = 0.0

    return df_pred


st.title("🚀 Momentum Winners")
st.write("Stocks demonstrating persistent multi-week upside trends with supportive momentum signals.")

try:
    featured = load_featured()
    prices = load_prices()

    # Sub-tabs: 5d / 15d / 30d Momentum
    import os
    import pandas as _pd
    mom5_path = os.path.join('data', 'momentum', 'top5d_performers.csv')
    mom15_path = os.path.join('data', 'momentum', 'top15d_performers.csv')
    mom30_path = os.path.join('data', 'momentum', 'top30d_performers.csv')

    df5 = _pd.read_csv(mom5_path) if os.path.exists(mom5_path) else _pd.DataFrame()
    df15 = _pd.read_csv(mom15_path) if os.path.exists(mom15_path) else _pd.DataFrame()
    df30 = _pd.read_csv(mom30_path) if os.path.exists(mom30_path) else _pd.DataFrame()

    # Ensure ticker is string for set ops
    for _df in (df5, df15, df30):
        if not _df.empty and 'ticker' in _df.columns:
            _df['ticker'] = _df['ticker'].astype(str).str.upper()

    s5 = set(df5['ticker']) if not df5.empty and 'ticker' in df5.columns else set()
    s15 = set(df15['ticker']) if not df15.empty and 'ticker' in df15.columns else set()
    s30 = set(df30['ticker']) if not df30.empty and 'ticker' in df30.columns else set()

    common_all = s5 & s15 & s30
    common_5_15_only = (s5 & s15) - s30
    common_15_30_only = (s15 & s30) - s5

    def _row_color_style(row):
        t = str(row.get('ticker', '')).upper()
        if t in common_all:
            color = '#c8e6c9'  # green
        elif t in common_5_15_only:
            color = '#bbdefb'  # blue
        elif t in common_15_30_only:
            color = '#fff9c4'  # yellow
        else:
            color = ''
        return [f'background-color: {color}' if color else '' for _ in row]

    tab5, tab15, tab30 = st.tabs(["Top 5d", "Top 15d", "Top 30d"])

    def _render_5d_tab():
        df_src = df5
        if df_src is None or df_src.empty:
            st.info("No data available for this timeframe.")
            return
        st.markdown(
            "Row color legend: "
            "<span style='background-color:#c8e6c9;padding:2px 6px;border-radius:3px;'>Green</span> = common in 5d, 15d, 30d; "
            "<span style='background-color:#bbdefb;padding:2px 6px;border-radius:3px;'>Blue</span> = common in 5d & 15d only; "
            "<span style='background-color:#fff9c4;padding:2px 6px;border-radius:3px;'>Yellow</span> = common in 15d & 30d only",
            unsafe_allow_html=True
        )
        def _row_color_style_5d(row):
            t = str(row.get('ticker', '')).upper()
            if t in common_all:
                color = '#c8e6c9'
            elif t in common_5_15_only:
                color = '#bbdefb'
            elif t in common_15_30_only:
                color = '#fff9c4'
            else:
                color = ''
            return [f'background-color: {color}' if color else '' for _ in row]
        try:
            styled = df_src.style.apply(_row_color_style_5d, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(df_src, use_container_width=True, hide_index=True)
        st.subheader("🔍 Analyze")
        tickers = df_src['ticker'].tolist() if 'ticker' in df_src.columns else []
        sel = st.selectbox("Select ticker", tickers, key="sel_5d")
        analyze = st.button("Analyze", type="primary", key="analyze_5d")
        if analyze and sel:
            try:
                feat_row = None
                if featured is not None and isinstance(featured, _pd.DataFrame) and not featured.empty:
                    _feat = featured[featured['ticker'].astype(str).str.upper() == sel]
                    if not _feat.empty:
                        feat_row = _feat.iloc[0]
                if feat_row is None:
                    feat_row = _pd.Series({'ticker': sel, 'close': _pd.NA})
                tpx = prices[prices['ticker'].astype(str).str.upper() == sel].copy() if prices is not None else _pd.DataFrame()
                if not tpx.empty:
                    tpx['date'] = _pd.to_datetime(tpx['date'])
                    tpx = tpx.sort_values('date')
                    start_dt = tpx['date'].max() - timedelta(days=90)
                    recent = tpx[tpx['date'] >= start_dt].copy()
                    chart = create_stock_price_chart(sel, recent, feat_row)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    st.warning(f"No price data found for {sel}")
            except Exception as _e:
                st.warning(f"Could not render analysis for {sel}: {str(_e)}")

    def _render_15d_tab():
        df_src = df15
        if df_src is None or df_src.empty:
            st.info("No data available for this timeframe.")
            return
        st.markdown(
            "Row color legend: "
            "<span style='background-color:#c8e6c9;padding:2px 6px;border-radius:3px;'>Green</span> = common in 5d, 15d, 30d; "
            "<span style='background-color:#bbdefb;padding:2px 6px;border-radius:3px;'>Blue</span> = common in 5d & 15d only; "
            "<span style='background-color:#fff9c4;padding:2px 6px;border-radius:3px;'>Yellow</span> = common in 15d & 30d only",
            unsafe_allow_html=True
        )
        def _row_color_style_15d(row):
            t = str(row.get('ticker', '')).upper()
            if t in common_all:
                color = '#c8e6c9'
            elif t in common_5_15_only:
                color = '#bbdefb'
            elif t in common_15_30_only:
                color = '#fff9c4'
            else:
                color = ''
            return [f'background-color: {color}' if color else '' for _ in row]
        try:
            styled = df_src.style.apply(_row_color_style_15d, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(df_src, use_container_width=True, hide_index=True)
        st.subheader("🔍 Analyze")
        tickers = df_src['ticker'].tolist() if 'ticker' in df_src.columns else []
        sel = st.selectbox("Select ticker", tickers, key="sel_15d")
        analyze = st.button("Analyze", type="primary", key="analyze_15d")
        if analyze and sel:
            try:
                feat_row = None
                if featured is not None and isinstance(featured, _pd.DataFrame) and not featured.empty:
                    _feat = featured[featured['ticker'].astype(str).str.upper() == sel]
                    if not _feat.empty:
                        feat_row = _feat.iloc[0]
                if feat_row is None:
                    feat_row = _pd.Series({'ticker': sel, 'close': _pd.NA})
                tpx = prices[prices['ticker'].astype(str).str.upper() == sel].copy() if prices is not None else _pd.DataFrame()
                if not tpx.empty:
                    tpx['date'] = _pd.to_datetime(tpx['date'])
                    tpx = tpx.sort_values('date')
                    start_dt = tpx['date'].max() - timedelta(days=90)
                    recent = tpx[tpx['date'] >= start_dt].copy()
                    chart = create_stock_price_chart(sel, recent, feat_row)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    st.warning(f"No price data found for {sel}")
            except Exception as _e:
                st.warning(f"Could not render analysis for {sel}: {str(_e)}")

    def _render_30d_tab():
        df_src = df30
        if df_src is None or df_src.empty:
            st.info("No data available for this timeframe.")
            return
        st.markdown(
            "Row color legend: "
            "<span style='background-color:#c8e6c9;padding:2px 6px;border-radius:3px;'>Green</span> = common in 5d, 15d, 30d; "
            "<span style='background-color:#bbdefb;padding:2px 6px;border-radius:3px;'>Blue</span> = common in 5d & 15d only; "
            "<span style='background-color:#fff9c4;padding:2px 6px;border-radius:3px;'>Yellow</span> = common in 15d & 30d only",
            unsafe_allow_html=True
        )
        def _row_color_style_30d(row):
            t = str(row.get('ticker', '')).upper()
            if t in common_all:
                color = '#c8e6c9'
            elif t in common_5_15_only:
                color = '#bbdefb'
            elif t in common_15_30_only:
                color = '#fff9c4'
            else:
                color = ''
            return [f'background-color: {color}' if color else '' for _ in row]
        try:
            styled = df_src.style.apply(_row_color_style_30d, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(df_src, use_container_width=True, hide_index=True)
        st.subheader("🔍 Analyze")
        tickers = df_src['ticker'].tolist() if 'ticker' in df_src.columns else []
        sel = st.selectbox("Select ticker", tickers, key="sel_30d")
        analyze = st.button("Analyze", type="primary", key="analyze_30d")
        if analyze and sel:
            try:
                feat_row = None
                if featured is not None and isinstance(featured, _pd.DataFrame) and not featured.empty:
                    _feat = featured[featured['ticker'].astype(str).str.upper() == sel]
                    if not _feat.empty:
                        feat_row = _feat.iloc[0]
                if feat_row is None:
                    feat_row = _pd.Series({'ticker': sel, 'close': _pd.NA})
                tpx = prices[prices['ticker'].astype(str).str.upper() == sel].copy() if prices is not None else _pd.DataFrame()
                if not tpx.empty:
                    tpx['date'] = _pd.to_datetime(tpx['date'])
                    tpx = tpx.sort_values('date')
                    start_dt = tpx['date'].max() - timedelta(days=90)
                    recent = tpx[tpx['date'] >= start_dt].copy()
                    chart = create_stock_price_chart(sel, recent, feat_row)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                else:
                    st.warning(f"No price data found for {sel}")
            except Exception as _e:
                st.warning(f"Could not render analysis for {sel}: {str(_e)}")

    with tab5:
        _render_5d_tab()
    with tab15:
        _render_15d_tab()
    with tab30:
        _render_30d_tab()

    # Removed Top 20 Momentum Winners and related components as requested
except Exception as e:
    st.error(f"Error rendering Momentum Winners page: {str(e)}")


