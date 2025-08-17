import os
from datetime import timedelta

import pandas as pd
import joblib
import streamlit as st

from charts.stock_charts import create_stock_price_chart


st.set_page_config(page_title="Momentum Winners", page_icon="🚀", layout="wide")


def load_featured() -> pd.DataFrame:
    featured_path = os.getenv('FEATURED_STOCKS_MOMENTUM_CSV', 'data/featured_stocks_momentum.csv')
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

    winners = compute_momentum_winners(featured)
    # Filter out low-priced stocks (< $2)
    winners['close'] = pd.to_numeric(winners['close'], errors='coerce')
    winners = winners[winners['close'] >= 2].copy()
    
    # Add weekly predictions
    winners = predict_weekly_returns(winners)
    # Sort primarily by predicted weekly return (desc), fallback to momentum score
    if 'predicted_weekly_return_pct' in winners.columns:
        winners = winners.sort_values('predicted_weekly_return_pct', ascending=False)
    elif 'mw_score' in winners.columns:
        winners = winners.sort_values('mw_score', ascending=False)

    if winners.empty:
        st.info("No momentum winners matched the current rules. Re-run analysis or refresh data.")
    else:
        # Top 20 table
        st.subheader("Top 20 Momentum Winners")
        top20 = winners.head(20).copy()
        cols = [
            c for c in [
                'ticker', 'close', 'predicted_weekly_return_pct', 'momentum_20d', 'momentum_30d', 'momentum_60d',
                'up_day_ratio_20d', 'rsi_14d', 'mw_score'
            ] if c in top20.columns
        ]
        table = top20[cols].copy()
        if 'close' in table.columns:
            table['close'] = table['close'].map(lambda x: f"${x:.2f}")
        if 'predicted_weekly_return_pct' in table.columns:
            table['predicted_weekly_return_pct'] = table['predicted_weekly_return_pct'].map(lambda x: f"{x:.2f}%")
        for c in ['momentum_20d', 'momentum_30d', 'momentum_60d', 'up_day_ratio_20d']:
            if c in table.columns:
                if c == 'up_day_ratio_20d':
                    table[c] = table[c].map(lambda x: f"{x*100:.1f}%")
                else:
                    table[c] = table[c].map(lambda x: f"{x*100:.1f}%")
        if 'mw_score' in table.columns:
            table['mw_score'] = table['mw_score'].map(lambda x: f"{x:.3f}")
        # Highlight rows that also appear in Top Stocks (from main page)
        top_set = set()
        try:
            if 'top_stocks_df' in st.session_state and not st.session_state.top_stocks_df.empty:
                top_set = set(st.session_state.top_stocks_df['ticker'].astype(str).tolist())
        except Exception:
            top_set = set()

        def _row_style(row: pd.Series):
            highlight = (row.get('ticker') in top_set)
            style = 'background-color: #2e7d32; color: white' if highlight else ''
            return [style] * len(row)

        try:
            styled = table.style.apply(_row_style, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(table, use_container_width=True, hide_index=True)

        # Selection for deep dive
        st.subheader("🔍 Analyze a Winner")
        sel = st.selectbox("Select ticker", top20['ticker'].tolist())

        if sel:
            row = winners[winners['ticker'] == sel].iloc[0]
            tpx = prices[prices['ticker'] == sel].copy()

            if not tpx.empty:
                tpx['date'] = pd.to_datetime(tpx['date'])
                tpx = tpx.sort_values('date')
                start_dt = tpx['date'].max() - timedelta(days=90)
                recent = tpx[tpx['date'] >= start_dt].copy()

                # Chart
                chart = create_stock_price_chart(sel, recent, row)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)

                # Metrics
                st.subheader("📈 Momentum Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Price", f"${row.get('close', 0):.2f}")
                with col2:
                    st.metric("Predicted Weekly Return", f"{row.get('predicted_weekly_return_pct', 0):.2f}%")
                with col3:
                    st.metric("30d Momentum", f"{row.get('momentum_30d', 0)*100:.1f}%")
                with col4:
                    st.metric("60d Momentum", f"{row.get('momentum_60d', 0)*100:.1f}%")

                colx1, colx2 = st.columns(2)
                with colx1:
                    st.metric("Up-Day Ratio (20d)", f"{row.get('up_day_ratio_20d', 0)*100:.1f}%")
                with colx2:
                    st.metric("Momentum (20d)", f"{row.get('momentum_20d', 0)*100:.1f}%")

                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    st.metric("RSI (14d)", f"{row.get('rsi_14d', 0):.1f}")
                with col6:
                    st.metric("MACD", f"{row.get('macd', 0):.3f}")
                with col7:
                    st.metric("Signal", f"{row.get('macd_signal', 0):.3f}")
                with col8:
                    st.metric("Trend Slope (30d)", f"{row.get('trend_slope_30d', 0):.3f}")

                # Explanations
                st.subheader("📝 Why it qualifies")
                reasons = []
                if row.get('momentum_60d', 0) > 0.5:
                    reasons.append("60d momentum > 50%")
                if row.get('momentum_30d', 0) > 0.2:
                    reasons.append("30d momentum > 20%")
                if row.get('up_day_ratio_20d', 0) > 0.55:
                    reasons.append("Up-day ratio > 55% over last 20d")
                if row.get('close', 0) > row.get('ma_20', 0):
                    reasons.append("Price above 20d MA")
                if row.get('macd', 0) > row.get('macd_signal', 0):
                    reasons.append("MACD > Signal (positive momentum)")
                if row.get('golden_cross', False):
                    reasons.append("Golden Cross detected")
                if not reasons:
                    reasons.append("Meets custom persistence criteria")
                st.write("- " + "\n- ".join(reasons))

            else:
                st.warning(f"No price data found for {sel}")
except Exception as e:
    st.error(f"Error rendering Momentum Winners page: {str(e)}")


