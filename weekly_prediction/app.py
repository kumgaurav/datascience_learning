import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from stock_selector_v2 import get_top_stocks, get_stock_analysis
from stock_selector import calculate_confidence_score, calculate_risk_score
from chatbot import create_chatbot_agent
import os
import joblib
from dotenv import load_dotenv
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Load environment variables from .env file
load_dotenv()

def create_stock_price_chart(ticker, stock_data, featured_data):
    """
    Create a 3-week price chart with resistance break and momentum highlights.
    
    Args:
        ticker (str): Stock ticker symbol
        stock_data (pd.DataFrame): Historical price data for the stock
        featured_data (pd.Series): Featured data containing technical indicators
    
    Returns:
        plotly.graph_objects.Figure: Interactive price chart
    """
    try:
        # Filter data for the specific ticker
        ticker_data = stock_data[stock_data['ticker'] == ticker].copy()
        
        if ticker_data.empty:
            return None
        
        # Convert date column to datetime
        ticker_data['date'] = pd.to_datetime(ticker_data['date'])
        
        # Sort by date and get last 3 weeks (21 trading days)
        ticker_data = ticker_data.sort_values('date')
        last_date = ticker_data['date'].max()
        start_date = last_date - timedelta(days=30)  # 30 days to ensure we get 21 trading days
        
        # Filter for last 3 weeks
        recent_data = ticker_data[ticker_data['date'] >= start_date].copy()
        
        if len(recent_data) < 5:  # Need at least 5 days for meaningful chart
            return None
        
        # Get technical indicators from featured data
        resistance_level = featured_data.get('resistance_20d', None)
        support_level = featured_data.get('support_20d', None)
        broke_resistance = featured_data.get('broke_resistance', False)
        momentum_5d = featured_data.get('momentum_5d', 0)
        momentum_10d = featured_data.get('momentum_10d', 0)
        rsi_14d = featured_data.get('rsi_14d', 50)
        
        # Create the main price chart
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=recent_data['date'],
            open=recent_data['open'],
            high=recent_data['high'],
            low=recent_data['low'],
            close=recent_data['close'],
            name='Price',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        ))
        
        # Add resistance line if available
        if resistance_level is not None and not pd.isna(resistance_level):
            fig.add_hline(
                y=resistance_level,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Resistance: ${resistance_level:.2f}",
                annotation_position="top right"
            )
            
            # Highlight resistance break if it occurred
            if broke_resistance:
                # Find the first day price closed above resistance
                break_data = recent_data[recent_data['close'] > resistance_level]
                if not break_data.empty:
                    break_date = break_data.iloc[0]['date']
                    break_price = break_data.iloc[0]['close']
                    
                    fig.add_annotation(
                        x=break_date,
                        y=break_price,
                        text=f"🟢 Resistance Break<br>${break_price:.2f}<br>Resistance: ${resistance_level:.2f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="green",
                        bgcolor="green",
                        bordercolor="white",
                        borderwidth=2
                    )
        
        # Add support line if available
        if support_level is not None and not pd.isna(support_level):
            fig.add_hline(
                y=support_level,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Support: ${support_level:.2f}",
                annotation_position="bottom right"
            )
        
        # Add momentum indicators
        if momentum_5d > 0 and momentum_10d > 0:
            # Find the day with strongest momentum
            momentum_data = recent_data.tail(5)  # Last 5 days
            if not momentum_data.empty:
                max_momentum_idx = momentum_data['close'].pct_change().idxmax()
                if not pd.isna(max_momentum_idx):
                    momentum_date = recent_data.loc[max_momentum_idx, 'date']
                    momentum_price = recent_data.loc[max_momentum_idx, 'close']
                    
                    fig.add_annotation(
                        x=momentum_date,
                        y=momentum_price,
                        text=f"🚀 Momentum<br>${momentum_price:.2f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="orange",
                        bgcolor="orange",
                        bordercolor="white",
                        borderwidth=2
                    )
        
        # Add RSI indicator
        if rsi_14d is not None and not pd.isna(rsi_14d):
            rsi_color = "green" if rsi_14d > 70 else "red" if rsi_14d < 30 else "gray"
            rsi_text = f"RSI: {rsi_14d:.1f}"
            
            # Add RSI annotation on the last day
            last_date = recent_data['date'].iloc[-1]
            last_price = recent_data['close'].iloc[-1]
            
            fig.add_annotation(
                x=last_date,
                y=last_price,
                text=rsi_text,
                showarrow=False,
                bgcolor=rsi_color,
                bordercolor="white",
                borderwidth=1,
                xanchor="left",
                yanchor="bottom"
            )
        
        # Update layout
        fig.update_layout(
            title=f"{ticker} - 3-Week Price Chart with Technical Analysis",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        # Add volume as subplot
        fig.add_trace(go.Bar(
            x=recent_data['date'],
            y=recent_data['volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3
        ))
        
        # Update layout to include volume subplot
        fig.update_layout(
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart for {ticker}: {str(e)}")
        return None

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Stock Screener Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Configuration ---
st.sidebar.title("🎯 Stock Screener Settings")

# Filter controls
st.sidebar.header("📊 Filter Settings")
min_confidence = st.sidebar.slider("Minimum Confidence Score", 0, 100, 30, help="Higher = more confident predictions")
max_risk = st.sidebar.slider("Maximum Risk Score", 0, 100, 70, help="Lower = less risky stocks")
diversify = st.sidebar.checkbox("Apply Portfolio Diversification", True, help="Limit exposure per sector/industry")
# Default to 25 stocks
num_stocks = st.sidebar.selectbox("Number of Stocks", [10, 15, 20, 25, 30], index=3)

# API Key Status
st.sidebar.header("🔑 API Configuration")
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    st.sidebar.success("✅ Google API Key Found")
    st.sidebar.info(f"Key: {api_key[:10]}...")
else:
    st.sidebar.error("❌ Google API Key Missing")
    st.sidebar.markdown("""
    **To enable AI chatbot:**
    1. Create a `.env` file in the project root
    2. Add: `GOOGLE_API_KEY=your_actual_api_key`
    3. Restart the app
    """)

# (Momentum Winners sidebar removed; now available as its own page under left navigation)

# --- App Title ---
st.title("📈 AI-Powered Stock Screener Pro")
st.write("Advanced stock analysis using XGBoost, technical indicators, and fundamental data.")

# --- Main Logic ---
# Use session state to store data and avoid re-running analysis on every interaction
if 'top_stocks_df' not in st.session_state:
    st.session_state.top_stocks_df = pd.DataFrame()
if 'agent' not in st.session_state:
    st.session_state.agent = None

# "Run Analysis" button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Find Top Stocks for This Week", type="primary", use_container_width=True):
        with st.spinner("Running advanced analysis... This may take a moment."):
            st.session_state.top_stocks_df = get_top_stocks(
                n=num_stocks, 
                min_confidence=min_confidence, 
                max_risk=max_risk, 
                diversify=diversify
            )
            
            # Debug information
            st.info(f"📊 Found {len(st.session_state.top_stocks_df)} stocks matching your criteria")
            if not st.session_state.top_stocks_df.empty:
                st.write(f"Top stock: {st.session_state.top_stocks_df.iloc[0]['ticker']} with {st.session_state.top_stocks_df.iloc[0]['predicted_return_pct']:.2f}% predicted return")
            
            # After getting stocks, create the chatbot agent if data is available
            if not st.session_state.top_stocks_df.empty:
                try:
                    st.session_state.agent = create_chatbot_agent(st.session_state.top_stocks_df)
                    if hasattr(st.session_state.agent, 'invoke'):
                        st.success("✅ Analysis complete! AI Chatbot initialized successfully!")
                    else:
                        st.warning("⚠️ Analysis complete! Using simplified analysis mode (no AI).")
                except Exception as e:
                    st.error(f"❌ Failed to initialize chatbot: {str(e)}")
                    st.session_state.agent = None

# --- Display Results ---
if not st.session_state.top_stocks_df.empty:

    
    # --- Summary Metrics ---
    st.header("📊 Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", len(st.session_state.top_stocks_df))
    with col2:
        avg_return = st.session_state.top_stocks_df['predicted_return_pct'].mean()
        st.metric("Avg Predicted Return", f"{avg_return:.2f}%")
    with col3:
        avg_confidence = st.session_state.top_stocks_df['confidence_score'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1f}/100")
    with col4:
        avg_risk = st.session_state.top_stocks_df['risk_score'].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.1f}/100")
    
    # --- Top Pick Highlight ---
    st.header("🏆 Top Pick")
    best_stock = st.session_state.top_stocks_df.iloc[0]
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.success(f"**{best_stock['ticker']}** - Predicted Return: **{best_stock['predicted_return_pct']:.2f}%**")
        st.write(f"Confidence: {best_stock['confidence_score']:.1f}/100 | Risk: {best_stock['risk_score']:.1f}/100")
    with col2:
        st.metric("Current Price", f"${best_stock['close']:.2f}")
    with col3:
        st.metric("Predicted Change", f"${best_stock['predicted_change']:+.2f}")
    
    # --- Detailed Stock Analysis ---
    if st.button("🔍 Analyze Top Pick"):
        analysis = get_stock_analysis(best_stock['ticker'])
        if analysis:
            st.subheader(f"📋 Detailed Analysis: {best_stock['ticker']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Technical Signals**")
                for signal, value in analysis['technical_signals'].items():
                    if isinstance(value, bool):
                        status = "✅" if value else "❌"
                        st.write(f"{status} {signal.replace('_', ' ').title()}")
                    else:
                        st.write(f"📊 {signal.replace('_', ' ').title()}: {value:.2f}")
            
            with col2:
                st.write("**Fundamental Signals**")
                for signal, value in analysis['fundamental_signals'].items():
                    if isinstance(value, bool):
                        status = "✅" if value else "❌"
                        st.write(f"{status} {signal.replace('_', ' ').title()}")
                    else:
                        st.write(f"📊 {signal.replace('_', ' ').title()}: {value:.2f}")
            
            with col3:
                st.write("**Risk Metrics**")
                for metric, value in analysis['risk_metrics'].items():
                    st.write(f"📊 {metric.replace('_', ' ').title()}: {value:.2f}")
    
    # --- Stock Price Chart Analysis ---
    st.header("📈 3-Week Price Chart Analysis")
    
    # Load historical price data
    try:
        stock_prices_path = os.getenv('STOCK_PRICES_CSV', 'data/stock_prices.csv')
        stock_data = pd.read_csv(stock_prices_path)
        st.success("✅ Historical price data loaded successfully")
        
        # Create stock selector
        available_tickers = st.session_state.top_stocks_df['ticker'].tolist()
        selected_ticker = st.selectbox(
            "Select a stock to analyze:",
            available_tickers,
            index=0,
            help="Choose a stock to view its 3-week price chart with technical analysis"
        )
        
        if selected_ticker:
            # Load complete featured data to get all technical indicators
            try:
                # Prefer merged weekly features+ranked output for richest technicals
                weekly_path = 'data/top/xgb_weekly_output.csv'
                if os.path.exists(weekly_path):
                    complete_featured_data = pd.read_csv(weekly_path)
                else:
                    # Env may point to legacy path; add robust fallbacks
                    env_path = os.getenv('FEATURED_STOCKS_CSV', 'data/featured_stocks_top.csv')
                    try_paths = [env_path, 'data/top/featured_stocks_top.csv', 'data/featured_stocks_top.csv']
                    picked = None
                    for _p in try_paths:
                        if os.path.exists(_p):
                            picked = _p
                            break
                    if picked is None:
                        raise FileNotFoundError(f"No featured stocks CSV found in {try_paths}")
                    complete_featured_data = pd.read_csv(picked)
                if 'ticker' in complete_featured_data.columns:
                    complete_featured_data['ticker'] = complete_featured_data['ticker'].astype(str).str.upper()
                stock_featured_data = complete_featured_data[complete_featured_data['ticker'] == selected_ticker].iloc[0]
                # Backfill support/resistance if missing from weekly output
                try:
                    need_sr = []
                    for _c in ['support_20d', 'resistance_20d']:
                        if _c not in stock_featured_data.index or pd.isna(stock_featured_data.get(_c)):
                            need_sr.append(_c)
                    if need_sr:
                        # Try from featured_stocks_top.csv
                        try:
                            _feat_top_path = os.getenv('FEATURED_STOCKS_CSV', 'data/featured_stocks_top.csv')
                            _feat_top = pd.read_csv(_feat_top_path)
                            if 'ticker' in _feat_top.columns:
                                _feat_top['ticker'] = _feat_top['ticker'].astype(str).str.upper()
                                _row = _feat_top[_feat_top['ticker'] == selected_ticker]
                                if not _row.empty:
                                    _row = _row.iloc[0]
                                    for _c in need_sr:
                                        if _c in _row.index and not pd.isna(_row.get(_c)):
                                            stock_featured_data[_c] = _row.get(_c)
                        except Exception:
                            pass
                        # If still missing, compute from last 20 closes
                        still_missing = [
                            _c for _c in ['support_20d', 'resistance_20d']
                            if _c not in stock_featured_data.index or pd.isna(stock_featured_data.get(_c))
                        ]
                        if still_missing:
                            try:
                                _td = stock_data[stock_data['ticker'] == selected_ticker].copy()
                                _td['date'] = pd.to_datetime(_td['date'], errors='coerce')
                                _td = _td.sort_values('date').tail(20)
                                if len(_td) > 0:
                                    if 'support_20d' in still_missing:
                                        stock_featured_data['support_20d'] = float(_td['close'].min())
                                    if 'resistance_20d' in still_missing:
                                        stock_featured_data['resistance_20d'] = float(_td['close'].max())
                            except Exception:
                                pass
                except Exception:
                    pass
                
                # Get predicted return and confidence data from filtered stocks if available
                if not st.session_state.top_stocks_df.empty:
                    filtered_stock_data = st.session_state.top_stocks_df[st.session_state.top_stocks_df['ticker'] == selected_ticker]
                    if not filtered_stock_data.empty:
                        filtered_row = filtered_stock_data.iloc[0]
                        # Add predicted return and confidence data to stock_featured_data
                        stock_featured_data['predicted_return_pct'] = filtered_row.get('predicted_return_pct', 'N/A')
                        stock_featured_data['confidence_score'] = filtered_row.get('confidence_score', 'N/A')
                        stock_featured_data['predicted_change'] = filtered_row.get('predicted_change', 'N/A')
                        stock_featured_data['risk_score'] = filtered_row.get('risk_score', 'N/A')
                        stock_featured_data['composite_score'] = filtered_row.get('composite_score', 'N/A')
                
                st.success(f"✅ Complete technical data loaded for {selected_ticker}")
            except Exception as e:
                # Fallback to filtered data if complete data not available
                stock_featured_data = st.session_state.top_stocks_df[st.session_state.top_stocks_df['ticker'] == selected_ticker].iloc[0]
                st.warning(f"⚠️ Using filtered data for {selected_ticker} (some technical indicators may be missing)")
            
            # Create tabs for different chart views
            candlestick_tab, price_graph_tab, earnings_tab = st.tabs(["📊 Candlestick Chart", "📈 Price Graph", "📋 Earnings History"])
            
            with candlestick_tab:
                # Create the price chart (current candlestick view)
                chart = create_stock_price_chart(selected_ticker, stock_data, stock_featured_data)
                
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                    
                    st.subheader(f"📊 Technical Analysis Summary: {selected_ticker}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${stock_featured_data['close']:.2f}")
                        if not pd.isna(stock_featured_data.get('resistance_20d')):
                            st.metric("Resistance Level", f"${stock_featured_data['resistance_20d']:.2f}")
                        if not pd.isna(stock_featured_data.get('support_20d')):
                            st.metric("Support Level", f"${stock_featured_data['support_20d']:.2f}")
                    
                    with col2:
                        rsi_value = stock_featured_data.get('rsi_14d')
                        rsi_display = f"{rsi_value:.1f}" if rsi_value is not None and not pd.isna(rsi_value) else "N/A"
                        st.metric("RSI (14d)", rsi_display)
                        
                        momentum_5d = stock_featured_data.get('momentum_5d')
                        momentum_5d_display = f"{momentum_5d:.2f}" if momentum_5d is not None and not pd.isna(momentum_5d) else "N/A"
                        st.metric("Momentum (5d)", momentum_5d_display)
                        
                        momentum_10d = stock_featured_data.get('momentum_10d')
                        momentum_10d_display = f"{momentum_10d:.2f}" if momentum_10d is not None and not pd.isna(momentum_10d) else "N/A"
                        st.metric("Momentum (10d)", momentum_10d_display)
                    
                    with col3:
                        resistance_status = "🟢 BROKEN" if stock_featured_data.get('broke_resistance', False) else "🔴 HOLDING"
                        st.metric("Resistance Status", resistance_status)
                        # Check if predicted_return_pct exists in the data
                        if 'predicted_return_pct' in stock_featured_data:
                            st.metric("Predicted Return", f"{stock_featured_data['predicted_return_pct']:.2f}%")
                        else:
                            st.metric("Predicted Return", "N/A")
                        # Check if confidence_score exists in the data
                        if 'confidence_score' in stock_featured_data:
                            st.metric("Confidence Score", f"{stock_featured_data['confidence_score']:.1f}/100")
                        else:
                            st.metric("Confidence Score", "N/A")
                    
                    # Add key price levels section
                    st.subheader("🎯 Key Price Levels & Validation")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Resistance break price
                        if stock_featured_data.get('broke_resistance', False):
                            resistance_level = stock_featured_data.get('resistance_20d')
                            current_price = stock_featured_data['close']
                            if resistance_level is not None and not pd.isna(resistance_level):
                                price_above_resistance = current_price - resistance_level
                                st.metric(
                                    "🟢 Resistance Break Price", 
                                    f"${resistance_level:.2f}",
                                    delta=f"+${price_above_resistance:.2f} above resistance"
                                )
                                # Add additional info about the resistance break
                                st.info(f"**Resistance Level**: ${resistance_level:.2f} was broken. Current price ${current_price:.2f} is ${price_above_resistance:.2f} above resistance.")
                        else:
                            resistance_level = stock_featured_data.get('resistance_20d')
                            current_price = stock_featured_data['close']
                            if resistance_level is not None and not pd.isna(resistance_level):
                                distance_to_resistance = resistance_level - current_price
                                st.metric(
                                    "🔴 Resistance Level", 
                                    f"${resistance_level:.2f}",
                                    delta=f"${distance_to_resistance:.2f} to break"
                                )
                                st.info(f"**Resistance Level**: ${resistance_level:.2f}. Current price ${current_price:.2f} needs to rise ${distance_to_resistance:.2f} to break resistance.")
                            else:
                                st.metric("🔴 Resistance Break Price", "Not broken yet")
                        
                        # Support level
                        support_level = stock_featured_data.get('support_20d')
                        if support_level is not None and not pd.isna(support_level):
                            current_price = stock_featured_data['close']
                            distance_from_support = current_price - support_level
                            st.metric(
                                "🟢 Support Level", 
                                f"${support_level:.2f}",
                                delta=f"+${distance_from_support:.2f} above support"
                            )
                            # Add additional info about support level
                            st.info(f"**Support Level**: ${support_level:.2f}. Current price ${current_price:.2f} is ${distance_from_support:.2f} above support.")
                        else:
                            st.metric("🔴 Support Level", "Not available")
                    
                    with col2:
                        # Momentum identification price
                        momentum_5d = stock_featured_data.get('momentum_5d')
                        momentum_10d = stock_featured_data.get('momentum_10d')
                        if momentum_5d is not None and momentum_10d is not None and momentum_5d > 0 and momentum_10d > 0:
                            current_price = stock_featured_data['close']
                            # Calculate momentum strength
                            momentum_strength = (momentum_5d + momentum_10d) / 2
                            st.metric(
                                "🚀 Momentum Identified", 
                                f"${current_price:.2f}",
                                delta=f"+{momentum_strength:.1f}% momentum"
                            )
                        else:
                            st.metric("📊 Momentum Status", "Mixed/Weak")
                        
                        # RSI interpretation
                        rsi_value = stock_featured_data.get('rsi_14d')
                        if rsi_value is not None and not pd.isna(rsi_value):
                            if rsi_value > 70:
                                st.metric("⚠️ RSI Status", "Overbought", delta=f"{rsi_value:.1f}")
                            elif rsi_value < 30:
                                st.metric("🟢 RSI Status", "Oversold", delta=f"{rsi_value:.1f}")
                            else:
                                st.metric("📊 RSI Status", "Neutral", delta=f"{rsi_value:.1f}")
                        else:
                            st.metric("📊 RSI Status", "N/A")
                    
                    # Add resistance price details
                    st.subheader("🎯 Resistance Price Analysis")
                    
                    resistance_level = stock_featured_data.get('resistance_20d')
                    current_price = stock_featured_data['close']
                    broke_resistance = stock_featured_data.get('broke_resistance', False)
                    
                    if resistance_level is not None and not pd.isna(resistance_level):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if broke_resistance:
                                price_above_resistance = current_price - resistance_level
                                st.success(f"**🟢 RESISTANCE BROKEN!**")
                                st.metric(
                                    "Resistance Price", 
                                    f"${resistance_level:.2f}",
                                    delta=f"BROKEN by ${price_above_resistance:.2f}"
                                )
                                st.write(f"**Current Price**: ${current_price:.2f}")
                                st.write(f"**Resistance Level**: ${resistance_level:.2f}")
                                st.write(f"**Price Above Resistance**: ${price_above_resistance:.2f}")
                            else:
                                distance_to_resistance = resistance_level - current_price
                                st.warning(f"**🔴 RESISTANCE NOT BROKEN**")
                                st.metric(
                                    "Resistance Price", 
                                    f"${resistance_level:.2f}",
                                    delta=f"${distance_to_resistance:.2f} to break"
                                )
                                st.write(f"**Current Price**: ${current_price:.2f}")
                                st.write(f"**Resistance Level**: ${resistance_level:.2f}")
                                st.write(f"**Distance to Break**: ${distance_to_resistance:.2f}")
                        
                        with col2:
                            # Show resistance break date if available
                            if broke_resistance:
                                st.info("**📅 Resistance Break Details**")
                                st.write("✅ **Status**: Resistance level has been successfully broken")
                                st.write("📈 **Signal**: Bullish - price is now above resistance")
                                st.write("🎯 **Next Target**: Look for continued upward momentum")
                            else:
                                st.info("**📅 Resistance Status**")
                                st.write("⏳ **Status**: Price is below resistance level")
                                st.write("📊 **Signal**: Waiting for breakout confirmation")
                                st.write("🎯 **Watch For**: Price breaking above resistance with volume")
                    else:
                        st.warning("Resistance level data not available for this stock.")
                    
                    # Add support level analysis
                    st.subheader("🎯 Support Level Analysis")
                    
                    support_level = stock_featured_data.get('support_20d')
                    current_price = stock_featured_data['close']
                    
                    if support_level is not None and not pd.isna(support_level):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            distance_from_support = current_price - support_level
                            if distance_from_support > 0:
                                st.success(f"**🟢 ABOVE SUPPORT**")
                                st.metric(
                                    "Support Price", 
                                    f"${support_level:.2f}",
                                    delta=f"${distance_from_support:.2f} above support"
                                )
                                st.write(f"**Current Price**: ${current_price:.2f}")
                                st.write(f"**Support Level**: ${support_level:.2f}")
                                st.write(f"**Distance Above Support**: ${distance_from_support:.2f}")
                            else:
                                st.error(f"**🔴 BELOW SUPPORT**")
                                st.metric(
                                    "Support Price", 
                                    f"${support_level:.2f}",
                                    delta=f"${abs(distance_from_support):.2f} below support"
                                )
                                st.write(f"**Current Price**: ${current_price:.2f}")
                                st.write(f"**Support Level**: ${support_level:.2f}")
                                st.write(f"**Distance Below Support**: ${abs(distance_from_support):.2f}")
                        
                        with col2:
                            if distance_from_support > 0:
                                st.info("**📅 Support Status**")
                                st.write("✅ **Status**: Price is above support level")
                                st.write("📈 **Signal**: Bullish - price has support below")
                                st.write("🎯 **Next Target**: Look for continued upward movement")
                            else:
                                st.warning("**📅 Support Status**")
                                st.write("⚠️ **Status**: Price is below support level")
                                st.write("📉 **Signal**: Bearish - price may continue falling")
                                st.write("🎯 **Watch For**: Price bouncing back above support")
                    else:
                        st.warning("Support level data not available for this stock.")
                    
                    # Add detailed technical insights
                    st.subheader("🔍 Technical Insights")
                    
                    insights = []
                    
                    # Resistance analysis
                    if stock_featured_data.get('broke_resistance', False):
                        insights.append("🟢 **Resistance Break**: Price has successfully broken above the 20-day resistance level, indicating bullish momentum.")
                    else:
                        insights.append("🔴 **Resistance Test**: Price is testing the resistance level. A break above could signal upward momentum.")
                    
                    # RSI analysis
                    rsi = stock_featured_data.get('rsi_14d')
                    if rsi is not None and not pd.isna(rsi):
                        if rsi > 70:
                            insights.append("⚠️ **Overbought**: RSI above 70 suggests the stock may be overbought and could face resistance.")
                        elif rsi < 30:
                            insights.append("🟢 **Oversold**: RSI below 30 suggests the stock may be oversold and could bounce back.")
                        else:
                            insights.append("📊 **Neutral RSI**: RSI in neutral territory, no extreme overbought/oversold conditions.")
                    else:
                        insights.append("📊 **RSI**: RSI data not available for analysis.")
                    
                    # Momentum analysis
                    momentum_5d = stock_featured_data.get('momentum_5d')
                    momentum_10d = stock_featured_data.get('momentum_10d')
                    
                    if momentum_5d is not None and momentum_10d is not None and not pd.isna(momentum_5d) and not pd.isna(momentum_10d):
                        if momentum_5d > 0 and momentum_10d > 0:
                            insights.append("🚀 **Positive Momentum**: Both 5-day and 10-day momentum are positive, indicating upward price movement.")
                        elif momentum_5d < 0 and momentum_10d < 0:
                            insights.append("📉 **Negative Momentum**: Both 5-day and 10-day momentum are negative, indicating downward pressure.")
                        else:
                            insights.append("📊 **Mixed Momentum**: Short-term and medium-term momentum are mixed, suggesting consolidation.")
                    else:
                        insights.append("📊 **Momentum**: Momentum data not available for analysis.")
                    
                    # Display insights
                    for insight in insights:
                        st.write(insight)
                    
                    # Add comprehensive chart interpretation guide
                    st.subheader("📚 How to Interpret the Chart for Investment Decisions")
                    
                    # Create expandable sections for different signal types
                    with st.expander("🟢 Bullish Signals (Good to Buy)", expanded=False):
                        st.markdown("""
                        **Resistance Break**: Price closes above the red resistance line
                        - Look for the 🟢 Resistance Break annotation on the chart
                        - This indicates strong buying pressure and potential upward movement
                        
                        **Green Candlesticks**: Multiple consecutive green days
                        - Shows consistent buying pressure
                        - Each green candle means the day closed higher than it opened
                        
                        **High Volume**: Volume increases with price gains
                        - Check the gray volume bars at the bottom
                        - Higher bars with price increases confirm strong buying interest
                        
                        **RSI 30-70**: Stock is not overbought
                        - RSI in the neutral range allows for continued upside
                        - Avoid stocks with RSI > 70 (overbought)
                        
                        **Momentum Point**: Recent strong upward movement
                        - Look for the 🚀 Momentum annotation on the chart
                        - Shows the peak of recent buying pressure
                        """)
                    
                    with st.expander("🔴 Bearish Signals (Avoid or Sell)", expanded=False):
                        st.markdown("""
                        **Support Break**: Price closes below the green support line
                        - This indicates potential further downside
                        - Support level acts as a floor for the stock price
                        
                        **Red Candlesticks**: Multiple consecutive red days
                        - Shows consistent selling pressure
                        - Each red candle means the day closed lower than it opened
                        
                        **High Volume Down**: High volume with price declines
                        - High volume bars with price drops indicate strong selling
                        - This confirms bearish sentiment
                        
                        **RSI >70**: Stock is overbought and may pull back
                        - RSI above 70 suggests the stock may be due for a correction
                        - Consider taking profits or waiting for a pullback
                        
                        **Low Volume**: Lack of buying interest
                        - Small volume bars indicate lack of conviction
                        - Price movements without volume are less reliable
                        """)
                    
                    with st.expander("📊 Neutral/Consolidation", expanded=False):
                        st.markdown("""
                        **Small Candlesticks**: Price moving sideways
                        - Small candle bodies indicate indecision
                        - Stock is consolidating before next move
                        
                        **Low Volume**: Lack of strong directional movement
                        - Low volume suggests lack of conviction
                        - Wait for volume confirmation before making decisions
                        
                        **RSI 40-60**: Neutral momentum
                        - RSI in middle range indicates balanced buying/selling
                        - Stock is not overbought or oversold
                        """)
                    
                    with st.expander("📋 Technical Analysis Summary Section", expanded=False):
                        st.markdown("""
                        **Current Price Metrics**
                        - **Current Price**: Latest closing price from the chart
                        - **Resistance Level**: Price level to watch for breakouts (red line)
                        - **Support Level**: Price level to watch for breakdowns (green line)
                        
                        **Momentum Indicators**
                        - **RSI (14d)**:
                          - 0-30: Oversold (potential buy opportunity)
                          - 30-70: Normal trading range
                          - 70-100: Overbought (potential sell signal)
                        - **Momentum (5d)**: Short-term price change percentage
                        - **Momentum (10d)**: Medium-term price change percentage
                        
                        **Status Indicators**
                        - **Resistance Status**:
                          - 🟢 BROKEN: Bullish signal - price above resistance
                          - 🔴 HOLDING: Price still below resistance level
                        - **Predicted Return**: AI model's forecast for price movement
                        - **Confidence Score**: How confident the model is (0-100 scale)
                        """)
                    
                    with st.expander("🔍 Technical Insights Section", expanded=False):
                        st.markdown("""
                        **Resistance Analysis**
                        - 🟢 **Resistance Break**: "Price has successfully broken above the 20-day resistance level, indicating bullish momentum"
                        - 🔴 **Resistance Test**: "Price is testing the resistance level. A break above could signal upward momentum"
                        
                        **RSI Analysis**
                        - ⚠️ **Overbought**: "RSI above 70 suggests the stock may be overbought and could face resistance"
                        - 🟢 **Oversold**: "RSI below 30 suggests the stock may be oversold and could bounce back"
                        - 📊 **Neutral RSI**: "RSI in neutral territory, no extreme overbought/oversold conditions"
                        
                        **Momentum Analysis**
                        - 🚀 **Positive Momentum**: "Both 5-day and 10-day momentum are positive, indicating upward price movement"
                        - 📉 **Negative Momentum**: "Both 5-day and 10-day momentum are negative, indicating downward pressure"
                        - 📊 **Mixed Momentum**: "Short-term and medium-term momentum are mixed, suggesting consolidation"
                        """)
                    
                    with st.expander("💡 Practical Investment Strategy", expanded=False):
                        st.markdown("""
                        **For Buying (Entry Points)**
                        1. **Wait for resistance break** with high volume
                        2. **Buy on pullbacks** to support level
                        3. **Look for oversold RSI** (<30) for bounce-back opportunities
                        4. **Confirm with positive momentum** (5d and 10d both positive)
                        
                        **For Selling (Exit Points)**
                        1. **Sell on resistance rejection** (price fails to break above resistance)
                        2. **Exit on support break** (price closes below support)
                        3. **Take profits on overbought RSI** (>70)
                        4. **Watch for negative momentum** (both 5d and 10d negative)
                        
                        **Risk Management**
                        - **Set stop-loss** below support level
                        - **Take partial profits** at resistance levels
                        - **Don't chase overbought stocks** (RSI >70)
                        - **Use volume confirmation** for major moves
                        """)
                    
            with price_graph_tab:
                st.subheader("📈 3-Week Price Graph")
                st.write("This tab will show a simple line chart of the 3-week price movement.")
                
                # Create a simple line chart for the 3-week period
                try:
                    # Filter data for the specific ticker
                    ticker_data = stock_data[stock_data['ticker'] == selected_ticker].copy()
                    
                    if not ticker_data.empty:
                        # Convert date column to datetime
                        ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                        
                        # Sort by date and get last 3 weeks (21 trading days)
                        ticker_data = ticker_data.sort_values('date')
                        last_date = ticker_data['date'].max()
                        start_date = last_date - timedelta(days=30)  # 30 days to ensure we get 21 trading days
                        
                        # Filter for last 3 weeks
                        recent_data = ticker_data[ticker_data['date'] >= start_date].copy()
                        
                        if len(recent_data) >= 5:  # Need at least 5 days for meaningful chart
                            # Create line chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=recent_data['date'],
                                y=recent_data['close'],
                                mode='lines+markers',
                                name='Close Price',
                                line=dict(color='#1f77b4', width=2),
                                marker=dict(size=6)
                            ))
                            
                            # Add moving averages
                            if len(recent_data) >= 20:
                                recent_data['MA20'] = recent_data['close'].rolling(window=20).mean()
                                fig.add_trace(go.Scatter(
                                    x=recent_data['date'],
                                    y=recent_data['MA20'],
                                    mode='lines',
                                    name='20-Day MA',
                                    line=dict(color='orange', width=1, dash='dash')
                                ))
                            
                            if len(recent_data) >= 50:
                                recent_data['MA50'] = recent_data['close'].rolling(window=50).mean()
                                fig.add_trace(go.Scatter(
                                    x=recent_data['date'],
                                    y=recent_data['MA50'],
                                    mode='lines',
                                    name='50-Day MA',
                                    line=dict(color='red', width=1, dash='dash')
                                ))
                            
                            fig.update_layout(
                                title=f"{selected_ticker} - 3-Week Price Movement",
                                xaxis_title="Date",
                                yaxis_title="Price ($)",
                                height=500,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add price statistics
                            st.subheader("📊 Price Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Start Price", f"${recent_data['close'].iloc[0]:.2f}")
                            with col2:
                                st.metric("End Price", f"${recent_data['close'].iloc[-1]:.2f}")
                            with col3:
                                price_change = recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]
                                price_change_pct = (price_change / recent_data['close'].iloc[0]) * 100
                                st.metric("Price Change", f"${price_change:.2f}", delta=f"{price_change_pct:.2f}%")
                            with col4:
                                st.metric("Highest Price", f"${recent_data['close'].max():.2f}")
                            
                            # Add volume analysis
                            st.subheader("📈 Volume Analysis")
                            if 'volume' in recent_data.columns:
                                avg_volume = recent_data['volume'].mean()
                                current_volume = recent_data['volume'].iloc[-1]
                                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Average Volume", f"{avg_volume:,.0f}")
                                with col2:
                                    st.metric("Current Volume", f"{current_volume:,.0f}")
                                with col3:
                                    st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
                                
                                # Volume trend
                                if volume_ratio > 1.5:
                                    st.success("📈 **High Volume**: Current volume is significantly above average, indicating strong interest.")
                                elif volume_ratio < 0.5:
                                    st.warning("📉 **Low Volume**: Current volume is below average, indicating weak interest.")
                                else:
                                    st.info("📊 **Normal Volume**: Current volume is within normal range.")
                        else:
                            st.warning("Insufficient data for 3-week analysis. Need at least 5 trading days.")
                    else:
                        st.error(f"No data found for {selected_ticker}")
                except Exception as e:
                    st.error(f"Error creating price graph: {str(e)}")
            
            with earnings_tab:
                st.subheader("📋 Quarterly Earnings & Revenue History")
                st.write("This tab shows how the company has performed against earnings and revenue expectations over the last year.")
                
                # Load earnings history data
                try:
                    earnings_history_path = os.getenv('EARNINGS_HISTORY_CSV', 'data/earnings_history.csv')
                    earnings_data = pd.read_csv(earnings_history_path)
                    earnings_ticker_data = earnings_data[earnings_data['ticker'] == selected_ticker]
                    
                    if not earnings_ticker_data.empty:
                        # Convert earnings_date to datetime
                        earnings_ticker_data['earnings_date'] = pd.to_datetime(earnings_ticker_data['earnings_date'])
                        
                        # Sort by date (most recent first)
                        earnings_ticker_data = earnings_ticker_data.sort_values('earnings_date', ascending=False)
                        
                        # Display earnings history table
                        st.subheader("📊 Earnings Performance History")
                        
                        # Format the data for display
                        display_data = earnings_ticker_data.copy()
                        display_data['earnings_date'] = display_data['earnings_date'].dt.strftime('%Y-%m-%d')
                        display_data['reported_eps'] = display_data['reported_eps'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                        display_data['estimate_eps'] = display_data['estimate_eps'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                        display_data['surprise_percentage'] = display_data['surprise_percentage'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                        
                        st.dataframe(display_data, use_container_width=True, hide_index=True)
                        
                        # Create earnings surprise chart
                        st.subheader("📈 Earnings Surprise Trend")
                        
                        # Filter out rows with missing data
                        chart_data = earnings_ticker_data.dropna(subset=['surprise_percentage'])
                        
                        if not chart_data.empty:
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                x=chart_data['earnings_date'].dt.strftime('%Y-%m'),
                                y=chart_data['surprise_percentage'],
                                name='Earnings Surprise %',
                                marker_color=['green' if x > 0 else 'red' for x in chart_data['surprise_percentage']]
                            ))
                            
                            fig.update_layout(
                                title=f"{selected_ticker} - Earnings Surprise Percentage",
                                xaxis_title="Quarter",
                                yaxis_title="Surprise Percentage (%)",
                                height=400,
                                showlegend=False
                            )
                            
                            # Add horizontal line at 0
                            fig.add_hline(y=0, line_dash="dash", line_color="black")
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Earnings analysis
                            st.subheader("📊 Earnings Analysis")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                positive_surprises = len(chart_data[chart_data['surprise_percentage'] > 0])
                                total_quarters = len(chart_data)
                                beat_rate = (positive_surprises / total_quarters * 100) if total_quarters > 0 else 0
                                st.metric("Beat Rate", f"{beat_rate:.1f}%")
                            
                            with col2:
                                avg_surprise = chart_data['surprise_percentage'].mean()
                                st.metric("Avg Surprise", f"{avg_surprise:.2f}%")
                            
                            with col3:
                                max_surprise = chart_data['surprise_percentage'].max()
                                st.metric("Best Surprise", f"{max_surprise:.2f}%")
                            
                            with col4:
                                min_surprise = chart_data['surprise_percentage'].min()
                                st.metric("Worst Surprise", f"{min_surprise:.2f}%")
                            
                            # Earnings consistency analysis
                            st.subheader("🎯 Earnings Consistency Analysis")
                            
                            if beat_rate >= 75:
                                st.success("🏆 **Excellent Track Record**: Company consistently beats earnings expectations.")
                            elif beat_rate >= 50:
                                st.info("📈 **Good Track Record**: Company beats expectations more often than not.")
                            elif beat_rate >= 25:
                                st.warning("⚠️ **Mixed Track Record**: Company has difficulty consistently beating expectations.")
                            else:
                                st.error("📉 **Poor Track Record**: Company rarely beats earnings expectations.")
                            
                            # Recent performance
                            if len(chart_data) >= 2:
                                recent_surprise = chart_data.iloc[0]['surprise_percentage']
                                previous_surprise = chart_data.iloc[1]['surprise_percentage']
                                
                                st.subheader("📅 Recent Performance")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Latest Quarter", f"{recent_surprise:.2f}%")
                                with col2:
                                    st.metric("Previous Quarter", f"{previous_surprise:.2f}%")
                                
                                if recent_surprise > previous_surprise:
                                    st.success("📈 **Improving**: Latest quarter showed better performance than the previous quarter.")
                                elif recent_surprise < previous_surprise:
                                    st.warning("📉 **Declining**: Latest quarter showed worse performance than the previous quarter.")
                                else:
                                    st.info("📊 **Stable**: Performance has remained consistent between quarters.")
                        else:
                            st.warning("No earnings surprise data available for analysis.")
                    else:
                        st.warning(f"No earnings history data found for {selected_ticker}")
                        
                        # Show available tickers
                        available_tickers = earnings_data['ticker'].unique()
                        st.info(f"Available tickers in earnings data: {', '.join(available_tickers)}")
                except Exception as e:
                    st.error(f"Error loading earnings data: {str(e)}")
                    st.info("Make sure the earnings_history.csv file is available in the data folder.")
                
                resistance_level = stock_featured_data.get('resistance_20d')
                current_price = stock_featured_data['close']
                broke_resistance = stock_featured_data.get('broke_resistance', False)
                
                if resistance_level is not None and not pd.isna(resistance_level):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if broke_resistance:
                            price_above_resistance = current_price - resistance_level
                            st.success(f"**🟢 RESISTANCE BROKEN!**")
                            st.metric(
                                "Resistance Price", 
                                f"${resistance_level:.2f}",
                                delta=f"BROKEN by ${price_above_resistance:.2f}"
                            )
                            st.write(f"**Current Price**: ${current_price:.2f}")
                            st.write(f"**Resistance Level**: ${resistance_level:.2f}")
                            st.write(f"**Price Above Resistance**: ${price_above_resistance:.2f}")
                        else:
                            distance_to_resistance = resistance_level - current_price
                            st.warning(f"**🔴 RESISTANCE NOT BROKEN**")
                            st.metric(
                                "Resistance Price", 
                                f"${resistance_level:.2f}",
                                delta=f"${distance_to_resistance:.2f} to break"
                            )
                            st.write(f"**Current Price**: ${current_price:.2f}")
                            st.write(f"**Resistance Level**: ${resistance_level:.2f}")
                            st.write(f"**Distance to Break**: ${distance_to_resistance:.2f}")
                    
                    with col2:
                        # Show resistance break date if available
                        if broke_resistance:
                            st.info("**📅 Resistance Break Details**")
                            st.write("✅ **Status**: Resistance level has been successfully broken")
                            st.write("📈 **Signal**: Bullish - price is now above resistance")
                            st.write("🎯 **Next Target**: Look for continued upward momentum")
                        else:
                            st.info("**📅 Resistance Status**")
                            st.write("⏳ **Status**: Price is below resistance level")
                            st.write("📊 **Signal**: Waiting for breakout confirmation")
                            st.write("🎯 **Watch For**: Price breaking above resistance with volume")
                else:
                    st.warning("Resistance level data not available for this stock.")
                
                # Add support level analysis
                st.subheader("🎯 Support Level Analysis")
                
                support_level = stock_featured_data.get('support_20d')
                current_price = stock_featured_data['close']
                
                if support_level is not None and not pd.isna(support_level):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        distance_from_support = current_price - support_level
                        if distance_from_support > 0:
                            st.success(f"**🟢 ABOVE SUPPORT**")
                            st.metric(
                                "Support Price", 
                                f"${support_level:.2f}",
                                delta=f"${distance_from_support:.2f} above support"
                            )
                            st.write(f"**Current Price**: ${current_price:.2f}")
                            st.write(f"**Support Level**: ${support_level:.2f}")
                            st.write(f"**Distance Above Support**: ${distance_from_support:.2f}")
                        else:
                            st.error(f"**🔴 BELOW SUPPORT**")
                            st.metric(
                                "Support Price", 
                                f"${support_level:.2f}",
                                delta=f"${abs(distance_from_support):.2f} below support"
                            )
                            st.write(f"**Current Price**: ${current_price:.2f}")
                            st.write(f"**Support Level**: ${support_level:.2f}")
                            st.write(f"**Distance Below Support**: ${abs(distance_from_support):.2f}")
                    
                    with col2:
                        if distance_from_support > 0:
                            st.info("**📅 Support Status**")
                            st.write("✅ **Status**: Price is above support level")
                            st.write("📈 **Signal**: Bullish - price has support below")
                            st.write("🎯 **Next Target**: Look for continued upward movement")
                        else:
                            st.warning("**📅 Support Status**")
                            st.write("⚠️ **Status**: Price is below support level")
                            st.write("📉 **Signal**: Bearish - price may continue falling")
                            st.write("🎯 **Watch For**: Price bouncing back above support")
                else:
                    st.warning("Support level data not available for this stock.")
                
                # Add detailed technical insights
                st.subheader("🔍 Technical Insights")
                
                insights = []
                
                # Resistance analysis
                if stock_featured_data.get('broke_resistance', False):
                    insights.append("🟢 **Resistance Break**: Price has successfully broken above the 20-day resistance level, indicating bullish momentum.")
                else:
                    insights.append("🔴 **Resistance Test**: Price is testing the resistance level. A break above could signal upward momentum.")
                
                # RSI analysis
                rsi = stock_featured_data.get('rsi_14d')
                if rsi is not None and not pd.isna(rsi):
                    if rsi > 70:
                        insights.append("⚠️ **Overbought**: RSI above 70 suggests the stock may be overbought and could face resistance.")
                    elif rsi < 30:
                        insights.append("🟢 **Oversold**: RSI below 30 suggests the stock may be oversold and could bounce back.")
                    else:
                        insights.append("📊 **Neutral RSI**: RSI in neutral territory, no extreme overbought/oversold conditions.")
                else:
                    insights.append("📊 **RSI**: RSI data not available for analysis.")
                
                # Momentum analysis
                momentum_5d = stock_featured_data.get('momentum_5d')
                momentum_10d = stock_featured_data.get('momentum_10d')
                
                if momentum_5d is not None and momentum_10d is not None and not pd.isna(momentum_5d) and not pd.isna(momentum_10d):
                    if momentum_5d > 0 and momentum_10d > 0:
                        insights.append("🚀 **Positive Momentum**: Both 5-day and 10-day momentum are positive, indicating upward price movement.")
                    elif momentum_5d < 0 and momentum_10d < 0:
                        insights.append("📉 **Negative Momentum**: Both 5-day and 10-day momentum are negative, indicating downward pressure.")
                    else:
                        insights.append("📊 **Mixed Momentum**: Short-term and medium-term momentum are mixed, suggesting consolidation.")
                else:
                    insights.append("📊 **Momentum**: Momentum data not available for analysis.")
                
                # Display insights
                for insight in insights:
                    st.write(insight)
                
                # Add comprehensive chart interpretation guide
                st.subheader("📚 How to Interpret the Chart for Investment Decisions")
                
                # Create expandable sections for different signal types
                with st.expander("🟢 Bullish Signals (Good to Buy)", expanded=False):
                    st.markdown("""
                    **Resistance Break**: Price closes above the red resistance line
                    - Look for the 🟢 Resistance Break annotation on the chart
                    - This indicates strong buying pressure and potential upward movement
                    
                    **Green Candlesticks**: Multiple consecutive green days
                    - Shows consistent buying pressure
                    - Each green candle means the day closed higher than it opened
                    
                    **High Volume**: Volume increases with price gains
                    - Check the gray volume bars at the bottom
                    - Higher bars with price increases confirm strong buying interest
                    
                    **RSI 30-70**: Stock is not overbought
                    - RSI in the neutral range allows for continued upside
                    - Avoid stocks with RSI > 70 (overbought)
                    
                    **Momentum Point**: Recent strong upward movement
                    - Look for the 🚀 Momentum annotation on the chart
                    - Shows the peak of recent buying pressure
                    """)
                
                with st.expander("🔴 Bearish Signals (Avoid or Sell)", expanded=False):
                    st.markdown("""
                    **Support Break**: Price closes below the green support line
                    - This indicates potential further downside
                    - Support level acts as a floor for the stock price
                    
                    **Red Candlesticks**: Multiple consecutive red days
                    - Shows consistent selling pressure
                    - Each red candle means the day closed lower than it opened
                    
                    **High Volume Down**: High volume with price declines
                    - High volume bars with price drops indicate strong selling
                    - This confirms bearish sentiment
                    
                    **RSI >70**: Stock is overbought and may pull back
                    - RSI above 70 suggests the stock may be due for a correction
                    - Consider taking profits or waiting for a pullback
                    
                    **Low Volume**: Lack of buying interest
                    - Small volume bars indicate lack of conviction
                    - Price movements without volume are less reliable
                    """)
                
                with st.expander("📊 Neutral/Consolidation", expanded=False):
                    st.markdown("""
                    **Small Candlesticks**: Price moving sideways
                    - Small candle bodies indicate indecision
                    - Stock is consolidating before next move
                    
                    **Low Volume**: Lack of strong directional movement
                    - Low volume suggests lack of conviction
                    - Wait for volume confirmation before making decisions
                    
                    **RSI 40-60**: Neutral momentum
                    - RSI in middle range indicates balanced buying/selling
                    - Stock is not overbought or oversold
                    """)
                
                with st.expander("📋 Technical Analysis Summary Section", expanded=False):
                    st.markdown("""
                    **Current Price Metrics**
                    - **Current Price**: Latest closing price from the chart
                    - **Resistance Level**: Price level to watch for breakouts (red line)
                    - **Support Level**: Price level to watch for breakdowns (green line)
                    
                    **Momentum Indicators**
                    - **RSI (14d)**:
                      - 0-30: Oversold (potential buy opportunity)
                      - 30-70: Normal trading range
                      - 70-100: Overbought (potential sell signal)
                    - **Momentum (5d)**: Short-term price change percentage
                    - **Momentum (10d)**: Medium-term price change percentage
                    
                    **Status Indicators**
                    - **Resistance Status**:
                      - 🟢 BROKEN: Bullish signal - price above resistance
                      - 🔴 HOLDING: Price still below resistance level
                    - **Predicted Return**: AI model's forecast for price movement
                    - **Confidence Score**: How confident the model is (0-100 scale)
                    """)
                
                with st.expander("🔍 Technical Insights Section", expanded=False):
                    st.markdown("""
                    **Resistance Analysis**
                    - 🟢 **Resistance Break**: "Price has successfully broken above the 20-day resistance level, indicating bullish momentum"
                    - 🔴 **Resistance Test**: "Price is testing the resistance level. A break above could signal upward momentum"
                    
                    **RSI Analysis**
                    - ⚠️ **Overbought**: "RSI above 70 suggests the stock may be overbought and could face resistance"
                    - 🟢 **Oversold**: "RSI below 30 suggests the stock may be oversold and could bounce back"
                    - 📊 **Neutral RSI**: "RSI in neutral territory, no extreme overbought/oversold conditions"
                    
                    **Momentum Analysis**
                    - 🚀 **Positive Momentum**: "Both 5-day and 10-day momentum are positive, indicating upward price movement"
                    - 📉 **Negative Momentum**: "Both 5-day and 10-day momentum are negative, indicating downward pressure"
                    - 📊 **Mixed Momentum**: "Short-term and medium-term momentum are mixed, suggesting consolidation"
                    """)
                
                with st.expander("💡 Practical Investment Strategy", expanded=False):
                    st.markdown("""
                    **For Buying (Entry Points)**
                    1. **Wait for resistance break** with high volume
                    2. **Buy on pullbacks** to support level
                    3. **Look for oversold RSI** (<30) for bounce-back opportunities
                    4. **Confirm with positive momentum** (5d and 10d both positive)
                    
                    **For Selling (Exit Points)**
                    1. **Sell on resistance rejection** (price fails to break above resistance)
                    2. **Exit on support break** (price closes below support)
                    3. **Take profits on overbought RSI** (>70)
                    4. **Watch for negative momentum** (both 5d and 10d negative)
                    
                    **Risk Management**
                    - **Set stop-loss** below support level
                    - **Take partial profits** at resistance levels
                    - **Don't chase overbought stocks** (RSI >70)
                    - **Use volume confirmation** for major moves
                    """)
                    
        else:
            st.warning(f"Could not create chart for {selected_ticker}. Insufficient historical data.")
                
    except Exception as e:
        st.error(f"❌ Error loading historical price data: {str(e)}")
        st.info("💡 Make sure the stock price data file is available in the data folder.")
    
    # --- Visualizations ---
    st.header("📈 Portfolio Analysis")
    
    # Create tabs for different visualizations (Momentum Winners moved to its own page)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Rankings", "🎯 Risk vs Return", "📈 Technical Signals", "🏢 Sector Analysis"])
    
    with tab1:
        # Stock rankings chart (exclude low-priced stocks < $3)
        _chart_df = st.session_state.top_stocks_df.copy()
        if 'close' in _chart_df.columns:
            try:
                _chart_df = _chart_df[pd.to_numeric(_chart_df['close'], errors='coerce') >= 3]
            except Exception:
                pass
        fig = px.bar(
            _chart_df.head(num_stocks),
            x='ticker',
            y='predicted_return_pct',
            color='confidence_score',
            title="Top 10 Stocks by Predicted Return",
            labels={'predicted_return_pct': 'Predicted Return (%)', 'confidence_score': 'Confidence Score'},
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(title=f"Top {num_stocks} Stocks by Predicted Return")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Risk vs Return scatter plot
        _df_plot = st.session_state.top_stocks_df.copy()
        # Plotly marker sizes must be non-negative
        if 'confidence_score' in _df_plot.columns:
            _df_plot['confidence_size'] = _df_plot['confidence_score'].abs()
        else:
            _df_plot['confidence_size'] = 0.0
        fig = px.scatter(
            _df_plot,
            x='risk_score',
            y='predicted_return_pct',
            size='confidence_size',
            color='composite_score',
            hover_data=['ticker'],
            title="Risk vs Return Analysis",
            labels={'risk_score': 'Risk Score', 'predicted_return_pct': 'Predicted Return (%)', 
                   'confidence_size': 'Confidence (|score|)', 'composite_score': 'Composite Score'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Technical signals heatmap
        if len(st.session_state.top_stocks_df) > 0:
            tech_signals = ['broke_resistance', 'post_earnings_dip_rally', 'strong_momentum', 
                           'breakout_confirmed', 'earnings_in_3_weeks', 'last_2q_positive_surprises']
            
            signal_data = []
            for _, stock in st.session_state.top_stocks_df.iterrows():
                for signal in tech_signals:
                    if signal in stock:
                        signal_data.append({
                            'ticker': stock['ticker'],
                            'signal': signal.replace('_', ' ').title(),
                            'value': 1 if stock[signal] else 0
                        })
            
            if signal_data:
                signal_df = pd.DataFrame(signal_data)
                
                # Handle duplicate entries by aggregating values
                try:
                    # Group by ticker and signal, then take the max value (in case of duplicates)
                    signal_df_agg = signal_df.groupby(['ticker', 'signal'])['value'].max().reset_index()
                    
                    # Create pivot table
                    pivot_df = signal_df_agg.pivot(index='ticker', columns='signal', values='value')
                    
                    # Fill NaN values with 0
                    pivot_df = pivot_df.fillna(0)
                    
                    fig = px.imshow(
                        pivot_df,
                        title="Technical Signals Heatmap",
                        color_continuous_scale='RdYlGn',
                        aspect='auto'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not create heatmap due to data structure: {str(e)}")
                    # Fallback: show a simple table
                    st.write("**Technical Signals Summary:**")
                    signal_summary = st.session_state.top_stocks_df[tech_signals].sum()
                    st.write(signal_summary)
    
    with tab4:
        # Sector analysis
        if 'sector' in st.session_state.top_stocks_df.columns:
            sector_counts = st.session_state.top_stocks_df['sector'].value_counts()
            fig = px.pie(
                values=sector_counts.values,
                names=sector_counts.index,
                title="Portfolio Sector Distribution"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sector information not available in the current dataset.")
    

    
    # --- Stock Table ---
    st.header("📋 Detailed Stock Recommendations")
    
    # Format the dataframe for better readability
    df_to_display = st.session_state.top_stocks_df.copy()
    # Optionally merge external model outputs (XGB weekly, LSTM predictions, Ensemble)
    try:
        import os as _os
        import pandas as _pd
        def _read_if_exists(path):
            try:
                return _pd.read_csv(path) if _os.path.exists(path) else None
            except Exception:
                return None
        # Prefer merged weekly XGB output if present, else raw ranked
        xgb_weekly = _read_if_exists('data/top/xgb_weekly_output.csv')
        use_weekly = xgb_weekly is not None and 'ticker' in xgb_weekly.columns
        xgb_csv = _read_if_exists('data/top/xgb_ranked_output.csv') or _read_if_exists('data/top/xgb_ranked.csv')
        lstm_csv = _read_if_exists('data/top/lstm_weekly_predictions_output.csv') or _read_if_exists('data/top/lstm_weekly_predictions.csv')
        ens_csv = _read_if_exists('data/top/ensemble_scores_output.csv') or _read_if_exists('data/top/final_ensemble_scores.csv')
        if use_weekly:
            df_to_display = xgb_weekly.copy()
            df_to_display['ticker'] = df_to_display['ticker'].astype(str).str.upper()
        else:
            # Fallback to building from session state + ranked + features
            # Prefer latest per-ticker features for clean 1:1 merges; fallback to full input
            xgb_feat_csv = _read_if_exists('data/top/xgb_features_latest.csv') or _read_if_exists('data/top/xgb_features_input.csv')
            # Prepare uppercase ticker keys
            df_to_display['ticker'] = df_to_display['ticker'].astype(str).str.upper()
            if xgb_csv is not None and 'ticker' in xgb_csv.columns:
                xgb_csv['ticker'] = xgb_csv['ticker'].astype(str).str.upper()
                # Disambiguate columns to avoid overwriting core fields
                xgb_ren = {}
                if 'confidence_score' in xgb_csv.columns:
                    xgb_ren['confidence_score'] = 'xgb_confidence_score'
                if 'predicted_return_pct' in xgb_csv.columns:
                    xgb_ren['predicted_return_pct'] = 'xgb_predicted_return_pct'
                xgb_csv = xgb_csv.rename(columns=xgb_ren)
                df_to_display = df_to_display.merge(xgb_csv, on='ticker', how='left')
            # Merge XGB feature inputs to expose all engineered columns in UI
            if xgb_feat_csv is not None and 'ticker' in xgb_feat_csv.columns:
                xgb_feat_csv['ticker'] = xgb_feat_csv['ticker'].astype(str).str.upper()
                # If multiple dates exist, keep the latest per ticker to avoid row explosion
                if 'date' in xgb_feat_csv.columns:
                    try:
                        xgb_feat_csv['date'] = _pd.to_datetime(xgb_feat_csv['date'], errors='coerce')
                        xgb_feat_csv = xgb_feat_csv.sort_values(['ticker', 'date']).drop_duplicates(subset=['ticker'], keep='last')
                    except Exception:
                        pass
                extra_cols = [c for c in xgb_feat_csv.columns if c != 'ticker' and c not in df_to_display.columns]
                if extra_cols:
                    df_to_display = df_to_display.merge(xgb_feat_csv[['ticker'] + extra_cols], on='ticker', how='left')
                    try:
                        st.caption(f"Merged XGB feature columns added: {len(extra_cols)}")
                    except Exception:
                        pass
                # Backfill overlapping columns (e.g., rsi_14d) where current values are NaN
                overlap_cols = [c for c in xgb_feat_csv.columns if c != 'ticker' and c in df_to_display.columns]
                if overlap_cols:
                    feat_subset = xgb_feat_csv[['ticker'] + overlap_cols]
                    df_to_display = df_to_display.merge(feat_subset, on='ticker', how='left', suffixes=("", "_feat"))
                    filled = 0
                    for col in overlap_cols:
                        feat_col = f"{col}_feat"
                        if feat_col in df_to_display.columns:
                            try:
                                before_na = df_to_display[col].isna().sum()
                                df_to_display[col] = df_to_display[col].combine_first(df_to_display[feat_col])
                                after_na = df_to_display[col].isna().sum()
                                filled += max(0, before_na - after_na)
                            except Exception:
                                pass
                            try:
                                df_to_display.drop(columns=[feat_col], inplace=True)
                            except Exception:
                                pass
                    try:
                        if filled > 0:
                            st.caption(f"Backfilled {filled} missing values from XGB features (RSI/Momentum/Volume/etc.)")
                    except Exception:
                        pass
        if lstm_csv is not None and 'ticker' in lstm_csv.columns:
            lstm_csv['ticker'] = lstm_csv['ticker'].astype(str).str.upper()
            df_to_display = df_to_display.merge(lstm_csv, on='ticker', how='left')
        if ens_csv is not None and 'ticker' in ens_csv.columns:
            ens_csv['ticker'] = ens_csv['ticker'].astype(str).str.upper()
            # Merge ensemble score and any available model columns if present
            ens_cols = [c for c in ['ensemble_score', 'xgb_pred', 'xgb_predicted_return_pct', 'lstm_predicted_return_pct'] if c in ens_csv.columns]
            merge_cols = ['ticker'] + ens_cols if ens_cols else ['ticker']
            df_to_display = df_to_display.merge(ens_csv[merge_cols], on='ticker', how='left')
    except Exception:
        pass
    
    # Format numeric columns
    if 'close' in df_to_display.columns:
        df_to_display['close'] = df_to_display['close'].apply(lambda x: f"${x:.2f}")
    if 'predicted_change' in df_to_display.columns:
        df_to_display['predicted_change'] = df_to_display['predicted_change'].apply(lambda x: f"${x:+.2f}")
    if 'predicted_return_pct' in df_to_display.columns:
        df_to_display['predicted_return_pct'] = df_to_display['predicted_return_pct'].apply(lambda x: f"{x:+.2f}%")
    if 'xgb_predicted_return_pct' in df_to_display.columns:
        df_to_display['xgb_predicted_return_pct'] = df_to_display['xgb_predicted_return_pct'].apply(lambda x: f"{x:+.2f}%")
    if 'lstm_predicted_return_pct' in df_to_display.columns:
        df_to_display['lstm_predicted_return_pct'] = df_to_display['lstm_predicted_return_pct'].apply(lambda x: f"{x:+.2f}%")
    if 'ensemble_score' in df_to_display.columns:
        df_to_display['ensemble_score'] = df_to_display['ensemble_score'].apply(lambda x: f"{x:+.3f}")
    if 'confidence_score' in df_to_display.columns:
        df_to_display['confidence_score'] = df_to_display['confidence_score'].apply(lambda x: f"{x:.1f}")
    if 'xgb_confidence_score' in df_to_display.columns:
        df_to_display['xgb_confidence_score'] = df_to_display['xgb_confidence_score'].apply(lambda x: f"{x:.1f}")
    if 'risk_score' in df_to_display.columns:
        df_to_display['risk_score'] = df_to_display['risk_score'].apply(lambda x: f"{x:.1f}")
    if 'composite_score' in df_to_display.columns:
        df_to_display['composite_score'] = df_to_display['composite_score'].apply(lambda x: f"{x:.1f}")
    
    # Highlight rows also present in Momentum page (green background)
    try:
        momentum_path = os.getenv('FEATURED_STOCKS_MOMENTUM_CSV', 'data/momentum/featured_stocks_momentum.csv')
        mom_df = pd.read_csv(momentum_path)
        # Build winners like the Momentum page
        for col in ['momentum_30d','momentum_60d','trend_slope_15d','ma_20','macd','macd_signal','close']:
            if col in mom_df.columns:
                mom_df[col] = pd.to_numeric(mom_df[col], errors='coerce')
        if 'momentum_winner' in mom_df.columns and mom_df['momentum_winner'].notna().any():
            mask = mom_df['momentum_winner'] == True
        else:
            mask = (
                ((mom_df.get('momentum_60d', 0) > 0.5) | (mom_df.get('momentum_30d', 0) > 0.2)) &
                (mom_df.get('trend_slope_15d', 0) > 0) &
                (mom_df.get('close', 0) > mom_df.get('ma_20', 0)) &
                (mom_df.get('macd', 0) > mom_df.get('macd_signal', 0))
            )
        winners = mom_df[mask].copy()
        # Score and take top 20
        if not winners.empty:
            def _mw_score(row):
                m60 = row.get('momentum_60d', 0) if pd.notna(row.get('momentum_60d', 0)) else 0
                m30 = row.get('momentum_30d', 0) if pd.notna(row.get('momentum_30d', 0)) else 0
                slope30 = row.get('trend_slope_30d', 0) if 'trend_slope_30d' in winners.columns else 0
                slope30 = slope30 if pd.notna(slope30) else 0
                return 0.6*m60 + 0.3*m30 + 0.1*slope30
            winners['mw_score'] = winners.apply(_mw_score, axis=1)
            winners = winners.sort_values('mw_score', ascending=False).head(20)
        momentum_tickers = set(winners['ticker'].astype(str).str.upper()) if not winners.empty else set()
    except Exception:
        momentum_tickers = set()

    def _highlight_momentum(row):
        t = str(row.get('ticker', '')).upper()
        color = '#e6ffed' if t in momentum_tickers else ''
        return [f'background-color: {color}'] * len(row)

    try:
        styled = df_to_display.style.apply(_highlight_momentum, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        # Fallback without styling
        st.dataframe(df_to_display, use_container_width=True, hide_index=True)
    
    # --- Export Options ---
    st.header("💾 Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = st.session_state.top_stocks_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"top_stocks_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Portfolio summary
        summary = f"""
Portfolio Summary - Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Total Stocks: {len(st.session_state.top_stocks_df)}
Average Predicted Return: {st.session_state.top_stocks_df['predicted_return_pct'].mean():.2f}%
Average Confidence Score: {st.session_state.top_stocks_df['confidence_score'].mean():.1f}
Average Risk Score: {st.session_state.top_stocks_df['risk_score'].mean():.1f}

Top 5 Picks:
"""
        for i, (_, stock) in enumerate(st.session_state.top_stocks_df.head().iterrows(), 1):
            summary += f"{i}. {stock['ticker']}: {stock['predicted_return_pct']:.2f}% return, {stock['confidence_score']:.1f} confidence\n"
        
        st.download_button(
            label="📄 Download Summary",
            data=summary,
            file_name=f"portfolio_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    # --- Chatbot Interface ---
    st.header("🤖 AI Stock Analyst Chat")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me about the stock recommendations, technical analysis, or portfolio insights..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                if st.session_state.agent is not None:
                    try:
                        response = st.session_state.agent.invoke({"input": prompt})
                        response_text = response.get('output', "Sorry, I encountered an error.")
                        
                        # Handle empty responses
                        if not response_text or response_text.strip() == "":
                            response_text = "I'm sorry, I couldn't generate a response. Please try asking a different question or rephrase your query."
                        
                        # Add debug info for troubleshooting
                        print(f"Chatbot response: {response_text[:200]}...")
                        
                    except Exception as e:
                        response_text = f"Sorry, I encountered an error: {str(e)}"
                        print(f"Chatbot error: {e}")
                else:
                    response_text = "The analysis must be run first to initialize the chatbot."
                st.markdown(response_text)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("🎯 Click the 'Find Top Stocks' button above to run the analysis and see stock recommendations.")
    
    # Show some helpful tips
    st.header("💡 How to Use This Screener")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Filter Settings**")
        st.write("• **Confidence Score**: Higher values = more confident predictions")
        st.write("• **Risk Score**: Lower values = less risky stocks")
        st.write("• **Diversification**: Limits exposure per sector/industry")
        st.write("• **Number of Stocks**: Choose how many recommendations to see")
    
    with col2:
        st.write("**📈 Analysis Features**")
        st.write("• **Technical Indicators**: RSI, MACD, Bollinger Bands, Volume")
        st.write("• **Fundamental Analysis**: Earnings surprises, growth rates")
        st.write("• **Risk Metrics**: Volatility, drawdown, Value at Risk")
        st.write("• **AI Chatbot**: Ask questions about recommendations")

## (Momentum Winners detailed section removed; available on its own page)

# --- Symbol Analysis Section (Always Available) ---
st.header("🔍 Individual Symbol Analysis")
st.write("Enter any stock symbol to analyze its current metrics and understand why it was or wasn't selected as a top pick.")

# Symbol input
col1, col2 = st.columns([2, 1])
with col1:
    symbol_input = st.text_input("Enter Stock Symbol (e.g., GOOGL, AAPL, TSLA):", placeholder="GOOGL", key="symbol_analysis_input")
with col2:
    analyze_button = st.button("🔍 Analyze Symbol", type="primary")

if analyze_button and symbol_input:
    symbol_input = symbol_input.upper().strip()
    
    # Load the featured data to get all available symbols
    try:
        # Try env path, then standardized top location, then legacy default
        env_path = os.getenv('FEATURED_STOCKS_CSV', 'data/featured_stocks_top.csv')
        try_paths = [env_path, 'data/top/featured_stocks_top.csv', 'data/featured_stocks_top.csv']
        picked = None
        for _p in try_paths:
            if os.path.exists(_p):
                picked = _p
                break
        if picked is None:
            raise FileNotFoundError(f"No featured stocks CSV found in {try_paths}")
        featured_data = pd.read_csv(picked)
        symbol_data = featured_data[featured_data['ticker'] == symbol_input]
        
        if symbol_data.empty:
            st.error(f"❌ Symbol '{symbol_input}' not found in the dataset. Available symbols: {', '.join(featured_data['ticker'].unique())}")
        else:
            # Get the symbol data
            stock_data = symbol_data.iloc[0]
            
            # Run the stock selector to get predictions
            try:
                top_stocks = get_top_stocks(n=50, min_confidence=0, max_risk=100, diversify=False)
                symbol_in_top = top_stocks[top_stocks['ticker'] == symbol_input]
                
                st.success(f"✅ Analysis complete for {symbol_input}")
                
                # Display current metrics
                st.subheader(f"📊 Current Metrics for {symbol_input}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${stock_data['close']:.2f}")
                with col2:
                    st.metric("RSI (14d)", f"{stock_data['rsi_14d']:.1f}")
                with col3:
                    st.metric("Volatility (30d)", f"{stock_data['volatility_30d']:.3f}")
                with col4:
                    st.metric("Volume Ratio", f"{stock_data['volume_ratio']:.2f}")
                
                # Display momentum metrics
                st.subheader("📈 Momentum Analysis")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("5-Day Momentum", f"{stock_data['momentum_5d']:.3f}")
                with col2:
                    st.metric("10-Day Momentum", f"{stock_data['momentum_10d']:.3f}")
                with col3:
                    st.metric("20-Day Momentum", f"{stock_data['momentum_20d']:.3f}")
                # Extended momentum
                col1, col2, col3 = st.columns(3)
                if 'momentum_30d' in stock_data.index:
                    with col1:
                        st.metric("30-Day Momentum", f"{stock_data['momentum_30d']:.3f}")
                if 'momentum_60d' in stock_data.index:
                    with col2:
                        st.metric("60-Day Momentum", f"{stock_data['momentum_60d']:.3f}")
                if 'up_day_ratio_20d' in stock_data.index:
                    with col3:
                        st.metric("Up-Day Ratio (20d)", f"{stock_data['up_day_ratio_20d']*100:.1f}%")
                
                # Display prediction results
                st.subheader("🎯 Model Prediction Results")
                
                if not symbol_in_top.empty:
                    prediction_data = symbol_in_top.iloc[0]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Predicted Return", f"{prediction_data['predicted_return_pct']:.2f}%", 
                                delta=f"{prediction_data['predicted_return_pct']:.2f}%")
                    with col2:
                        st.metric("Confidence Score", f"{prediction_data['confidence_score']:.1f}/100")
                    with col3:
                        st.metric("Risk Score", f"{prediction_data['risk_score']:.1f}/100")
                    with col4:
                        st.metric("Composite Score", f"{prediction_data['composite_score']:.1f}")
                    
                    # Show ranking
                    rank = top_stocks[top_stocks['ticker'] == symbol_input].index[0] + 1
                    st.info(f"🏆 **Ranking**: {symbol_input} is ranked #{rank} out of {len(top_stocks)} stocks analyzed")
                    
                else:
                    st.warning(f"⚠️ {symbol_input} was not selected in the top picks analysis")
                    
                    # Show why it wasn't selected
                    st.subheader("🔍 Why Wasn't It Selected?")
                    
                    # Calculate what the prediction would be
                    try:
                        # Get model predictions for this symbol (Top Stocks model)
                        model = joblib.load('models/stock_predictor_top.joblib')
                        # Use the model's expected training columns
                        training_cols = model.get_booster().feature_names
                        # Build a single-row feature frame matching the model schema
                        row_dict = {}
                        for col in training_cols:
                            val = stock_data.get(col, 0)
                            try:
                                val = float(val)
                            except Exception:
                                val = 0.0
                            row_dict[col] = val
                        import pandas as _pd_alias  # local alias to avoid shadowing
                        X_symbol = _pd_alias.DataFrame([row_dict])

                        # Make prediction (percentage over ~5 trading days)
                        predicted_return_pct = float(model.predict(X_symbol)[0])

                        st.metric("Model Prediction", f"{predicted_return_pct:.2f}%",
                                  delta=f"{predicted_return_pct:.2f}%")
                        
                        # Calculate confidence and risk scores
                        confidence_score = calculate_confidence_score(stock_data)
                        risk_score = calculate_risk_score(stock_data)
                        composite_score = (predicted_return_pct * 0.4) + (confidence_score * 0.4) + ((100 - risk_score) * 0.2)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Calculated Confidence", f"{confidence_score:.1f}/100")
                        with col2:
                            st.metric("Calculated Risk", f"{risk_score:.1f}/100")
                        with col3:
                            st.metric("Calculated Composite", f"{composite_score:.1f}")
                        
                    except Exception as e:
                        st.error(f"Could not calculate model prediction: {str(e)}")
                
                # Technical Analysis Summary
                st.subheader("📊 Technical Analysis Summary")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Key Technical Signals:**")
                    signals = []
                    if stock_data.get('broke_resistance', False):
                        signals.append("✅ Broke Resistance")
                    else:
                        signals.append("❌ No Resistance Break")
                    
                    if stock_data.get('post_earnings_dip_rally', False):
                        signals.append("✅ Post-Earnings Dip Rally")
                    else:
                        signals.append("❌ No Post-Earnings Rally")
                    
                    if stock_data.get('strong_momentum', False):
                        signals.append("✅ Strong Momentum")
                    else:
                        signals.append("❌ Weak Momentum")
                    
                    if stock_data.get('breakout_confirmed', False):
                        signals.append("✅ Breakout Confirmed")
                    else:
                        signals.append("❌ No Breakout Confirmation")
                    
                    for signal in signals:
                        st.write(f"• {signal}")
                
                with col2:
                    st.write("**Risk Assessment:**")
                    volatility = stock_data['volatility_30d']
                    if volatility > 0.5:
                        st.write("• ⚠️ High Volatility")
                    elif volatility > 0.3:
                        st.write("• 📊 Moderate Volatility")
                    else:
                        st.write("• ✅ Low Volatility")
                    
                    rsi = stock_data['rsi_14d']
                    if rsi > 70:
                        st.write("• ⚠️ Overbought (RSI > 70)")
                    elif rsi < 30:
                        st.write("• 🟢 Oversold (RSI < 30)")
                    else:
                        st.write("• 📊 Neutral RSI")
                
                # Detailed explanation
                st.subheader("💡 Analysis Explanation")
                
                explanation = f"""
                **Analysis for {symbol_input} (${stock_data['close']:.2f})**
                
                **Current Technical Position:**
                - **RSI**: {stock_data['rsi_14d']:.1f} ({'Overbought' if stock_data['rsi_14d'] > 70 else 'Oversold' if stock_data['rsi_14d'] < 30 else 'Neutral'})
                - **Momentum**: 5d={stock_data['momentum_5d']:.3f}, 10d={stock_data['momentum_10d']:.3f}, 20d={stock_data['momentum_20d']:.3f}
                - **Volatility**: {stock_data['volatility_30d']:.3f} ({'High' if stock_data['volatility_30d'] > 0.5 else 'Moderate' if stock_data['volatility_30d'] > 0.3 else 'Low'})
                - **Volume**: {stock_data['volume_ratio']:.2f}x average ({'Above' if stock_data['volume_ratio'] > 1.2 else 'Below' if stock_data['volume_ratio'] < 0.8 else 'Normal'} average)
                
                **Key Signals:**
                - **Resistance Break**: {'✅ Yes' if stock_data.get('broke_resistance', False) else '❌ No'}
                - **Strong Momentum**: {'✅ Yes' if stock_data.get('strong_momentum', False) else '❌ No'}
                - **Breakout Confirmed**: {'✅ Yes' if stock_data.get('breakout_confirmed', False) else '❌ No'}
                
                **Why This Matters:**
                The model looks for stocks with strong technical signals, positive momentum, and confirmed breakouts. 
                Stocks with low momentum, no resistance breaks, or poor technical signals typically score lower 
                in the confidence and composite scoring system.
                """
                
                st.markdown(explanation)
                
            except Exception as e:
                st.error(f"Error analyzing symbol: {str(e)}")
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

elif analyze_button and not symbol_input:
    st.warning("Please enter a stock symbol to analyze.")