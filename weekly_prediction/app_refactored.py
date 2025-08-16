import streamlit as st
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Import modular components
from charts.stock_charts import create_stock_price_chart, create_price_line_chart, create_earnings_chart
from ui_components.stock_analysis import (
    display_technical_analysis_summary, display_key_price_levels, 
    display_resistance_analysis, display_support_analysis,
    display_technical_insights, display_chart_interpretation_guide
)
from ui_components.portfolio_analysis import (
    display_stock_rankings, display_risk_return_analysis, 
    display_technical_signals_heatmap, display_sector_analysis,
    display_portfolio_metrics, display_top_stocks_table, display_export_options
)
from utils.data_utils import (
    load_stock_data, load_featured_stocks_data, load_earnings_history_data,
    get_stock_featured_data, get_earnings_data_for_ticker,
    format_price_stats, calculate_price_movement_summary
)

# Import existing modules
import stock_selector
import chatbot

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Stock Screener & Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Stock Screener & Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key Status
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            st.success("✅ Google API Key Found")
        else:
            st.error("❌ Google API Key Missing")
            st.info("Add GOOGLE_API_KEY to your .env file")
        
        # Model Settings
        st.subheader("🎯 Model Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0, 100, 30, help="Minimum confidence score for stock selection")
        risk_threshold = st.slider("Risk Threshold", 0, 100, 70, help="Maximum risk score for stock selection")
        num_stocks = st.slider("Number of Stocks", 5, 20, 10, help="Number of top stocks to display")
        
        # Analysis Settings
        st.subheader("📊 Analysis Settings")
        show_technical_analysis = st.checkbox("Show Technical Analysis", value=True)
        show_portfolio_analysis = st.checkbox("Show Portfolio Analysis", value=True)
        show_chatbot = st.checkbox("Show AI Chatbot", value=True)
    
    # Initialize session state
    if 'top_stocks_df' not in st.session_state:
        st.session_state.top_stocks_df = pd.DataFrame()
    if 'chatbot_agent' not in st.session_state:
        st.session_state.chatbot_agent = None
    
    # Load data and run analysis
    if st.button("🚀 Run Stock Analysis", type="primary"):
        with st.spinner("Loading data and running analysis..."):
            try:
                # Get top stocks
                top_stocks = stock_selector.get_top_stocks(
                    n=num_stocks,
                    confidence_threshold=confidence_threshold,
                    risk_threshold=risk_threshold
                )
                
                if not top_stocks.empty:
                    st.session_state.top_stocks_df = top_stocks
                    st.success(f"✅ Analysis complete! Found {len(top_stocks)} stocks.")
                else:
                    st.warning("⚠️ No stocks found matching the criteria. Try adjusting the thresholds.")
                    
            except Exception as e:
                st.error(f"❌ Error running analysis: {str(e)}")
    
    # Display results if available
    if not st.session_state.top_stocks_df.empty:
        # Portfolio Analysis Section
        if show_portfolio_analysis:
            st.header("📈 Portfolio Analysis")
            
            # Portfolio metrics
            display_portfolio_metrics(st.session_state.top_stocks_df)
            
            # Top stocks table
            display_top_stocks_table(st.session_state.top_stocks_df)
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Rankings", "🎯 Risk vs Return", "📈 Technical Signals", "🏢 Sector Analysis"])
            
            with tab1:
                display_stock_rankings(st.session_state.top_stocks_df)
            
            with tab2:
                display_risk_return_analysis(st.session_state.top_stocks_df)
            
            with tab3:
                display_technical_signals_heatmap(st.session_state.top_stocks_df)
            
            with tab4:
                display_sector_analysis(st.session_state.top_stocks_df)
            
            # Export options
            display_export_options(st.session_state.top_stocks_df)
        
        # Individual Stock Analysis Section
        st.header("🔍 Individual Stock Analysis")
        
        # Stock selector
        available_tickers = st.session_state.top_stocks_df['ticker'].tolist()
        selected_ticker = st.selectbox("Select a stock to analyze:", available_tickers)
        
        if selected_ticker:
            # Load data for selected stock
            stock_data = load_stock_data(selected_ticker)
            featured_data = load_featured_stocks_data()
            earnings_data = load_earnings_history_data()
            
            if stock_data is not None and featured_data is not None:
                stock_featured_data = get_stock_featured_data(
                    featured_data, selected_ticker, st.session_state.top_stocks_df
                )
                
                if stock_featured_data is not None:
                    # Create tabs for different chart views
                    candlestick_tab, price_graph_tab, earnings_tab = st.tabs([
                        "📊 Candlestick Chart", "📈 Price Graph", "📋 Earnings History"
                    ])
                    
                    with candlestick_tab:
                        # Create the candlestick chart
                        chart = create_stock_price_chart(selected_ticker, stock_data, stock_featured_data)
                        
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                            
                            # Technical analysis summary
                            if show_technical_analysis:
                                display_technical_analysis_summary(stock_featured_data)
                                display_key_price_levels(stock_featured_data)
                                display_resistance_analysis(stock_featured_data)
                                display_support_analysis(stock_featured_data)
                                display_technical_insights(stock_featured_data)
                                display_chart_interpretation_guide()
                        else:
                            st.warning(f"Could not create chart for {selected_ticker}. Insufficient historical data.")
                    
                    with price_graph_tab:
                        st.subheader("📈 3-Week Price Graph")
                        
                        # Create line chart
                        line_chart = create_price_line_chart(selected_ticker, stock_data)
                        if line_chart:
                            st.plotly_chart(line_chart, use_container_width=True)
                        
                        # Price statistics
                        price_stats = format_price_stats(stock_data)
                        if price_stats:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Current Price", f"${price_stats['current_price']:.2f}")
                                st.metric("Period High", f"${price_stats['period_high']:.2f}")
                            
                            with col2:
                                st.metric("Price Change", f"${price_stats['price_change']:.2f}", 
                                        delta=f"{price_stats['price_change_pct']:.2f}%")
                                st.metric("Period Low", f"${price_stats['period_low']:.2f}")
                            
                            with col3:
                                st.metric("Current Volume", f"{price_stats['current_volume']:,.0f}")
                                st.metric("Volume Change", f"{price_stats['volume_change']:.1f}%")
                        
                        # Price movement summary
                        movement_summary = calculate_price_movement_summary(stock_data)
                        if movement_summary:
                            st.subheader("📊 Price Movement Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Return", f"{movement_summary['total_return']:.2f}%")
                            with col2:
                                st.metric("Volatility", f"{movement_summary['volatility']:.2f}%")
                            with col3:
                                st.metric("Positive Days", movement_summary['positive_days'])
                            with col4:
                                st.metric("Negative Days", movement_summary['negative_days'])
                    
                    with earnings_tab:
                        st.subheader("📋 Quarterly Earnings & Revenue History")
                        
                        if earnings_data is not None:
                            ticker_earnings = get_earnings_data_for_ticker(earnings_data, selected_ticker)
                            
                            if ticker_earnings is not None:
                                # Earnings chart
                                earnings_chart = create_earnings_chart(ticker_earnings, selected_ticker)
                                if earnings_chart:
                                    st.plotly_chart(earnings_chart, use_container_width=True)
                                
                                # Earnings table
                                st.subheader("📊 Earnings Performance")
                                earnings_display = ticker_earnings[['earnings_date', 'reported_eps', 'estimate_eps', 'surprise_percentage']].copy()
                                earnings_display['earnings_date'] = earnings_display['earnings_date'].dt.strftime('%Y-%m-%d')
                                earnings_display.columns = ['Earnings Date', 'Reported EPS', 'Estimate EPS', 'Surprise %']
                                st.dataframe(earnings_display, use_container_width=True)
                                
                                # Earnings consistency analysis
                                st.subheader("📈 Earnings Consistency Analysis")
                                positive_surprises = len(ticker_earnings[ticker_earnings['surprise_percentage'] > 0])
                                total_quarters = len(ticker_earnings)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Positive Surprises", positive_surprises)
                                with col2:
                                    st.metric("Total Quarters", total_quarters)
                                with col3:
                                    consistency_rate = (positive_surprises / total_quarters) * 100 if total_quarters > 0 else 0
                                    st.metric("Consistency Rate", f"{consistency_rate:.1f}%")
                            else:
                                st.warning(f"No earnings history data found for {selected_ticker}")
                        else:
                            st.warning("Earnings history data not available")
                else:
                    st.error(f"Could not load featured data for {selected_ticker}")
            else:
                st.error(f"Could not load data for {selected_ticker}")
    
    # Individual Symbol Analysis Section
    st.header("🔍 Individual Symbol Analysis")
    
    # Symbol input
    symbol_input = st.text_input("Enter stock symbol (e.g., AAPL, GOOGL):", placeholder="AAPL")
    
    if st.button("Analyze Symbol") and symbol_input:
        with st.spinner(f"Analyzing {symbol_input.upper()}..."):
            try:
                # Load data for the input symbol
                stock_data = load_stock_data(symbol_input.upper())
                featured_data = load_featured_stocks_data()
                
                if stock_data is not None and featured_data is not None:
                    stock_featured_data = get_stock_featured_data(featured_data, symbol_input.upper())
                    
                    if stock_featured_data is not None:
                        st.success(f"✅ Analysis complete for {symbol_input.upper()}")
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Price", f"${stock_featured_data['close']:.2f}")
                            if 'predicted_return_pct' in stock_featured_data:
                                st.metric("Predicted Return", f"{stock_featured_data['predicted_return_pct']:.2f}%")
                        
                        with col2:
                            rsi_value = stock_featured_data.get('rsi_14d')
                            rsi_display = f"{rsi_value:.1f}" if rsi_value is not None and not pd.isna(rsi_value) else "N/A"
                            st.metric("RSI (14d)", rsi_display)
                            
                            if 'confidence_score' in stock_featured_data:
                                st.metric("Confidence Score", f"{stock_featured_data['confidence_score']:.1f}/100")
                        
                        with col3:
                            resistance_status = "🟢 BROKEN" if stock_featured_data.get('broke_resistance', False) else "🔴 HOLDING"
                            st.metric("Resistance Status", resistance_status)
                            
                            if 'risk_score' in stock_featured_data:
                                st.metric("Risk Score", f"{stock_featured_data['risk_score']:.1f}/100")
                        
                        # Detailed analysis
                        display_technical_analysis_summary(stock_featured_data)
                        display_key_price_levels(stock_featured_data)
                        display_resistance_analysis(stock_featured_data)
                        display_support_analysis(stock_featured_data)
                        display_technical_insights(stock_featured_data)
                        
                    else:
                        st.warning(f"No featured data available for {symbol_input.upper()}")
                else:
                    st.error(f"Could not load data for {symbol_input.upper()}")
                    
            except Exception as e:
                st.error(f"Error analyzing {symbol_input.upper()}: {str(e)}")
    
    # AI Chatbot Section
    if show_chatbot:
        st.header("🤖 AI Stock Analysis Assistant")
        
        # Initialize chatbot if we have data and it's not already initialized
        if not st.session_state.top_stocks_df.empty and st.session_state.chatbot_agent is None:
            try:
                st.session_state.chatbot_agent = chatbot.create_chatbot_agent(st.session_state.top_stocks_df)
                st.success("✅ Chatbot initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize chatbot: {str(e)}")
        
        # Show message if no data available
        if st.session_state.top_stocks_df.empty:
            st.info("💡 Run the stock analysis first to enable the AI chatbot with your data.")
        
        # Chat interface
        if st.session_state.chatbot_agent is not None:
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me about stocks, technical analysis, or investment strategies..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get chatbot response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.chatbot_agent.invoke(prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
