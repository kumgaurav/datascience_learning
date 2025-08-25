import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
from stock_analysis import calculate_rsi, generate_trading_signals, NeuroEvolutionAgent
from config_manager import ConfigManager, DatabaseManager
from logger_config import logger

# Page configuration - COMMENTED OUT (using app_refactored.py as main)
# st.set_page_config(
#     page_title="Stock Change Tracker",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# Custom CSS for styling
st.markdown("""
<style>
    .green-symbol {
        color: #00FF00 !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .tab-content {
        padding: 20px 0;
    }
    .connection-success {
        color: #28a745;
        font-weight: bold;
    }
    .connection-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize configuration and database managers
@st.cache_resource
def get_managers():
    """Initialize configuration and database managers"""
    logger.info("Creating ConfigManager...")
    try:
        config_manager = ConfigManager()
        logger.info("ConfigManager created successfully")
        
        logger.info("Creating DatabaseManager...")
        db_manager = DatabaseManager(config_manager)
        logger.info("DatabaseManager created successfully")
        
        return config_manager, db_manager
    except Exception as e:
        logger.error(f"Error creating managers: {e}", exc_info=True)
        raise

def is_earning_within_3_weeks(earning_date):
    """Check if earning date is within next 3 weeks"""
    if pd.isna(earning_date):
        return False
    
    try:
        earning_dt = pd.to_datetime(earning_date)
        today = datetime.now()
        three_weeks_later = today + timedelta(weeks=3)
        return today <= earning_dt <= three_weeks_later
    except:
        return False

def display_connection_status(db_manager):
    """Display database connection status"""
    st.sidebar.subheader("🔗 Database Status")
    
    # Test connection
    is_connected, message = db_manager.test_connection()
    
    if is_connected:
        st.sidebar.markdown(f'<p class="connection-success">✅ {message}</p>', unsafe_allow_html=True)
        
        # Display table information
        table_info = db_manager.get_table_info()
        if table_info:
            st.sidebar.subheader("📊 Tables")
            for table_key, info in table_info.items():
                if 'error' in info:
                    st.sidebar.error(f"{info['name']}: {info['error']}")
                else:
                    st.sidebar.success(f"{info['name']}: {info['count']} rows")
        
        # Test earnings join
        st.sidebar.subheader("🔗 Earnings Join Test")
        join_success, join_message = db_manager.test_earnings_join()
        if join_success:
            st.sidebar.success(join_message)
        else:
            st.sidebar.error(join_message)
        
        # Show table schema for debugging
        if st.sidebar.checkbox("Show Table Schema", value=False):
            st.sidebar.subheader("📋 Table Schema")
            schema = db_manager.get_table_schema('stocksinfp')
            st.sidebar.write("stocksinfp schema:", schema)
            
    else:
        st.sidebar.markdown(f'<p class="connection-error">❌ {message}</p>', unsafe_allow_html=True)
        st.sidebar.error("Please check your database configuration in conf/config.ini")

def display_home_page(db_manager):
    logger.info("=== DISPLAY HOME PAGE STARTED ===")
    st.title("📈 Stock Change Tracker Dashboard")
    st.markdown("---")
    
    # Initialize session state for pagination and search
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'search_term' not in st.session_state:
        st.session_state.search_term = ""
    if 'page_size' not in st.session_state:
        st.session_state.page_size = 50
    
    # Search and pagination controls
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search_input = st.text_input("🔍 Search stocks by symbol:", 
                                   value=st.session_state.search_term,
                                   placeholder="Enter symbol (e.g., AAPL, MSFT)")
        if search_input != st.session_state.search_term:
            st.session_state.search_term = search_input
            st.session_state.current_page = 1  # Reset to first page on new search
            st.rerun()
    
    with col2:
        page_size_options = [25, 50, 100, 200]
        page_size = st.selectbox("Items per page:", 
                               page_size_options, 
                               index=page_size_options.index(st.session_state.page_size))
        if page_size != st.session_state.page_size:
            st.session_state.page_size = page_size
            st.session_state.current_page = 1  # Reset to first page on page size change
            st.rerun()
    
    with col3:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Get total count and statistics for pagination and metrics
    total_stocks = db_manager.get_stock_count(st.session_state.search_term)
    total_pages = max(1, (total_stocks + st.session_state.page_size - 1) // st.session_state.page_size)
    gainers_count, losers_count, avg_change = db_manager.get_gainers_losers_count(st.session_state.search_term)
    
    # Load data from MySQL with pagination and search
    logger.info(f"Loading stock change tracker data... Page: {st.session_state.current_page}, Search: '{st.session_state.search_term}'")
    with st.spinner("Loading stock data..."):
        df = db_manager.load_stock_change_tracker(
            page=st.session_state.current_page,
            page_size=st.session_state.page_size,
            search_term=st.session_state.search_term
        )
    logger.info(f"Loaded {len(df)} rows from stock_change_tracker")
    
    if df.empty:
        if st.session_state.search_term:
            st.warning(f"No stocks found matching '{st.session_state.search_term}'. Try a different search term.")
        else:
            st.warning("No data available. Please check your database connection and ensure the stock_change_tracker table exists with data.")
            st.info("Expected columns: symbol, price_when_added, current_price, change_in_percent")
        
        # Show available symbols for debugging
        available_symbols = db_manager.get_available_symbols()
        if available_symbols:
            st.info(f"Available symbols in database: {', '.join(available_symbols[:10])}{'...' if len(available_symbols) > 10 else ''}")
        
        return
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", total_stocks)
    with col2:
        gainers_pct = f"{gainers_count/total_stocks*100:.1f}%" if total_stocks > 0 else "0%"
        st.metric("Gainers", gainers_count, delta=gainers_pct)
    with col3:
        losers_pct = f"{losers_count/total_stocks*100:.1f}%" if total_stocks > 0 else "0%"
        st.metric("Losers", losers_count, delta=losers_pct)
    with col4:
        st.metric("Avg Change %", f"{avg_change:.2f}%")
    
    st.markdown("---")
    
    # Display the table with conditional formatting
    st.subheader("Stock Performance Overview")
    st.caption("🟢 = Earnings within next 3 weeks | Click symbol for detailed analysis")
    
    # Create header
    header_cols = st.columns([2, 2, 2, 2, 2])
    headers = ["Symbol", "Price when added", "Current Price", "Change %", "Earnings Date"]
    
    for col, header in zip(header_cols, headers):
        col.markdown(f"**{header}**")
    
    st.markdown("---")
    
    # Display each row
    for idx, row in df.iterrows():
        cols = st.columns([2, 2, 2, 2, 2])
        
        with cols[0]:
            symbol_color = "🟢" if is_earning_within_3_weeks(row['earning_date']) else "⚪"
            button_key = f"btn_{row['symbol']}_{idx}"
            if st.button(f"{symbol_color} {row['symbol']}", key=button_key):
                try:
                    st.session_state.selected_symbol = row['symbol']
                    st.session_state.page = 'stock_detail'
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clicking symbol {row['symbol']}: {e}")
        
        with cols[1]:
            st.write(f"${row['price_when_added']:.2f}")
        
        with cols[2]:
            st.write(f"${row['current_price']:.2f}")
        
        with cols[3]:
            change_color = "🟢" if row['change_in_percent'] > 0 else "🔴"
            st.write(f"{change_color} {row['change_in_percent']:.2f}%")
        
        with cols[4]:
            earning_text = row['earning_date'] if pd.notna(row['earning_date']) else 'N/A'
            st.write(earning_text)
    
    # Pagination controls
    st.markdown("---")
    
    # Show current page info
    start_item = (st.session_state.current_page - 1) * st.session_state.page_size + 1
    end_item = min(st.session_state.current_page * st.session_state.page_size, total_stocks)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.session_state.current_page > 1:
            if st.button("⬅️ Previous", key="prev_page"):
                st.session_state.current_page -= 1
                st.rerun()
    
    with col2:
        st.markdown(f"<div style='text-align: center;'>Showing {start_item}-{end_item} of {total_stocks} stocks | Page {st.session_state.current_page} of {total_pages}</div>", 
                   unsafe_allow_html=True)
        
        # Page number input for quick navigation
        if total_pages > 1:
            page_input = st.number_input("Go to page:", 
                                       min_value=1, 
                                       max_value=total_pages, 
                                       value=st.session_state.current_page,
                                       key="page_input")
            if page_input != st.session_state.current_page:
                st.session_state.current_page = page_input
                st.rerun()
    
    with col3:
        if st.session_state.current_page < total_pages:
            if st.button("Next ➡️", key="next_page"):
                st.session_state.current_page += 1
                st.rerun()

def display_about_page():
    st.title("About Stock Change Tracker")
    st.markdown("""
    ## Features
    
    1. **Home Dashboard**: View all stocks ordered by change percentage
    2. **Earnings Highlight**: Stocks with earnings in next 3 weeks shown in green
    3. **Detailed Analysis**: Click any stock symbol for detailed analysis
    4. **RSI Analysis**: 14-day RSI calculations and visualization
    5. **Extreme RSI**: Analysis of periods when RSI > 75
    6. **Neuro-Evolution Signals**: Advanced trading signals using AI
    
    ## Technology Stack
    
    - **Frontend**: Streamlit
    - **Database**: MySQL (configured via conf/config.ini)
    - **Charts**: Plotly
    - **Technical Analysis**: Custom RSI implementation
    - **AI Signals**: Custom Neuro-Evolution Algorithm
    
    ## Database Configuration
    
    The application reads database settings from `conf/config.ini`:
    
    ```ini
    [mysql]
    url=localhost
    username=your_username
    password=your_password
    database=stocksdb
    table=stocksinfp
    stock_change_tracker_table=stock_change_tracker
    ```
    
    ## Expected Database Schema
    
    ### stock_change_tracker table:
    - symbol (VARCHAR)
    - change_in_percent (DECIMAL)
    - current_price (DECIMAL)
    - price_when_added (DECIMAL)
    
    ### stocks_earnings table:
    - symbol (VARCHAR)
    - earnings_date (DATE)
    
    ### stocksinfp table:
    - symbol (VARCHAR)
    - date (DATE)
    - open (DECIMAL)
    - high (DECIMAL)
    - low (DECIMAL)
    - close (DECIMAL)
    - volume (BIGINT)
    """)

def display_stock_detail_page(db_manager):
    if 'selected_symbol' not in st.session_state:
        st.error("No symbol selected. Please go back to home page.")
        return
    
    symbol = st.session_state.selected_symbol
    st.title(f"📊 {symbol} - Detailed Analysis")
    
    # Back button
    if st.button("← Back to Home", key="back_to_home_btn"):
        st.session_state.page = 'home'
        st.rerun()
    
    # Load stock data from MySQL
    stock_data = db_manager.load_stock_data(symbol)
    
    if stock_data.empty:
        st.warning(f"No historical data available for {symbol}")
        return
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["14-Day RSI", "RSI Analysis (>75)", "Neuro-Evolution Signals"])
    
    with tab1:
        st.subheader(f"{symbol} - 14-Day RSI Analysis")
        
        if len(stock_data) >= 14:
            # Calculate RSI
            rsi_values = calculate_rsi(stock_data['close'].values, period=14)
            
            if len(rsi_values) > 0:
                # Create RSI chart
                fig = go.Figure()
                
                # Add price line
                fig.add_trace(go.Scatter(
                    x=stock_data['date'].iloc[14:],
                    y=stock_data['close'].iloc[14:],
                    mode='lines',
                    name='Price',
                    yaxis='y2'
                ))
                
                # Add RSI line
                fig.add_trace(go.Scatter(
                    x=stock_data['date'].iloc[14:],
                    y=rsi_values,
                    mode='lines',
                    name='RSI',
                    yaxis='y'
                ))
                
                # Add RSI thresholds
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                
                # Update layout for dual y-axis
                fig.update_layout(
                    title=f'{symbol} - Price and RSI',
                    xaxis_title='Date',
                    yaxis=dict(title='RSI', side='left', range=[0, 100]),
                    yaxis2=dict(title='Price ($)', side='right', overlaying='y'),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # RSI Statistics
                current_rsi = rsi_values[-1]
                avg_rsi = np.mean(rsi_values)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current RSI", f"{current_rsi:.2f}")
                with col2:
                    st.metric("Average RSI", f"{avg_rsi:.2f}")
                with col3:
                    overbought_count = len(rsi_values[rsi_values > 70])
                    st.metric("Overbought Days", overbought_count)
                with col4:
                    oversold_count = len(rsi_values[rsi_values < 30])
                    st.metric("Oversold Days", oversold_count)
            else:
                st.error("Could not calculate RSI values")
        else:
            st.error("Insufficient data for RSI calculation (need at least 14 days)")
    
    with tab2:
        st.subheader(f"{symbol} - RSI > 75 Analysis")
        
        if len(stock_data) >= 14:
            rsi_values = calculate_rsi(stock_data['close'].values, period=14)
            
            if len(rsi_values) > 0:
                # Find periods where RSI > 75
                high_rsi_indices = np.where(rsi_values > 75)[0]
                
                if len(high_rsi_indices) > 0:
                    st.success(f"Found {len(high_rsi_indices)} periods with RSI > 75")
                    
                    # Create DataFrame for high RSI periods
                    high_rsi_data = []
                    for idx in high_rsi_indices:
                        data_idx = idx + 14  # Adjust for RSI calculation offset
                        if data_idx < len(stock_data):
                            high_rsi_data.append({
                                'Date': stock_data.iloc[data_idx]['date'],
                                'Price': stock_data.iloc[data_idx]['close'],
                                'RSI': rsi_values[idx]
                            })
                    
                    if high_rsi_data:
                        high_rsi_df = pd.DataFrame(high_rsi_data)
                        st.dataframe(high_rsi_df, use_container_width=True)
                        
                        # Chart for high RSI periods
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=high_rsi_df['Date'],
                            y=high_rsi_df['Price'],
                            mode='markers+lines',
                            name='Price during High RSI',
                            marker=dict(color='red', size=8)
                        ))
                        
                        fig.update_layout(
                            title=f'{symbol} - Price during RSI > 75 Periods',
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No periods found with RSI > 75")
            else:
                st.error("Could not calculate RSI values")
        else:
            st.error("Insufficient data for RSI calculation")
    
    with tab3:
        st.subheader(f"{symbol} - Neuro-Evolution Trading Signals")
        
        # Option to enable/disable neuro-evolution to prevent crashes
        enable_neuro = st.checkbox("Enable Neuro-Evolution Analysis (May be resource intensive)", value=False)
        
        if enable_neuro:
            if len(stock_data) >= 30:
                try:
                    with st.spinner("Generating neuro-evolution signals..."):
                        signals = generate_trading_signals(symbol, stock_data)
                    
                    if signals is not None and not signals.empty:
                        # Display signals
                        st.success(f"Generated {len(signals)} trading signals")
                        st.dataframe(signals.tail(20), use_container_width=True)
                        
                        # Plot signals on price chart
                        fig = go.Figure()
                        
                        # Price line
                        fig.add_trace(go.Scatter(
                            x=stock_data['date'],
                            y=stock_data['close'],
                            mode='lines',
                            name='Price',
                            line=dict(color='black', width=2)
                        ))
                        
                        # Buy signals
                        buy_signals = signals[signals['signal'] == 'BUY']
                        if not buy_signals.empty:
                            fig.add_trace(go.Scatter(
                                x=buy_signals['date'],
                                y=buy_signals['close'],
                                mode='markers',
                                name='Buy Signal',
                                marker=dict(color='green', size=10, symbol='triangle-up')
                            ))
                        
                        # Sell signals
                        sell_signals = signals[signals['signal'] == 'SELL']
                        if not sell_signals.empty:
                            fig.add_trace(go.Scatter(
                                x=sell_signals['date'],
                                y=sell_signals['close'],
                                mode='markers',
                                name='Sell Signal',
                                marker=dict(color='red', size=10, symbol='triangle-down')
                            ))
                        
                        fig.update_layout(
                            title=f'{symbol} - Neuro-Evolution Trading Signals',
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No trading signals generated")
                        
                except Exception as e:
                    st.error(f"Error generating neuro-evolution signals: {e}")
                    st.info("Showing simulated signals instead...")
                    
                    # Show simulated results
                    np.random.seed(hash(symbol) % 2**32)
                    n_signals = min(10, len(stock_data))
                    signal_indices = np.random.choice(len(stock_data), n_signals, replace=False)
                    signal_indices.sort()
                    
                    sim_signals = []
                    for idx in signal_indices:
                        signal_type = np.random.choice(['BUY', 'SELL'], p=[0.6, 0.4])
                        confidence = np.random.uniform(70, 95)
                        sim_signals.append({
                            'Date': stock_data.iloc[idx]['date'].strftime('%Y-%m-%d'),
                            'Price': f"${stock_data.iloc[idx]['close']:.2f}",
                            'Signal': signal_type,
                            'Confidence': f"{confidence:.1f}%"
                        })
                    
                    st.dataframe(pd.DataFrame(sim_signals), use_container_width=True)
            else:
                st.error("Insufficient data for neuro-evolution analysis (need at least 30 days)")
        else:
            st.info("Enable the checkbox above to run neuro-evolution analysis.")

# Main application logic
def main():
    logger.info("=== MAIN FUNCTION STARTED ===")
    
    try:
        # Show progress in UI
        with st.spinner("Initializing application..."):
            logger.info("Initializing managers...")
            config_manager, db_manager = get_managers()
            logger.info("Managers initialized successfully")
        
        # Display connection status in sidebar
        logger.info("Displaying connection status...")
        display_connection_status(db_manager)
        
        # Initialize session state
        if 'page' not in st.session_state:
            st.session_state.page = 'home'
            logger.info("Session state initialized to 'home'")
        
        logger.info(f"Current page: {st.session_state.page}")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        
        # Only update page from radio button if we're not already on stock_detail page
        if st.session_state.page != 'stock_detail':
            page = st.sidebar.radio("Go to", ["Home", "About"], key="main_navigation")
            logger.info(f"Navigation selection: {page}")
            
            if page == "Home":
                st.session_state.page = 'home'
            elif page == "About":
                st.session_state.page = 'about'
        else:
            # Show current navigation but don't let it override stock_detail
            if st.session_state.page == 'stock_detail':
                st.sidebar.radio("Go to", ["Home", "About"], key="main_navigation", index=0, disabled=False)
                st.sidebar.info("📊 Viewing stock details")
                if st.sidebar.button("← Back to Home"):
                    st.session_state.page = 'home'
                    st.rerun()
        
        # Display appropriate page
        logger.info(f"Displaying page: {st.session_state.page}")
        
        if st.session_state.page == 'home':
            logger.info("Loading home page...")
            display_home_page(db_manager)
            logger.info("Home page loaded successfully")
        elif st.session_state.page == 'stock_detail':
            logger.info("Loading stock detail page...")
            display_stock_detail_page(db_manager)
            logger.info("Stock detail page loaded successfully")
        elif st.session_state.page == 'about':
            logger.info("Loading about page...")
            display_about_page()
            logger.info("About page loaded successfully")
            
    except Exception as e:
        logger.error(f"CRITICAL ERROR in main(): {e}", exc_info=True)
        st.error(f"Application Error: {e}")
        st.error("Check the logs for detailed error information")
        
    logger.info("=== MAIN FUNCTION COMPLETED ===")

if __name__ == "__main__":
    main() 