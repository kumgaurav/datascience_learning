"""
⚡ Volatile Stocks Page

Displays the top 25 most volatile stocks with positive returns over the last 4 weeks.
This is a standalone Streamlit page that gets automatically detected.
"""

import streamlit as st
import pandas as pd
import logging
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger_config import setup_logger
from data import DatabaseManager
from pages.tabs import VolatileStocksTab

# Setup logging
setup_logger()
logger = logging.getLogger('StockApp')

# Note: Page configuration is handled by the main app (app_refactored.py)

@st.cache_resource
def get_database_manager():
    """Initialize and cache the database manager"""
    try:
        return DatabaseManager()
    except Exception as e:
        st.error(f"Failed to initialize database: {e}")
        st.stop()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_volatile_stocks_data():
    """Get cached volatile stocks data to avoid repeated DB queries"""
    try:
        # Create a fresh database manager for this cached call
        db_manager = DatabaseManager()
        logger.info("[CACHED CALL] Fetching volatile stocks data")
        
        from pages.tabs import VolatileStocksTab
        volatile_stocks_tab = VolatileStocksTab(db_manager)
        result = volatile_stocks_tab._get_volatile_stocks()
        
        logger.info(f"[CACHED CALL] Returning {len(result)} volatile stocks")
        return result
    except Exception as e:
        logger.error(f"Error getting cached volatile stocks data: {e}")
        return pd.DataFrame()

def main():
    """Main function for the Volatile Stocks page"""
    logger.info("=== VOLATILE STOCKS PAGE STARTED ===")
    
    # Initialize database manager
    with st.spinner("Initializing database connection..."):
        db_manager = get_database_manager()
    
    # Page header
    st.title("⚡ Top 25 Most Volatile Stocks")
    st.markdown("**Analysis of the most volatile stocks with positive returns over the last 4 weeks**")
    
    # Add some context and cache controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info("📊 **Data Source**\nBased on historical price data with volatility calculations")
    
    with col2:
        st.info("📅 **Time Period**\nLast 4 weeks (28 days) of trading data")
    
    with col3:
        st.info("🎯 **Criteria**\nHigh volatility stocks with positive returns only")
    
    with col4:
        if st.button("🔄 Clear Cache", help="Refresh data from database", key="volatile_stocks_clear_cache"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Display connection status
    try:
        status = db_manager.get_connection_status()
        if status['status'] == 'Connected':
            st.sidebar.success(f"🟢 Database Connected")
        else:
            st.sidebar.error(f"🔴 Database Error: {status.get('error', 'Unknown error')}")
    except Exception as e:
        st.sidebar.error("🔴 Connection Status Unknown")
    
    # First, get the actual analysis data - this contains all we need (cached)
    with st.spinner("Loading volatile stocks data..."):
        volatile_stocks_tab = VolatileStocksTab(db_manager)
        volatile_stocks_data = get_cached_volatile_stocks_data()
    
    # Extract symbols directly from the analysis results - no additional DB queries needed
    if not volatile_stocks_data.empty:
        # Use all symbols from analysis as available options (sorted)
        symbol_list = sorted(volatile_stocks_data['symbol'].tolist())
        # Top 25 (or however many we got) as defaults
        default_symbols = volatile_stocks_data['symbol'].tolist()
        
        st.sidebar.success(f"✅ Found {len(volatile_stocks_data)} volatile stocks")
    else:
        symbol_list = []
        default_symbols = []
        st.sidebar.warning("⚠️ No volatility data available")
    
    # Add symbol filter and chart options
    with st.sidebar:
        st.markdown("### 🎯 Symbol Filter")
        
        # Symbol multi-select
        selected_symbols = st.multiselect(
            "Select symbols to analyze:",
            options=symbol_list,
            default=default_symbols,
            help="Top 25 volatile stocks pre-selected. Modify selection to focus on specific stocks."
        )
        
        # Chart columns option
        st.markdown("### 📊 Chart Layout")
        chart_columns = st.selectbox(
            "Chart columns:",
            options=[1, 2, 3, 4],
            index=0,  # Default to 1 column
            help="Choose number of columns for individual stock charts"
        )
    
    # Render the volatile stocks analysis
    try:
        # Pass the already calculated data to avoid re-querying
        volatile_stocks_tab.render("MARKET_WIDE", pd.DataFrame(), selected_symbols, chart_columns, volatile_stocks_data)
    except Exception as e:
        logger.error(f"Error rendering volatile stocks: {e}")
        st.error(f"Error loading analysis: {e}")
    
    # Navigation info and home button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🏠 Back to Home", type="primary", use_container_width=True, key="volatile_stocks_home_btn"):
            st.switch_page("app_refactored.py")
    
    st.info("💡 **Navigation Tip**: Use the sidebar to navigate between pages, or use the button above to return to the main dashboard.")
    
    logger.info("Volatile stocks page rendered successfully")

# Run the main function
if __name__ == "__main__":
    main()
else:
    # This runs when the page is loaded by Streamlit's multipage system
    main() 