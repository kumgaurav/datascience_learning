"""
📅 Earnings Calendar Page

Dedicated page for earnings calendar and analysis.
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
from pages.tabs import EarningsStocksTab

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
def get_cached_earnings_stocks_data():
    """Get cached earnings stocks data to avoid repeated DB queries"""
    try:
        # Create a fresh database manager for this cached call
        db_manager = DatabaseManager()
        logger.info("[CACHED CALL] Fetching earnings stocks data")
        
        from pages.tabs import EarningsStocksTab
        earnings_tab = EarningsStocksTab(db_manager)
        result = earnings_tab._get_earnings_stocks()
        
        logger.info(f"[CACHED CALL] Returning {len(result)} earnings stocks")
        return result
    except Exception as e:
        logger.error(f"Error getting cached earnings stocks data: {e}")
        return pd.DataFrame()

def main():
    """Main function for the Earnings Calendar page"""
    logger.info("=== EARNINGS PAGE STARTED ===")
    
    # Initialize database manager
    with st.spinner("Initializing database connection..."):
        db_manager = get_database_manager()
    
    # Page header
    st.title("📅 Earnings Calendar & Analysis")
    st.markdown("""
    **Comprehensive earnings analysis for stocks with upcoming earnings announcements**
    
    This page shows stocks with earnings in the next 4 weeks, along with:
    - 📅 Weekly earnings calendar
    - ⚡ Volatility analysis and scoring
    - 📈 Pre-earnings price charts
    - 📊 Detailed analysis table
    - 💡 Trading insights and tips
    """)
    
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
    
    # Add some earnings-specific controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Clear Cache", help="Refresh data from database", key="earnings_clear_cache"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        show_only_high_vol = st.checkbox("⚡ High Volatility Only", help="Show only stocks with above-average volatility")
    
    with col3:
        show_calendar_view = st.checkbox("📅 Calendar View", value=True, help="Show calendar layout")
    
    with col4:
        st.info("💡 **Cache Status**\nData cached for 1 hour")
    
    st.markdown("---")
    
    # First, get the actual analysis data - this contains all we need (cached)
    with st.spinner("Loading earnings stocks data..."):
        earnings_tab = EarningsStocksTab(db_manager)
        earnings_stocks_data = get_cached_earnings_stocks_data()
    
    # Extract symbols directly from the analysis results - no additional DB queries needed
    if not earnings_stocks_data.empty:
        # Use all symbols from analysis as available options (sorted)
        symbol_list = sorted(earnings_stocks_data['symbol'].tolist())
        # All earnings stocks as defaults
        default_symbols = earnings_stocks_data['symbol'].tolist()
        
        st.sidebar.success(f"✅ Found {len(earnings_stocks_data)} earnings stocks")
    else:
        symbol_list = []
        default_symbols = []
        st.sidebar.warning("⚠️ No earnings data available")
    
    # Add symbol filter and chart options
    with st.sidebar:
        st.markdown("### 🎯 Symbol Filter")
        
        # Symbol multi-select
        selected_symbols = st.multiselect(
            "Select symbols to analyze:",
            options=symbol_list,
            default=default_symbols,
            help="Earnings stocks pre-selected. Modify selection to focus on specific stocks."
        )
        
        # Chart columns option
        st.markdown("### 📊 Chart Layout")
        chart_columns = st.selectbox(
            "Chart columns:",
            options=[1, 2, 3, 4],
            index=0,  # Default to 1 column
            help="Choose number of columns for individual stock charts"
        )
    
    # Render the earnings analysis
    try:
        # Pass the already calculated data to avoid re-querying
        empty_stock_data = pd.DataFrame()
        earnings_tab.render("", empty_stock_data, selected_symbols, chart_columns, earnings_stocks_data)
    except Exception as e:
        logger.error(f"Error rendering earnings analysis: {e}")
        st.error(f"Error loading earnings analysis: {e}")
    
    # Navigation info and home button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🏠 Back to Home", type="primary", use_container_width=True, key="earnings_home_btn"):
            st.switch_page("app_refactored.py")
    
    st.info("💡 **Navigation Tip**: Use the sidebar to navigate between pages, or use the button above to return to the main dashboard.")
    
    logger.info("Earnings page rendered successfully")

# Run the main function
if __name__ == "__main__":
    main()
else:
    # This runs when the page is loaded by Streamlit's multipage system
    main() 