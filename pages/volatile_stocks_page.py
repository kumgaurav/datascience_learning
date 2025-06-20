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

# Page configuration
st.set_page_config(
    page_title="Volatile Stocks",
    page_icon="⚡",
    layout="wide"
)

@st.cache_resource
def get_database_manager():
    """Initialize and cache the database manager"""
    try:
        return DatabaseManager()
    except Exception as e:
        st.error(f"Failed to initialize database: {e}")
        st.stop()

def main():
    """Main function for the Volatile Stocks page"""
    logger.info("=== VOLATILE STOCKS PAGE STARTED ===")
    
    # Initialize database manager
    with st.spinner("Initializing database connection..."):
        db_manager = get_database_manager()
    
    # Page header
    st.title("⚡ Top 25 Most Volatile Stocks")
    st.markdown("**Analysis of the most volatile stocks with positive returns over the last 4 weeks**")
    
    # Add some context
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📊 **Data Source**\nBased on historical price data with volatility calculations")
    
    with col2:
        st.info("📅 **Time Period**\nLast 4 weeks (28 days) of trading data")
    
    with col3:
        st.info("🎯 **Criteria**\nHigh volatility stocks with positive returns only")
    
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
    
    # Render the volatile stocks analysis
    try:
        volatile_stocks_tab = VolatileStocksTab(db_manager)
        # We pass a dummy symbol since this analysis is market-wide
        volatile_stocks_tab.render("MARKET_WIDE", pd.DataFrame())
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