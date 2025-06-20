"""
🏆 Top 25 Performers Page

Displays the top 25 performing stocks over the last 3 months.
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
from pages.tabs import BestPerformersTab

# Setup logging
setup_logger()
logger = logging.getLogger('StockApp')

# Page configuration
st.set_page_config(
    page_title="Top 25 Performers",
    page_icon="🏆",
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
    """Main function for the Top Performers page"""
    logger.info("=== TOP PERFORMERS PAGE STARTED ===")
    
    # Initialize database manager
    with st.spinner("Initializing database connection..."):
        db_manager = get_database_manager()
    
    # Page header
    st.title("🏆 Top 25 Best Performing Stocks")
    st.markdown("**Analysis of the best performing stocks over the last 3 months**")
    
    # Add some context
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📊 **Data Source**\nBased on historical price data from your stocksinfp table")
    
    with col2:
        st.info("📅 **Time Period**\nLast 3 months of trading data")
    
    with col3:
        st.info("🎯 **Methodology**\nMonthly gains aggregated, only positive contributions counted")
    
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
    
    # Render the best performers analysis
    try:
        best_performers_tab = BestPerformersTab(db_manager)
        # We pass a dummy symbol since this analysis is market-wide
        best_performers_tab.render("MARKET_WIDE", pd.DataFrame())
    except Exception as e:
        logger.error(f"Error rendering best performers: {e}")
        st.error(f"Error loading analysis: {e}")
    
    # Navigation info and home button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🏠 Back to Home", type="primary", use_container_width=True, key="top_performers_home_btn"):
            st.switch_page("app_refactored.py")
    
    st.info("💡 **Navigation Tip**: Use the sidebar to navigate between pages, or use the button above to return to the main dashboard.")
    
    logger.info("Top performers page rendered successfully")

# Run the main function
if __name__ == "__main__":
    main()
else:
    # This runs when the page is loaded by Streamlit's multipage system
    main() 