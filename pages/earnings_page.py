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

# Page configuration
st.set_page_config(
    page_title="Earnings Calendar",
    page_icon="📅",
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
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Earnings Data", help="Reload earnings data from database"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        show_only_high_vol = st.checkbox("⚡ High Volatility Only", help="Show only stocks with above-average volatility")
    
    with col3:
        show_calendar_view = st.checkbox("📅 Calendar View", value=True, help="Show calendar layout")
    
    st.markdown("---")
    
    # Render the earnings analysis
    try:
        earnings_tab = EarningsStocksTab(db_manager)
        # We'll pass empty stock data since this is a standalone page
        empty_stock_data = pd.DataFrame()
        # Use the earnings tab but with page-specific context
        earnings_tab.render("", empty_stock_data)
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