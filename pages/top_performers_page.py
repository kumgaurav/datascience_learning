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
def get_cached_best_performers_data():
    """Get cached best performers data to avoid repeated DB queries"""
    try:
        # Create a fresh database manager for this cached call
        db_manager = DatabaseManager()
        logger.info("[CACHED CALL] Fetching best performers data")
        
        from pages.tabs import BestPerformersTab
        best_performers_tab = BestPerformersTab(db_manager)
        result = best_performers_tab._get_best_performers()
        
        logger.info(f"[CACHED CALL] Returning {len(result)} best performers")
        return result
    except Exception as e:
        logger.error(f"Error getting cached best performers data: {e}")
        return pd.DataFrame()

def main():
    """Main function for the Top Performers page"""
    logger.info("=== TOP PERFORMERS PAGE STARTED ===")
    
    # Initialize database manager
    with st.spinner("Initializing database connection..."):
        db_manager = get_database_manager()
    
    # Page header
    st.title("🏆 Top 25 Best Performing Stocks")
    st.markdown("**Analysis of the best performing stocks over the last 3 months**")
    
    # Add some context and cache controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info("📊 **Data Source**\nBased on historical price data from your stocksinfp table")
    
    with col2:
        st.info("📅 **Time Period**\nLast 3 months of trading data")
    
    with col3:
        st.info("🎯 **Methodology**\nMonthly gains aggregated, only positive contributions counted")
    
    with col4:
        if st.button("🔄 Clear Cache", help="Refresh data from database", key="top_performers_clear_cache"):
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
    with st.spinner("Loading top performers data..."):
        best_performers_tab = BestPerformersTab(db_manager)
        best_performers_data = get_cached_best_performers_data()
    
    # Extract symbols directly from the analysis results - no additional DB queries needed
    if not best_performers_data.empty:
        # Use all symbols from analysis as available options (sorted)
        symbol_list = sorted(best_performers_data['symbol'].tolist())
        # Top 25 (or however many we got) as defaults
        default_symbols = best_performers_data['symbol'].tolist()
        
        st.sidebar.success(f"✅ Found {len(best_performers_data)} top performing stocks")
    else:
        symbol_list = []
        default_symbols = []
        st.sidebar.warning("⚠️ No performance data available")
    
    # Add symbol filter and chart options
    with st.sidebar:
        st.markdown("### 🎯 Symbol Filter")
        
        # Symbol multi-select
        selected_symbols = st.multiselect(
            "Select symbols to analyze:",
            options=symbol_list,
            default=default_symbols,
            help="Top 25 performers pre-selected. Modify selection to focus on specific stocks."
        )
        
        # Chart columns option
        st.markdown("### 📊 Chart Layout")
        chart_columns = st.selectbox(
            "Chart columns:",
            options=[1, 2, 3, 4],
            index=0,  # Default to 1 column
            help="Choose number of columns for individual stock charts"
        )
        
        # Data reduction sensitivity control
        st.markdown("### 🎯 Data Detail Level")
        data_reduction_sensitivity = st.selectbox(
            "Chart detail level:",
            options=[
                ("High Detail", 1.5),
                ("Medium Detail", 3.0), 
                ("Low Detail", 5.0),
                ("Minimal Detail", 8.0)
            ],
            index=1,  # Default to Medium Detail (3%)
            format_func=lambda x: x[0],
            help="Choose how much detail to show in charts:\n• High Detail: Shows more price movements (>1.5% changes)\n• Medium Detail: Shows significant movements (>3% changes)\n• Low Detail: Shows major movements only (>5% changes)\n• Minimal Detail: Shows only major trends (>8% changes)"
        )
        
        # Store the sensitivity value for use by the tab
        st.session_state['data_reduction_threshold'] = data_reduction_sensitivity[1]
        
        # Additional option for volatile stocks
        st.markdown("### ⚡ Volatile Stock Handling")
        apply_strict_filtering = st.checkbox(
            "Apply strict filtering for volatile stocks",
            value=True,
            help="When enabled, applies additional filtering for very volatile stocks (like SEZL, CRWV) to ensure charts remain readable. This enforces a hard cap of 15 data points per stock."
        )
        
        # Store the strict filtering preference
        st.session_state['apply_strict_volatile_filtering'] = apply_strict_filtering
    
    # Render the best performers analysis
    try:
        # Pass the already calculated data to avoid re-querying
        best_performers_tab.render("MARKET_WIDE", pd.DataFrame(), selected_symbols, chart_columns, best_performers_data)
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