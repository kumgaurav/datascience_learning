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
def get_cached_volatile_stocks_data(time_period_days: int = 28):
    """Get cached volatile stocks data to avoid repeated DB queries"""
    try:
        # Create a fresh database manager for this cached call
        db_manager = DatabaseManager()
        logger.info(f"[CACHED CALL] Fetching volatile stocks data for {time_period_days} days")
        
        from pages.tabs import VolatileStocksTab
        volatile_stocks_tab = VolatileStocksTab(db_manager, time_period_days)
        result = volatile_stocks_tab._get_volatile_stocks()
        
        logger.info(f"[CACHED CALL] Returning {len(result)} volatile stocks")
        return result
    except Exception as e:
        logger.error(f"Error getting cached volatile stocks data: {e}")
        return pd.DataFrame()

def render_volatile_stocks_tab(time_period_days: int, tab_name: str):
    """Render volatile stocks analysis for a specific time period"""
    # Initialize database manager
    db_manager = get_database_manager()
    
    # Get the actual analysis data for this time period (cached)
    with st.spinner(f"Loading volatile stocks data for {tab_name.lower()}..."):
        volatile_stocks_data = get_cached_volatile_stocks_data(time_period_days)
        volatile_stocks_tab = VolatileStocksTab(db_manager, time_period_days)
    
    # Extract symbols directly from the analysis results - no additional DB queries needed
    if not volatile_stocks_data.empty:
        # Use all symbols from analysis as available options (sorted)
        symbol_list = sorted(volatile_stocks_data['symbol'].tolist())
        # Top 25 (or however many we got) as defaults
        default_symbols = volatile_stocks_data['symbol'].tolist()
    else:
        symbol_list = []
        default_symbols = []
        st.warning(f"⚠️ No volatility data available for {tab_name.lower()}")
        return
    
    # Add symbol filter and chart options in columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Symbol multi-select for individual charts only
        selected_symbols = st.multiselect(
            f"Select symbols for individual chart analysis:",
            options=symbol_list,
            default=[],  # Empty by default - user chooses which ones to analyze individually
            help=f"Choose specific stocks from the top 25 volatile stocks to see detailed individual charts. Leave empty to see only the main analysis.",
            key=f"symbols_{time_period_days}"
        )
    
    with col2:
        # Chart columns option
        chart_columns = st.selectbox(
            "Chart columns:",
            options=[1, 2, 3, 4],
            index=0,  # Default to 1 column (1 chart per row)
            help="Choose number of columns for individual stock charts (1 = one chart per row)",
            key=f"chart_cols_{time_period_days}"
        )
    
    # Render the volatile stocks analysis
    try:
        # Pass the already calculated data to avoid re-querying
        # Don't pass selected_symbols to main render - this would filter main analysis to only selected stocks
        # The main analysis should show all 25 stocks, selected_symbols is only for individual charts
        volatile_stocks_tab.render("MARKET_WIDE", pd.DataFrame(), None, chart_columns, volatile_stocks_data)
        
        # If symbols are selected, show individual analysis
        if selected_symbols:
            st.markdown("---")
            st.markdown("### 🔍 Individual Stock Analysis")
            st.info(f"📊 Showing detailed charts for {len(selected_symbols)} selected symbols")
            volatile_stocks_tab._display_individual_volatile_charts(selected_symbols, chart_columns, volatile_stocks_data)
            
    except Exception as e:
        logger.error(f"Error rendering volatile stocks for {tab_name}: {e}")
        st.error(f"Error loading analysis: {e}")

def render_all_time_volatile_tab():
    """Render the All Time volatile stocks analysis"""
    # Initialize database manager
    db_manager = get_database_manager()
    
    # Create volatile stocks tab instance
    volatile_stocks_tab = VolatileStocksTab(db_manager, 28)  # Use 28 days as base period
    
    # Add symbol filter and chart options in columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Note: We'll populate the symbol list after getting the all-time data
        # For now, create a placeholder that will be updated
        selected_symbols = st.multiselect(
            f"Select symbols for individual chart analysis:",
            options=[],  # Will be populated after data is fetched
            default=[],  # Empty by default
            help=f"Choose specific stocks from the all-time performers to see detailed individual charts. Leave empty to see only the main analysis.",
            key=f"symbols_all_time"
        )
    
    with col2:
        # Chart columns option
        chart_columns = st.selectbox(
            "Chart columns:",
            options=[1, 2, 3, 4],
            index=0,  # Default to 1 column
            help="Choose number of columns for individual stock charts (1 = one chart per row)",
            key=f"chart_cols_all_time"
        )
    
    # Render the all time volatile stocks analysis
    try:
        with st.spinner("Loading all time volatile performers..."):
            # Get all time performers data
            all_time_data = volatile_stocks_tab._get_all_time_performers()
            
            if not all_time_data.empty:
                # Update the symbol list for the multiselect
                symbol_list = sorted(all_time_data['symbol'].tolist())
                
                # Re-render the multiselect with the actual symbols
                # Note: This is a workaround for Streamlit's limitation
                # In a real app, you might want to use session state to manage this
                if symbol_list:
                    with col1:
                        selected_symbols = st.multiselect(
                            f"Select symbols for individual chart analysis:",
                            options=symbol_list,
                            default=[],
                            help=f"Choose specific stocks from the {len(symbol_list)} all-time performers to see detailed individual charts.",
                            key=f"symbols_all_time_updated"
                        )
                
                # Render the all time analysis
                volatile_stocks_tab.render("MARKET_WIDE", pd.DataFrame(), selected_symbols, chart_columns, None, show_all_time=True)
                
            else:
                st.warning("⚠️ No stocks found that appear in all three time periods")
                st.info("""
                **Why might this happen?**
                - Market conditions change rapidly
                - Different stocks are volatile at different times
                - Very strict criteria (stocks must appear in ALL three periods)
                
                **Suggestion**: Check individual time period tabs to see volatile stocks for specific periods.
                """)
                
    except Exception as e:
        logger.error(f"Error rendering all time volatile stocks: {e}")
        st.error(f"Error loading all time analysis: {e}")

def main():
    """Main function for the Volatile Stocks page"""
    logger.info("=== VOLATILE STOCKS PAGE STARTED ===")
    
    # Initialize database manager
    with st.spinner("Initializing database connection..."):
        db_manager = get_database_manager()
    
    # Page header
    st.title("⚡ Top 25 Most Volatile Stocks")
    st.markdown("**Analysis of the most volatile stocks with positive returns across different time periods**")
    
    # Add some context and cache controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info("📊 **Data Source**\nBased on historical price data with volatility calculations")
    
    with col2:
        st.info("📅 **Multiple Periods**\n4 weeks, 2 weeks, current week, and all-time analysis")
    
    with col3:
        st.info("🎯 **Criteria**\nHigh volatility stocks with positive returns only")
    
    with col4:
        if st.button("🔄 Clear Cache", help="Refresh data from database", key="volatile_stocks_clear_cache"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Display connection status in sidebar
    try:
        status = db_manager.get_connection_status()
        if status['status'] == 'Connected':
            st.sidebar.success(f"🟢 Database Connected")
        else:
            st.sidebar.error(f"🔴 Database Error: {status.get('error', 'Unknown error')}")
    except Exception as e:
        st.sidebar.error("🔴 Connection Status Unknown")
    
    # Create tabs for different time periods
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Last 4 Weeks", 
        "📈 Last 2 Weeks", 
        "⚡ Current Week",
        "🏆 All Time"
    ])
    
    with tab1:
        st.markdown("### 🗓️ 4-Week Volatility Analysis")
        st.markdown("Analysis of the most volatile stocks with positive returns over the **last 4 weeks (28 days)**")
        st.markdown("---")
        render_volatile_stocks_tab(28, "Last 4 Weeks")
    
    with tab2:
        st.markdown("### 📊 2-Week Volatility Analysis") 
        st.markdown("Analysis of the most volatile stocks with positive returns over the **last 2 weeks (14 days)**")
        st.markdown("---")
        render_volatile_stocks_tab(14, "Last 2 Weeks")
    
    with tab3:
        st.markdown("### ⚡ Current Week Volatility Analysis")
        st.markdown("Analysis of the most volatile stocks with positive returns during the **current week (7 days)**")
        st.markdown("---")
        render_volatile_stocks_tab(7, "Current Week")
    
    with tab4:
        st.markdown("### 🏆 All Time Volatile Performers")
        st.markdown("**Elite stocks that consistently appear in ALL three time periods:**")
        st.markdown("- ⚡ Current Week (7 days)")
        st.markdown("- 📈 Last 2 Weeks (14 days)")
        st.markdown("- 📅 Last 4 Weeks (28 days)")
        st.markdown("---")
        render_all_time_volatile_tab()
    
    # Navigation info and home button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🏠 Back to Home", type="primary", use_container_width=True, key="volatile_stocks_home_btn"):
            st.switch_page("app_refactored.py")
    
    st.info("💡 **Navigation Tip**: Use the tabs above to switch between different time periods, or use the button above to return to the main dashboard.")
    
    logger.info("Volatile stocks page rendered successfully")

# Run the main function
if __name__ == "__main__":
    main()
else:
    # This runs when the page is loaded by Streamlit's multipage system
    main() 