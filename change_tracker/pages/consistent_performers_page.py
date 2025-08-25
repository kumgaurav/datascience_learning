"""
🏆 Consistent Performers Page

Displays the top 25 most consistent performing stocks over the last 3 months.
Uses statistical models to measure consistency rather than just total returns.
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
from pages.tabs import ConsistentPerformersTab

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
def get_cached_consistent_performers_data():
    """Get cached consistent performers data to avoid repeated DB queries"""
    try:
        # Create a fresh database manager for this cached call
        db_manager = DatabaseManager()
        
        # Execute the query directly here to avoid multiple levels of function calls
        from datetime import datetime, timedelta
        
        # Calculate date range for last 3 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        logger.info(f"[CACHED CALL] Fetching consistent performers data from {start_date.date()} to {end_date.date()}")
        
        # Single database call to get all data with consistent filtering
        query = """
        SELECT p.symbol, date, close, volume
        FROM stocksinfp p
        INNER JOIN stock_change_tracker s ON s.symbol = p.symbol 
            AND s.is_active = 1 
            AND s.current_price > 5
        WHERE date >= %(start_date)s AND date <= %(end_date)s
        AND close IS NOT NULL AND close != '' AND close != '0'
        ORDER BY symbol, date
        """
        
        all_data = db_manager.execute_query(query, {
            'start_date': start_date,
            'end_date': end_date
        })
        
        if all_data.empty:
            logger.warning("[CACHED CALL] No data found for the date range")
            return pd.DataFrame()
        
        logger.info(f"[CACHED CALL] Loaded {len(all_data)} rows for consistency analysis")
        
        # Process the data using the new utility-based approach
        from pages.tabs.consistent_performers_tab import ConsistentPerformersTab
        from utils.consistency import ConsistencyCalculator
        
        temp_tab = ConsistentPerformersTab(db_manager)
        
        # Convert data types
        all_data['date'] = pd.to_datetime(all_data['date'])
        all_data['close'] = pd.to_numeric(all_data['close'], errors='coerce')
        all_data['volume'] = pd.to_numeric(all_data['volume'], errors='coerce')
        
        # Remove any rows where conversion failed
        all_data = all_data.dropna(subset=['close'])
        all_data = all_data[all_data['close'] > 0]
        
        # Calculate consistency statistics using the new utility approach
        calculator = ConsistencyCalculator()
        consistency_results = calculator.calculate_consistency_statistics(all_data, months=3)
        
        if consistency_results.empty:
            logger.warning("[CACHED CALL] No consistency results calculated")
            return pd.DataFrame()
        
        # Get top 25 consistent performers
        top_25 = consistency_results.nlargest(25, 'consistency_index')
        
        logger.info(f"[CACHED CALL] Returning {len(top_25)} consistent performers")
        return top_25
        
    except Exception as e:
        logger.error(f"Error in cached consistent performers data: {e}")
        return pd.DataFrame()

def main():
    """Main function for the Consistent Performers page"""
    logger.info("=== CONSISTENT PERFORMERS PAGE STARTED ===")
    
    # Initialize database manager
    with st.spinner("Initializing database connection..."):
        db_manager = get_database_manager()
    
    # Page header
    st.title("🎯 Top 25 Most Consistent Performing Stocks")
    st.markdown("**Statistical analysis of stocks with the most consistent performance over the last 3 months**")
    
    # Add some context and cache controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info("📊 **Consistency Model**\nBased on statistical analysis combining positive months ratio, volatility, and geometric mean returns")
    
    with col2:
        st.info("📅 **Time Period**\nLast 3 months of trading data")
    
    with col3:
        st.info("🎯 **Focus**\nStocks that performed well in ALL 3 months, not just highest total returns")
    
    with col4:
        if st.button("🔄 Clear Cache", help="Refresh data from database", key="consistent_performers_clear_cache"):
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
    with st.spinner("Loading consistent performers data..."):
        consistent_performers_tab = ConsistentPerformersTab(db_manager)
        consistent_performers_data = get_cached_consistent_performers_data()
    
    # Extract symbols directly from the analysis results - no additional DB queries needed
    if not consistent_performers_data.empty:
        # Use all symbols from analysis as available options (sorted)
        symbol_list = sorted(consistent_performers_data['symbol'].tolist())
        # Top 25 (or however many we got) as defaults
        default_symbols = consistent_performers_data['symbol'].tolist()
        
        st.sidebar.success(f"✅ Found {len(consistent_performers_data)} consistent performers")
    else:
        symbol_list = []
        default_symbols = []
        st.sidebar.warning("⚠️ No consistency data available")
    
    # Add symbol filter and chart options
    with st.sidebar:
        st.markdown("### 🎯 Symbol Filter")
        
        # Symbol multi-select
        selected_symbols = st.multiselect(
            "Select symbols to analyze:",
            options=symbol_list,
            default=default_symbols,
            help="Top 25 consistent performers pre-selected. Modify selection to focus on specific stocks."
        )
        
        # Chart columns option
        st.markdown("### 📊 Chart Layout")
        chart_columns = st.selectbox(
            "Chart columns:",
            options=[1, 2, 3, 4],
            index=0,  # Default to 1 column
            help="Choose number of columns for individual stock charts"
        )
    
    # Add methodology info in sidebar
    st.sidebar.markdown("### 📊 Methodology")
    st.sidebar.markdown("""
    **Consistency Index Components:**
    - **40%** Positive Months Ratio
    - **30%** Volatility Score (lower is better)  
    - **30%** Geometric Mean Return
    
    This rewards stocks like TMC and SEZL that performed well consistently across all months.
    """)
    

    
    # Render the consistent performers analysis
    try:
        # Pass the already calculated data to avoid re-querying
        consistent_performers_tab.render("MARKET_WIDE", pd.DataFrame(), selected_symbols, chart_columns, consistent_performers_data)
    except Exception as e:
        logger.error(f"Error rendering consistent performers: {e}")
        st.error(f"Error loading analysis: {e}")
    
    # Navigation info and home button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🏠 Back to Home", type="primary", use_container_width=True, key="consistent_performers_home_btn"):
            st.switch_page("app_refactored.py")
    
    st.info("💡 **Navigation Tip**: Use the sidebar to navigate between pages, or use the button above to return to the main dashboard.")
    
    logger.info("Consistent performers page rendered successfully")

# Run the main function
if __name__ == "__main__":
    main()
else:
    # This runs when the page is loaded by Streamlit's multipage system
    main() 