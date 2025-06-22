"""
Home Page Component

Handles the main dashboard display with stock change tracker table.
"""

import streamlit as st
import pandas as pd
import logging
from typing import Optional
from data import DatabaseManager
from utils.helpers import format_currency, format_percentage

logger = logging.getLogger('StockApp')


class HomePage:
    """
    Home page component for displaying stock change tracker
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize home page
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
    
    def render(self):
        """Render the home page"""
        logger.info("=== DISPLAY HOME PAGE STARTED ===")
        
        st.title("📈 Stock Analysis Dashboard")
        st.markdown("---")
        
        # Navigation buttons and cache control
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("🏆 Top 25 Performers", type="secondary", use_container_width=True, key="home_top_performers"):
                st.switch_page("pages/top_performers_page.py")
        
        with col2:
            if st.button("⚡ Top 25 Volatile Stocks", type="secondary", use_container_width=True, key="home_volatile_stocks"):
                st.switch_page("pages/volatile_stocks_page.py")
        
        with col3:
            if st.button("📅 Earnings Calendar", type="secondary", use_container_width=True, key="home_earnings"):
                st.switch_page("pages/earnings_page.py")
        
        with col4:
            if st.button("🔄 Clear Cache", help="Refresh all cached data", key="home_clear_cache"):
                st.cache_data.clear()
                st.rerun()
        
        st.markdown("---")
        
        # Search and pagination controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_term = st.text_input("🔍 Search Symbol", 
                                      value=st.session_state.get('search_term', ''),
                                      key='search_input')
        
        with col2:
            page_size = st.selectbox("📄 Rows per page", [25, 50, 100], 
                                   index=1, key='page_size_select')
        
        with col3:
            page = st.number_input("📖 Page", min_value=1, value=1, 
                                 key='page_input')
        
        # Update session state
        st.session_state['search_term'] = search_term
        
        # Load and display data
        self._display_stock_table(page, page_size, search_term)
        
        logger.info("Home page loaded successfully")
    
    def _display_stock_table(self, page: int, page_size: int, search: str):
        """
        Display the stock change tracker table
        
        Args:
            page: Current page number
            page_size: Number of rows per page
            search: Search term
        """
        logger.info(f"Loading stock change tracker data... Page: {page}, Search: '{search}'")
        
        try:
            # Load data from database (cached)
            df = self.db_manager.get_stock_change_tracker(page, page_size, search)
            
            if df.empty:
                st.warning("No data found matching your criteria.")
                return
            
            logger.info(f"Loaded {len(df)} rows from stock_change_tracker")
            
            # Process data for display
            display_df = self._prepare_display_data(df)
            
            # Display table
            st.subheader(f"📊 Stock Change Tracker (Page {page})")
            
            # Create interactive table
            self._render_interactive_table(display_df)
            
        except Exception as e:
            logger.error(f"Error loading stock data: {e}")
            st.error(f"Error loading data: {e}")
    
    def _prepare_display_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for display with formatting
        
        Args:
            df: Raw data from database
        
        Returns:
            Formatted DataFrame for display
        """
        display_df = df.copy()
        
        # Format numeric columns
        display_df['Change %'] = display_df['change_in_percent'].apply(format_percentage)
        display_df['Current Price'] = display_df['current_price'].apply(format_currency)
        display_df['Initial Price'] = display_df['price_when_added'].apply(format_currency)
        
        # Format earnings date
        display_df['Earnings Date'] = pd.to_datetime(display_df['earnings_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        display_df['Earnings Date'] = display_df['Earnings Date'].fillna('N/A')
        
        # Check if earnings is within 3 weeks
        display_df['Earnings Soon'] = display_df['earnings_date'].apply(
            lambda x: self.db_manager.is_earnings_within_weeks(x, 3)
        )
        
        # Select and reorder columns for display
        display_columns = ['symbol', 'Change %', 'Current Price', 'Initial Price', 
                          'Earnings Date', 'Earnings Soon']
        
        return display_df[display_columns]
    
    def _render_interactive_table(self, df: pd.DataFrame):
        """
        Render interactive table with clickable symbols
        
        Args:
            df: Formatted DataFrame for display
        """
        # Prepare display DataFrame
        display_df = df.copy()
        
        # Remove the 'Earnings Soon' column for display
        display_df = display_df.drop('Earnings Soon', axis=1)
        
        # Display the dataframe without complex styling to avoid issues
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Change %": st.column_config.TextColumn("Change %", width="small"),
                "Current Price": st.column_config.TextColumn("Current Price", width="medium"),
                "Initial Price": st.column_config.TextColumn("Initial Price", width="medium"),
                "Earnings Date": st.column_config.TextColumn("Earnings Date", width="medium")
            }
        )
        
        # Add earnings info separately
        earnings_symbols = df[df['Earnings Soon'] == True]['symbol'].tolist()
        if earnings_symbols:
            st.success(f"🟢 Earnings within 3 weeks: {', '.join(earnings_symbols)}")
        
        # Handle symbol clicks
        self._handle_symbol_selection(df)
    

    
    def _handle_symbol_selection(self, df: pd.DataFrame):
        """
        Handle symbol selection for navigation
        
        Args:
            df: DataFrame with symbol data
        """
        # Create symbol selection dropdown as alternative to clicking
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_symbol = st.selectbox(
                "📋 Select Symbol for Detailed Analysis:",
                options=[''] + df['symbol'].tolist(),
                key='symbol_select'
            )
        
        with col2:
            if selected_symbol:
                if st.button("🔍 Analyze", key='analyze_button'):
                    st.session_state['selected_symbol'] = selected_symbol
                    st.session_state['page'] = 'stock_detail'
                    st.rerun() 