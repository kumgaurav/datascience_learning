"""
Stock Detail Page Component

Handles the stock detail page with multiple analysis tabs.
"""

import streamlit as st
import pandas as pd
import logging
from typing import Optional
from data import DatabaseManager
from pages.tabs import (
    RSITab, RSIAnalysisTab, TradingSignalsTab,
    PriceMABBTab, RSIStochTab, MACDTab, VolumeOBVTab, ATRTab,
    EarningsStocksTab
)
from utils.helpers import format_currency, format_percentage

logger = logging.getLogger('StockApp')


class StockDetailPage:
    """
    Stock detail page component with multiple analysis tabs
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize stock detail page
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.rsi_tab = RSITab(db_manager)
        self.rsi_analysis_tab = RSIAnalysisTab(db_manager)
        self.trading_signals_tab = TradingSignalsTab(db_manager)
        self.price_ma_bb_tab = PriceMABBTab(db_manager)
        self.rsi_stoch_tab = RSIStochTab(db_manager)
        self.macd_tab = MACDTab(db_manager)
        self.volume_obv_tab = VolumeOBVTab(db_manager)
        self.atr_tab = ATRTab(db_manager)
        self.earnings_tab = EarningsStocksTab(db_manager)
    
    def render(self, symbol: str):
        """
        Render the stock detail page
        
        Args:
            symbol: Stock symbol to analyze
        """
        logger.info(f"=== STOCK DETAIL PAGE STARTED for {symbol} ===")
        
        # Load stock data
        stock_data = self._load_stock_data(symbol)
        stock_info = self._load_stock_info(symbol)
        
        if stock_data.empty:
            st.error(f"No data found for symbol: {symbol}")
            self._render_back_button()
            return
        
        # Display stock header
        self._render_stock_header(symbol, stock_info, stock_data)
        
        # Display tabs
        self._render_tabs(symbol, stock_data, stock_info)
        
        # Back button
        self._render_back_button()
        
        logger.info(f"Stock detail page loaded successfully for {symbol}")
    
    def _load_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Load historical stock data
        
        Args:
            symbol: Stock symbol
        
        Returns:
            DataFrame with stock data
        """
        try:
            logger.info(f"Loading stock data for {symbol}")
            stock_data = self.db_manager.get_stock_data(symbol, limit=1000)
            logger.info(f"Loaded {len(stock_data)} rows of data for {symbol}")
            return stock_data
        except Exception as e:
            logger.error(f"Error loading stock data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _load_stock_info(self, symbol: str) -> dict:
        """
        Load basic stock information
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with stock info
        """
        try:
            logger.info(f"Loading stock info for {symbol}")
            stock_info = self.db_manager.get_stock_info(symbol)
            return stock_info
        except Exception as e:
            logger.error(f"Error loading stock info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def _render_stock_header(self, symbol: str, stock_info: dict, stock_data: pd.DataFrame):
        """
        Render stock information header
        
        Args:
            symbol: Stock symbol
            stock_info: Basic stock information
            stock_data: Historical stock data
        """
        st.title(f"📊 {symbol} - Stock Analysis")
        
        # Create header columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = stock_info.get('current_price', 0)
            st.metric(
                label="Current Price",
                value=format_currency(current_price)
            )
        
        with col2:
            change_percent = stock_info.get('change_in_percent', 0)
            # Safely convert to float for delta calculation
            try:
                change_float = float(change_percent) if change_percent else 0
                delta_str = f"{change_float:.2f}%" if change_float != 0 else None
            except (ValueError, TypeError):
                change_float = 0
                delta_str = None
            
            st.metric(
                label="Change %",
                value=format_percentage(change_percent),
                delta=delta_str
            )
        
        with col3:
            if not stock_data.empty:
                latest_volume = stock_data.iloc[-1].get('volume', 0)
                try:
                    volume_float = float(latest_volume) if latest_volume else 0
                    volume_str = f"{volume_float:,.0f}" if volume_float > 0 else "N/A"
                except (ValueError, TypeError):
                    volume_str = "N/A"
                st.metric(
                    label="Latest Volume",
                    value=volume_str
                )
            else:
                st.metric(label="Latest Volume", value="N/A")
        
        with col4:
            earnings_date = stock_info.get('earnings_date')
            if earnings_date and not pd.isna(earnings_date):
                earnings_str = pd.to_datetime(earnings_date).strftime('%Y-%m-%d')
                is_soon = self.db_manager.is_earnings_within_weeks(earnings_date, 3)
                st.metric(
                    label="Earnings Date",
                    value=earnings_str,
                    delta="📅 Soon" if is_soon else None
                )
            else:
                st.metric(label="Earnings Date", value="N/A")
        
        # Additional info
        st.markdown("---")
        
        if not stock_data.empty:
            data_range = f"{stock_data.iloc[0]['date']} to {stock_data.iloc[-1]['date']}"
            st.caption(f"📈 Data Range: {data_range} | 📊 Total Records: {len(stock_data)}")
    
    def _render_tabs(self, symbol: str, stock_data: pd.DataFrame, stock_info: dict):
        """
        Render analysis tabs
        
        Args:
            symbol: Stock symbol
            stock_data: Historical stock data
            stock_info: Basic stock information
        """
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📈 14-Day RSI",
            "🔍 RSI > 75 Analysis", 
            "🤖 AI Trading Signals",
            "📊 Price & MAs & BBs",
            "⚡ RSI & Stochastic",
            "📈 MACD",
            "📊 Volume & OBV",
            "🎯 ATR",
            "📅 Earnings Calendar"
        ])
        
        # Render tab contents
        with tab1:
            self.rsi_tab.render(symbol, stock_data)
        
        with tab2:
            self.rsi_analysis_tab.render(symbol, stock_data)
        
        with tab3:
            self.trading_signals_tab.render(symbol, stock_data)
        
        with tab4:
            self.price_ma_bb_tab.render(symbol, stock_data)
        
        with tab5:
            self.rsi_stoch_tab.render(symbol, stock_data)
        
        with tab6:
            self.macd_tab.render(symbol, stock_data)
        
        with tab7:
            self.volume_obv_tab.render(symbol, stock_data)
        
        with tab8:
            self.atr_tab.render(symbol, stock_data)
        
        with tab9:
            self.earnings_tab.render(symbol, stock_data)
    
    def _render_back_button(self):
        """Render back to home button"""
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🏠 Back to Home", key="back_to_home", use_container_width=True):
                st.session_state['page'] = 'home'
                st.session_state.pop('selected_symbol', None)
                st.rerun() 