"""
Consistent Performers Tab Component - Refactored Version

Uses utility classes for clean separation of concerns.
Reduced from 2,218 lines to ~300 lines while maintaining all functionality.
"""

# Standard library
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any

# Third-party packages
import streamlit as st
import pandas as pd

# Local imports
from data import DatabaseManager
from utils.consistency import (
    ConsistencyDataFetcher,
    ConsistencyCalculator, 
    ConsistencyVisualizer,
    ConsistencyUIRenderer
)

# Custom Exception Classes
class DataFetchError(Exception):
    """Raised when data cannot be fetched from database"""
    pass

class DataValidationError(Exception):
    """Raised when data fails validation checks"""
    pass

class CalculationError(Exception):
    """Raised when statistical calculations fail"""
    pass

class ChartCreationError(Exception):
    """Raised when chart/visualization creation fails"""
    pass

# Configuration constants
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds

logger = logging.getLogger('StockApp')


class ConsistentPerformersTab:
    """
    Refactored Consistent Performers Tab using utility classes.
    
    Orchestrates data fetching, calculations, visualizations, and UI rendering
    through specialized utility classes for better separation of concerns.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize with utility class dependencies.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        
        # Initialize utility classes
        self.data_fetcher = ConsistencyDataFetcher(db_manager)
        self.calculator = ConsistencyCalculator()
        self.visualizer = ConsistencyVisualizer()
        self.ui_renderer = ConsistencyUIRenderer()
        
        logger.info("[ConsistentPerformersTab] Initialized with utility classes")
    
    @st.cache_data(ttl=DEFAULT_CACHE_TTL)
    def _get_cached_price_data_for_symbols(_self, symbols: list, months: int = 3) -> pd.DataFrame:
        """
        Get cached price data for selected symbols using data fetcher utility.
        
        Args:
            symbols: List of stock symbols
            months: Number of months to analyze
            
        Returns:
            DataFrame with price data
        """
        try:
            # Create fresh data fetcher for cached call to avoid pickling issues
            from data import DatabaseManager
            from utils.consistency import ConsistencyDataFetcher
            
            db_manager = DatabaseManager()
            data_fetcher = ConsistencyDataFetcher(db_manager)
            
            logger.info(f"[get_cached_price_data] Fetching data for {len(symbols)} symbols over {months} months")
            return data_fetcher.get_price_data_for_symbols(symbols, months)
            
        except Exception as e:
            logger.error(f"Error in cached price data fetch: {e}")
            st.error(f"Unable to fetch price data: {e}")
            return pd.DataFrame()
    
    def render(self, symbol: str, stock_data: pd.DataFrame, selected_symbols: Optional[List[str]] = None, 
               chart_columns: int = 1, pre_calculated_data: Optional[pd.DataFrame] = None) -> None:
        """
        Render the Consistent Performers tab with multiple time period sub-tabs.
        
        Args:
            symbol: Currently selected stock symbol
            stock_data: Stock price data (not used in this implementation)
            selected_symbols: Optional list of symbols to analyze
            chart_columns: Number of columns for chart layout
            pre_calculated_data: Optional pre-calculated consistency data for 3-month analysis
        """
        logger.info(f"[render] Rendering Consistent Performers tab for symbol: {symbol}")
        
        # Render UI controls using UI renderer utility
        debug_mode = self.ui_renderer.render_control_panel()
        
        # Create sub-tabs for different time periods including All-Time Performers
        tab1, tab2, tab3, tab4 = st.tabs([
            "📅 3-Month Consistent Performers", 
            "📅 2-Month Consistent Performers", 
            "📅 Current Month Performers",
            "🏆 All-Time Champions"
        ])
        
        with tab1:
            st.markdown("### 🏆 Top 25 Most Consistent Performers - Last 3 Months")
            st.caption("Long-term consistency analysis using 3-month statistical models")
            self._render_period_analysis(3, symbol, selected_symbols, chart_columns, pre_calculated_data, debug_mode)
        
        with tab2:
            st.markdown("### 🏆 Top 25 Most Consistent Performers - Last 2 Months")
            st.caption("Medium-term consistency analysis using 2-month statistical models")
            self._render_period_analysis(2, symbol, selected_symbols, chart_columns, None, debug_mode)
        
        with tab3:
            st.markdown("### 🏆 Top 25 Most Consistent Performers - Current Month")
            st.caption("Short-term consistency analysis using current month or recent trading days performance")
            
            # Add helpful info for current month tab
            if datetime.now().day <= 10:
                st.info("📅 **Note**: Since we're early in the month, this analysis may include data from recent trading days to ensure meaningful results.")
            
            self._render_period_analysis(1, symbol, selected_symbols, chart_columns, None, debug_mode)
        
        with tab4:
            st.markdown("### 🏆 All-Time Champions - Ultimate Consistency")
            st.caption("Elite stocks appearing in ALL 3 time periods - the cream of the crop!")
            self._render_all_time_performers_analysis(symbol, selected_symbols, chart_columns, pre_calculated_data, debug_mode)
    
    def _render_period_analysis(self, months: int, symbol: str, selected_symbols: Optional[List[str]] = None, 
                               chart_columns: int = 1, pre_calculated_data: Optional[pd.DataFrame] = None, 
                               debug_mode: bool = False) -> None:
        """
        Render analysis for a specific time period using utility classes.
        
        Args:
            months: Number of months to analyze (1, 2, or 3)
            symbol: Currently selected stock symbol
            selected_symbols: Optional list of symbols to analyze
            chart_columns: Number of columns for chart layout
            pre_calculated_data: Optional pre-calculated consistency data
            debug_mode: Whether to show debug information
        """
        try:
            # Get consistent performing stocks using calculator utility
            consistent_performers, raw_data = self._get_consistency_data_with_utilities(months, pre_calculated_data)
            
            if consistent_performers.empty:
                st.warning(f"No performance data available for the last {months} month(s)")
                return
            
            # Show debug information using UI renderer
            if debug_mode:
                self.ui_renderer.display_period_debug_info(consistent_performers, months, raw_data)
            
            # Display visualizations using visualizer utility
            self.visualizer.display_consistency_chart(consistent_performers, months)
            
            # Display symbol selection interface and individual charts
            self._display_symbol_selection_and_charts(consistent_performers, raw_data, months, selected_symbols, chart_columns, "period")
            
            # Display analysis components using UI renderer utility
            self.ui_renderer.display_consistency_table(consistent_performers, months)
            self.ui_renderer.display_insights(consistent_performers, symbol, months)
            
        except (DataFetchError, DataValidationError, CalculationError) as e:
            logger.error(f"Error in {months}-month analysis: {e}")
            st.error(f"Unable to complete {months}-month analysis: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in {months}-month analysis: {e}")
            st.error(f"An unexpected error occurred during {months}-month analysis. Please try again.")
    
    def _render_all_time_performers_analysis(self, symbol: str, selected_symbols: Optional[List[str]] = None, 
                                           chart_columns: int = 1, pre_calculated_data: Optional[pd.DataFrame] = None, 
                                           debug_mode: bool = False) -> None:
        """
        Render All-Time Performers analysis using utility classes.
        
        Args:
            symbol: Currently selected stock symbol
            selected_symbols: Optional list of symbols to analyze
            chart_columns: Number of columns for chart layout
            pre_calculated_data: Optional pre-calculated consistency data for 3-month analysis
            debug_mode: Whether to show debug information
        """
        try:
            logger.info("[_render_all_time_performers_analysis] Starting All-Time Performers analysis")
            
            # === STEP 1: GET DATA FOR ALL 3 PERIODS ===
            st.info("🔄 Analyzing consistency across all time periods... This may take a moment.")
            
            # Get data for all periods using utility classes
            consistent_1m, _ = self._get_consistency_data_with_utilities(1, None)
            consistent_2m, _ = self._get_consistency_data_with_utilities(2, None)
            consistent_3m, raw_data = self._get_consistency_data_with_utilities(3, pre_calculated_data)
            
            # Check if we have data for all periods
            if any(df.empty for df in [consistent_1m, consistent_2m, consistent_3m]):
                self.ui_renderer.display_insufficient_data_warning(consistent_1m, consistent_2m, consistent_3m)
                return
            
            # === STEP 2: FIND ALL-TIME PERFORMERS ===
            all_time_performers = self.calculator.find_all_time_performers(consistent_1m, consistent_2m, consistent_3m)
            
            if all_time_performers.empty:
                self.ui_renderer.display_no_all_time_performers_found(consistent_1m, consistent_2m, consistent_3m)
                return
            
            # === STEP 3: SUCCESS AND DEBUG ===
            st.success(f"🎉 Found {len(all_time_performers)} All-Time Champions!")
            st.write(f"**These {len(all_time_performers)} stocks appear in ALL 3 time periods and represent the most consistent performers.**")
            
            if debug_mode:
                self.ui_renderer.display_all_time_debug_info(all_time_performers, consistent_1m, consistent_2m, consistent_3m)
            
            # === STEP 4: VISUALIZATIONS ===
            self.visualizer.display_consistency_chart(all_time_performers, months=0)  # 0 indicates all-time
            
            # Display symbol selection interface and individual charts
            self._display_symbol_selection_and_charts(all_time_performers, raw_data, 3, selected_symbols, chart_columns, "alltime")
            
            # === STEP 5: UI COMPONENTS ===
            self.ui_renderer.display_all_time_performers_table(all_time_performers)
            self.ui_renderer.display_all_time_insights(all_time_performers, symbol)
            
            # === STEP 6: CROSS-PERIOD COMPARISON ===
            self._display_cross_period_comparison(all_time_performers, consistent_1m, consistent_2m, consistent_3m)
            
            logger.info(f"[_render_all_time_performers_analysis] Successfully completed for {len(all_time_performers)} stocks")
            
        except Exception as e:
            logger.error(f"Error in all-time performers analysis: {e}")
            st.error(f"An error occurred during all-time performers analysis: {e}")
    
    def _get_consistency_data_with_utilities(self, months: int, pre_calculated_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Get consistency data using utility classes.
        
        Args:
            months: Number of months to analyze
            pre_calculated_data: Optional pre-calculated data for 3-month analysis
            
        Returns:
            Tuple of (consistency_data, raw_data)
        """
        try:
            # Determine data source
            use_precalculated = (months == 3 and pre_calculated_data is not None and not pre_calculated_data.empty)
            
            if use_precalculated:
                logger.info(f"[get_consistency_data] Using pre-calculated data for {months} months")
                # Process pre-calculated data using calculator utility
                consistent_performers = self.calculator.process_precalculated_data(pre_calculated_data)
                raw_data = None
            else:
                logger.info(f"[get_consistency_data] Fetching fresh data for {months} months")
                # Get fresh data using data fetcher and calculator utilities
                all_data = self.data_fetcher.get_all_stock_data(months)
                
                if all_data.empty:
                    return pd.DataFrame(), pd.DataFrame()
                
                # Calculate consistency using calculator utility
                consistent_performers = self.calculator.calculate_consistency_statistics(all_data, months)
                raw_data = all_data
            
            return consistent_performers, raw_data
            
        except Exception as e:
            logger.error(f"Error getting consistency data with utilities: {e}")
            raise DataFetchError(f"Failed to get consistency data for {months} months: {e}") from e
    

    
    def _display_individual_charts_with_utilities(self, selected_symbols: List[str], chart_columns: int, 
                                                 raw_data: Optional[pd.DataFrame], consistency_data: pd.DataFrame, 
                                                 months: int) -> None:
        """Display individual stock charts using visualizer utility."""
        try:
            if raw_data is not None and not raw_data.empty:
                # Use raw data from main analysis
                self.visualizer.create_individual_price_charts(raw_data, selected_symbols, chart_columns, months, consistency_data)
            else:
                # Fallback: fetch data specifically for charts
                chart_data = self._get_cached_price_data_for_symbols(selected_symbols, months=months)
                if not chart_data.empty:
                    self.visualizer.create_individual_price_charts(chart_data, selected_symbols, chart_columns, months, consistency_data)
                else:
                    st.warning("Unable to fetch data for individual charts")
                    
        except ChartCreationError as e:
            logger.error(f"Chart creation failed: {e}")
            st.error("Unable to create individual stock charts. Please try with fewer symbols.")
        except Exception as e:
            logger.error(f"Unexpected error in individual charts: {e}")
            st.error("An unexpected error occurred while creating charts.")
    
    def _display_cross_period_comparison(self, all_time_performers: pd.DataFrame, consistent_1m: pd.DataFrame, 
                                       consistent_2m: pd.DataFrame, consistent_3m: pd.DataFrame) -> None:
        """Display cross-period performance comparison using UI renderer."""
        st.markdown("---")
        st.subheader("📊 Cross-Period Performance Comparison")
        
        if all_time_performers.empty:
            return
        
        # Calculate rankings using calculator utility
        comparison_data = self.calculator.calculate_cross_period_rankings(
            all_time_performers, consistent_1m, consistent_2m, consistent_3m
        )
        
        # Display using UI renderer utility
        self.ui_renderer.display_cross_period_comparison_table(comparison_data)
    
    def _display_symbol_selection_and_charts(self, consistency_data: pd.DataFrame, raw_data: Optional[pd.DataFrame], 
                                           months: int, external_selected_symbols: Optional[List[str]] = None, 
                                           chart_columns: int = 1, analysis_type: str = "period") -> None:
        """
        Display symbol selection interface and individual charts.
        
        Args:
            consistency_data: Consistency data for symbol selection
            raw_data: Raw stock data for charts
            months: Number of months analyzed
            external_selected_symbols: Externally provided selected symbols (may not match available data)
            chart_columns: Number of chart columns
            analysis_type: Type of analysis ("period" or "alltime") to make keys unique
        """
        if consistency_data.empty:
            return
        
        st.markdown("---")
        st.subheader("📈 Individual Stock Analysis")
        
        # Get available symbols from consistency data
        available_symbols = consistency_data.nlargest(25, 'consistency_index')['symbol'].tolist()
        
        # Create symbol selection interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Check if external symbols are available in our data
            default_symbols = []
            if external_selected_symbols:
                # Use external symbols if they exist in our data
                matching_symbols = [sym for sym in external_selected_symbols if sym in available_symbols]
                if matching_symbols:
                    default_symbols = matching_symbols
                    st.info(f"📊 Using {len(matching_symbols)} symbols from your selection that appear in consistency data")
                else:
                    st.info("📊 Your selected symbols don't appear in consistency data. Choose from top performers below.")
            
            # If no valid external symbols, use top performers as default
            if not default_symbols:
                default_symbols = available_symbols[:25]  # Top 25 by consistency index
            
            # Multi-select for symbols
            selected_symbols = st.multiselect(
                "🎯 Select Consistent Performers to Analyze:",
                options=available_symbols,
                default=default_symbols,
                help="Select up to 25 stocks to display detailed price charts. Stocks are ordered by Consistency Index.",
                key=f"consistent_performers_symbols_{analysis_type}_{months}_{len(available_symbols)}"
            )
        
        with col2:
            # Chart configuration
            chart_columns = st.selectbox(
                "Chart Columns:",
                options=[1, 2, 3, 4],
                index=0,  # Default to 1 column
                help="Number of columns in the chart grid",
                key=f"consistent_performers_chart_columns_{analysis_type}_{months}_{len(available_symbols)}"
            )
        
        # Display charts if symbols are selected
        if selected_symbols:
            # Limit to 25 symbols for performance (consistent with top performers count)
            if len(selected_symbols) > 25:
                st.warning("⚠️ Please select maximum 25 symbols for better performance")
                selected_symbols = selected_symbols[:25]
            
            # st.info(f"📊 Showing analysis for {len(selected_symbols)} selected symbols")
            
            # Display individual charts
            self._display_individual_charts_with_utilities(
                selected_symbols, chart_columns, raw_data, consistency_data, months
            )
        else:
            st.info("👆 Select one or more symbols above to view detailed price charts")


# Backward compatibility wrapper
def ConsistentPerformersTabClass(db_manager):
    """Backward compatibility wrapper for the refactored class."""
    return ConsistentPerformersTab(db_manager) 