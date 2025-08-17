"""
Earnings Stocks Tab Component - Refactored Version

Uses utility classes for clean separation of concerns.
Reduced from 1,763 lines to ~300 lines while maintaining all functionality.
"""

# Standard library
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Third-party packages
import streamlit as st
import pandas as pd

# Local imports
from data import DatabaseManager
from utils.earnings import (
    EarningsDataFetcher,
    EarningsCalculator, 
    EarningsVisualizer,
    EarningsUIRenderer
)

# Custom Exception Classes
class EarningsDataFetchError(Exception):
    """Raised when earnings data cannot be fetched from database"""
    pass

class EarningsDataValidationError(Exception):
    """Raised when earnings data fails validation checks"""
    pass

class EarningsCalculationError(Exception):
    """Raised when earnings calculations fail"""
    pass

class EarningsVisualizationError(Exception):
    """Raised when earnings visualization creation fails"""
    pass

# Configuration constants
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds

logger = logging.getLogger('StockApp')


class EarningsStocksTab:
    """
    Refactored Earnings Stocks Tab using utility classes.
    
    Orchestrates earnings data fetching, calculations, visualizations, and UI rendering
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
        self.data_fetcher = EarningsDataFetcher(db_manager)
        self.calculator = EarningsCalculator()
        self.visualizer = EarningsVisualizer()
        self.ui_renderer = EarningsUIRenderer()
        
        logger.info("[EarningsStocksTab] Initialized with utility classes")
    
    def render(self, symbol: str, stock_data: pd.DataFrame, selected_symbols: Optional[List[str]] = None, 
               chart_columns: int = 1, pre_calculated_data: Optional[pd.DataFrame] = None) -> None:
        """
        Render the Earnings Stocks tab with weekly sub-tabs.
        
        Args:
            symbol: Currently selected stock symbol
            stock_data: Stock price data (not used in this implementation)
            selected_symbols: Optional list of symbols to analyze
            chart_columns: Number of columns for chart layout
            pre_calculated_data: Optional pre-calculated earnings data
        """
        logger.info(f"[render] Rendering Earnings Stocks tab for symbol: {symbol}")
        
        # Render UI controls using UI renderer utility
        debug_mode, chart_columns = self.ui_renderer.render_control_panel()
        
        try:
            # Get earnings data using data fetcher utility
            earnings_data = self._get_earnings_data_with_utilities(pre_calculated_data)
            
            if earnings_data.empty:
                self.ui_renderer.display_no_earnings_message()
                return
            
            # Filter by selected symbols if provided
            if selected_symbols:
                earnings_data = self._filter_by_selected_symbols(earnings_data, selected_symbols)
                if earnings_data.empty:
                    return
            
            # Calculate performance metrics using calculator utility
            earnings_with_metrics = self._calculate_earnings_metrics(earnings_data)
            
            # Categorize earnings by week using calculator utility
            categorized_earnings = self.calculator.categorize_earnings_by_week(earnings_with_metrics)
            
            # Calculate statistics using calculator utility
            calendar_stats = self.calculator.calculate_earnings_calendar_stats(categorized_earnings)
            
            # Display summary using UI renderer
            self.ui_renderer.display_earnings_summary(categorized_earnings, calendar_stats)
            
            # Display weekly tabs with content
            self._display_weekly_tabs(categorized_earnings, calendar_stats, selected_symbols, chart_columns, debug_mode)
            
            # Display overall visualizations
            self._display_overall_visualizations(earnings_with_metrics, categorized_earnings, calendar_stats)
            
            # Display insights
            self.ui_renderer.display_performance_insights(categorized_earnings, calendar_stats)
            
            # Display debug information if enabled
            if debug_mode:
                self.ui_renderer.display_debug_info(earnings_data, categorized_earnings, calendar_stats)
            
        except (EarningsDataFetchError, EarningsDataValidationError, EarningsCalculationError, EarningsVisualizationError) as e:
            logger.error(f"Error in earnings analysis: {e}")
            st.error(f"Unable to complete earnings analysis: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in earnings analysis: {e}")
            st.error(f"An unexpected error occurred during earnings analysis. Please try again.")
    
    def _get_earnings_data_with_utilities(self, pre_calculated_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get earnings data using utility classes.
        
        Args:
            pre_calculated_data: Optional pre-calculated earnings data
            
        Returns:
            DataFrame with earnings data
        """
        try:
            # Use pre-calculated data if available
            if pre_calculated_data is not None and not pre_calculated_data.empty:
                logger.info(f"[get_earnings_data] Using pre-calculated data with {len(pre_calculated_data)} records")
                return pre_calculated_data
            
            # Fetch fresh data using data fetcher utility
            logger.info("[get_earnings_data] Fetching fresh earnings data")
            earnings_data = self.data_fetcher.fetch_earnings_data()
            
            return earnings_data
            
        except Exception as e:
            logger.error(f"Error getting earnings data with utilities: {e}")
            raise EarningsDataFetchError(f"Failed to get earnings data: {e}") from e
    
    def _calculate_earnings_metrics(self, earnings_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance metrics for earnings data using calculator utility.
        
        Args:
            earnings_data: DataFrame with earnings data
            
        Returns:
            DataFrame with added performance metrics
        """
        try:
            if earnings_data.empty:
                return earnings_data
            
            logger.info(f"[calculate_earnings_metrics] Processing {len(earnings_data)} earnings records")
            
            # Check if volatility metrics are already calculated (from data fetcher)
            required_columns = ['volatility_score', 'total_return', 'max_daily_gain', 'max_daily_loss', 'volatility_std']
            has_volatility_metrics = all(col in earnings_data.columns for col in required_columns)
            
            if has_volatility_metrics:
                # Metrics already calculated by data fetcher, no need to recalculate
                logger.info("[calculate_earnings_metrics] Using pre-calculated volatility metrics from data fetcher")
                non_zero_returns = len(earnings_data[earnings_data['total_return'] != 0])
                non_zero_volatility = len(earnings_data[earnings_data['volatility_score'] != 0])
                logger.info(f"[calculate_earnings_metrics] Found {non_zero_returns} symbols with non-zero returns, {non_zero_volatility} with non-zero volatility")
                return earnings_data
            
            # Fallback: calculate metrics if not already present (shouldn't happen in normal flow)
            logger.warning("[calculate_earnings_metrics] Volatility metrics missing, calculating them now using earnings-specific method")
            
            # Get symbols for metric calculation
            symbols = earnings_data['symbol'].unique().tolist()
            
            # Get price data for metrics calculation using earnings-specific method (8 weeks for consistency)
            price_data = self.data_fetcher.get_earnings_price_data_for_symbols(symbols, weeks=8)
            
            if price_data.empty:
                logger.warning("[calculate_earnings_metrics] No price data available, using default values")
                # Add default metric columns
                for col in required_columns:
                    if col not in earnings_data.columns:
                        earnings_data[col] = 0.0
                return earnings_data
            
            # Calculate pre-earnings metrics using calculator utility
            metrics_data = self.calculator.calculate_pre_earnings_metrics(price_data, symbols)
            
            # Merge earnings data with metrics using calculator utility
            earnings_with_metrics = self.calculator.merge_earnings_with_metrics(earnings_data, metrics_data)
            
            return earnings_with_metrics
            
        except Exception as e:
            logger.error(f"Error calculating earnings metrics: {e}")
            # Ensure default columns exist
            required_columns = ['volatility_score', 'total_return', 'max_daily_gain', 'max_daily_loss', 'volatility_std']
            for col in required_columns:
                if col not in earnings_data.columns:
                    earnings_data[col] = 0.0
            return earnings_data
    
    def _filter_by_selected_symbols(self, earnings_data: pd.DataFrame, selected_symbols: List[str]) -> pd.DataFrame:
        """Filter earnings data by selected symbols."""
        try:
            filtered_data = earnings_data[earnings_data['symbol'].isin(selected_symbols)]
            if filtered_data.empty:
                st.warning(f"No earnings data available for selected symbols: {', '.join(selected_symbols)}")
                return pd.DataFrame()
            
            st.info(f"📊 Showing earnings analysis for {len(selected_symbols)} selected symbols")
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error filtering by selected symbols: {e}")
            return earnings_data
    
    def _display_weekly_tabs(self, categorized_earnings: Dict[str, pd.DataFrame], 
                           calendar_stats: Dict[str, Dict[str, float]], 
                           selected_symbols: Optional[List[str]], 
                           chart_columns: int, debug_mode: bool) -> None:
        """
        Display weekly earnings tabs with content.
        
        Args:
            categorized_earnings: Dictionary with categorized earnings data
            calendar_stats: Dictionary with calendar statistics
            selected_symbols: Optional list of selected symbols
            chart_columns: Number of chart columns
            debug_mode: Whether to show debug information
        """
        try:
            # Always create the 3 main tabs (matching original behavior)
            week_info = [
                ('current_week', '📅 This Week (0-7 days)', 'This Week'),
                ('next_week', '📅 Next Week (8-14 days)', 'Next Week'),
                ('week_after', '📅 Week 3 (15-21 days)', 'Week 3')
            ]
            
            # Create tab labels with counts
            tab_labels = []
            for week_key, tab_label, week_title in week_info:
                week_earnings = categorized_earnings.get(week_key, pd.DataFrame())
                count = len(week_earnings) if not week_earnings.empty else 0
                tab_labels.append(f"{tab_label} ({count})")
            
            # Create and render tabs
            tabs = st.tabs(tab_labels)
            
            for i, (week_key, _, week_title) in enumerate(week_info):
                with tabs[i]:
                    week_earnings = categorized_earnings.get(week_key, pd.DataFrame())
                    self._render_week_content(week_earnings, week_title, week_key, 
                                            calendar_stats.get(week_key, {}), 
                                            selected_symbols, chart_columns, debug_mode)
                    
        except Exception as e:
            logger.error(f"Error displaying weekly tabs: {e}")
            st.error("Failed to display weekly earnings tabs")
    
    def _render_week_content(self, week_data: pd.DataFrame, week_title: str, week_key: str,
                           week_stats: Dict[str, float], selected_symbols: Optional[List[str]], 
                           chart_columns: int, debug_mode: bool) -> None:
        """
        Render content for a specific week.
        
        Args:
            week_data: DataFrame with week's earnings data
            week_title: Title for the week
            week_key: Key for the week category
            week_stats: Statistics for the week
            selected_symbols: Optional list of selected symbols
            chart_columns: Number of chart columns
            debug_mode: Whether to show debug information
        """
        try:
            # Handle empty data case (like original implementation)
            if week_data.empty:
                st.warning(f"📅 No stocks with earnings scheduled for {week_title.lower()}")
                st.info("""
                **This could mean:**
                - No earnings announcements scheduled for this week
                - All stocks filtered out by selection criteria
                - Data may not be available for this time period
                """)
                return
            
            # Filter by selected symbols if provided
            if selected_symbols:
                filtered_week_data = week_data[week_data['symbol'].isin(selected_symbols)]
                if filtered_week_data.empty:
                    st.warning(f"No selected symbols have earnings in {week_title.lower()}")
                    return
                week_data = filtered_week_data
            
            # Display success message
            st.success(f"✅ Found {len(week_data)} stocks with earnings in {week_title.lower()}")
            
            # Display week statistics using UI renderer
            self.ui_renderer.display_week_statistics(week_stats, week_title)
            
            # Display earnings table using UI renderer
            self.ui_renderer.display_earnings_table(week_data, week_title)
            
            # Display symbol selection and individual charts
            self._display_symbol_selection_and_charts(week_data, week_title, selected_symbols, chart_columns)
            
        except Exception as e:
            logger.error(f"Error rendering week content for {week_title}: {e}")
            st.error(f"Failed to render content for {week_title}")
    
    def _display_symbol_selection_and_charts(self, week_data: pd.DataFrame, week_title: str,
                                           external_selected_symbols: Optional[List[str]], 
                                           chart_columns: int) -> None:
        """
        Display symbol selection interface and individual charts for a week.
        
        Args:
            week_data: DataFrame with week's earnings data (already filtered by week)
            week_title: Title for the week
            external_selected_symbols: Externally provided selected symbols
            chart_columns: Number of chart columns
        """
        try:
            if week_data.empty:
                return
            
            # Get available symbols from week data (already filtered to specific week)
            available_symbols = week_data['symbol'].tolist()
            logger.info(f"[_display_symbol_selection_and_charts] Processing {len(available_symbols)} symbols for {week_title}")
            
            # Sort symbols by returns (highest first) and select top 25 as default
            if 'total_return' in week_data.columns:
                logger.info(f"[_display_symbol_selection_and_charts] Sorting {len(available_symbols)} symbols by returns")
                
                # Sort week_data by total_return (highest first)
                sorted_week_data = week_data.sort_values('total_return', ascending=False)
                sorted_symbols = sorted_week_data['symbol'].tolist()
                
                # Log the sorted order for debugging
                logger.info("[_display_symbol_selection_and_charts] Top performers by return:")
                for i, (_, row) in enumerate(sorted_week_data.head(10).iterrows(), 1):
                    logger.info(f"[_display_symbol_selection_and_charts] {i}. {row['symbol']}: {row['total_return']:.2f}%")
                
                # Take top 25 performers as default (or all if less than 25)
                default_count = min(25, len(sorted_symbols))
                default_symbols = sorted_symbols[:default_count]
                
                logger.info(f"[_display_symbol_selection_and_charts] Selected top {len(default_symbols)} performers as default")
                
            else:
                logger.warning("[_display_symbol_selection_and_charts] total_return column missing, using original order")
                sorted_symbols = available_symbols
                default_count = min(10, len(available_symbols))  # Fallback to 10
                default_symbols = available_symbols[:default_count]
            
            # Display symbol selection interface using UI renderer with pre-sorted symbols
            selected_symbols, chart_columns = self.ui_renderer.display_symbol_selection_interface_with_defaults(
                available_symbols=sorted_symbols,  # Use sorted order for the dropdown
                default_symbols=default_symbols,   # Use top performers as defaults
                unique_id=week_title.replace(' ', '_').lower()
            )
            
            # Debug: Log the earnings data being passed to visualizer
            logger.info(f"[_display_symbol_selection_and_charts] About to create charts for {len(selected_symbols)} symbols")
            logger.info(f"[_display_symbol_selection_and_charts] Selected symbols: {selected_symbols}")
            logger.info(f"[_display_symbol_selection_and_charts] Week data shape: {week_data.shape}")
            
            if 'total_return' in week_data.columns:
                logger.info("[_display_symbol_selection_and_charts] Week data contains total_return column")
                # Show returns for selected symbols
                for symbol in selected_symbols[:5]:  # Show first 5 for debugging
                    symbol_data = week_data[week_data['symbol'] == symbol]
                    if not symbol_data.empty:
                        return_val = symbol_data['total_return'].iloc[0]
                        logger.info(f"[_display_symbol_selection_and_charts] {symbol} total_return: {return_val:.2f}%")
                    else:
                        logger.warning(f"[_display_symbol_selection_and_charts] {symbol} not found in week_data")
            else:
                logger.error("[_display_symbol_selection_and_charts] total_return column MISSING from week_data!")
                logger.info(f"[_display_symbol_selection_and_charts] Available columns: {list(week_data.columns)}")
            
            # Display individual charts if symbols are selected
            if selected_symbols:
                # Limit to 20 symbols for performance
                if len(selected_symbols) > 20:
                    st.warning("⚠️ Please select maximum 20 symbols for better performance")
                    selected_symbols = selected_symbols[:20]
                
                st.info(f"📊 Showing price charts for {len(selected_symbols)} selected symbols")
                
                # Get price data for selected symbols using earnings-specific method (8 weeks)
                price_data = self.data_fetcher.get_cached_earnings_price_data_for_symbols(selected_symbols, weeks=8)
                
                # Create earnings price charts using visualizer utility
                self.visualizer.create_earnings_price_charts(
                    price_data, selected_symbols, chart_columns, weeks=8, earnings_data=week_data
                )
            else:
                st.info("👆 Select one or more symbols above to view detailed price charts")
                
        except Exception as e:
            logger.error(f"Error displaying symbol selection for {week_title}: {e}")
            st.error(f"Failed to display symbol selection for {week_title}")
    
    def _display_overall_visualizations(self, earnings_data: pd.DataFrame, 
                                      categorized_earnings: Dict[str, pd.DataFrame],
                                      calendar_stats: Dict[str, Dict[str, float]]) -> None:
        """
        Display overall earnings visualizations.
        
        Args:
            earnings_data: Complete earnings data
            categorized_earnings: Categorized earnings data
            calendar_stats: Calendar statistics
        """
        try:
            st.markdown("---")
            st.markdown("## 📊 Overall Earnings Analysis")
            
            # Create columns for different visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📅 Earnings Calendar")
                # Display earnings calendar chart using visualizer utility
                self.visualizer.create_earnings_calendar_chart(categorized_earnings)
            
            with col2:
                st.markdown("### 📈 Performance Comparison")
                # Display performance comparison chart using visualizer utility
                self.visualizer.create_performance_comparison_chart(categorized_earnings, calendar_stats)
            
            # Display volatility distribution chart
            st.markdown("### 📊 Volatility Analysis")
            self.visualizer.create_volatility_distribution_chart(earnings_data)
            
        except Exception as e:
            logger.error(f"Error displaying overall visualizations: {e}")
            st.error("Failed to display overall earnings visualizations")


# Backward compatibility wrapper
def EarningsStocksTabClass(db_manager):
    """Backward compatibility wrapper for the refactored class."""
    return EarningsStocksTab(db_manager) 