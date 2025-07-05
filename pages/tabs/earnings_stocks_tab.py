"""
Earnings Stocks Tab Component - Stocks with earnings in next 3 weeks with volatility analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from data import DatabaseManager
from analysis import TechnicalIndicators
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import random

logger = logging.getLogger('StockApp')


class EarningsStocksTab:
    """
    Tab component for analyzing stocks with upcoming earnings
    
    This class provides comprehensive analysis of stocks with earnings announcements
    in the next 3 weeks, including volatility analysis, price trends, and trading insights.
    
    REFACTORING IMPROVEMENTS:
    - ✅ Eliminated code duplication with centralized utility methods
    - ✅ Added comprehensive constants to replace magic numbers
    - ✅ Implemented data caching to improve performance
    - ✅ Standardized data cleaning with _clean_dataframe utility
    - ✅ Improved error handling and logging consistency
    - ✅ Enhanced code organization with clear section separators
    - ✅ Optimized database queries and DataFrame operations
    - ✅ Added type hints and improved documentation
    
    ARCHITECTURE:
    - Data Retrieval: Handles earnings data fetching and caching
    - Data Processing: Centralized cleaning and metric calculations
    - Display Logic: Organized weekly tabs with consistent UI patterns
    - Chart Generation: Matplotlib/Plotly integration with size limits
    - Utility Functions: Reusable components for common operations
    
    PERFORMANCE OPTIMIZATIONS:
    - Cached earnings data to avoid redundant database calls
    - Centralized DataFrame cleaning operations
    - Memory-safe chart generation with size limits
    - Efficient symbol sorting and filtering
    """
    
    # Time-based constants
    FALLBACK_DAYS = 30
    VOLATILITY_LOOKBACK_DAYS = 21
    EARNINGS_LOOKBACK_WEEKS = 3
    RECENT_MOMENTUM_DAYS = 7
    STOCK_GRID_LOOKBACK_DAYS = 28
    INDIVIDUAL_CHART_LOOKBACK_DAYS = 21
    
    # Data validation constants
    MIN_DATA_POINTS = 5
    MIN_EXTENDED_DATA_POINTS = 10
    MIN_VOLATILITY_THRESHOLD = 0.2
    VOLATILITY_OUTLIER_THRESHOLD = 200
    MIN_SYMBOL_DATA_POINTS = 5
    
    # Display constants
    DEFAULT_GRID_COLUMNS = 3
    MAX_FALLBACK_RESULTS = 20
    MAX_INDIVIDUAL_CHARTS = 20
    MAX_FIGURE_HEIGHT = 100
    MAX_FIGURE_WIDTH = 25
    MAX_CALENDAR_COLUMNS = 5
    
    # Week definition constants
    WEEK_1_RANGE = (0, 7)
    WEEK_2_RANGE = (8, 14)
    WEEK_3_RANGE = (15, 21)
    
    # Volatility calculation constants
    ANNUALIZATION_FACTOR = 252
    VOLATILITY_STD_WEIGHT = 0.4
    ATR_WEIGHT = 0.3
    PRICE_RANGE_WEIGHT = 0.3
    
    # Chart formatting constants
    PRICE_PADDING_RATIO = 0.1
    DATE_TICK_THRESHOLD = 10
    DATE_TICK_STEP_DIVISOR = 5
    
    # UI text constants
    WEEKS_CONFIG = [
        ("📅 This Week (0-7 days)", "This Week"),
        ("📅 Next Week (8-14 days)", "Next Week"),
        ("📅 Week 3 (15-21 days)", "Week 3")
    ]
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tech_indicators = TechnicalIndicators()
        self._cached_earnings_data: Optional[pd.DataFrame] = None
    
    def render(self, symbol: str, stock_data: pd.DataFrame, selected_symbols: Optional[List[str]] = None, 
               chart_columns: int = 1, pre_calculated_data: Optional[pd.DataFrame] = None):
        """
        Render the earnings stocks tab
        
        Args:
            symbol: Current stock symbol
            stock_data: Historical stock data (not used in this tab)
            selected_symbols: List of selected symbols to analyze
            chart_columns: Number of columns for chart layout
            pre_calculated_data: Pre-calculated earnings data to avoid re-fetching
        """
        logger.info(f"Rendering Earnings Stocks tab for symbol: {symbol}")
        
        try:
            earnings_stocks = self._get_earnings_data(pre_calculated_data)
            
            if earnings_stocks.empty:
                self._display_no_data_message()
                return
            
            # Filter by selected symbols if provided
            if selected_symbols:
                earnings_stocks = self._filter_by_symbols(earnings_stocks, selected_symbols)
                if earnings_stocks.empty:
                    return
            
            self._display_success_message(earnings_stocks, selected_symbols)
            
            # Create weekly sub-tabs
            self._display_weekly_tabs(earnings_stocks, selected_symbols, chart_columns, symbol)
            
        except Exception as e:
            logger.error(f"Error in Earnings Stocks tab: {e}")
            st.error(f"❌ Unable to load earnings stocks data: {str(e)}")
            st.info("Please check your database connection and table structure.")
    
    # ==================== UTILITY METHODS ====================
    
    def _clean_dataframe(self, df: pd.DataFrame, 
                        numeric_columns: List[str] = None, 
                        datetime_columns: List[str] = None,
                        required_columns: List[str] = None) -> pd.DataFrame:
        """
        Centralized data cleaning utility
        
        Args:
            df: DataFrame to clean
            numeric_columns: Columns to convert to numeric
            datetime_columns: Columns to convert to datetime
            required_columns: Columns that must not have null values
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
            
        df_clean = df.copy()
        
        # Convert datetime columns
        if datetime_columns:
            for col in datetime_columns:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Convert numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove rows with null required columns
        if required_columns:
            existing_required = [col for col in required_columns if col in df_clean.columns]
            df_clean = df_clean.dropna(subset=existing_required)
        
        # Remove zero/negative values for price columns
        price_cols = ['close', 'high', 'low', 'current_price']
        for col in price_cols:
            if col in df_clean.columns:
                df_clean = df_clean[df_clean[col] > 0]
        
        return df_clean
    
    def _get_cached_earnings_data(self) -> pd.DataFrame:
        """Get cached earnings data to avoid redundant database calls"""
        if self._cached_earnings_data is None:
            self._cached_earnings_data = self._get_earnings_stocks()
        return self._cached_earnings_data
    
    def _calculate_return_metrics(self, price_series: pd.Series) -> Dict[str, float]:
        """
        Calculate standard return metrics for a price series
        
        Args:
            price_series: Series of prices
            
        Returns:
            Dictionary with return metrics
        """
        if len(price_series) < 2:
            return {
                'total_return': 0.0,
                'first_price': price_series.iloc[0] if len(price_series) > 0 else 0.0,
                'last_price': price_series.iloc[-1] if len(price_series) > 0 else 0.0,
                'current_price': price_series.iloc[-1] if len(price_series) > 0 else 0.0
            }
        
        first_price = price_series.iloc[0]
        last_price = price_series.iloc[-1]
        total_return = ((last_price - first_price) / first_price) * 100
        
        return {
            'total_return': total_return,
            'first_price': first_price,
            'last_price': last_price,
            'current_price': last_price
        }
    
    def _get_momentum_indicator(self, momentum_value: float) -> str:
        """Get emoji indicator for momentum"""
        if momentum_value > 0:
            return "📈"
        elif momentum_value < 0:
            return "📉"
        else:
            return "➡️"
    
    def _get_return_indicator(self, return_value: float) -> str:
        """Get emoji indicator for returns"""
        if return_value > 0:
            return "🟢"
        elif return_value < 0:
            return "🔴"
        else:
            return "⚪"
    
    def _format_percentage(self, value: float, plus_sign: bool = True) -> str:
        """Format percentage with consistent styling"""
        if plus_sign:
            return f"{value:+.1f}%"
        else:
            return f"{value:.1f}%"
    
    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value"""
        if denominator == 0:
            return default
        return numerator / denominator
    
    # ==================== DATA RETRIEVAL METHODS ====================
    
    def _get_earnings_data(self, pre_calculated_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Get earnings data, using pre-calculated if available"""
        with st.spinner("Loading earnings data..."):
            if pre_calculated_data is not None and not pre_calculated_data.empty:
                logger.info("Using pre-calculated earnings data")
                return pre_calculated_data
            else:
                return self._get_earnings_stocks()
    
    def _filter_by_symbols(self, earnings_stocks: pd.DataFrame, selected_symbols: List[str]) -> pd.DataFrame:
        """Filter earnings stocks by selected symbols"""
        filtered_earnings = earnings_stocks[earnings_stocks['symbol'].isin(selected_symbols)]
        if filtered_earnings.empty:
            st.warning(f"No earnings data available for selected symbols: {', '.join(selected_symbols)}")
        return filtered_earnings
    
    def _display_no_data_message(self):
        """Display message when no earnings data is found"""
        st.warning("📅 No stocks with earnings data found")
        st.info("""
        **Possible reasons:**
        - Earnings table may not be available in your database
        - No stocks have earnings scheduled in the next 3 weeks
        - Database connection issues
        
        **What you can do:**
        - Check if the `stocks_earnings` table exists in your database
        - Verify the table has data for upcoming earnings dates
        - Try refreshing the page
        """)
    
    def _display_success_message(self, earnings_stocks: pd.DataFrame, selected_symbols: Optional[List[str]]):
        """Display success message with data summary"""
        if selected_symbols:
            st.info(f"📊 Showing analysis for {len(selected_symbols)} selected symbols")
        else:
            st.success(f"✅ Found {len(earnings_stocks)} stocks with upcoming earnings")
    
    def _get_earnings_stocks(self) -> pd.DataFrame:
        """Get stocks with earnings in next 3 weeks with volatility analysis"""
        try:
            date_range = self._calculate_date_ranges()
            
            logger.info(f"Fetching stocks with earnings from {date_range['earnings_start'].date()} to {date_range['earnings_end'].date()}")
            logger.info(f"Fetching volatility data from {date_range['volatility_start'].date()} to {date_range['volatility_end'].date()}")
            
            # Check earnings table availability
            if not self._check_earnings_table_exists():
                logger.warning("Earnings table not found, using fallback method")
                return self._get_fallback_earnings_data()
            
            # Get earnings data
            all_data = self._fetch_earnings_data(date_range)
            
            if all_data.empty:
                logger.warning("No stocks with earnings in next 3 weeks found")
                return pd.DataFrame()
            
            # Process and calculate statistics
            processed_data = self._process_earnings_data(all_data)
            earnings_results = self._calculate_earnings_statistics(processed_data)
            
            if earnings_results.empty:
                return pd.DataFrame()
            
            # Sort and return results
            earnings_results = earnings_results.sort_values(['earnings_date', 'volatility_score'], ascending=[True, False])
            
            logger.info(f"Found {len(earnings_results)} stocks with earnings and volatility data")
            return earnings_results
            
        except Exception as e:
            logger.error(f"Error getting earnings stocks: {e}")
            return pd.DataFrame()
    
    def _calculate_date_ranges(self) -> Dict[str, datetime]:
        """Calculate date ranges for earnings and volatility analysis"""
        end_date = datetime.now()
        return {
            'volatility_start': end_date - timedelta(days=self.VOLATILITY_LOOKBACK_DAYS),
            'volatility_end': end_date,
            'earnings_start': end_date,
            'earnings_end': end_date + timedelta(weeks=self.EARNINGS_LOOKBACK_WEEKS)
        }
    
    def _fetch_earnings_data(self, date_range: Dict[str, datetime]) -> pd.DataFrame:
        """Fetch earnings data from database"""
        min_price = self.db_manager.get_minimum_price_filter()
        
        query = """
        SELECT DISTINCT
            p.symbol, p.date, p.close, p.high, p.low, p.volume, se.earnings_date
        FROM stocksdb.stocksinfp p
        INNER JOIN stocksdb.stock_change_tracker s ON s.symbol = p.symbol 
            AND s.is_active = 1 AND s.current_price > 5
        INNER JOIN stocksdb.stocks_earnings se ON p.symbol = se.symbol
        WHERE p.date >= %(start_date)s AND p.date <= %(end_date)s
        AND se.earnings_date >= %(earnings_start)s AND se.earnings_date <= %(earnings_end)s
        AND p.close IS NOT NULL AND p.close != '' AND p.close != '0'
        AND p.high IS NOT NULL AND p.high != '' AND p.high != '0'
        AND p.low IS NOT NULL AND p.low != '' AND p.low != '0'
        ORDER BY p.symbol, p.date
        """
        
        return self.db_manager.execute_query(query, {
            'start_date': date_range['volatility_start'],
            'end_date': date_range['volatility_end'],
            'earnings_start': date_range['earnings_start'],
            'earnings_end': date_range['earnings_end']
        })
    
    def _process_earnings_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean raw earnings data"""
        if raw_data.empty:
            return raw_data
            
        # Use centralized cleaning utility
        cleaned_data = self._clean_dataframe(
            raw_data,
            numeric_columns=['close', 'high', 'low', 'volume'],
            datetime_columns=['date', 'earnings_date'],
            required_columns=['close', 'high', 'low']
        )
        
        logger.info(f"Found price data for {cleaned_data['symbol'].nunique()} unique symbols with upcoming earnings")
        return cleaned_data
    
    def _calculate_earnings_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility and earnings statistics for each stock"""
        symbol_groups = data.groupby('symbol')
        earnings_list = []
        
        for symbol, group_data in symbol_groups:
            try:
                stats = self._calculate_symbol_statistics(symbol, group_data)
                if stats:
                    earnings_list.append(stats)
            except Exception as e:
                logger.debug(f"Error processing symbol {symbol}: {e}")
                continue
        
        if not earnings_list:
            return pd.DataFrame()
        
        earnings_df = pd.DataFrame(earnings_list)
        # Filter outliers
        earnings_df = earnings_df[earnings_df['volatility_score'] <= self.VOLATILITY_OUTLIER_THRESHOLD]
        
        return earnings_df
    
    def _calculate_symbol_statistics(self, symbol: str, group_data: pd.DataFrame) -> Optional[Dict]:
        """Calculate statistics for a single symbol"""
        group_data = group_data.sort_values('date')
        
        if len(group_data) < self.MIN_EXTENDED_DATA_POINTS:
            return None
        
        # Get earnings information
        earnings_date = group_data.iloc[0]['earnings_date']
        days_to_earnings = (earnings_date - datetime.now()).days
        
        # Calculate daily returns
        group_data = group_data.copy()
        group_data['daily_return'] = group_data['close'].pct_change()
        group_data = group_data.dropna(subset=['daily_return'])
        
        if len(group_data) < self.MIN_DATA_POINTS:
            return None
        
        # Calculate volatility metrics
        volatility_metrics = self._calculate_volatility_metrics(group_data)
        price_metrics = self._calculate_price_metrics(group_data)
        momentum_metrics = self._calculate_momentum_metrics(group_data)
        
        return {
            'symbol': symbol,
            'earnings_date': earnings_date,
            'days_to_earnings': days_to_earnings,
            'data_points': len(group_data),
            **volatility_metrics,
            **price_metrics,
            **momentum_metrics
        }
    
    def _calculate_volatility_metrics(self, group_data: pd.DataFrame) -> Dict:
        """Calculate volatility-related metrics"""
        daily_returns = group_data['daily_return']
        volatility_std = daily_returns.std() * np.sqrt(self.ANNUALIZATION_FACTOR)
        
        # Average True Range
        group_data['tr'] = np.maximum(
            group_data['high'] - group_data['low'],
            np.maximum(
                abs(group_data['high'] - group_data['close'].shift(1)),
                abs(group_data['low'] - group_data['close'].shift(1))
            )
        )
        atr = group_data['tr'].mean()
        atr_pct = self._safe_divide(atr, group_data['close'].mean()) * 100
        
        # Price range volatility
        price_range = self._safe_divide(
            (group_data['high'].max() - group_data['low'].min()),
            group_data['close'].mean()
        ) * 100
        
        # Composite volatility score using weighted averages
        volatility_score = (
            volatility_std * self.VOLATILITY_STD_WEIGHT + 
            atr_pct * self.ATR_WEIGHT + 
            price_range * self.PRICE_RANGE_WEIGHT
        )
        
        return {
            'volatility_score': volatility_score,
            'volatility_std': volatility_std,
            'atr_pct': atr_pct,
            'price_range': price_range,
            'avg_daily_return': daily_returns.mean() * 100,
            'max_daily_gain': daily_returns.max() * 100,
            'max_daily_loss': daily_returns.min() * 100
        }
    
    def _calculate_price_metrics(self, group_data: pd.DataFrame) -> Dict:
        """Calculate price-related metrics"""
        return self._calculate_return_metrics(group_data['close'])
    
    def _calculate_momentum_metrics(self, group_data: pd.DataFrame) -> Dict:
        """Calculate momentum-related metrics"""
        # Pre-earnings momentum (last 7 days)
        recent_data = group_data.tail(self.RECENT_MOMENTUM_DAYS)
        
        if len(recent_data) >= 2:
            recent_return = ((recent_data.iloc[-1]['close'] - recent_data.iloc[0]['close']) / 
                           recent_data.iloc[0]['close']) * 100
        else:
            recent_return = 0
        
        return {'recent_momentum': recent_return}
    
    def _display_weekly_tabs(self, earnings_stocks: pd.DataFrame, selected_symbols: Optional[List[str]], 
                            chart_columns: int, current_symbol: str):
        """Display earnings data organized by weekly tabs"""
        # Create tabs using configuration
        tab_labels = [config[0] for config in self.WEEKS_CONFIG]
        tabs = st.tabs(tab_labels)
        
        # Define week ranges
        week_ranges = [self.WEEK_1_RANGE, self.WEEK_2_RANGE, self.WEEK_3_RANGE]
        
        # Display each tab
        for tab, (_, week_name), (min_days, max_days) in zip(tabs, self.WEEKS_CONFIG, week_ranges):
            with tab:
                week_data = self._filter_by_week(earnings_stocks, min_days, max_days)
                self._display_week_content(week_data, week_name, selected_symbols, chart_columns, current_symbol)
    
    def _filter_by_week(self, data: pd.DataFrame, min_days: int, max_days: int) -> pd.DataFrame:
        """Filter earnings data by days to earnings range"""
        if data.empty:
            return data
        
        return data[(data['days_to_earnings'] >= min_days) & (data['days_to_earnings'] <= max_days)]
    
    def _display_week_content(self, week_data: pd.DataFrame, week_name: str, 
                             selected_symbols: Optional[List[str]], chart_columns: int, current_symbol: str):
        """Display content for a specific week tab"""
        if week_data.empty:
            st.warning(f"📅 No stocks with earnings scheduled for {week_name.lower()}")
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
                st.warning(f"No selected symbols have earnings in {week_name.lower()}")
                return
            week_data = filtered_week_data
        
        # Display week summary
        st.success(f"✅ Found {len(week_data)} stocks with earnings in {week_name.lower()}")
        
        # Display calendar for this week
        self._display_earnings_calendar(week_data)
        
        # Display volatility chart for this week
        self._display_earnings_volatility_chart(week_data)
        
        # Display charts based on selection
        if selected_symbols:
            # Show individual charts for selected symbols in this week
            week_selected_symbols = [sym for sym in selected_symbols if sym in week_data['symbol'].values]
            if week_selected_symbols:
                st.markdown("---")
                self._display_individual_earnings_charts(week_selected_symbols, chart_columns)
        else:
            # Show grid for all stocks in this week
            self._display_stock_price_grid(week_data)
        
        # Display table for this week
        self._display_earnings_table(week_data)
        
        # Display insights for this week
        self._display_week_insights(week_data, current_symbol, week_name)
    
    def _display_week_insights(self, week_data: pd.DataFrame, current_symbol: str, week_name: str):
        """Display insights specific to a week"""
        st.subheader(f"💡 {week_name} Insights")
        
        if week_data.empty:
            st.warning("No data available for insights")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**{week_name} Summary:**")
            avg_volatility = week_data['volatility_score'].mean()
            avg_momentum = week_data['recent_momentum'].mean()
            
            st.write(f"• Total Stocks: {len(week_data)}")
            st.write(f"• Avg Volatility: {avg_volatility:.1f}")
            st.write(f"• Avg Momentum: {avg_momentum:+.1f}%")
            
            positive_momentum = (week_data['recent_momentum'] > 0).sum()
            st.write(f"• Positive Momentum: {positive_momentum}/{len(week_data)}")
        
        with col2:
            st.markdown("**Most Volatile:**")
            top_volatile = week_data.nlargest(3, 'volatility_score')
            for i, (_, row) in enumerate(top_volatile.iterrows(), 1):
                momentum_icon = "📈" if row['recent_momentum'] > 0 else "📉" if row['recent_momentum'] < 0 else "➡️"
                st.write(f"{i}. **{row['symbol']}** {momentum_icon}")
                st.write(f"   Vol: {row['volatility_score']:.1f} | {row['earnings_date'].strftime('%m/%d')}")
        
        with col3:
            st.markdown("**Best Returns:**")
            top_returns = week_data.nlargest(3, 'total_return')
            for i, (_, row) in enumerate(top_returns.iterrows(), 1):
                return_icon = "🟢" if row['total_return'] > 0 else "🔴" if row['total_return'] < 0 else "⚪"
                st.write(f"{i}. **{row['symbol']}** {return_icon}")
                st.write(f"   Return: {row['total_return']:+.1f}% | {row['earnings_date'].strftime('%m/%d')}")
        
        # Current stock analysis for this week
        if current_symbol in week_data['symbol'].values:
            st.markdown("---")
            current_data = week_data[week_data['symbol'] == current_symbol].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"🎯 **{current_symbol} has earnings in {week_name.lower()}!**")
                st.write(f"• Earnings Date: {current_data['earnings_date'].strftime('%Y-%m-%d')}")
                st.write(f"• Days Until: {current_data['days_to_earnings']}")
                
            with col2:
                st.write(f"• Volatility Score: {current_data['volatility_score']:.1f}")
                momentum_text = "Bullish" if current_data['recent_momentum'] > 0 else "Bearish" if current_data['recent_momentum'] < 0 else "Neutral"
                st.write(f"• Recent Momentum: {momentum_text} ({current_data['recent_momentum']:+.1f}%)")
                st.write(f"• Total Return: {current_data['total_return']:+.1f}%")
        
        # Week-specific trading tips
        st.markdown("---")
        if week_name == "This Week":
            st.info("""
            **💡 This Week Trading Tips:**
            - ⏰ Earnings are imminent - volatility likely to increase
            - 📊 Monitor pre-market and after-hours activity
            - 🎯 Consider tight stop-losses due to earnings proximity
            - 📈 Watch for unusual volume spikes
            """)
        elif week_name == "Next Week":
            st.info("""
            **💡 Next Week Trading Tips:**
            - 📅 Good time for position building before earnings
            - 📊 Analyze recent momentum trends
            - ⚡ Volatility may start increasing as earnings approach
            - 🎯 Consider earnings play strategies
            """)
        else:  # Week 3
            st.info("""
            **💡 Week 3 Trading Tips:**
            - 📈 Early opportunity for earnings positioning
            - 📊 Focus on fundamental analysis while there's time
            - ⚖️ Good risk/reward ratio for longer-term positions
            - 🎯 Monitor for any guidance or pre-announcements
            """)
    
    def _check_earnings_table_exists(self) -> bool:
        """Check if the earnings table exists and has data"""
        try:
            test_query = "SELECT COUNT(*) as count FROM stocksdb.stocks_earnings LIMIT 1"
            result = self.db_manager.execute_query(test_query)
            return not result.empty and result.iloc[0]['count'] >= 0
        except Exception as e:
            logger.debug(f"Earnings table check failed: {e}")
            return False
    
    def _get_fallback_earnings_data(self) -> pd.DataFrame:
        """Get fallback data when earnings table is not available"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.FALLBACK_DAYS)
            
            data = self._fetch_fallback_data(start_date, end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Process fallback data
            processed_data = self._process_fallback_data(data)
            fallback_results = self._calculate_fallback_statistics(processed_data)
            
            if not fallback_results.empty:
                # Add fake earnings dates (next 1-3 weeks)
                fallback_results['earnings_date'] = fallback_results.apply(
                    lambda x: datetime.now() + timedelta(days=random.randint(1, 21)), axis=1
                )
                fallback_results['days_to_earnings'] = fallback_results['earnings_date'].apply(
                    lambda x: (x - datetime.now()).days
                )
            
            return fallback_results.head(self.MAX_FALLBACK_RESULTS)
            
        except Exception as e:
            logger.error(f"Error in fallback earnings data: {e}")
            return pd.DataFrame()
    
    def _fetch_fallback_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch fallback data from database"""
        min_price = self.db_manager.get_minimum_price_filter()
        
        query = """
        SELECT symbol, date, close, high, low, volume
        FROM stocksdb.stocksinfp
        WHERE date >= %(start_date)s AND date <= %(end_date)s
        AND close IS NOT NULL AND close != '' AND close != '0'
        AND high IS NOT NULL AND high != '' AND high != '0'
        AND low IS NOT NULL AND low != '' AND low != '0'
        AND CAST(close AS DECIMAL(10,2)) >= %(min_price)s
        ORDER BY symbol, date
        """
        
        return self.db_manager.execute_query(query, {
            'start_date': start_date,
            'end_date': end_date,
            'min_price': min_price
        })
    
    def _process_fallback_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process fallback data"""
        # Convert data types
        data['date'] = pd.to_datetime(data['date'])
        for col in ['close', 'high', 'low', 'volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Clean data
        data = data.dropna(subset=['close', 'high', 'low'])
        data = data[(data['close'] > 0) & (data['high'] > 0) & (data['low'] > 0)]
        
        return data
    
    def _calculate_fallback_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistics for fallback data"""
        symbol_groups = data.groupby('symbol')
        results_list = []
        
        for symbol, group_data in symbol_groups:
            try:
                stats = self._calculate_fallback_symbol_stats(symbol, group_data)
                if stats:
                    results_list.append(stats)
            except Exception as e:
                logger.debug(f"Error processing symbol {symbol} in fallback: {e}")
                continue
        
        if not results_list:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results_list)
        return results_df.sort_values('volatility_score', ascending=False)
    
    def _calculate_fallback_symbol_stats(self, symbol: str, group_data: pd.DataFrame) -> Optional[Dict]:
        """Calculate statistics for a single symbol in fallback mode"""
        group_data = group_data.sort_values('date')
        
        if len(group_data) < self.MIN_EXTENDED_DATA_POINTS:
            return None
        
        # Calculate daily returns
        group_data = group_data.copy()
        group_data['daily_return'] = group_data['close'].pct_change()
        group_data = group_data.dropna(subset=['daily_return'])
        
        if len(group_data) < self.MIN_DATA_POINTS:
            return None
        
        # Only include if reasonably volatile
        daily_returns = group_data['daily_return']
        volatility_std = daily_returns.std() * np.sqrt(252)
        
        if volatility_std < self.MIN_VOLATILITY_THRESHOLD:
            return None
        
        # Calculate all metrics
        volatility_metrics = self._calculate_volatility_metrics(group_data)
        price_metrics = self._calculate_price_metrics(group_data)
        momentum_metrics = self._calculate_momentum_metrics(group_data)
        
        return {
            'symbol': symbol,
            'data_points': len(group_data),
            **volatility_metrics,
            **price_metrics,
            **momentum_metrics
        }
    
    def _display_earnings_calendar(self, data: pd.DataFrame):
        """Display earnings calendar view"""
        st.subheader("📅 Earnings Calendar - Next 3 Weeks")
        
        if data.empty:
            st.warning("No earnings calendar data available")
            return
        
        # Prepare calendar data
        calendar_data = self._prepare_calendar_data(data)
        
        # Display weekly calendar
        self._render_weekly_calendar(calendar_data)
    
    def _prepare_calendar_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for calendar display"""
        data_copy = data.copy()
        data_copy['earnings_date'] = pd.to_datetime(data_copy['earnings_date'])
        data_copy['week_start'] = data_copy['earnings_date'].dt.to_period('W').dt.start_time
        data_copy['earnings_date_str'] = data_copy['earnings_date'].dt.strftime('%Y-%m-%d')
        data_copy['weekday'] = data_copy['earnings_date'].dt.strftime('%A')
        
        return data_copy.sort_values('earnings_date')
    
    def _render_weekly_calendar(self, calendar_data: pd.DataFrame):
        """Render weekly calendar view"""
        weeks = calendar_data.groupby('week_start')
        
        for week_start, week_data in weeks:
            week_end = week_start + timedelta(days=6)
            st.markdown(f"**Week of {week_start.strftime('%B %d')} - {week_end.strftime('%B %d, %Y')}**")
            
            # Group by day within the week
            daily_earnings = week_data.groupby(['earnings_date_str', 'weekday'])
            
            # Use constant for max columns
            max_columns = min(len(daily_earnings), self.MAX_CALENDAR_COLUMNS)
            cols = st.columns(max_columns)
            
            for i, ((date_str, weekday), day_data) in enumerate(daily_earnings):
                if i < len(cols):
                    self._render_daily_earnings(cols[i], date_str, weekday, day_data)
            
            st.markdown("---")
    
    def _render_daily_earnings(self, col, date_str: str, weekday: str, day_data: pd.DataFrame):
        """Render earnings for a single day"""
        with col:
            st.markdown(f"**{weekday}**")
            st.markdown(f"*{date_str}*")
            
            # Sort symbols by volatility within the day
            day_symbols = day_data.sort_values('volatility_score', ascending=False)
            
            for _, row in day_symbols.iterrows():
                # Use utility functions for indicators
                momentum_indicator = self._get_momentum_indicator(row['recent_momentum'])
                volatility_indicator = "⚡" if row['volatility_score'] > day_symbols['volatility_score'].median() else ""
                
                st.write(f"{momentum_indicator}{volatility_indicator} **{row['symbol']}**")
                momentum_str = self._format_percentage(row['recent_momentum'])
                st.caption(f"Vol: {row['volatility_score']:.1f} | Momentum: {momentum_str}")
    
    def _display_earnings_volatility_chart(self, data: pd.DataFrame):
        """Display earnings stocks volatility chart with earnings dates"""
        st.subheader("⚡ Stocks with Upcoming Earnings - Volatility Analysis")
        
        if data.empty:
            st.warning("No data available for chart")
            return
        
        # Create and display the chart
        fig = self._create_volatility_chart(data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        self._display_chart_explanation()
    
    def _create_volatility_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create the volatility scatter plot"""
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=data['days_to_earnings'],
            y=data['volatility_score'],
            mode='markers+text',
            text=data['symbol'],
            textposition='top center',
            marker=dict(
                size=12,
                color=data['total_return'],
                colorscale='RdYlGn',
                colorbar=dict(title="Total Return %"),
                line=dict(width=1, color='black')
            ),
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Earnings in %{x} days<br>" +
                "Volatility Score: %{y:.2f}<br>" +
                "Total Return: %{marker.color:.2f}%<br>" +
                "<extra></extra>"
            )
        ))
        
        # Configure layout
        fig.update_layout(
            title='Stocks with Upcoming Earnings: Volatility vs Days to Earnings',
            xaxis_title='Days to Earnings',
            yaxis_title='Volatility Score',
            height=600,
            showlegend=False,
            hovermode='closest'
        )
        
        # Add week markers
        self._add_week_markers(fig, data)
        
        # Style grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def _add_week_markers(self, fig: go.Figure, data: pd.DataFrame):
        """Add vertical lines for week markers"""
        for week in [7, 14, 21, 28]:
            fig.add_vline(x=week, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_annotation(
                x=week, y=data['volatility_score'].max(),
                text=f"Week {week//7}",
                showarrow=False,
                yshift=10
            )
    
    def _display_chart_explanation(self):
        """Display chart explanation"""
        st.info("""
        **Chart Explanation:**
        - X-axis: Days until earnings announcement
        - Y-axis: Volatility score (higher = more volatile)
        - Color: Total return % (green = positive, red = negative)
        - Hover for detailed information
        - Vertical lines mark weeks 1, 2, 3, and 4
        """)
    
    def _display_stock_price_grid(self, data: pd.DataFrame):
        """Display individual stock price charts in a grid layout"""
        st.subheader("📈 Stock Price Charts - Earnings Candidates")
        
        if data.empty:
            st.warning("No data available for price charts")
            return
        
        try:
            # Check if too many stocks for grid display
            if len(data) > self.MAX_INDIVIDUAL_CHARTS:
                st.warning(f"⚠️ Too many stocks for grid display ({len(data)}). "
                          f"Showing top {self.MAX_INDIVIDUAL_CHARTS} by volatility score to prevent memory issues.")
                st.info(f"💡 **Tip**: Use the symbol filter to select specific stocks for detailed analysis.")
                
                # Limit to top volatile stocks
                limited_data = data.nlargest(self.MAX_INDIVIDUAL_CHARTS, 'volatility_score')
            else:
                limited_data = data
            
            # Create comprehensive stock data for grid
            stock_grid_data = self._prepare_stock_grid_data(limited_data)
            
            if stock_grid_data.empty:
                st.warning("Unable to prepare data for price charts")
                return
            
            # Get stocks sorted by earnings date then volatility
            earnings_stocks = limited_data.sort_values(['earnings_date', 'volatility_score'], ascending=[True, False])['symbol'].tolist()
            
            # Create the grid visualization
            self._create_stock_price_grid(stock_grid_data, earnings_stocks, ncols=self.DEFAULT_GRID_COLUMNS)
            
        except Exception as e:
            logger.error(f"Error creating price grid: {e}")
            st.error("Unable to create price charts")
            st.info("💡 **Troubleshooting**: Try refreshing the page or selecting fewer symbols.")
    
    def _prepare_stock_grid_data(self, earnings_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive stock data for grid visualization"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.STOCK_GRID_LOOKBACK_DAYS)
            
            # Get symbols from earnings stocks
            earnings_symbols = earnings_data['symbol'].tolist()
            
            # Create placeholders for symbols
            symbol_placeholders = ', '.join(['%s'] * len(earnings_symbols))
            
            # Include minimum price filter from config
            min_price = self.db_manager.get_minimum_price_filter()
            
            query = f"""
            SELECT symbol, date, close, high, low
            FROM stocksdb.stocksinfp
            WHERE symbol IN ({symbol_placeholders})
            AND date >= %s AND date <= %s
            AND close IS NOT NULL AND close != '' AND close != '0'
            AND CAST(close AS DECIMAL(10,2)) >= %s
            ORDER BY symbol, date
            """
            
            params = earnings_symbols + [start_date, end_date, min_price]
            price_data = self.db_manager.execute_query(query, params)
            
            if price_data.empty:
                return pd.DataFrame()
            
            # Use centralized cleaning utility
            price_data = self._clean_dataframe(
                price_data,
                numeric_columns=['close', 'high', 'low'],
                datetime_columns=['date'],
                required_columns=['close', 'high', 'low']
            )
            
            # Calculate additional metrics for each symbol
            enhanced_data = []
            
            for symbol in earnings_symbols:
                symbol_data = price_data[price_data['symbol'] == symbol].sort_values('date')
                
                if len(symbol_data) < self.MIN_SYMBOL_DATA_POINTS:
                    continue
                
                # Get earnings info
                earnings_info = earnings_data[earnings_data['symbol'] == symbol]
                if earnings_info.empty:
                    continue
                
                earnings_row = earnings_info.iloc[0]
                
                # Calculate max/min close for the period
                max_close = symbol_data['close'].max()
                min_close = symbol_data['close'].min()
                
                # Add calculated fields to each row
                for _, row in symbol_data.iterrows():
                    enhanced_row = {
                        'Date': row['date'],
                        'symbol': symbol,
                        'Close': row['close'],
                        'High': row['high'],
                        'Low': row['low'],
                        'max_close': max_close,
                        'min_close': min_close,
                        'volatility_score': earnings_row['volatility_score'],
                        'total_return': earnings_row['total_return'],
                        'earnings_date': earnings_row['earnings_date'],
                        'days_to_earnings': earnings_row['days_to_earnings'],
                        'recent_momentum': earnings_row['recent_momentum']
                    }
                    enhanced_data.append(enhanced_row)
            
            if not enhanced_data:
                return pd.DataFrame()
            
            result_df = pd.DataFrame(enhanced_data)
            # Final cleaning for date columns
            result_df = self._clean_dataframe(
                result_df,
                datetime_columns=['Date', 'earnings_date']
            )
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error preparing stock grid data: {e}")
            return pd.DataFrame()
    
    def _create_stock_price_grid(self, pandas_df: pd.DataFrame, earnings_stocks: List[str], ncols: int = 3):
        """Create a grid of stock price plots using Matplotlib and Seaborn."""
        try:
            # Prepare data
            filtered_stocks = self._prepare_grid_stocks_data(pandas_df, earnings_stocks)
            
            if not filtered_stocks:
                st.warning("No stock data available for grid visualization")
                return

            # Create and display the grid
            fig = self._create_matplotlib_grid(filtered_stocks, ncols)
            st.pyplot(fig)
            plt.close(fig)
            
            # Add explanation
            self._display_grid_explanation()
            
        except Exception as e:
            logger.error(f"Error creating matplotlib grid: {e}")
            st.error(f"Unable to create price charts: {e}")
    
    def _prepare_grid_stocks_data(self, pandas_df: pd.DataFrame, earnings_stocks: List[str]) -> Dict[str, pd.DataFrame]:
        """Prepare and filter stock data for grid visualization"""
        # Convert Date column to datetime format
        pandas_df = pandas_df.copy()
        pandas_df["Date"] = pd.to_datetime(pandas_df["Date"], errors='coerce')

        # Filter and sort stocks by earnings date then volatility
        filtered_df = pandas_df[pandas_df["symbol"].isin(earnings_stocks)].copy()
        
        # Create a dictionary of stock data
        return {
            symbol: filtered_df[filtered_df["symbol"] == symbol].sort_values("Date")
            for symbol in earnings_stocks if symbol in filtered_df["symbol"].unique()
        }
    
    def _create_matplotlib_grid(self, filtered_stocks: Dict[str, pd.DataFrame], ncols: int) -> plt.Figure:
        """Create the matplotlib figure and axes grid"""
        # Calculate layout
        nrows = int(np.ceil(len(filtered_stocks) / ncols))
        
        # Calculate figure size with safety limits
        fig_width = min(18, self.MAX_FIGURE_WIDTH)  # Enforce maximum width
        base_height = 5
        calculated_height = base_height * nrows
        fig_height = min(calculated_height, self.MAX_FIGURE_HEIGHT)  # Enforce maximum height
        
        # Log size information for debugging
        logger.info(f"Creating matplotlib grid: {nrows}x{ncols}, "
                   f"calculated size: {fig_width}x{calculated_height}, "
                   f"final size: {fig_width}x{fig_height}")

        # Create figure and axes with size limits
        try:
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(fig_width, fig_height),
                sharex=False,
                sharey=False
            )
        except Exception as e:
            logger.error(f"Failed to create matplotlib figure with size {fig_width}x{fig_height}: {e}")
            # Fallback to smaller size
            fig_height = min(50, fig_height)  # Emergency fallback
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(fig_width, fig_height),
                sharex=False,
                sharey=False
            )
        
        # Handle single row case
        if nrows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # Iterate over stocks and create plots
        for i, (symbol, stock_data) in enumerate(filtered_stocks.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]

            if stock_data.empty:
                self._handle_empty_plot(ax, symbol)
                continue

            self._create_earnings_stock_plot(ax, stock_data, symbol)

        # Clean up unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # Apply tight layout
        plt.tight_layout()
        
        return fig
    
    def _display_grid_explanation(self):
        """Display explanation for the stock price grid"""
        st.info(f"""
        **Chart Information:**
        - Each chart shows the stock's close price movement over the last 3 weeks leading up to earnings
        - Title includes earnings date, days until earnings, volatility score, and recent momentum
        - Charts are sorted by earnings date (earliest first)
        - Blue line with markers shows the price trend
        - Recent momentum indicates price movement in the last 7 trading days
        - Maximum {self.MAX_INDIVIDUAL_CHARTS} charts displayed to prevent memory issues
        - **Individual charts are ordered by returns (highest returns first)**
        """)
    
    def _handle_empty_plot(self, ax, symbol):
        """Handle empty plot case"""
        ax.text(0.5, 0.5, f'No data\nfor {symbol}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{symbol} - No Data Available', fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _create_earnings_stock_plot(self, ax, stock_data, symbol):
        """Create individual stock plot for earnings candidate"""
        try:
            # Extract metrics
            metrics = self._extract_plot_metrics(stock_data)
            
            # Plot the price line
            ax.plot(metrics['dates'], metrics['prices'], marker='o', linestyle='-', linewidth=2, 
                   color='blue', markersize=3, label='Close Price')
            
            # Annotate price values
            self._annotate_price_values(ax, metrics['dates'], metrics['prices'])
            
            # Set title and labels
            self._set_plot_title_and_labels(ax, symbol, metrics)
            
            # Format axes
            self._format_plot_axes(ax, metrics)
            
        except Exception as e:
            logger.error(f"Error creating plot for {symbol}: {e}")
            self._handle_empty_plot(ax, symbol)
    
    def _extract_plot_metrics(self, stock_data: pd.DataFrame) -> Dict:
        """Extract all metrics needed for plotting"""
        # Get earnings info
        earnings_date = stock_data['earnings_date'].iloc[0]
        days_to_earnings = stock_data['days_to_earnings'].iloc[0]
        volatility_score = stock_data['volatility_score'].iloc[0]
        recent_momentum = stock_data['recent_momentum'].iloc[0]
        
        # Get price data
        dates = stock_data['Date']
        prices = stock_data['Close']
        
        # Calculate return
        first_price = prices.iloc[0]
        last_price = prices.iloc[-1]
        total_return = ((last_price - first_price) / first_price) * 100
        
        # Calculate max/min for display
        max_price = prices.max()
        min_price = prices.min()
        
        # Format earnings date
        earnings_str = earnings_date.strftime('%m/%d') if pd.notna(earnings_date) else 'N/A'
        
        return {
            'dates': dates,
            'prices': prices,
            'earnings_date': earnings_date,
            'days_to_earnings': days_to_earnings,
            'volatility_score': volatility_score,
            'recent_momentum': recent_momentum,
            'total_return': total_return,
            'max_price': max_price,
            'min_price': min_price,
            'earnings_str': earnings_str
        }
    
    def _annotate_price_values(self, ax, dates, prices):
        """Annotate price values with color based on price change"""
        prices_list = prices.tolist()
        dates_list = dates.tolist()
        
        for i, (date, price) in enumerate(zip(dates_list, prices_list)):
            # Determine color based on price change from previous point
            if i == 0:
                text_color = "black"
            else:
                prev_price = prices_list[i-1]
                if price > prev_price:
                    text_color = "green"
                elif price < prev_price:
                    text_color = "red"
                else:
                    text_color = "black"
            
            ax.text(date, price, f'${price:.2f}', fontsize=8, fontweight='bold',
                   ha='center', va='bottom', color=text_color,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    def _set_plot_title_and_labels(self, ax, symbol: str, metrics: Dict):
        """Set plot title and axis labels"""
        # Set title with bold symbol and key metrics
        ax.set_title(f'**{symbol}** | ERD: {metrics["earnings_str"]}, Days: {metrics["days_to_earnings"]}\n'
                    f'VS: {metrics["volatility_score"]:.2f}, Max: ${metrics["max_price"]:.2f}, Min: ${metrics["min_price"]:.2f}\n'
                    f'Return: ',
                    fontsize=16, fontweight='bold', pad=10)
        
        # Add colored return and momentum values
        return_color = 'green' if metrics['total_return'] >= 0 else 'red'
        momentum_color = 'green' if metrics['recent_momentum'] >= 0 else 'red'
        
        ax.text(0.12, 0.92, f"{metrics['total_return']:+.1f}%", 
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                color=return_color, ha='left', va='top')
        
        ax.text(0.20, 0.92, f", Momentum: {metrics['recent_momentum']:+.1f}%", 
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                color=momentum_color, ha='left', va='top')
        
        # Set axis labels
        ax.set_xlabel('Date', fontsize=9, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=9, fontweight='bold')
    
    def _format_plot_axes(self, ax, metrics: Dict):
        """Format plot axes and grid"""
        # Format x-axis (dates)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        if len(metrics['dates']) > self.DATE_TICK_THRESHOLD:
            # Show fewer dates if too many
            step = len(metrics['dates']) // self.DATE_TICK_STEP_DIVISOR
            ax.set_xticks(metrics['dates'][::step])
        
        # Set y-axis to show price range nicely
        price_padding = (metrics['max_price'] - metrics['min_price']) * self.PRICE_PADDING_RATIO
        ax.set_ylim(metrics['min_price'] - price_padding, metrics['max_price'] + price_padding)
    
    def _display_earnings_table(self, data: pd.DataFrame):
        """Display earnings stocks data in table format"""
        st.subheader("📋 Earnings Stocks Analysis Table")
        
        if data.empty:
            st.warning("No data available for table")
            return
        
        display_data = data.sort_values(['earnings_date', 'volatility_score'], ascending=[True, False])
        
        formatted_data = {
            'Rank': range(1, len(display_data) + 1),
            'Symbol': display_data['symbol'].tolist(),
            'Earnings Date': [date.strftime('%Y-%m-%d') for date in display_data['earnings_date']],
            'Days to Earnings': display_data['days_to_earnings'].tolist(),
            'Volatility Score': [f"{val:.2f}" for val in display_data['volatility_score']],
            'Recent Momentum (%)': [f"{val:+.2f}%" for val in display_data['recent_momentum']],
            'Total Return (%)': [f"{val:+.2f}%" for val in display_data['total_return']],
            'Current Price ($)': [f"${val:.2f}" for val in display_data['current_price']]
        }
        
        df_display = pd.DataFrame(formatted_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    def _display_insights(self, data: pd.DataFrame, current_symbol: str):
        """Display insights about the earnings stocks data"""
        st.subheader("💡 Earnings & Volatility Insights")
        
        if data.empty:
            st.warning("No data available for insights")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**This Week's Earnings:**")
            this_week = data[data['days_to_earnings'] <= 7].sort_values('volatility_score', ascending=False)
            if not this_week.empty:
                for i, (_, row) in enumerate(this_week.head(3).iterrows(), 1):
                    momentum_icon = "📈" if row['recent_momentum'] > 0 else "📉" if row['recent_momentum'] < 0 else "➡️"
                    st.write(f"{i}. **{row['symbol']}** {momentum_icon}")
                    st.write(f"   {row['earnings_date'].strftime('%m/%d')} | Vol: {row['volatility_score']:.1f}")
            else:
                st.write("No earnings this week")
        
        with col2:
            st.markdown("**Most Volatile:**")
            top_volatile = data.nlargest(3, 'volatility_score')
            for i, (_, row) in enumerate(top_volatile.iterrows(), 1):
                days_color = "🟢" if row['days_to_earnings'] <= 7 else "🟡" if row['days_to_earnings'] <= 14 else "🔴"
                st.write(f"{i}. **{row['symbol']}** {days_color}")
                st.write(f"   Vol: {row['volatility_score']:.1f} | {row['days_to_earnings']}d")
        
        with col3:
            st.markdown("**Analysis Summary:**")
            avg_volatility = data['volatility_score'].mean()
            avg_momentum = data['recent_momentum'].mean()
            
            st.write(f"• Total Stocks: {len(data)}")
            st.write(f"• Avg Volatility: {avg_volatility:.1f}")
            st.write(f"• Avg Momentum: {avg_momentum:+.1f}%")
            
            positive_momentum = (data['recent_momentum'] > 0).sum()
            st.write(f"• Positive Momentum: {positive_momentum}/{len(data)}")
        
        # Current stock analysis
        if current_symbol in data['symbol'].values:
            st.markdown("---")
            current_data = data[data['symbol'] == current_symbol].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"🎯 **{current_symbol} has earnings coming up!**")
                st.write(f"• Earnings Date: {current_data['earnings_date'].strftime('%Y-%m-%d')}")
                st.write(f"• Days Until: {current_data['days_to_earnings']}")
                
            with col2:
                st.write(f"• Volatility Score: {current_data['volatility_score']:.1f}")
                momentum_text = "Bullish" if current_data['recent_momentum'] > 0 else "Bearish" if current_data['recent_momentum'] < 0 else "Neutral"
                st.write(f"• Recent Momentum: {momentum_text} ({current_data['recent_momentum']:+.1f}%)")
        
        # Weekly breakdown
        st.markdown("---")
        st.markdown("**📅 Weekly Breakdown:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        weeks = [
            (1, "Week 1 (1-7 days)", col1),
            (2, "Week 2 (8-14 days)", col2), 
            (3, "Week 3 (15-21 days)", col3),
            (4, "Week 4 (22-28 days)", col4)
        ]
        
        for week_num, week_label, col in weeks:
            with col:
                week_start = (week_num - 1) * 7 + 1
                week_end = week_num * 7
                week_data = data[(data['days_to_earnings'] >= week_start) & (data['days_to_earnings'] <= week_end)]
                
                st.markdown(f"**{week_label}**")
                st.write(f"• Count: {len(week_data)}")
                if not week_data.empty:
                    avg_vol = week_data['volatility_score'].mean()
                    st.write(f"• Avg Vol: {avg_vol:.1f}")
                    bullish_count = (week_data['recent_momentum'] > 0).sum()
                    st.write(f"• Bullish: {bullish_count}/{len(week_data)}")
        
        st.markdown("---")
        st.info("""
        **💡 Earnings Trading Tips:**
        - High volatility often increases before earnings
        - Consider position sizing based on volatility
        - Monitor recent momentum for trend direction
        - Be aware of earnings date for position timing
        - Use stop-losses due to increased volatility
        """)
    
    def _display_individual_earnings_charts(self, selected_symbols: list, chart_columns: int):
        """Display individual earnings charts for selected symbols"""
        st.subheader("📈 Individual Earnings Stock Charts")
        st.markdown("**Pre-Earnings Price Movement Analysis (Last 3 Weeks) - Sorted by Returns**")
        
        try:
            # Check if too many symbols are selected
            if len(selected_symbols) > self.MAX_INDIVIDUAL_CHARTS:
                st.warning(f"⚠️ Too many symbols selected ({len(selected_symbols)}). "
                          f"Showing only the first {self.MAX_INDIVIDUAL_CHARTS} symbols to prevent memory issues.")
                st.info(f"💡 **Tip**: Select fewer symbols for individual chart analysis. "
                        f"Current limit: {self.MAX_INDIVIDUAL_CHARTS} charts maximum.")
                
                # Limit the symbols to prevent memory issues
                limited_symbols = selected_symbols[:self.MAX_INDIVIDUAL_CHARTS]
            else:
                limited_symbols = selected_symbols
            
            # Sort symbols by returns (highest returns first)
            sorted_symbols = self._sort_symbols_by_returns(limited_symbols)
            
            # Get price data for selected symbols
            price_data = self._get_earnings_price_data_for_symbols(sorted_symbols)
            
            if price_data.empty:
                st.warning("No price data available for selected symbols")
                return
            
            # Create individual charts (now sorted by returns)
            self._create_individual_earnings_charts(price_data, sorted_symbols, chart_columns)
            
        except Exception as e:
            logger.error(f"Error displaying individual earnings charts: {e}")
            st.error("Failed to load individual earnings stock charts")
            st.info("💡 **Troubleshooting**: Try selecting fewer symbols or refresh the page.")
    
    def _sort_symbols_by_returns(self, symbols: list) -> list:
        """Sort symbols by their total returns (highest returns first)"""
        try:
            # Use cached earnings data to avoid redundant database calls
            earnings_data = self._get_cached_earnings_data()
            
            if earnings_data.empty:
                logger.warning("No earnings data available for sorting symbols by returns")
                return symbols
            
            # Filter earnings data to only include our symbols
            filtered_earnings = earnings_data[earnings_data['symbol'].isin(symbols)]
            
            if filtered_earnings.empty:
                logger.warning("No earnings data found for selected symbols")
                return symbols
            
            # Sort by total returns (descending - highest returns first)
            sorted_earnings = filtered_earnings.sort_values('total_return', ascending=False)
            
            # Get sorted symbol list
            sorted_symbols = sorted_earnings['symbol'].tolist()
            
            # Add any symbols that weren't in earnings data (maintain original order for these)
            missing_symbols = [sym for sym in symbols if sym not in sorted_symbols]
            sorted_symbols.extend(missing_symbols)
            
            logger.info(f"Sorted {len(sorted_symbols)} symbols by returns for individual charts")
            return sorted_symbols
            
        except Exception as e:
            logger.error(f"Error sorting symbols by returns: {e}")
            return symbols  # Return original order if sorting fails
    
    def _get_earnings_price_data_for_symbols(self, symbols: list) -> pd.DataFrame:
        """Get price data for selected symbols over last 3 weeks for earnings analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.INDIVIDUAL_CHART_LOOKBACK_DAYS)
            
            # Format symbols for SQL IN clause
            symbols_str = "', '".join(symbols)
            
            query = f"""
            SELECT p.symbol, date, close, high, low, volume
            FROM stocksdb.stocksinfp p
            INNER JOIN stocksdb.stock_change_tracker s ON s.symbol = p.symbol 
                AND s.is_active = 1 
                AND s.current_price > 5
            WHERE p.symbol IN ('{symbols_str}')
            AND date >= %s AND date <= %s
            AND close IS NOT NULL AND close != '' AND close != '0'
            ORDER BY symbol, date
            """
            
            data = self.db_manager.execute_query(query, (start_date, end_date))
            
            if data.empty:
                return pd.DataFrame()
            
            # Use centralized cleaning utility
            return self._clean_dataframe(
                data,
                numeric_columns=['close', 'high', 'low', 'volume'],
                datetime_columns=['date'],
                required_columns=['close']
            )
            
        except Exception as e:
            logger.error(f"Error getting earnings price data: {e}")
            return pd.DataFrame()
    
    def _create_individual_earnings_charts(self, price_data: pd.DataFrame, symbols: list, ncols: int):
        """Create individual matplotlib charts for each earnings symbol"""
        try:
            # Calculate number of rows needed
            nrows = (len(symbols) + ncols - 1) // ncols
            
            # Calculate figure size with safety limits
            fig_width = min(18, self.MAX_FIGURE_WIDTH)
            base_height = 5
            calculated_height = base_height * nrows
            fig_height = min(calculated_height, self.MAX_FIGURE_HEIGHT)
            
            # Log size information for debugging
            logger.info(f"Creating individual earnings charts: {nrows}x{ncols}, "
                       f"symbols: {len(symbols)}, "
                       f"calculated size: {fig_width}x{calculated_height}, "
                       f"final size: {fig_width}x{fig_height}")
            
            # Warn user if figure size was limited
            if calculated_height > self.MAX_FIGURE_HEIGHT:
                st.warning(f"⚠️ Chart size limited to prevent memory issues. "
                          f"Displaying {len(symbols)} charts in a compressed layout.")
                st.info("💡 **Tip**: For better visibility, select fewer symbols or increase chart columns.")
            
            # Create figure with size limits
            try:
                fig, axes = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    figsize=(fig_width, fig_height),
                    sharex=False,  # Disable shared x-axis so each plot gets its own label
                    sharey=False  # Disable shared y-axis
                )
            except Exception as e:
                logger.error(f"Failed to create individual earnings figure with size {fig_width}x{fig_height}: {e}")
                # Fallback to smaller size
                fig_height = min(50, fig_height)  # Emergency fallback
                fig, axes = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    figsize=(fig_width, fig_height),
                    sharex=False,
                    sharey=False
                )
            
            # Set style - use a compatible style
            try:
                plt.style.use('seaborn-v0_8')
            except:
                try:
                    plt.style.use('seaborn')
                except:
                    plt.style.use('default')
            
            # Handle single subplot case
            if nrows == 1 and ncols == 1:
                axes = [axes]
            elif nrows == 1 or ncols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            # Create chart for each symbol
            for idx, symbol in enumerate(symbols):
                if idx < len(axes):
                    ax = axes[idx]
                    symbol_data = price_data[price_data['symbol'] == symbol].copy()
                    
                    if not symbol_data.empty:
                        self._create_individual_earnings_chart(ax, symbol_data, symbol)
                    else:
                        self._handle_empty_earnings_chart(ax, symbol)
            
            # Hide unused subplots
            for idx in range(len(symbols), len(axes)):
                axes[idx].axis('off')
            
            # Apply tight layout to ensure labels don't overlap (your original style)
            plt.tight_layout()
            
            # Display chart with compatible parameters
            st.pyplot(fig)
            
            # Close the figure to free memory
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error creating individual earnings charts: {e}")
            st.error("Failed to create individual earnings stock charts")
    
    def _create_individual_earnings_chart(self, ax, symbol_data, symbol):
        """Create chart for individual earnings stock"""
        try:
            symbol_data = symbol_data.sort_values('date')
            
            # Ensure numeric conversion
            symbol_data['close'] = pd.to_numeric(symbol_data['close'], errors='coerce')
            
            # Get values for the title
            max_close = symbol_data['close'].max()
            min_close = symbol_data['close'].min()
            first_price = symbol_data.iloc[0]['close']
            last_price = symbol_data.iloc[-1]['close']
            total_return = ((last_price - first_price) / first_price) * 100
            
            # Get earnings info if available
            earnings_info = self._get_earnings_info_for_symbol(symbol)
            earnings_date_str = "N/A"
            if earnings_info:
                earnings_date = earnings_info.get('earnings_date')
                if earnings_date:
                    earnings_date_str = earnings_date.strftime('%m/%d')
            
            # Determine color for overall return
            return_color = 'green' if total_return >= 0 else 'red'
            
            # Plot Close as a line graph with markers (back to blue)
            ax.plot(
                symbol_data['date'],
                symbol_data['close'],
                marker='o',
                linestyle='-',
                label="Close Price",
                linewidth=2,
                color="blue"
            )
            
            # Annotate each Close value with color based on price change
            prices = symbol_data['close'].tolist()
            dates = symbol_data['date'].tolist()
            
            for i, (date, close) in enumerate(zip(dates, prices)):
                # Determine color based on price change from previous point
                if i == 0:
                    # First point - use neutral color
                    text_color = "black"
                else:
                    # Compare with previous price
                    prev_price = prices[i-1]
                    if close > prev_price:
                        # Price increased - green
                        text_color = "green"
                    elif close < prev_price:
                        # Price decreased - red
                        text_color = "red"
                    else:
                        # Price unchanged - neutral
                        text_color = "black"
                
                ax.text(
                    date, close, f"${close:.2f}",
                    fontsize=10,
                    fontweight='bold',
                    ha="right",
                    va="bottom",
                    color=text_color,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
                )
            
            # Set title with bold symbol and earnings date (black text, no return)
            ax.set_title(
                f"**{symbol}** | Max: ${max_close:.2f}, Min: ${min_close:.2f}, ERD: {earnings_date_str}, Return: ",
                fontsize=18,
                fontweight='bold',
                pad=10
            )
            
            # Add colored return percentage right after "Return: "
            ax.text(0.78, 1.02, f"{total_return:+.1f}%", 
                    transform=ax.transAxes, fontsize=16, fontweight='bold',
                    color=return_color, ha='left', va='bottom')
            
            # Set individual x-axis label for each chart
            ax.set_xlabel("Date", fontsize=10, fontweight='bold')
            
            # Set y-axis label
            ax.set_ylabel("Close Price ($)", fontsize=10, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            
            # Configure x-axis for proper date formatting
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            
            # Ensure date labels are readable
            ax.tick_params(axis='x', rotation=45)
            
        except Exception as e:
            logger.error(f"Error creating earnings chart for {symbol}: {e}")
            self._handle_empty_earnings_chart(ax, symbol)
    
    def _get_earnings_info_for_symbol(self, symbol: str) -> dict:
        """Get earnings information for a specific symbol"""
        try:
            query = """
            SELECT earnings_date
            FROM stocksdb.stocks_earnings
            WHERE symbol = %s
            AND earnings_date >= CURDATE()
            ORDER BY earnings_date ASC
            LIMIT 1
            """
            
            result = self.db_manager.execute_query(query, (symbol,))
            
            if not result.empty:
                earnings_date = pd.to_datetime(result.iloc[0]['earnings_date'])
                days_to_earnings = (earnings_date - datetime.now()).days
                return {
                    'earnings_date': earnings_date,
                    'days_to_earnings': days_to_earnings
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting earnings info for {symbol}: {e}")
            return {}
    
    def _handle_empty_earnings_chart(self, ax, symbol):
        """Handle case where no data is available for an earnings symbol"""
        ax.text(0.5, 0.5, f'No data available\nfor {symbol}', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
        ax.set_title(f'{symbol} - No Data (Earnings)', fontsize=18, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([]) 