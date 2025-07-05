"""
Volatile Stocks Tab Component - Top 25 volatile stocks with positive returns
Optimized version with reduced code duplication and improved performance
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import logging
from datetime import datetime, timedelta
from data import DatabaseManager
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger('StockApp')

class VolatileStocksTab:
    """Optimized Volatile Stocks Tab with improved performance and code organization"""
    
    # Constants
    TOP_STOCKS_LIMIT = 25
    CACHE_TTL = 3600  # 1 hour
    VOLATILITY_OUTLIER_THRESHOLD = 200
    CHART_FIGURE_SIZE = (18, 6)
    
    # Time period thresholds and requirements
    TIME_PERIOD_THRESHOLDS = {
        'current_week': 7,
        'two_weeks': 14,
        'four_weeks': 28
    }
    
    # Minimum data requirements by time period
    MIN_DATA_REQUIREMENTS = {
        'analysis': {
            'current_week': {'initial': 3, 'returns': 2},
            'two_weeks': {'initial': 6, 'returns': 3},
            'four_weeks': {'initial': 10, 'returns': 5}
        },
        'charts': {
            'current_week': 2,
            'two_weeks': 3,
            'four_weeks': 5
        }
    }
    
    def __init__(self, db_manager: DatabaseManager, time_period_days: int = 28):
        self.db_manager = db_manager
        self.time_period_days = time_period_days
    
    def render(self, symbol: str, stock_data: pd.DataFrame, selected_symbols: Optional[List[str]] = None, 
               chart_columns: int = 1, pre_calculated_data: Optional[pd.DataFrame] = None,
               show_all_time: bool = False):
        """Main render method with improved error handling and performance"""
        
        if show_all_time:
            logger.info("Rendering All Time Volatile Stocks analysis")
            self._render_all_time_performers(symbol, selected_symbols, chart_columns)
        else:
            logger.info(f"Rendering Volatile Stocks tab for {self.time_period_days} days")
            self._render_time_period_analysis(symbol, stock_data, selected_symbols, chart_columns, pre_calculated_data)
    
    def _render_time_period_analysis(self, symbol: str, stock_data: pd.DataFrame, 
                                   selected_symbols: Optional[List[str]], chart_columns: int, 
                                   pre_calculated_data: Optional[pd.DataFrame]):
        """Render analysis for a specific time period"""
        try:
            # Get volatile stocks data
            volatile_stocks = self._get_data_source(pre_calculated_data)
            if volatile_stocks.empty:
                st.warning(f"No volatile stocks data available for {self._get_time_period_text()}")
                return
            
            # Apply symbol filtering if provided
            if selected_symbols:
                volatile_stocks = self._filter_by_symbols(volatile_stocks, selected_symbols)
                if volatile_stocks.empty:
                    return
            
            # Display all components
            self._display_all_components(volatile_stocks, symbol, chart_columns, selected_symbols)
            
        except Exception as e:
            logger.error(f"Error in Volatile Stocks tab: {e}")
            st.error("Unable to load volatile stocks data. Please try again later.")
    
    def _render_all_time_performers(self, symbol: str, selected_symbols: Optional[List[str]], 
                                  chart_columns: int):
        """Render All Time performers - stocks that appear in all three time periods"""
        try:
            st.subheader("🏆 All Time Volatile Performers")
            st.markdown("**Stocks that consistently appear in all three time periods:**")
            
            # Get all time performers
            all_time_data = self._get_all_time_performers()
            
            if all_time_data.empty:
                st.warning("No stocks found that appear in all three time periods (Current Week, Last 2 Weeks, Last 4 Weeks)")
                return
            
            # Apply symbol filtering if provided
            if selected_symbols:
                all_time_data = self._filter_by_symbols_all_time(all_time_data, selected_symbols)
                if all_time_data.empty:
                    return
            
            # Display all time components
            self._display_all_time_components(all_time_data, symbol, chart_columns, selected_symbols)
            
        except Exception as e:
            logger.error(f"Error in All Time Volatile Stocks analysis: {e}")
            st.error("Unable to load all time volatile stocks data. Please try again later.")
    
    def _get_all_time_performers(self) -> pd.DataFrame:
        """Get stocks that appear in all three time periods"""
        try:
            logger.info("Fetching all time volatile performers...")
            
            # Store original period to restore later
            original_period = self.time_period_days
            
            # Get volatile stocks for all three time periods
            periods = [7, 14, 28]  # Current Week, Last 2 Weeks, Last 4 Weeks
            period_data = {}
            
            for period_days in periods:
                # Temporarily change the time period
                self.time_period_days = period_days
                
                period_stocks = self._get_volatile_stocks()
                if not period_stocks.empty:
                    period_data[period_days] = set(period_stocks['symbol'].tolist())
                else:
                    period_data[period_days] = set()
            
            # Find intersection of all three periods
            if all(period_data.values()):
                common_symbols = period_data[7] & period_data[14] & period_data[28]
            else:
                common_symbols = set()
            
            logger.info(f"Found {len(common_symbols)} stocks appearing in all three time periods")
            
            if not common_symbols:
                # Restore original period before returning
                self.time_period_days = original_period
                return pd.DataFrame()
            
            # Get detailed data for common symbols using 4-week period
            self.time_period_days = 28  # Use 4-week data for detailed analysis
            all_volatile_stocks = self._get_volatile_stocks()
            
            # Restore original period
            self.time_period_days = original_period
            
            if all_volatile_stocks.empty:
                return pd.DataFrame()
            
            # Filter to only common symbols
            all_time_performers = all_volatile_stocks[
                all_volatile_stocks['symbol'].isin(common_symbols)
            ].copy()
            
            # Add period appearance info
            all_time_performers['appears_in_periods'] = 'All 3 Periods'
            all_time_performers['period_consistency'] = 'High'
            
            return all_time_performers.sort_values('total_return', ascending=False)
            
        except Exception as e:
            logger.error(f"Error getting all time performers: {e}")
            # Restore original period in case of error
            self.time_period_days = original_period
            return pd.DataFrame()
    
    def _filter_by_symbols_all_time(self, data: pd.DataFrame, selected_symbols: List[str]) -> pd.DataFrame:
        """Filter all time data by selected symbols"""
        filtered_data = data[data['symbol'].isin(selected_symbols)]
        if filtered_data.empty:
            st.warning(f"None of the selected symbols appear in all three time periods: {', '.join(selected_symbols)}")
            return pd.DataFrame()
        
        st.info(f"🏆 Showing {len(filtered_data)} selected symbols that appear in all time periods")
        return filtered_data
    
    def _display_all_time_components(self, data: pd.DataFrame, symbol: str, 
                                   chart_columns: int, selected_symbols: Optional[List[str]]):
        """Display all components for all time performers"""
        # Summary info
        self._display_all_time_summary(data)
        
        # Main chart
        self._display_all_time_chart(data)
        
        st.markdown("---")
        # Price grid using 4-week data
        self._display_stock_price_grid(data, chart_columns)
        
        # Individual charts if symbols selected
        if selected_symbols:
            st.markdown("---")
            self._display_individual_volatile_charts(selected_symbols, chart_columns, data)
        
        # Table and insights
        self._display_all_time_table(data)
        self._display_all_time_insights(data, symbol)
    
    def _display_all_time_summary(self, data: pd.DataFrame):
        """Display summary information for all time performers"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏆 All Time Performers", len(data))
        
        with col2:
            avg_return = data['total_return'].mean()
            st.metric("📈 Avg Return (4W)", f"{avg_return:.1f}%")
        
        with col3:
            avg_volatility = data['volatility_score'].mean()
            st.metric("⚡ Avg Volatility", f"{avg_volatility:.1f}")
        
        with col4:
            max_return = data['total_return'].max()
            best_performer = data.loc[data['total_return'].idxmax(), 'symbol']
            st.metric("🥇 Best Performer", f"{best_performer}: {max_return:.1f}%")
        
        st.info("""
        **🏆 All Time Performers Criteria:**
        - Stocks that appear in **all three time periods**:
          - ⚡ Current Week (7 days)
          - 📈 Last 2 Weeks (14 days)  
          - 📅 Last 4 Weeks (28 days)
        - Shows **consistent volatility** across different time frames
        - Analysis based on **4-week performance data**
        """)
    
    def _display_all_time_chart(self, data: pd.DataFrame):
        """Display chart for all time performers"""
        st.subheader("🏆 All Time Volatile Performers - Total Returns")
        
        if data.empty:
            st.warning("No data available for all time chart")
            return
        
        # Sort by total return (highest to lowest for chart display)
        data_sorted = data.sort_values('total_return', ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Total Return (4 Weeks)',
            x=data_sorted['total_return'],
            y=data_sorted['symbol'],
            orientation='h',
            marker_color='#FFD700',  # Gold color for all time performers
            text=[f"{val:.1f}%" for val in data_sorted['total_return']],
            textposition='inside',
            textfont=dict(color='black', size=10),
            customdata=list(zip(data_sorted['volatility_score'], data_sorted['total_return'])),
            hovertemplate='<b>%{y}</b><br>' +
                         'Total Return (4W): %{x:.1f}%<br>' +
                         'Volatility Score: %{customdata[0]:.1f}<br>' +
                         'Consistency: All 3 Periods<extra></extra>'
        ))
        
        fig.update_layout(
            title='All Time Volatile Performers - 4 Week Returns<br><sub>Stocks appearing in Current Week, 2 Weeks, AND 4 Weeks</sub>',
            xaxis_title='Total Return - 4 Weeks (%)',
            yaxis_title='Stock Symbol (All Time Performers)',
            height=max(400, len(data_sorted) * 25),
            showlegend=False,
            hovermode='y unified'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_all_time_table(self, data: pd.DataFrame):
        """Display all time performers table"""
        st.subheader("📋 All Time Performers Analysis Table")
        
        if data.empty:
            st.warning("No data available for all time table")
            return
        
        # Sort by total return (highest first)
        display_data = data.sort_values('total_return', ascending=False)
        
        formatted_data = {
            'Rank': range(1, len(display_data) + 1),
            'Symbol': display_data['symbol'].tolist(),
            'Consistency': ['All 3 Periods'] * len(display_data),
            'Total Return - 4W (%)': [f"{val:.2f}%" for val in display_data['total_return']],
            'Volatility Score': [f"{val:.2f}" for val in display_data['volatility_score']],
            'Std Dev (%)': [f"{val:.1f}%" for val in display_data['volatility_std']],
            'Max Daily Gain (%)': [f"{val:.2f}%" for val in display_data['max_daily_gain']],
            'Max Daily Loss (%)': [f"{val:.2f}%" for val in display_data['max_daily_loss']]
        }
        
        df_display = pd.DataFrame(formatted_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    def _display_all_time_insights(self, data: pd.DataFrame, current_symbol: str):
        """Display insights for all time performers"""
        st.subheader("💡 All Time Performers Insights")
        
        if data.empty:
            st.warning("No data available for all time insights")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏆 Top All Time Performers:**")
            top_3 = data.nlargest(3, 'total_return')
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                st.write(f"{i}. **{row['symbol']}**: +{row['total_return']:.1f}%")
                st.write(f"   Volatility: {row['volatility_score']:.1f}")
        
        with col2:
            st.markdown("**📊 Consistency Analysis:**")
            total_performers = len(data)
            avg_return = data['total_return'].mean()
            avg_volatility = data['volatility_score'].mean()
            
            st.write(f"• All Time Performers: {total_performers}")
            st.write(f"• Avg 4W Return: +{avg_return:.1f}%")
            st.write(f"• Avg Volatility: {avg_volatility:.1f}")
        
        with col3:
            st.markdown("**🎯 Current Stock Status:**")
            if current_symbol in data['symbol'].values:
                current_data = data[data['symbol'] == current_symbol].iloc[0]
                rank = (data['total_return'] > current_data['total_return']).sum() + 1
                
                st.success(f"🏆 **{current_symbol} is an All Time Performer!**")
                st.write(f"• Rank: #{rank} of {len(data)}")
                st.write(f"• 4W Return: +{current_data['total_return']:.1f}%")
                st.write(f"• Volatility: {current_data['volatility_score']:.1f}")
            else:
                st.info(f"📊 {current_symbol} is not an All Time Performer")
                st.write("(Not in all 3 time periods)")
        
        st.markdown("---")
        st.success(f"""
        **🏆 All Time Performers Significance:**
        - These {len(data)} stocks show **consistent high volatility** across all time periods
        - They appear in **Current Week (7d)**, **Last 2 Weeks (14d)**, AND **Last 4 Weeks (28d)**
        - **Higher reliability** for volatility-based trading strategies
        - **Sustained momentum** and market interest
        
        **⚠️ Investment Note:** Consistent volatility = Both consistent opportunity AND risk.
        """)
    
    def _get_data_source(self, pre_calculated_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Get data source with fallback to calculation"""
        if pre_calculated_data is not None and not pre_calculated_data.empty:
            logger.info("Using pre-calculated volatility data")
            return pre_calculated_data
        return self._get_volatile_stocks()
    
    def _filter_by_symbols(self, data: pd.DataFrame, selected_symbols: List[str]) -> pd.DataFrame:
        """Filter data by selected symbols with validation"""
        filtered_data = data[data['symbol'].isin(selected_symbols)]
        if filtered_data.empty:
            st.warning(f"No volatility data available for selected symbols: {', '.join(selected_symbols)}")
            return pd.DataFrame()
        
        st.info(f"📊 Showing analysis for {len(selected_symbols)} selected symbols")
        return filtered_data
    
    def _display_all_components(self, volatile_stocks: pd.DataFrame, symbol: str, 
                              chart_columns: int, selected_symbols: Optional[List[str]]):
        """Display all UI components in order"""
        # Main analysis components
        self._display_volatility_chart(volatile_stocks)
        
        st.markdown("---")
        self._display_stock_price_grid(volatile_stocks, chart_columns)
        
        # Individual charts if symbols selected
        if selected_symbols:
            st.markdown("---")
            self._display_individual_volatile_charts(selected_symbols, chart_columns, volatile_stocks)
        
        # Data table and insights
        self._display_volatility_table(volatile_stocks)
        self._display_insights(volatile_stocks, symbol)
    
    def _get_volatile_stocks(self) -> pd.DataFrame:
        """Get most volatile stocks with optimized database query"""
        try:
            date_range = self._get_date_range()
            logger.info(f"Fetching stock data from {date_range[0].date()} to {date_range[1].date()}")
            
            # Single optimized query
            all_data = self._fetch_stock_data(*date_range)
            if all_data.empty:
                return pd.DataFrame()
            
            # Process data
            clean_data = self._clean_stock_data(all_data)
            volatility_results = self._calculate_volatility_statistics(clean_data)
            
            if volatility_results.empty:
                return pd.DataFrame()
            
            # Filter and return top stocks
            return self._filter_and_rank_stocks(volatility_results)
            
        except Exception as e:
            logger.error(f"Error getting volatile stocks: {e}")
            return pd.DataFrame()
    
    def _get_date_range(self) -> Tuple[datetime, datetime]:
        """Get start and end dates for the time period"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.time_period_days)
        return start_date, end_date
    
    def _fetch_stock_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch stock data with optimized query"""
        query = """
        SELECT p.symbol, date, close, high, low, volume
        FROM stocksinfp p
        INNER JOIN stock_change_tracker s ON s.symbol = p.symbol 
            AND s.is_active = 1 
            AND s.current_price > 5
        WHERE date >= %s AND date <= %s
        AND close IS NOT NULL AND close != '' AND close != '0'
        AND high IS NOT NULL AND high != '' AND high != '0'
        AND low IS NOT NULL AND low != '' AND low != '0'
        ORDER BY symbol, date
        """
        return self.db_manager.execute_query(query, (start_date, end_date))
    
    def _clean_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare stock data with optimized processing"""
        # Convert data types in batch
        data['date'] = pd.to_datetime(data['date'])
        numeric_columns = ['close', 'high', 'low', 'volume']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Clean data
        data = data.dropna(subset=['close', 'high', 'low'])
        data = data[(data['close'] > 0) & (data['high'] > 0) & (data['low'] > 0)]
        
        return data
    
    def _get_minimum_requirements(self, requirement_type: str) -> Dict[str, int]:
        """Get minimum data requirements based on time period"""
        if self.time_period_days <= self.TIME_PERIOD_THRESHOLDS['current_week']:
            period_key = 'current_week'
        elif self.time_period_days <= self.TIME_PERIOD_THRESHOLDS['two_weeks']:
            period_key = 'two_weeks'
        else:
            period_key = 'four_weeks'
        
        if requirement_type == 'analysis':
            return self.MIN_DATA_REQUIREMENTS['analysis'][period_key]
        else:
            return {'min_points': self.MIN_DATA_REQUIREMENTS['charts'][period_key]}
    
    def _calculate_volatility_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility statistics with optimized processing"""
        symbol_groups = data.groupby('symbol')
        volatility_list = []
        
        # Get minimum requirements
        min_reqs = self._get_minimum_requirements('analysis')
        min_initial_points = min_reqs['initial']
        min_return_points = min_reqs['returns']
        
        logger.info(f"Processing {len(symbol_groups)} symbols with min requirements: {min_initial_points} initial, {min_return_points} returns")
        
        for symbol, group_data in symbol_groups:
            try:
                volatility_stats = self._calculate_symbol_volatility(group_data, symbol, min_initial_points, min_return_points)
                if volatility_stats:
                    volatility_list.append(volatility_stats)
            except Exception as e:
                logger.debug(f"Error processing symbol {symbol}: {e}")
                continue
        
        if not volatility_list:
            logger.warning(f"No stocks passed volatility analysis for {self.time_period_days} days period")
            return pd.DataFrame()
        
        # Create DataFrame and filter outliers
        volatility_df = pd.DataFrame(volatility_list)
        logger.info(f"Found {len(volatility_df)} stocks before outlier filtering")
        
        volatility_df = volatility_df[volatility_df['volatility_score'] <= self.VOLATILITY_OUTLIER_THRESHOLD]
        logger.info(f"Found {len(volatility_df)} stocks after outlier filtering")
        
        return volatility_df
    
    def _calculate_symbol_volatility(self, group_data: pd.DataFrame, symbol: str, 
                                   min_initial_points: int, min_return_points: int) -> Optional[Dict]:
        """Calculate volatility metrics for a single symbol"""
        group_data = group_data.sort_values('date')
        
        if len(group_data) < min_initial_points:
            logger.debug(f"Skipping {symbol}: only {len(group_data)} data points, need {min_initial_points}")
            return None
        
        # Calculate daily returns
        group_data = group_data.copy()
        group_data['daily_return'] = group_data['close'].pct_change()
        group_data = group_data.dropna(subset=['daily_return'])
        
        if len(group_data) < min_return_points:
            return None
        
        # Calculate volatility metrics
        daily_returns = group_data['daily_return']
        volatility_std = daily_returns.std() * np.sqrt(252)  # Annualized
        
        # Average True Range
        group_data['tr'] = np.maximum(
            group_data['high'] - group_data['low'],
            np.maximum(
                abs(group_data['high'] - group_data['close'].shift(1)),
                abs(group_data['low'] - group_data['close'].shift(1))
            )
        )
        atr = group_data['tr'].mean()
        atr_pct = (atr / group_data['close'].mean()) * 100
        
        # Price range volatility
        price_range = (group_data['high'].max() - group_data['low'].min()) / group_data['close'].mean() * 100
        
        # Total return
        first_price = group_data.iloc[0]['close']
        last_price = group_data.iloc[-1]['close']
        total_return = ((last_price - first_price) / first_price) * 100
        
        # Composite volatility score
        volatility_score = (volatility_std * 0.4 + atr_pct * 0.3 + price_range * 0.3)
        
        return {
            'symbol': symbol,
            'volatility_score': volatility_score,
            'volatility_std': volatility_std,
            'atr_pct': atr_pct,
            'price_range': price_range,
            'total_return': total_return,
            'first_price': first_price,
            'last_price': last_price,
            'data_points': len(group_data),
            'avg_daily_return': daily_returns.mean() * 100,
            'max_daily_gain': daily_returns.max() * 100,
            'max_daily_loss': daily_returns.min() * 100
        }
    
    def _filter_and_rank_stocks(self, volatility_results: pd.DataFrame) -> pd.DataFrame:
        """Filter for positive returns and get top stocks"""
        logger.info(f"Total volatility results: {len(volatility_results)} stocks")
        
        positive_returns = volatility_results[volatility_results['total_return'] > 0]
        logger.info(f"Stocks with positive returns: {len(positive_returns)} stocks")
        
        if positive_returns.empty:
            logger.warning(f"No stocks with positive returns found for {self.time_period_days} days period")
            return pd.DataFrame()
        
        top_stocks = positive_returns.nlargest(self.TOP_STOCKS_LIMIT, 'volatility_score')
        logger.info(f"Found {len(top_stocks)} volatile stocks with positive returns")
        
        return top_stocks
    
    def _get_time_period_text(self) -> str:
        """Get formatted time period text"""
        if self.time_period_days == 7:
            return "Current Week"
        elif self.time_period_days == 14:
            return "Last 2 Weeks"
        elif self.time_period_days == 28:
            return "Last 4 Weeks"
        else:
            return f"{self.time_period_days} Days"
    
    def _display_volatility_chart(self, data: pd.DataFrame):
        """Display volatility visualization chart"""
        time_period_text = self._get_time_period_text()
        st.subheader(f"⚡ Top {self.TOP_STOCKS_LIMIT} Most Volatile Stocks - {time_period_text} (Positive Returns Only)")
        
        if data.empty:
            st.warning("No data available for chart")
            return
        
        # Sort by total return (highest to lowest for chart display)
        data = data.sort_values('total_return', ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Total Return',
            x=data['total_return'],
            y=data['symbol'],
            orientation='h',
            marker_color='#2e8b57',  # Sea green color for positive returns
            text=[f"{val:.1f}%" for val in data['total_return']],
            textposition='inside',
            textfont=dict(color='white', size=10),
            customdata=list(zip(data['volatility_score'], data['total_return'])),
            hovertemplate='<b>%{y}</b><br>' +
                         'Total Return: %{x:.1f}%<br>' +
                         'Volatility Score: %{customdata[0]:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Total Return by Stock (Highest Returns First)<br><sub>Selected from Top {self.TOP_STOCKS_LIMIT} Most Volatile Stocks</sub>',
            xaxis_title='Total Return (%)',
            yaxis_title='Stock Symbol (ordered by Total Return)',
            height=800,
            showlegend=False,
            hovermode='y unified'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_stock_price_grid(self, data: pd.DataFrame, chart_columns: int = 1):
        """Display individual stock price charts in a grid layout"""
        time_period_text = self._get_time_period_text()
        st.subheader(f"📈 Individual Stock Price Charts - {time_period_text}")
        
        if data.empty:
            st.warning("No data available for price charts")
            return
        
        try:
            # Sort data by total return (highest first) before getting symbols
            data_sorted = data.sort_values('total_return', ascending=False)
            
            # Get top stocks and prepare chart data
            top_stocks = data_sorted.head(self.TOP_STOCKS_LIMIT)['symbol'].tolist()
            chart_data = self._prepare_chart_data(top_stocks, data_sorted)
            
            if chart_data.empty:
                st.warning("Unable to prepare data for price charts")
                return
            
            # Create the grid visualization
            self._create_matplotlib_grid(chart_data, top_stocks, chart_columns)
            
        except Exception as e:
            logger.error(f"Error creating price grid: {e}")
            st.error("Unable to create price charts")
    
    def _prepare_chart_data(self, symbols: List[str], volatility_data: pd.DataFrame) -> pd.DataFrame:
        """Unified method to prepare chart data for any set of symbols"""
        try:
            start_date, end_date = self._get_date_range()
            
            # Create SQL query for selected symbols
            symbol_placeholders = ', '.join(['%s'] * len(symbols))
            query = f"""
            SELECT p.symbol, date, close, high, low
            FROM stocksinfp p
            INNER JOIN stock_change_tracker s ON s.symbol = p.symbol 
                AND s.is_active = 1 
                AND s.current_price > 5
            WHERE p.symbol IN ({symbol_placeholders})
            AND date >= %s AND date <= %s
            AND close IS NOT NULL AND close != '' AND close != '0'
            ORDER BY symbol, date
            """
            
            params = symbols + [start_date, end_date]
            price_data = self.db_manager.execute_query(query, params)
            
            if price_data.empty:
                return pd.DataFrame()
            
            # Clean price data
            price_data = self._clean_price_data(price_data)
            
            # Enhance with volatility metrics
            return self._enhance_with_volatility_data(price_data, volatility_data, symbols)
            
        except Exception as e:
            logger.error(f"Error preparing chart data: {e}")
            return pd.DataFrame()
    
    def _clean_price_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Clean price data for charts"""
        # Convert data types
        price_data['date'] = pd.to_datetime(price_data['date'])
        for col in ['close', 'high', 'low']:
            price_data[col] = pd.to_numeric(price_data[col], errors='coerce')
        
        # Clean data
        return price_data.dropna(subset=['close', 'high', 'low'])
    
    def _enhance_with_volatility_data(self, price_data: pd.DataFrame, volatility_data: pd.DataFrame, 
                                    symbols: List[str]) -> pd.DataFrame:
        """Enhance price data with volatility metrics"""
        enhanced_data = []
        min_reqs = self._get_minimum_requirements('charts')
        min_chart_points = min_reqs['min_points']
        
        for symbol in symbols:
            symbol_price_data = price_data[price_data['symbol'] == symbol].sort_values('date')
            
            if len(symbol_price_data) < min_chart_points:
                logger.debug(f"Skipping {symbol} for chart: only {len(symbol_price_data)} data points, need {min_chart_points}")
                continue
            
            # Get volatility info
            volatility_info = volatility_data[volatility_data['symbol'] == symbol]
            if volatility_info.empty:
                continue
            
            volatility_row = volatility_info.iloc[0]
            
            # Calculate period max/min for the chart
            max_close = symbol_price_data['close'].max()
            min_close = symbol_price_data['close'].min()
            
            # Add enhanced fields to each price row
            for _, row in symbol_price_data.iterrows():
                enhanced_row = {
                    'Date': row['date'],
                    'symbol': symbol,
                    'Close': row['close'],
                    'High': row['high'],
                    'Low': row['low'],
                    'max_close': max_close,
                    'min_close': min_close,
                    'volatility_score': volatility_row['volatility_score'],
                    'total_return': volatility_row['total_return'],
                    'first_price': volatility_row['first_price'],
                    'last_price': volatility_row['last_price']
                }
                enhanced_data.append(enhanced_row)
        
        if not enhanced_data:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(enhanced_data)
        result_df['Date'] = pd.to_datetime(result_df['Date'])
        
        return result_df
    
    def _create_matplotlib_grid(self, chart_data: pd.DataFrame, symbols: List[str], ncols: int = 1):
        """Unified method to create matplotlib grid for any set of symbols"""
        try:
            # Convert Date column to datetime
            chart_data = chart_data.copy()
            chart_data["Date"] = pd.to_datetime(chart_data["Date"], errors='coerce')
            
            # Filter and sort by total_return (descending) - highest returns first
            sorted_df = chart_data[chart_data["symbol"].isin(symbols)].copy()
            
            # Get unique symbols with their return values, then sort by return
            symbol_returns = sorted_df.groupby('symbol')['total_return'].first().sort_values(ascending=False)
            symbols_by_return = symbol_returns.index.tolist()
            
            # Create dictionary of stock data ordered by return (highest first)
            stock_data_dict = {
                symbol: sorted_df[sorted_df["symbol"] == symbol].sort_values("Date")
                for symbol in symbols_by_return
            }
            
            if not stock_data_dict:
                st.warning("No stock data available for grid visualization")
                return
            
            # Create matplotlib figure
            self._create_matplotlib_figure(stock_data_dict, ncols)
            
        except Exception as e:
            logger.error(f"Error creating matplotlib grid: {e}")
            st.error(f"Unable to create price charts: {e}")
    
    def _create_matplotlib_figure(self, stock_data_dict: Dict[str, pd.DataFrame], ncols: int):
        """Create and display matplotlib figure"""
        # Calculate layout
        nrows = int(np.ceil(len(stock_data_dict) / ncols))
        
        # Create figure and axes
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(self.CHART_FIGURE_SIZE[0], self.CHART_FIGURE_SIZE[1] * nrows),
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
        
        # Create plots
        for i, (symbol, stock_data) in enumerate(stock_data_dict.items()):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            if stock_data.empty:
                self._handle_empty_plot(ax, symbol)
                continue
            
            self._create_stock_plot(ax, stock_data, symbol)
        
        # Clean up unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        # Apply layout
        plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Display and cleanup
        st.pyplot(fig)
        plt.close(fig)
        
        # Add explanation
        self._add_chart_explanation()
    
    def _add_chart_explanation(self):
        """Add chart explanation info box"""
        time_period_text = self._get_time_period_text().lower()
        
        st.info(f"""
        **📊 Chart Information (2-Step Process):**
        - **Step 1**: Selected top {self.TOP_STOCKS_LIMIT} stocks by volatility score (highest volatility first)
        - **Step 2**: Charts are sorted by **Total Return** (highest returns first)
        - Each chart shows the stock's close price movement over the {time_period_text}
        - Price values are annotated on each data point
        - Title includes Max/Min close prices, volatility score, and total return
        - Blue line with markers shows the price trend
        - Focus is on **volatile stocks with the highest returns**
        """)
    
    def _handle_empty_plot(self, ax, symbol):
        """Handle cases where there's no data for a symbol"""
        ax.text(
            0.5, 0.5, f"No Data for {symbol}",
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=12,
            color='red'
        )
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _create_stock_plot(self, ax, stock_data, symbol):
        """Create individual stock plot with volatility metrics"""
        try:
            # Ensure numeric conversion
            stock_data = stock_data.copy()
            numeric_fields = ["Close", "max_close", "min_close"]
            for field in numeric_fields:
                stock_data[field] = pd.to_numeric(stock_data[field], errors='coerce')
            
            # Drop NaN values
            stock_data = stock_data.dropna(subset=["Close"])
            
            if stock_data.empty:
                self._handle_empty_plot(ax, symbol)
                return
            
            # Get metrics for title
            max_close = stock_data["max_close"].iloc[0]
            min_close = stock_data["min_close"].iloc[0]
            volatility_score = stock_data["volatility_score"].iloc[0]
            total_return = stock_data["total_return"].iloc[0]
            
            # Plot line and annotations
            self._plot_price_line(ax, stock_data)
            self._add_price_annotations(ax, stock_data)
            self._format_plot_title_and_axes(ax, symbol, max_close, min_close, total_return)
            
        except Exception as e:
            logger.error(f"Error creating plot for {symbol}: {e}")
            self._handle_empty_plot(ax, symbol)
    
    def _plot_price_line(self, ax, stock_data):
        """Plot the price line with markers"""
        ax.plot(
            stock_data["Date"],
            stock_data["Close"],
            marker='o',
            linestyle='-',
            label="Close Price",
            linewidth=2.5,
            color="blue",
            markersize=4,
            alpha=0.8
        )
    
    def _add_price_annotations(self, ax, stock_data):
        """Add price value annotations with color coding"""
        prices = stock_data["Close"].tolist()
        dates = stock_data["Date"].tolist()
        
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Determine color based on price change
            if i == 0:
                text_color = "black"
            else:
                prev_price = prices[i-1]
                if close > prev_price:
                    text_color = "green"
                elif close < prev_price:
                    text_color = "red"
                else:
                    text_color = "black"
            
            ax.text(
                date, close, f"${close:.2f}",
                fontsize=9,
                fontweight='bold',
                ha="center",
                va="bottom",
                color=text_color,
                bbox=dict(facecolor='white', edgecolor='gray', 
                         boxstyle='round,pad=0.2', alpha=0.8)
            )
    
    def _format_plot_title_and_axes(self, ax, symbol, max_close, min_close, total_return):
        """Format plot title and axes"""
        return_color = 'green' if total_return >= 0 else 'red'
        
        # Set title
        ax.set_title(
            f"**{symbol}** | Max: ${max_close:.2f}, Min: ${min_close:.2f}, Return: ",
            fontsize=18,
            fontweight='bold',
            pad=10
        )
        
        # Add colored return percentage
        ax.text(0.72, 1.02, f"{total_return:+.1f}%", 
                transform=ax.transAxes, fontsize=16, fontweight='bold',
                color=return_color, ha='left', va='bottom')
        
        # Set axis labels
        ax.set_xlabel("Date", fontsize=10, fontweight='bold')
        ax.set_ylabel("Close Price ($)", fontsize=10, fontweight='bold')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=8)
        
        # Configure date formatting
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.margins(x=0.02, y=0.1)
    
    def _display_volatility_table(self, data: pd.DataFrame):
        """Display volatility data in table format"""
        st.subheader("📋 Volatility Analysis Table")
        
        if data.empty:
            st.warning("No data available for table")
            return
        
        # Sort by total return (highest first)
        display_data = data.sort_values('total_return', ascending=False)
        
        formatted_data = {
            'Rank': range(1, len(display_data) + 1),
            'Symbol': display_data['symbol'].tolist(),
            'Volatility Score': [f"{val:.2f}" for val in display_data['volatility_score']],
            'Total Return (%)': [f"{val:.2f}%" for val in display_data['total_return']],
            'Std Dev (%)': [f"{val:.1f}%" for val in display_data['volatility_std']],
            'ATR (%)': [f"{val:.2f}%" for val in display_data['atr_pct']],
            'Max Daily Gain (%)': [f"{val:.2f}%" for val in display_data['max_daily_gain']],
            'Max Daily Loss (%)': [f"{val:.2f}%" for val in display_data['max_daily_loss']]
        }
        
        df_display = pd.DataFrame(formatted_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    def _display_insights(self, data: pd.DataFrame, current_symbol: str):
        """Display insights about the volatility data"""
        st.subheader("💡 Volatility Insights")
        
        if data.empty:
            st.warning("No data available for insights")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Highest Returns:**")
            top_3 = data.nlargest(3, 'total_return')
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                st.write(f"{i}. **{row['symbol']}**: +{row['total_return']:.1f}%")
                st.write(f"   Volatility: {row['volatility_score']:.1f}")
        
        with col2:
            st.markdown("**Analysis:**")
            avg_volatility = data['volatility_score'].mean()
            avg_return = data['total_return'].mean()
            
            st.write(f"• Avg Volatility: {avg_volatility:.1f}")
            st.write(f"• Avg Return: +{avg_return:.1f}%")
            st.write(f"• Stocks Analyzed: {len(data)}")
        
        with col3:
            st.markdown("**Current Stock:**")
            if current_symbol in data['symbol'].values:
                current_data = data[data['symbol'] == current_symbol].iloc[0]
                rank = (data['total_return'] > current_data['total_return']).sum() + 1
                
                st.success(f"📈 **{current_symbol} is #{rank}!**")
                st.write(f"• Return: +{current_data['total_return']:.1f}%")
                st.write(f"• Volatility: {current_data['volatility_score']:.1f}")
            else:
                st.info(f"📊 {current_symbol} not in top {self.TOP_STOCKS_LIMIT}")
        
        st.markdown("---")
        st.info(f"""
        **📊 Selection Process:**
        1. **Step 1**: Selected top {self.TOP_STOCKS_LIMIT} stocks by volatility score (highest volatility first)
        2. **Step 2**: Sorted by **Total Return** (highest returns first)
        
        **⚠️ Risk Warning:** High volatility = High risk & High opportunity.
        Always use proper risk management and position sizing.
        """)
    
    def _display_individual_volatile_charts(self, selected_symbols: List[str], chart_columns: int, 
                                          volatility_data: pd.DataFrame):
        """Display individual stock charts for selected symbols"""
        time_period_text = self._get_time_period_text()
        st.subheader("📈 Individual Stock Performance Charts")
        st.markdown(f"**{time_period_text} Volatility Analysis**")
        
        try:
            # Use the unified chart preparation method
            chart_data = self._prepare_chart_data(selected_symbols, volatility_data)
            
            if chart_data.empty:
                st.warning("Unable to prepare chart data for selected symbols")
                return
            
            # Create the matplotlib grid
            self._create_matplotlib_grid(chart_data, selected_symbols, chart_columns)
            
        except Exception as e:
            logger.error(f"Error displaying individual volatile charts: {e}")
            st.error("Failed to load individual stock charts") 