"""
Best Performers Tab Component - Top 25 stocks for last 3 months
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import logging
from datetime import datetime, timedelta
from data import DatabaseManager
from analysis import TechnicalIndicators
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from utils.chart_utils import ChartUtils

logger = logging.getLogger('StockApp')

class BestPerformersTab:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tech_indicators = TechnicalIndicators()
    
    # Add caching for price data to avoid repeated DB queries
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def _get_cached_price_data_for_symbols(_self, symbols: list) -> pd.DataFrame:
        """Get cached price data for selected symbols over last 3 months"""
        try:
            # Create fresh db manager for cached call to avoid pickling issues
            from data import DatabaseManager
            db_manager = DatabaseManager()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            logger.info(f"[CACHED CALL] Fetching price data for {len(symbols)} symbols")
            
            # Format symbols for SQL IN clause
            symbols_str = "', '".join(symbols)
            
            # Include minimum price filter from config
            min_price = db_manager.get_minimum_price_filter()
            
            query = f"""
            SELECT p.symbol, date, close, high, low, volume
            FROM stocksinfp as p
            LEFT JOIN stock_change_tracker as s ON s.symbol = p.symbol
            WHERE p.symbol IN ('{symbols_str}')
            AND date >= %s AND date <= %s
            AND close IS NOT NULL AND close != '' AND close != '0'
            AND CAST(close AS DECIMAL(10,2)) >= %s
            AND s.is_active = 1
            ORDER BY symbol, date
            """
            
            data = db_manager.execute_query(query, (start_date, end_date, min_price))
            
            if data.empty:
                logger.info("[CACHED CALL] No price data found")
                return pd.DataFrame()
            
            # Convert data types
            data['date'] = pd.to_datetime(data['date'])
            for col in ['close', 'high', 'low', 'volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Clean data
            data = data.dropna(subset=['close'])
            data = data[data['close'] > 0]
            
            logger.info(f"[CACHED CALL] Returning {len(data)} rows of price data")
            return data
            
        except Exception as e:
            logger.error(f"Error getting cached price data: {e}")
            return pd.DataFrame()
    
    def render(self, symbol: str, stock_data: pd.DataFrame, selected_symbols: list = None, chart_columns: int = 1, pre_calculated_data: pd.DataFrame = None):
        logger.info(f"Rendering Best Performers tab")
        
        try:
            # Use pre-calculated data if provided, otherwise get fresh data
            if pre_calculated_data is not None and not pre_calculated_data.empty:
                best_performers = pre_calculated_data.copy()
                logger.info("Using pre-calculated performance data")
            else:
                # Get best performing stocks
                best_performers = self._get_best_performers()
                logger.info("Fetched fresh performance data")
            
            if best_performers.empty:
                st.warning("No performance data available for the last 3 months")
                return
            
            # Filter by selected symbols if provided
            if selected_symbols:
                filtered_performers = best_performers[best_performers['symbol'].isin(selected_symbols)]
                if filtered_performers.empty:
                    st.warning(f"No performance data available for selected symbols: {', '.join(selected_symbols)}")
                    return
                best_performers = filtered_performers
                st.info(f"📊 Showing analysis for {len(selected_symbols)} selected symbols")
            
            # Display chart
            self._display_performance_chart(best_performers)
            
            # Display individual stock charts if symbols are selected
            if selected_symbols:
                st.markdown("---")
                self._display_individual_charts(selected_symbols, chart_columns)
            
            # Display analysis table
            self._display_performance_table(best_performers)
            
            # Display insights
            self._display_insights(best_performers, symbol)
            
        except Exception as e:
            logger.error(f"Error in Best Performers tab: {e}")
            st.error("Unable to load performance data. Please try again later.")
    
    def _get_best_performers(self) -> pd.DataFrame:
        """
        Get best performing stocks for the last 3 months using pandas statistical calculations
        
        Returns:
            DataFrame with performance data by month
        """
        try:
            # Calculate date range for last 3 months
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
            logger.info(f"Fetching all stock data from {start_date.date()} to {end_date.date()}")
            
            # Get ALL stock data for the last 3 months in one query
            # Include minimum price filter from config
            min_price = self.db_manager.get_minimum_price_filter()
            
            query = """
            SELECT p.symbol, date, close, volume
            FROM stocksinfp as p
            LEFT JOIN stock_change_tracker as s ON s.symbol = p.symbol
            WHERE date >= %(start_date)s AND date <= %(end_date)s
            AND close IS NOT NULL AND close != '' AND close != '0'
            AND CAST(close AS DECIMAL(10,2)) >= %(min_price)s
            AND s.is_active = 1
            ORDER BY symbol, date
            """
            
            # Single database call to get all data
            all_data = self.db_manager.execute_query(query, {
                'start_date': start_date,
                'end_date': end_date,
                'min_price': min_price
            })
            
            if all_data.empty:
                logger.warning("No data found for the date range")
                return pd.DataFrame()
            
            logger.info(f"Loaded {len(all_data)} rows of stock data for analysis")
            
            # Convert data types
            all_data['date'] = pd.to_datetime(all_data['date'])
            all_data['close'] = pd.to_numeric(all_data['close'], errors='coerce')
            all_data['volume'] = pd.to_numeric(all_data['volume'], errors='coerce')
            
            # Remove any rows where conversion failed
            all_data = all_data.dropna(subset=['close'])
            all_data = all_data[all_data['close'] > 0]
            
            logger.info(f"After cleaning: {len(all_data)} rows")
            
            # Calculate performance using pandas statistical functions
            performance_results = self._calculate_performance_statistics(all_data, start_date, end_date)
            
            if performance_results.empty:
                logger.warning("No performance results calculated")
                return pd.DataFrame()
            
            # Get top 25 performers
            top_25 = performance_results.nlargest(25, 'total_performance')
            
            logger.info(f"Top 5 performers: {top_25.head()['symbol'].tolist()}")
            logger.info(f"Top performer: {top_25.iloc[0]['symbol']} with {top_25.iloc[0]['total_performance']:.2f}% return")
            
            return top_25
            
        except Exception as e:
            logger.error(f"Error getting best performers: {e}")
            return pd.DataFrame()
    
    def _calculate_performance_statistics(self, data: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Calculate performance statistics using pandas operations
        
        Args:
            data: DataFrame with all stock data
            start_date: Start date for analysis
            end_date: End date for analysis
        
        Returns:
            DataFrame with performance statistics
        """
        try:
            # Group by symbol to calculate per-symbol statistics
            symbol_groups = data.groupby('symbol')
            
            performance_list = []
            
            for symbol, group_data in symbol_groups:
                try:
                    # Ensure data is sorted by date
                    group_data = group_data.sort_values('date')
                    
                    # Need at least 10 data points for meaningful analysis
                    if len(group_data) < 10:
                        continue
                    
                    # Calculate total return (simple approach)
                    first_price = group_data.iloc[0]['close']
                    last_price = group_data.iloc[-1]['close']
                    total_return = ((last_price - first_price) / first_price) * 100
                    
                    # Calculate monthly performance for visualization
                    monthly_performance = self._calculate_monthly_stats(group_data, start_date, end_date)
                    
                    # Create result record
                    result = {
                        'symbol': symbol,
                        'total_performance': total_return,
                        'first_price': first_price,
                        'last_price': last_price,
                        'data_points': len(group_data),
                        'first_date': group_data.iloc[0]['date'].strftime('%Y-%m-%d'),
                        'last_date': group_data.iloc[-1]['date'].strftime('%Y-%m-%d')
                    }
                    
                    # Add monthly performance for visualization
                    result.update(monthly_performance)
                    
                    performance_list.append(result)
                    
                except Exception as e:
                    logger.debug(f"Error processing symbol {symbol}: {e}")
                    continue
            
            if not performance_list:
                return pd.DataFrame()
            
            # Convert to DataFrame
            performance_df = pd.DataFrame(performance_list)
            
            # Filter out any extreme outliers (optional)
            # Remove stocks with returns > 1000% (likely data errors)
            performance_df = performance_df[performance_df['total_performance'] <= 1000]
            
            logger.info(f"Calculated performance for {len(performance_df)} symbols")
            
            return performance_df
            
        except Exception as e:
            logger.error(f"Error in performance statistics calculation: {e}")
            return pd.DataFrame()
    
    def _calculate_monthly_stats(self, group_data: pd.DataFrame, start_date: datetime, end_date: datetime) -> dict:
        """
        Calculate monthly statistics for a single symbol using pandas operations
        
        Args:
            group_data: DataFrame for a single symbol
            start_date: Start date
            end_date: End date
        
        Returns:
            Dictionary with monthly performance data
        """
        try:
            # Add month-year column
            group_data = group_data.copy()
            group_data['month_year'] = group_data['date'].dt.to_period('M')
            
            # Group by month and get first/last prices
            monthly_stats = group_data.groupby('month_year')['close'].agg(['first', 'last', 'count']).reset_index()
            
            # Calculate monthly returns
            monthly_stats['monthly_return'] = ((monthly_stats['last'] - monthly_stats['first']) / monthly_stats['first']) * 100
            
            # Get the last 3 months of data
            monthly_stats = monthly_stats.tail(3)
            
            # Initialize default values
            result = {
                'month_1_pct': 0.0,
                'month_2_pct': 0.0, 
                'month_3_pct': 0.0,
                'month_1_name': 'N/A',
                'month_2_name': 'N/A',
                'month_3_name': 'N/A'
            }
            
            # Fill in actual values (only positive contributions for visualization)
            for i, (_, row) in enumerate(monthly_stats.iterrows()):
                month_key = f'month_{min(i+1, 3)}_pct'
                month_name_key = f'month_{min(i+1, 3)}_name'
                result[month_key] = max(0, row['monthly_return'])  # Only positive contributions
                result[month_name_key] = str(row['month_year'])
            
            return result
            
        except Exception as e:
            logger.debug(f"Error calculating monthly stats: {e}")
            return {
                'month_1_pct': 0.0,
                'month_2_pct': 0.0,
                'month_3_pct': 0.0,
                'month_1_name': 'N/A',
                'month_2_name': 'N/A',
                'month_3_name': 'N/A'
            }
    
    def _display_performance_chart(self, data: pd.DataFrame):
        """
        Display stacked bar chart of performance
        
        Args:
            data: Performance data
        """
        st.subheader("📊 Top 25 Best Performing Stocks - Last 3 Months")
        
        if data.empty:
            st.warning("No data available for chart")
            return
        
        # Sort by total performance
        data = data.sort_values('total_performance', ascending=True)
        
        # Create stacked bar chart
        fig = go.Figure()
        
        # Get month names for legend (use the first row)
        month_names = [
            data.iloc[0]['month_3_name'],
            data.iloc[0]['month_2_name'], 
            data.iloc[0]['month_1_name']
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
        
        # Add bars for each month
        fig.add_trace(go.Bar(
            name=f'Month {month_names[0]}',
            x=data['month_3_pct'],
            y=data['symbol'],
            orientation='h',
            marker_color=colors[0],
            text=[f"{val:.1f}%" for val in data['month_3_pct']],
            textposition='inside',
            textfont=dict(color='white', size=10)
        ))
        
        fig.add_trace(go.Bar(
            name=f'Month {month_names[1]}',
            x=data['month_2_pct'],
            y=data['symbol'],
            orientation='h',
            marker_color=colors[1],
            text=[f"{val:.1f}%" for val in data['month_2_pct']],
            textposition='inside',
            textfont=dict(color='white', size=10)
        ))
        
        fig.add_trace(go.Bar(
            name=f'Month {month_names[2]}',
            x=data['month_1_pct'],
            y=data['symbol'],
            orientation='h',
            marker_color=colors[2],
            text=[f"{val:.1f}%" for val in data['month_1_pct']],
            textposition='inside',
            textfont=dict(color='white', size=10)
        ))
        
        fig.update_layout(
            title='Average Percentage Increase by Stock and Month',
            xaxis_title='Average Percentage Increase',
            yaxis_title='Stock Symbol (ordered by overall increase)',
            barmode='stack',
            height=800,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            margin=dict(r=150),  # Extra margin for legend
            hovermode='y unified'
        )
        
        # Customize axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_performance_table(self, data: pd.DataFrame):
        """
        Display performance data in table format
        
        Args:
            data: Performance data
        """
        st.subheader("📋 Performance Summary Table")
        
        if data.empty:
            st.warning("No data available for table")
            return
        
        # Prepare display data
        display_data = data.copy()
        display_data = display_data.sort_values('total_performance', ascending=False)
        
        # Format the data for display
        formatted_data = {
            'Rank': range(1, len(display_data) + 1),
            'Symbol': display_data['symbol'].tolist(),
            'Total Return (%)': [f"{val:.2f}%" for val in display_data['total_performance']],
            'First Price ($)': [f"${val:.2f}" for val in display_data['first_price']],
            'Last Price ($)': [f"${val:.2f}" for val in display_data['last_price']],
            'Data Points': display_data['data_points'].tolist(),
            'Period': [f"{row['first_date']} to {row['last_date']}" for _, row in display_data.iterrows()]
        }
        
        df_display = pd.DataFrame(formatted_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    def _display_insights(self, data: pd.DataFrame, current_symbol: str):
        """
        Display insights about the performance data
        
        Args:
            data: Performance data
            current_symbol: Currently selected symbol
        """
        st.subheader("💡 Performance Insights")
        
        if data.empty:
            st.warning("No data available for insights")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Top Performers:**")
            top_3 = data.nlargest(3, 'total_performance')
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                st.write(f"{i}. **{row['symbol']}**: {row['total_performance']:.1f}%")
        
        with col2:
            st.markdown("**Monthly Analysis:**")
            month_1_avg = data['month_1_pct'].mean()
            month_2_avg = data['month_2_pct'].mean()
            month_3_avg = data['month_3_pct'].mean()
            
            st.write(f"• Latest Month Avg: {month_1_avg:.1f}%")
            st.write(f"• Middle Month Avg: {month_2_avg:.1f}%")
            st.write(f"• Earliest Month Avg: {month_3_avg:.1f}%")
            
            # Determine best performing month
            month_avgs = [month_3_avg, month_2_avg, month_1_avg]
            month_names = ['Earliest', 'Middle', 'Latest']
            best_month_idx = np.argmax(month_avgs)
            st.write(f"• **Best Month**: {month_names[best_month_idx]}")
        
        with col3:
            st.markdown("**Current Stock Analysis:**")
            if current_symbol in data['symbol'].values:
                current_data = data[data['symbol'] == current_symbol].iloc[0]
                rank = (data['total_performance'] > current_data['total_performance']).sum() + 1
                
                st.success(f"🎯 **{current_symbol} is ranked #{rank} out of top 25!**")
                st.write(f"• Total Gain: {current_data['total_performance']:.1f}%")
                st.write(f"• Best Month: {max(current_data['month_1_pct'], current_data['month_2_pct'], current_data['month_3_pct']):.1f}%")
            else:
                st.info(f"📊 {current_symbol} is not in the top 25 performers")
                st.write("• Consider analyzing top performers")
                st.write("• Look for similar patterns")
        
        # Market insights
        st.markdown("---")
        st.markdown("**📈 Market Insights:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Consistency analysis
            consistent_performers = 0
            for _, row in data.iterrows():
                if row['month_1_pct'] > 5 and row['month_2_pct'] > 5 and row['month_3_pct'] > 5:
                    consistent_performers += 1
            
            st.write(f"• **Consistent Performers** (>5% each month): {consistent_performers}")
            st.write(f"• **Average Total Gain**: {data['total_performance'].mean():.1f}%")
            st.write(f"• **Median Total Gain**: {data['total_performance'].median():.1f}%")
        
        with col2:
            # Volatility insights
            total_gains = data['total_performance']
            st.write(f"• **Highest Gain**: {total_gains.max():.1f}%")
            st.write(f"• **Lowest Gain**: {total_gains.min():.1f}%")
            st.write(f"• **Standard Deviation**: {total_gains.std():.1f}%")
    
    def _display_individual_charts(self, selected_symbols: list, chart_columns: int):
        """Display individual stock charts for selected symbols"""
        st.subheader("📈 Individual Stock Performance Charts")
        st.markdown("**3-Month Price Movement Analysis**")
        
        try:
            # Get price data for selected symbols (using cached version)
            price_data = self._get_cached_price_data_for_symbols(selected_symbols)
            
            if price_data.empty:
                st.warning("No price data available for selected symbols")
                return
            
            # Create individual charts
            self._create_individual_price_charts(price_data, selected_symbols, chart_columns)
            
        except Exception as e:
            logger.error(f"Error displaying individual charts: {e}")
            st.error("Failed to load individual stock charts")
    
    def _reduce_data_for_meaningful_decisions(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce data points to show only important movements for meaningful decision-making.
        
        Strategy:
        1. Always include first and last data points
        2. Include significant peaks and valleys (local min/max)
        3. Include points with major percentage changes (user-configurable threshold)
        4. Include weekly representatives when data is too dense
        5. Apply hard cap of 15 points per symbol for volatile stocks
        6. Use intelligent spacing for very volatile stocks
        
        Args:
            price_data: DataFrame with columns ['symbol', 'date', 'close', etc.]
            
        Returns:
            DataFrame with reduced data points showing only important movements
        """
        try:
            if price_data.empty:
                return price_data
            
            reduced_data_list = []
            
            # Process each symbol separately
            for symbol in price_data['symbol'].unique():
                symbol_data = price_data[price_data['symbol'] == symbol].copy()
                
                if len(symbol_data) <= 10:
                    # If already few data points, keep all
                    reduced_data_list.append(symbol_data)
                    continue
                
                # Sort by date to ensure proper order
                symbol_data = symbol_data.sort_values('date').reset_index(drop=True)
                
                # Initialize with first and last points (always include)
                important_indices = {0, len(symbol_data) - 1}
                
                # Use user-configurable threshold or default to 3%
                threshold = getattr(st.session_state, 'data_reduction_threshold', 3.0)
                
                # Find significant percentage changes
                significant_changes = set()
                for i in range(1, len(symbol_data)):
                    current_price = symbol_data.iloc[i]['close']
                    prev_price = symbol_data.iloc[i-1]['close']
                    pct_change = abs((current_price - prev_price) / prev_price * 100)
                    
                    if pct_change >= threshold:
                        significant_changes.add(i)
                
                # Find local peaks and valleys (important turning points)
                closes = symbol_data['close'].values
                peaks_valleys = set()
                for i in range(1, len(closes) - 1):
                    # Local maximum (peak)
                    if closes[i] > closes[i-1] and closes[i] > closes[i+1]:
                        peaks_valleys.add(i)
                    # Local minimum (valley)
                    elif closes[i] < closes[i-1] and closes[i] < closes[i+1]:
                        peaks_valleys.add(i)
                
                # Combine all important points
                important_indices.update(significant_changes)
                important_indices.update(peaks_valleys)
                
                # For volatile stocks (many significant changes), apply additional filtering
                if len(important_indices) > 20:
                    logger.info(f"Applying volatile stock filtering for {symbol}: {len(important_indices)} initial points")
                    
                    # Check if strict filtering is enabled
                    strict_filtering_enabled = getattr(st.session_state, 'apply_strict_volatile_filtering', True)
                    
                    if strict_filtering_enabled:
                        # Strategy 1: Keep only the most significant changes (top 50%)
                        if len(significant_changes) > 10:
                            significant_list = list(significant_changes)
                            changes_with_magnitude = []
                            
                            for i in significant_list:
                                if i < len(symbol_data) - 1:  # Avoid index error
                                    current_price = symbol_data.iloc[i]['close']
                                    prev_price = symbol_data.iloc[i-1]['close']
                                    pct_change = abs((current_price - prev_price) / prev_price * 100)
                                    changes_with_magnitude.append((i, pct_change))
                            
                            # Sort by magnitude and keep top 50%
                            changes_with_magnitude.sort(key=lambda x: x[1], reverse=True)
                            top_changes = [x[0] for x in changes_with_magnitude[:len(changes_with_magnitude)//2]]
                            
                            # Remove less significant changes from important_indices
                            important_indices = {idx for idx in important_indices if idx not in significant_changes or idx in top_changes}
                        
                        # Strategy 2: Apply intelligent spacing (no points within 2 days of each other)
                        if len(important_indices) > 15:
                            sorted_indices = sorted(list(important_indices))
                            spaced_indices = {0, len(symbol_data) - 1}  # Always keep first and last
                            
                            for i, idx in enumerate(sorted_indices):
                                if idx in {0, len(symbol_data) - 1}:
                                    continue  # Already included
                                
                                # Check if this point is at least 2 days away from any already selected point
                                min_distance = min([abs(idx - selected) for selected in spaced_indices])
                                if min_distance >= 2:
                                    spaced_indices.add(idx)
                                    
                                # Hard cap: stop if we have 15 points
                                if len(spaced_indices) >= 15:
                                    break
                            
                            important_indices = spaced_indices
                    else:
                        # Strict filtering disabled, just apply basic spacing
                        # Keep more points but ensure minimum spacing of 1 day
                        if len(important_indices) > 25:
                            sorted_indices = sorted(list(important_indices))
                            spaced_indices = {0, len(symbol_data) - 1}  # Always keep first and last
                            
                            for i, idx in enumerate(sorted_indices):
                                if idx in {0, len(symbol_data) - 1}:
                                    continue  # Already included
                                
                                # Check if this point is at least 1 day away from any already selected point
                                min_distance = min([abs(idx - selected) for selected in spaced_indices])
                                if min_distance >= 1:
                                    spaced_indices.add(idx)
                                    
                                # Looser cap: stop if we have 25 points
                                if len(spaced_indices) >= 25:
                                    break
                            
                            important_indices = spaced_indices
                
                # If still too many points, apply final filtering with fixed intervals (only if strict filtering enabled)
                strict_filtering_enabled = getattr(st.session_state, 'apply_strict_volatile_filtering', True)
                max_points = 15 if strict_filtering_enabled else 25
                
                if len(important_indices) > max_points:
                    # Calculate interval to get approximately 12-15 points
                    interval = max(2, len(symbol_data) // 12)
                    
                    # Keep first, last, and evenly spaced points
                    final_indices = {0, len(symbol_data) - 1}
                    for i in range(interval, len(symbol_data) - interval, interval):
                        final_indices.add(i)
                    
                    # Add a few of the most significant peaks/valleys if we have room
                    remaining_space = max_points - len(final_indices)
                    if remaining_space > 0 and peaks_valleys:
                        sorted_peaks_valleys = sorted(list(peaks_valleys))[:remaining_space]
                        final_indices.update(sorted_peaks_valleys)
                    
                    important_indices = final_indices
                
                # If still too few points, add some intermediate points for trend visibility
                if len(important_indices) < 8:
                    # Add some evenly distributed points
                    step = max(1, len(symbol_data) // 8)
                    for i in range(0, len(symbol_data), step):
                        important_indices.add(i)
                
                # Convert to sorted list and get the important data points
                important_indices = sorted(list(important_indices))
                reduced_symbol_data = symbol_data.iloc[important_indices].copy()
                
                # Add a flag to indicate this is reduced data
                reduced_symbol_data['is_reduced'] = True
                reduced_symbol_data['original_count'] = len(symbol_data)
                reduced_symbol_data['reduced_count'] = len(reduced_symbol_data)
                
                reduced_data_list.append(reduced_symbol_data)
                
                logger.info(f"Reduced {symbol} data from {len(symbol_data)} to {len(reduced_symbol_data)} points")
            
            # Combine all reduced data
            if reduced_data_list:
                result = pd.concat(reduced_data_list, ignore_index=True)
                logger.info(f"Total data reduction: {len(price_data)} -> {len(result)} points")
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in data reduction: {e}")
            # Return original data if reduction fails
            return price_data
    
    def _create_individual_price_charts(self, price_data: pd.DataFrame, symbols: list, ncols: int):
        """Create individual matplotlib charts for each symbol using centralized chart utils"""
        try:
            # Apply data reduction to show only meaningful movements
            reduced_price_data = self._reduce_data_for_meaningful_decisions(price_data)
            
            if reduced_price_data.empty:
                st.warning("No data available after reduction")
                return
            
            # Show data reduction info
            if not reduced_price_data.empty and 'original_count' in reduced_price_data.columns:
                original_total = reduced_price_data['original_count'].sum()
                reduced_total = len(reduced_price_data)
                reduction_pct = (1 - reduced_total/max(original_total, 1)) * 100
                
                # Check if we applied volatile stock filtering
                strict_filtering_applied = getattr(st.session_state, 'apply_strict_volatile_filtering', True)
                
                # Count how many symbols were heavily reduced (>70% reduction)
                heavily_reduced_stocks = []
                for symbol in reduced_price_data['symbol'].unique():
                    symbol_reduced = reduced_price_data[reduced_price_data['symbol'] == symbol]
                    if not symbol_reduced.empty:
                        original_count = symbol_reduced.iloc[0]['original_count']
                        reduced_count = len(symbol_reduced)
                        if original_count > 0 and (1 - reduced_count/original_count) > 0.7:
                            heavily_reduced_stocks.append(symbol)
                
                # Create info message
                info_msg = f"📊 **Data Optimization**: Showing {reduced_total} key data points out of {original_total} total points ({reduction_pct:.1f}% reduction) to highlight important price movements for better decision-making."
                
                if heavily_reduced_stocks and strict_filtering_applied:
                    info_msg += f"\n\n⚡ **Volatile Stock Filtering Applied**: Extra filtering applied to {len(heavily_reduced_stocks)} volatile stocks ({', '.join(heavily_reduced_stocks[:3])}{', ...' if len(heavily_reduced_stocks) > 3 else ''}) to maintain chart readability."
                
                st.info(info_msg)
            
            # Convert DataFrame to dict for caching (Streamlit can't cache DataFrames with complex objects)
            price_data_dict = reduced_price_data.to_dict('records') if not reduced_price_data.empty else {}
            
            # Use centralized chart utils for figure generation
            fig = ChartUtils.create_cached_matplotlib_figure(
                'individual', price_data_dict, symbols, ncols
            )
            
            if fig is not None:
                # Display the cached figure
                st.pyplot(fig, clear_figure=True)
                
                # Add explanatory note about the data reduction
                st.markdown(f"""
                **📈 Chart Information:**
                - Charts show **key price movements only** for clearer decision-making
                - Includes: Start/end points, significant changes (>{st.session_state.get('data_reduction_threshold', 3.0):.1f}%), peaks & valleys
                - **Green annotations**: Price increases | **Red annotations**: Price decreases
                - Focus on trend patterns rather than daily noise for investment decisions
                - **Tip**: Adjust "Chart detail level" in sidebar to show more/fewer data points
                """)
            else:
                st.error("Failed to create individual stock charts")
            
        except Exception as e:
            logger.error(f"Error creating individual charts: {e}")
            st.error("Failed to create individual stock charts")
    
    # Chart creation methods moved to centralized utils/chart_utils.py 