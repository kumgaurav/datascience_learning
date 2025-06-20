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

logger = logging.getLogger('StockApp')

class BestPerformersTab:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tech_indicators = TechnicalIndicators()
    
    def render(self, symbol: str, stock_data: pd.DataFrame):
        logger.info(f"Rendering Best Performers tab")
        
        try:
            # Get best performing stocks
            best_performers = self._get_best_performers()
            
            if best_performers.empty:
                st.warning("No performance data available for the last 3 months")
                return
            
            # Display chart
            self._display_performance_chart(best_performers)
            
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
            query = """
            SELECT p.symbol, date, close, volume
            FROM stocksinfp as p
            LEFT JOIN stock_change_tracker as s ON s.symbol = p.symbol
            WHERE date >= %(start_date)s AND date <= %(end_date)s
            AND close IS NOT NULL AND close != '' AND close != '0'
            AND s.is_active = 1
            ORDER BY symbol, date
            """
            
            # Single database call to get all data
            all_data = self.db_manager.execute_query(query, {
                'start_date': start_date,
                'end_date': end_date
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