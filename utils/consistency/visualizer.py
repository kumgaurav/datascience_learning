"""
Consistency Visualizer Utility

Handles all chart creation and data visualization for consistency analysis.
Extracted from ConsistentPerformersTab for better separation of concerns.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from utils.chart_utils import ChartUtils

# Configuration constants
TOP_PERFORMERS_COUNT = 25
MAX_CHART_SYMBOLS = 15
CHART_FIGURE_SIZE = (18, 6)  # Width, height per row
CHART_SPACING = {'pad': 3.0, 'h_pad': 4.0, 'w_pad': 3.0}
MAX_CHART_TICKS = 6

# Custom Exception Classes
class ChartCreationError(Exception):
    """Raised when chart/visualization creation fails"""
    pass

logger = logging.getLogger('StockApp')


class ConsistencyVisualizer:
    """
    Handles all visualization and chart creation for consistency analysis.
    
    Responsibilities:
    - Plotly chart creation
    - Matplotlib figure generation
    - Data reduction for charts
    - Chart layout and formatting
    """
    
    def display_consistency_chart(self, data: pd.DataFrame, months: int = 3) -> None:
        """
        Display consistency chart with monthly performance.
        
        Args:
            data: Consistency data with performance metrics
            months: Number of months analyzed (1, 2, 3, or 0 for all-time)
        """
        # Handle all-time performers case
        if months == 0:
            st.subheader(f"📊 All-Time Champions - Elite Consistency Across All Periods")
            st.caption("Ranked by Average Consistency Index: stocks appearing in ALL 3 time periods")
            period_text = "All-Time"
        else:
            period_text = "Month" if months == 1 else "Months"
            st.subheader(f"📊 Top {TOP_PERFORMERS_COUNT} Most Consistent Performers - Last {months} {period_text}")
            st.caption("Ranked by Consistency Index: combines positive months ratio, low volatility, and geometric mean return")
        
        if data.empty:
            st.warning("No data available for chart")
            return
        
        try:
            # Sort by total return (highest to lowest for chart display)
            data = data.sort_values('total_return', ascending=True)
            
            # Create chart showing monthly returns
            fig = go.Figure()
            
            # Get month names for legend - only show months that are relevant for the period
            month_names = []
            month_columns = []
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
            
            # Determine which months to show based on analysis period
            if months == 0:  # All-time performers
                # For all-time, show 3-month data but with different labels
                month_names = ['3M Return', '2M Return', '1M Return']
                month_columns = ['total_return_3m', 'total_return_2m', 'total_return_1m']
            elif months >= 3:
                month_names = [data.iloc[0]['month_3_name'], data.iloc[0]['month_2_name'], data.iloc[0]['month_1_name']]
                month_columns = ['month_3_pct', 'month_2_pct', 'month_1_pct']
            elif months == 2:
                month_names = [data.iloc[0]['month_2_name'], data.iloc[0]['month_1_name']]
                month_columns = ['month_2_pct', 'month_1_pct']
                colors = colors[:2]
            else:  # months == 1
                month_names = [data.iloc[0]['month_1_name']]
                month_columns = ['month_1_pct']
                colors = colors[:1]
            
            # Add bars for each month
            for i, (month_col, month_name, color) in enumerate(zip(month_columns, month_names, colors)):
                fig.add_trace(go.Bar(
                    name=f'{month_name}',
                    x=data[month_col],
                    y=data['symbol'],
                    orientation='h',
                    marker_color=color,
                    text=[f"{val:.1f}%" for val in data[month_col]],
                    textposition='inside',
                    textfont=dict(color='white', size=9),
                    customdata=list(zip(data['consistency_index'], data['total_return'], data['positive_months_ratio'])),
                    hovertemplate=f'<b>%{{y}}</b><br>' +
                                 f'{month_name}: %{{x:.1f}}%<br>' +
                                 'Consistency Index: %{customdata[0]:.3f}<br>' +
                                 'Total Return: %{customdata[1]:.1f}%<br>' +
                                 'Positive Months: %{customdata[2]:.0%}<extra></extra>'
                ))
            
            # Add consistency index annotations
            max_values = []
            for month_col in month_columns:
                max_values.extend(data[month_col].tolist())
            max_x = max(max_values) if max_values else 0
            
            for i, row in data.iterrows():
                fig.add_annotation(
                    x=max_x + 5,
                    y=row['symbol'],
                    text=f"CI: {row['consistency_index']:.3f}<br>TR: {row['total_return']:.0f}%",
                    showarrow=False,
                    font=dict(size=8, color='black'),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='gray',
                    borderwidth=1,
                    xanchor='left'
                )
            
            if months == 0:
                chart_title = 'Cross-Period Performance of All-Time Champions'
            elif months == 1:
                chart_title = 'Monthly Performance of Most Consistent Stocks'
            else:
                chart_title = f'{period_text[:-1]}ly Performance of Most Consistent Stocks'
            
            fig.update_layout(
                title=f'{chart_title}<br><sub>Ordered by Total Return (TR) • Consistency Index (CI) shown on right</sub>',
                xaxis_title='Monthly Return (%)',
                yaxis_title='Stock Symbol (ordered by Total Return)',
                barmode='group',
                height=800,
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                margin=dict(r=200),
                hovermode='y unified'
            )
            
            # Add vertical line at 0%
            fig.add_vline(x=0, line_dash="dash", line_color="red", 
                         annotation_text="0% Return", annotation_position="top")
            
            # Customize axes
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=True, zerolinecolor='red')
            fig.update_yaxes(showgrid=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            self._display_chart_explanation(months)
            
        except Exception as e:
            logger.error(f"Error creating consistency chart: {e}")
            raise ChartCreationError(f"Failed to create consistency chart: {e}") from e
    
    def create_individual_price_charts(self, price_data: pd.DataFrame, symbols: List[str], chart_columns: int, months: int = 3, consistency_data: Optional[pd.DataFrame] = None) -> None:
        """
        Create individual matplotlib charts for each symbol using centralized chart utils.
        
        Args:
            price_data: Price data for all symbols
            symbols: List of symbols to create charts for
            chart_columns: Number of columns in chart grid
            months: Number of months analyzed (1, 2, or 3)
            consistency_data: Optional consistency data for sorting by return
        """
        try:
            from utils.chart_utils import ChartUtils
            
            # Sort symbols by total return if consistency data is available
            if consistency_data is not None and not consistency_data.empty:
                # Filter consistency data for the selected symbols
                selected_returns = consistency_data[consistency_data['symbol'].isin(symbols)][['symbol', 'total_return']]
                if not selected_returns.empty:
                    # Sort by total return (highest first) for proper chart ordering
                    symbols_ordered_by_return = selected_returns.sort_values('total_return', ascending=False)['symbol'].tolist()
                    logger.info(f"[create_individual_price_charts] Sorted symbols by total return: {symbols_ordered_by_return}")
                    symbols = symbols_ordered_by_return
                else:
                    logger.warning(f"[create_individual_price_charts] No consistency data found for selected symbols: {symbols}")
            else:
                logger.info(f"[create_individual_price_charts] No consistency data provided, using original symbol order: {symbols}")
            
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
                period_text = f"{months} {'month' if months == 1 else 'months'}"
                info_msg = f"📊 **Data Optimization**: Showing {reduced_total} key data points out of {original_total} total points ({reduction_pct:.1f}% reduction) to highlight important price movements for {period_text} consistency analysis."
                
                if heavily_reduced_stocks and strict_filtering_applied:
                    info_msg += f"\n\n⚡ **Volatile Stock Filtering Applied**: Extra filtering applied to {len(heavily_reduced_stocks)} volatile stocks ({', '.join(heavily_reduced_stocks[:3])}{', ...' if len(heavily_reduced_stocks) > 3 else ''}) to maintain chart readability."
                
                st.info(info_msg)
            
            # Convert DataFrame to dict for caching (Streamlit can't cache DataFrames with complex objects)
            price_data_dict = reduced_price_data.to_dict('records') if not reduced_price_data.empty else {}
            
            # Use centralized chart utils for figure generation
            try:
                fig = ChartUtils.create_cached_matplotlib_figure(
                    'individual', price_data_dict, symbols, chart_columns
                )
            except Exception as chart_error:
                logger.error(f"Chart creation failed: {chart_error}")
                st.error(f"Failed to create matplotlib figure: {chart_error}")
                return
            
            if fig is not None:
                # Display the cached figure
                st.pyplot(fig, clear_figure=True)
                
                # Add explanatory note about the data reduction
                st.markdown(f"""
                **📈 Consistency Chart Information:**
                - Charts show **key price movements only** for clearer consistency analysis over {months} {'month' if months == 1 else 'months'}
                - Includes: Start/end points, significant changes (>{st.session_state.get('data_reduction_threshold', 3.0):.1f}%), peaks & valleys
                - **Green annotations**: Price increases | **Red annotations**: Price decreases
                - Focus on trend stability and consistent growth patterns for investment decisions
                - **Tip**: Adjust "Chart detail level" in sidebar to show more/fewer data points
                """)
            else:
                st.error("Failed to create individual stock charts")
            
        except Exception as e:
            logger.error(f"Unexpected error creating individual charts: {e}")
            st.error("An unexpected error occurred while creating charts. Please try again.")
    
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
                    logger.debug(f"[_reduce_data_for_meaningful_decisions] Applying volatile stock filtering for {symbol}: {len(important_indices)} initial points")
                    
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
                
                logger.debug(f"[_reduce_data_for_meaningful_decisions] Reduced {symbol} data from {len(symbol_data)} to {len(reduced_symbol_data)} points")
            
            # Combine all reduced data
            if reduced_data_list:
                result = pd.concat(reduced_data_list, ignore_index=True)
                logger.info(f"[_reduce_data_for_meaningful_decisions] Total data reduction: {len(price_data)} -> {len(result)} points")
                return result
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in data reduction: {e}")
            # Return original data if reduction fails
            return price_data
    
    def _display_chart_explanation(self, months: int) -> None:
        """Display explanation for the consistency chart."""
        if months == 0:  # All-time performers
            st.info(f"""
            **📈 How to Read This All-Time Champions Chart:**
            - **Elite stocks appearing in ALL 3 time periods** (1M, 2M, 3M)
            - **Ordered by Average Consistency Index** (highest consistency at top)
            - Each colored bar shows the **total return** for that specific period
            - **TR**: 3-Month Total Return (primary sort for display)
            - **CI**: Average Consistency Index across all periods
            - **Green bars** = 3-month returns, **Orange bars** = 2-month returns, **Blue bars** = 1-month returns
            - Shows which all-time performers have the **best cross-period performance**
            
            **📋 Elite Selection Process:**
            1. **Step 1**: Find top performers in each period individually (1M, 2M, 3M)
            2. **Step 2**: Take intersection - only stocks appearing in ALL periods
            3. **Step 3**: Calculate average consistency across all periods
            4. **Step 4**: Sort by average consistency (highest first)
            
            **📋 All-Time Champion Criteria:**
            - **Must appear in ALL 3 time periods** (strictest filter)
            - **Must pass individual consistency analysis** for each period
            - **Represents the most stable, reliable performers** across all timeframes
            """)
        else:
            period_text = "months" if months > 1 else "month"
            positive_text = "ALL positive bars" if months > 1 else "positive bar"
            performance_text = f"ALL {months} {period_text}" if months > 1 else "the current month"
            
            st.info(f"""
            **📈 How to Read This Chart:**
            - **Top 25 most consistent stocks**, now **ordered by Total Return (TR)** (highest returns at top)
            - Each colored bar shows the monthly return for that month with percentage
            - **TR**: Total Return over {months} {period_text} (primary sort) 
            - **CI**: Consistency Index (0-1 scale, higher = more consistent)
            - Stocks with **{positive_text}** performed well in {performance_text}
            - This shows which consistent performers also have the **highest total returns**
            
            **📋 Selection Process:**
            1. **Step 1**: Find top 25 stocks by Consistency Index (statistical model)
            2. **Step 2**: Sort these 25 by Total Return (highest first)
            
            **📋 Eligibility Criteria:**
            - Only stocks with **positive total returns** and **at least 1 positive month** are included
            - Must pass consistency analysis with sufficient data points
            """)
    
    def _display_data_reduction_info(self, reduced_price_data: pd.DataFrame, months: int) -> None:
        """Display information about data reduction applied."""
        if not reduced_price_data.empty and 'original_count' in reduced_price_data.columns:
            original_total = reduced_price_data['original_count'].sum()
            reduced_total = len(reduced_price_data)
            reduction_pct = (1 - reduced_total/max(original_total, 1)) * 100
            
            strict_filtering_applied = getattr(st.session_state, 'apply_strict_volatile_filtering', True)
            
            # Count heavily reduced stocks
            heavily_reduced_stocks = []
            for symbol in reduced_price_data['symbol'].unique():
                symbol_reduced = reduced_price_data[reduced_price_data['symbol'] == symbol]
                if not symbol_reduced.empty:
                    original_count = symbol_reduced.iloc[0]['original_count']
                    reduced_count = len(symbol_reduced)
                    if original_count > 0 and (1 - reduced_count/original_count) > 0.7:
                        heavily_reduced_stocks.append(symbol)
            
            period_text = f"{months} {'month' if months == 1 else 'months'}"
            info_msg = f"📊 **Data Optimization**: Showing {reduced_total} key data points out of {original_total} total points ({reduction_pct:.1f}% reduction) to highlight important price movements for {period_text} consistency analysis."
            
            if heavily_reduced_stocks and strict_filtering_applied:
                info_msg += f"\n\n⚡ **Volatile Stock Filtering Applied**: Extra filtering applied to {len(heavily_reduced_stocks)} volatile stocks ({', '.join(heavily_reduced_stocks[:3])}{', ...' if len(heavily_reduced_stocks) > 3 else ''}) to maintain chart readability."
            
            st.info(info_msg)
    
    def _display_chart_information(self, months: int) -> None:
        """Display information about the consistency charts."""
        if months == 0:  # All-time performers
            st.markdown(f"""
            **📈 All-Time Champions Chart Information:**
            - Charts show **key price movements only** for clearer consistency analysis over the 3-month period
            - **These are the elite stocks** that appear in ALL 3 time periods (1M, 2M, 3M)
            - Includes: Start/end points, significant changes (>{st.session_state.get('data_reduction_threshold', 3.0):.1f}%), peaks & valleys
            - **Green annotations**: Price increases | **Red annotations**: Price decreases
            - Focus on **cross-period stability** and consistent growth patterns for premium investment decisions
            - **Tip**: These represent the most reliable, stable performers across all timeframes
            """)
        else:
            st.markdown(f"""
            **📈 Consistency Chart Information:**
            - Charts show **key price movements only** for clearer consistency analysis over {months} {'month' if months == 1 else 'months'}
            - Includes: Start/end points, significant changes (>{st.session_state.get('data_reduction_threshold', 3.0):.1f}%), peaks & valleys
            - **Green annotations**: Price increases | **Red annotations**: Price decreases
            - Focus on trend stability and consistent growth patterns for investment decisions
            - **Tip**: Adjust "Chart detail level" in sidebar to show more/fewer data points
            """) 