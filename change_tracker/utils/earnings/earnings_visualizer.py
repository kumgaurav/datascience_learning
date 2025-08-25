"""
Earnings Visualizer Utility

Handles all chart creation and visualization for earnings analysis.
Extracted from EarningsStocksTab for better separation of concerns.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import numpy as np

# Custom Exception Classes
class EarningsVisualizationError(Exception):
    """Raised when earnings visualization creation fails"""
    pass

logger = logging.getLogger('StockApp')


class EarningsVisualizer:
    """
    Handles all chart creation and visualization for earnings analysis.
    
    Responsibilities:
    - Price charts for earnings stocks
    - Volatility visualization
    - Earnings calendar charts
    - Performance comparison charts
    """
    
    def __init__(self):
        """Initialize the earnings visualizer."""
        logger.debug("[EarningsVisualizer] Initialized")
        
        # Set matplotlib backend for Streamlit
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
        except Exception as e:
            logger.warning(f"Could not set matplotlib backend: {e}")
    
    def create_earnings_price_charts(self, price_data: pd.DataFrame, selected_symbols: List[str], 
                                   chart_columns: int = 1, weeks: int = 3, 
                                   earnings_data: Optional[pd.DataFrame] = None) -> None:
        """
        Create individual earnings price charts for selected symbols.
        
        Args:
            price_data: DataFrame with price data
            selected_symbols: List of symbols to chart
            chart_columns: Number of columns in chart grid
            weeks: Number of weeks of data
            earnings_data: Optional earnings data for annotation
        """
        try:
            if price_data.empty or not selected_symbols:
                st.warning("⚠️ No price data available for selected symbols")
                return
            
            logger.debug(f"[create_earnings_price_charts] Creating individual charts for {len(selected_symbols)} symbols")
            
            # Limit symbols for performance
            MAX_INDIVIDUAL_CHARTS = 20
            if len(selected_symbols) > MAX_INDIVIDUAL_CHARTS:
                st.warning(f"⚠️ Too many symbols selected ({len(selected_symbols)}). "
                          f"Showing only the first {MAX_INDIVIDUAL_CHARTS} symbols to prevent memory issues.")
                st.info(f"💡 **Tip**: Select fewer symbols for individual chart analysis. "
                        f"Current limit: {MAX_INDIVIDUAL_CHARTS} charts maximum.")
                selected_symbols = selected_symbols[:MAX_INDIVIDUAL_CHARTS]
            
            # Sort symbols by returns using pre-calculated values if available
            if earnings_data is not None and not earnings_data.empty and 'total_return' in earnings_data.columns:
                # Use pre-calculated returns from earnings data for accurate sorting
                logger.debug("[create_earnings_price_charts] Using pre-calculated returns for sorting")
                logger.debug(f"[create_earnings_price_charts] Earnings data contains {len(earnings_data)} records")
                logger.debug(f"[create_earnings_price_charts] Earnings data columns: {list(earnings_data.columns)}")
                
                # Show sample of earnings data
                if len(earnings_data) > 0:
                    logger.debug("[create_earnings_price_charts] Sample earnings data:")
                    for idx, row in earnings_data.head(3).iterrows():
                        if 'symbol' in row and 'total_return' in row:
                            logger.debug(f"[create_earnings_price_charts] Sample: {row['symbol']}: {row['total_return']:.2f}%")
                
                sorted_symbols = self._sort_symbols_by_precalculated_returns(selected_symbols, earnings_data)
            else:
                # Fallback to price data calculation (less accurate)
                logger.warning("[create_earnings_price_charts] Using price data calculation for sorting - this may be inaccurate!")
                if earnings_data is None:
                    logger.warning("[create_earnings_price_charts] earnings_data is None")
                elif earnings_data.empty:
                    logger.warning("[create_earnings_price_charts] earnings_data is empty")
                elif 'total_return' not in earnings_data.columns:
                    logger.warning(f"[create_earnings_price_charts] total_return not in earnings_data columns: {list(earnings_data.columns) if earnings_data is not None else 'None'}")
                
                sorted_symbols = self._sort_symbols_by_returns(selected_symbols, price_data)
            
            # Create individual matplotlib charts
            self._create_individual_earnings_charts(price_data, sorted_symbols, chart_columns, earnings_data)
            
            # Add chart information
            st.markdown(f"""
            **📈 Individual Earnings Stock Charts Information:**
            - Charts show **{weeks}-week price movements** leading up to earnings
            - **Blue line with markers**: Stock price trend
            - **Price annotations**: Color-coded (green=up, red=down, black=unchanged)
            - **Title includes**: Max/Min prices, earnings date, and total return
            - **Charts are sorted by returns** (highest returns first)
            - **Useful for**: Analyzing pre-earnings price momentum and volatility
            """)
            
        except Exception as e:
            logger.error(f"Error creating earnings price charts: {e}")
            st.error(f"Failed to create earnings price charts: {e}")
    
    def create_volatility_distribution_chart(self, earnings_data: pd.DataFrame) -> None:
        """
        Create a volatility distribution chart for earnings stocks.
        
        Args:
            earnings_data: DataFrame with volatility metrics
        """
        try:
            if earnings_data.empty or 'volatility_score' not in earnings_data.columns:
                st.warning("⚠️ No volatility data available")
                return
            
            logger.debug(f"[create_volatility_distribution_chart] Creating volatility chart for {len(earnings_data)} stocks")
            
            # Clean and validate data
            clean_data = earnings_data.copy()
            
            # Replace infinite values with NaN
            clean_data = clean_data.replace([np.inf, -np.inf], np.nan)
            
            # Ensure minimum data requirements
            if len(clean_data.dropna(subset=['volatility_score'])) < 2:
                st.warning("⚠️ Insufficient data for volatility analysis")
                return
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Volatility distribution histogram
            volatility_scores = clean_data['volatility_score'].dropna()
            
            if len(volatility_scores) > 0:
                # Handle edge case where all values are the same
                if volatility_scores.nunique() == 1:
                    # Single value - show as single bar
                    ax1.bar([volatility_scores.iloc[0]], [len(volatility_scores)], 
                           width=0.1, alpha=0.7, color='skyblue', edgecolor='black')
                else:
                    # Multiple values - show histogram
                    ax1.hist(volatility_scores, bins=min(20, len(volatility_scores)), 
                            edgecolor='black', alpha=0.7, color='skyblue')
                
                ax1.set_title('Volatility Score Distribution', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Volatility Score')
                ax1.set_ylabel('Number of Stocks')
                ax1.grid(True, alpha=0.3)
            
            # Returns vs Volatility scatter plot
            if 'total_return' in clean_data.columns:
                # Get clean data for both columns
                scatter_data = clean_data.dropna(subset=['total_return', 'volatility_score'])
                
                if len(scatter_data) > 0:
                    x_vals = scatter_data['volatility_score']
                    y_vals = scatter_data['total_return']
                    
                    ax2.scatter(x_vals, y_vals, alpha=0.6, color='coral')
                    ax2.set_title('Returns vs Volatility', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Volatility Score')
                    ax2.set_ylabel('Total Return (%)')
                    ax2.grid(True, alpha=0.3)
                    
                    # Add trend line with robust error handling
                    if len(x_vals) > 1 and len(np.unique(x_vals)) > 1:
                        try:
                            # Additional validation for polynomial fit
                            x_clean = np.array(x_vals)
                            y_clean = np.array(y_vals)
                            
                            # Check for numerical issues
                            if (not np.any(np.isnan(x_clean)) and not np.any(np.isnan(y_clean)) and
                                not np.any(np.isinf(x_clean)) and not np.any(np.isinf(y_clean)) and
                                np.std(x_clean) > 1e-10):  # Ensure sufficient variation
                                
                                z = np.polyfit(x_clean, y_clean, 1)
                                p = np.poly1d(z)
                                ax2.plot(x_clean, p(x_clean), "r--", alpha=0.8, linewidth=2)
                            else:
                                logger.warning("Data not suitable for trend line fitting")
                        except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
                            logger.warning(f"Could not fit trend line: {e}")
                            # Continue without trend line
                else:
                    ax2.text(0.5, 0.5, 'No valid data for scatter plot', 
                            transform=ax2.transAxes, ha='center', va='center')
            
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            
        except Exception as e:
            logger.error(f"Error creating volatility distribution chart: {e}")
            st.error(f"Failed to create volatility distribution chart: {e}")
    
    def create_earnings_calendar_chart(self, categorized_earnings: Dict[str, pd.DataFrame]) -> None:
        """
        Create a visual earnings calendar chart.
        
        Args:
            categorized_earnings: Dictionary with categorized earnings data
        """
        try:
            logger.debug("[create_earnings_calendar_chart] Creating earnings calendar chart")
            
            # Count earnings by week
            week_counts = {}
            week_labels = {
                'current_week': 'Current Week',
                'next_week': 'Next Week', 
                'week_after': 'Week After',
                'later': 'Later (3+ weeks)'
            }
            
            for week, data in categorized_earnings.items():
                week_counts[week_labels.get(week, week)] = len(data)
            
            if not any(week_counts.values()):
                st.info("📅 No upcoming earnings in the next few weeks")
                return
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            weeks = list(week_counts.keys())
            counts = list(week_counts.values())
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            bars = ax.bar(weeks, counts, color=colors[:len(weeks)], alpha=0.8, edgecolor='black')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            ax.set_title('Upcoming Earnings Calendar', fontsize=16, fontweight='bold', pad=20)
            ax.set_ylabel('Number of Stocks', fontsize=12)
            ax.set_xlabel('Week', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Style improvements
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig, clear_figure=True)
            
        except Exception as e:
            logger.error(f"Error creating earnings calendar chart: {e}")
            st.error(f"Failed to create earnings calendar chart: {e}")
    
    def _sort_symbols_by_returns(self, symbols: List[str], price_data: pd.DataFrame) -> List[str]:
        """
        Sort symbols by their total returns (highest first).
        
        Args:
            symbols: List of symbols to sort
            price_data: DataFrame with price data
            
        Returns:
            List of symbols sorted by returns
        """
        try:
            symbol_returns = []
            
            for symbol in symbols:
                symbol_data = price_data[price_data['symbol'] == symbol]
                if not symbol_data.empty:
                    symbol_data = symbol_data.sort_values('date')
                    first_price = symbol_data.iloc[0]['close']
                    last_price = symbol_data.iloc[-1]['close']
                    total_return = ((last_price - first_price) / first_price) * 100
                    symbol_returns.append((symbol, total_return))
                else:
                    symbol_returns.append((symbol, 0))
            
            # Sort by returns (highest first)
            symbol_returns.sort(key=lambda x: x[1], reverse=True)
            
            return [symbol for symbol, _ in symbol_returns]
            
        except Exception as e:
            logger.error(f"Error sorting symbols by returns: {e}")
            return symbols

    def _sort_symbols_by_precalculated_returns(self, symbols: List[str], earnings_data: pd.DataFrame) -> List[str]:
        """
        Sort symbols by their pre-calculated total returns from earnings data (highest first).
        This is more accurate than recalculating from price data.
        
        Args:
            symbols: List of symbols to sort
            earnings_data: DataFrame with earnings data containing total_return column
            
        Returns:
            List of symbols sorted by pre-calculated returns
        """
        try:
            logger.debug(f"[_sort_symbols_by_precalculated_returns] Starting sort for {len(symbols)} symbols: {symbols}")
            
            # Debug: Check if earnings_data has the required columns
            if 'total_return' not in earnings_data.columns:
                logger.error("[_sort_symbols_by_precalculated_returns] total_return column missing from earnings_data")
                logger.debug(f"[_sort_symbols_by_precalculated_returns] Available columns: {list(earnings_data.columns)}")
                return symbols
            
            # Debug: Show earnings data for all symbols
            logger.debug(f"[_sort_symbols_by_precalculated_returns] Earnings data shape: {earnings_data.shape}")
            for symbol in symbols:
                symbol_earnings = earnings_data[earnings_data['symbol'] == symbol]
                if not symbol_earnings.empty:
                    total_return = symbol_earnings['total_return'].iloc[0]
                    logger.debug(f"[_sort_symbols_by_precalculated_returns] {symbol}: total_return = {total_return}")
                else:
                    logger.warning(f"[_sort_symbols_by_precalculated_returns] {symbol}: NO DATA FOUND in earnings_data")
            
            symbol_returns = []
            
            for symbol in symbols:
                # Get pre-calculated return from earnings data
                symbol_earnings = earnings_data[earnings_data['symbol'] == symbol]
                if not symbol_earnings.empty and 'total_return' in symbol_earnings.columns:
                    total_return = symbol_earnings['total_return'].iloc[0]
                    if pd.notna(total_return):
                        symbol_returns.append((symbol, float(total_return)))
                        logger.debug(f"[_sort_symbols_by_precalculated_returns] Added {symbol} with return {total_return:.2f}%")
                    else:
                        symbol_returns.append((symbol, 0.0))
                        logger.warning(f"[_sort_symbols_by_precalculated_returns] {symbol} has NaN return, using 0.0")
                else:
                    symbol_returns.append((symbol, 0.0))
                    logger.warning(f"[_sort_symbols_by_precalculated_returns] {symbol} not found or missing total_return, using 0.0")
            
            # Sort by pre-calculated returns (highest first)
            symbol_returns.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"[_sort_symbols_by_precalculated_returns] SORTED ORDER:")
            for i, (symbol, return_val) in enumerate(symbol_returns, 1):
                logger.debug(f"[_sort_symbols_by_precalculated_returns] {i}. {symbol}: {return_val:.2f}%")
            
            sorted_symbols = [symbol for symbol, _ in symbol_returns]
            logger.debug(f"[_sort_symbols_by_precalculated_returns] Final sorted symbols: {sorted_symbols}")
            
            return sorted_symbols
            
        except Exception as e:
            logger.error(f"Error sorting symbols by pre-calculated returns: {e}")
            logger.exception("Full traceback:")
            return symbols
    
    def _create_individual_earnings_charts(self, price_data: pd.DataFrame, symbols: List[str], 
                                         chart_columns: int, earnings_data: Optional[pd.DataFrame] = None) -> None:
        """
        Create individual matplotlib charts for each earnings symbol.
        
        Args:
            price_data: DataFrame with price data
            symbols: List of symbols to chart
            chart_columns: Number of columns in grid
            earnings_data: Optional earnings data for annotation
        """
        try:
            if price_data.empty or not symbols:
                st.warning("⚠️ No data available for chart creation")
                return
            
            logger.debug(f"[_create_individual_earnings_charts] Creating {len(symbols)} individual charts")
            
            # Calculate grid dimensions
            n_charts = len(symbols)
            n_rows = (n_charts + chart_columns - 1) // chart_columns
            
            # Calculate figure size with safety limits
            MAX_FIGURE_WIDTH = 25
            MAX_FIGURE_HEIGHT = 100
            
            fig_width = min(18, MAX_FIGURE_WIDTH)
            base_height = 5
            calculated_height = base_height * n_rows
            fig_height = min(calculated_height, MAX_FIGURE_HEIGHT)
            
            # Log size information for debugging
            logger.debug(f"Creating individual earnings charts: {n_rows}x{chart_columns}, "
                         f"symbols: {len(symbols)}, "
                         f"calculated size: {fig_width}x{calculated_height}, "
                         f"final size: {fig_width}x{fig_height}")
            
            # Warn user if figure size was limited
            if calculated_height > MAX_FIGURE_HEIGHT:
                st.warning(f"⚠️ Chart size limited to prevent memory issues. "
                          f"Displaying {len(symbols)} charts in a compressed layout.")
                st.info("💡 **Tip**: For better visibility, select fewer symbols or increase chart columns.")
            
            # Create figure with size limits
            try:
                fig, axes = plt.subplots(
                    nrows=n_rows,
                    ncols=chart_columns,
                    figsize=(fig_width, fig_height),
                    sharex=False,  # Each plot gets its own label
                    sharey=False
                )
            except Exception as e:
                logger.error(f"Failed to create individual earnings figure with size {fig_width}x{fig_height}: {e}")
                # Fallback to smaller size
                fig_height = min(50, fig_height)
                fig, axes = plt.subplots(
                    nrows=n_rows,
                    ncols=chart_columns,
                    figsize=(fig_width, fig_height),
                    sharex=False,
                    sharey=False
                )
            
            # Set style
            try:
                plt.style.use('seaborn-v0_8')
            except:
                try:
                    plt.style.use('seaborn')
                except:
                    plt.style.use('default')
            
            # Handle single subplot case
            if n_rows == 1 and chart_columns == 1:
                axes = [axes]
            elif n_rows == 1 or chart_columns == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            # Create chart for each symbol
            for idx, symbol in enumerate(symbols):
                if idx < len(axes):
                    ax = axes[idx]
                    symbol_data = price_data[price_data['symbol'] == symbol].copy()
                    
                    if not symbol_data.empty:
                        self._create_individual_earnings_chart(ax, symbol_data, symbol, earnings_data)
                    else:
                        self._handle_empty_earnings_chart(ax, symbol)
            
            # Hide unused subplots
            for idx in range(len(symbols), len(axes)):
                axes[idx].axis('off')
            
            # Apply tight layout
            plt.tight_layout()
            
            # Display chart
            st.pyplot(fig, clear_figure=True)
            
            # Close the figure to free memory
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error creating individual earnings charts: {e}")
            st.error("Failed to create individual earnings stock charts")
    
    def create_performance_comparison_chart(self, categorized_earnings: Dict[str, pd.DataFrame], 
                                          stats: Dict[str, Dict[str, float]]) -> None:
        """
        Create a performance comparison chart across different earnings weeks.
        
        Args:
            categorized_earnings: Dictionary with categorized earnings data
            stats: Dictionary with statistics for each category
        """
        try:
            logger.debug("[create_performance_comparison_chart] Creating performance comparison chart")
            
            # Prepare data for plotting
            weeks = []
            avg_returns = []
            avg_volatilities = []
            stock_counts = []
            
            week_labels = {
                'current_week': 'Current Week',
                'next_week': 'Next Week', 
                'week_after': 'Week After',
                'later': 'Later'
            }
            
            for week, week_stats in stats.items():
                if week_stats.get('total_stocks', 0) > 0:
                    weeks.append(week_labels.get(week, week))
                    avg_returns.append(week_stats.get('avg_return', 0))
                    avg_volatilities.append(week_stats.get('avg_volatility', 0))
                    stock_counts.append(week_stats.get('total_stocks', 0))
            
            if not weeks:
                st.info("📊 No data available for performance comparison")
                return
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Average Returns by Week
            bars1 = ax1.bar(weeks, avg_returns, color='lightcoral', alpha=0.8, edgecolor='black')
            ax1.set_title('Average Returns by Earnings Week', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Average Return (%)')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars1, avg_returns):
                if abs(val) > 0.01:  # Only show if meaningful value
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if val >= 0 else -0.3),
                            f'{val:.1f}%', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')
            
            # 2. Average Volatility by Week
            bars2 = ax2.bar(weeks, avg_volatilities, color='lightblue', alpha=0.8, edgecolor='black')
            ax2.set_title('Average Volatility by Earnings Week', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Average Volatility Score')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars2, avg_volatilities):
                if val > 0.01:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # 3. Stock Count by Week
            bars3 = ax3.bar(weeks, stock_counts, color='lightgreen', alpha=0.8, edgecolor='black')
            ax3.set_title('Number of Stocks by Earnings Week', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Number of Stocks')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars3, stock_counts):
                if val > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(val), ha='center', va='bottom', fontweight='bold')
            
            # 4. Positive Return Percentage by Week
            positive_pcts = [stats[week].get('positive_return_pct', 0) for week in stats.keys() 
                           if stats[week].get('total_stocks', 0) > 0]
            
            if positive_pcts:
                bars4 = ax4.bar(weeks, positive_pcts, color='gold', alpha=0.8, edgecolor='black')
                ax4.set_title('Positive Return Percentage by Week', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Positive Return %')
                ax4.grid(True, alpha=0.3, axis='y')
                ax4.set_ylim(0, 100)
                
                # Add value labels
                for bar, val in zip(bars4, positive_pcts):
                    if val > 0:
                        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                f'{val:.0f}%', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-axis labels for better readability
            for ax in [ax1, ax2, ax3, ax4]:
                ax.tick_params(axis='x', rotation=45)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            
        except Exception as e:
            logger.error(f"Error creating performance comparison chart: {e}")
            st.error(f"Failed to create performance comparison chart: {e}")
    
    def _create_individual_earnings_chart(self, ax, symbol_data: pd.DataFrame, symbol: str, 
                                        earnings_data: Optional[pd.DataFrame] = None) -> None:
        """
        Create chart for individual earnings stock.
        
        Args:
            ax: matplotlib axis object
            symbol_data: DataFrame with price data for the symbol
            symbol: Stock symbol
            earnings_data: Optional earnings data for annotation
        """
        try:
            # Sort by date and prepare data
            symbol_data = symbol_data.sort_values('date')
            symbol_data['close'] = pd.to_numeric(symbol_data['close'], errors='coerce')
            symbol_data = symbol_data.dropna(subset=['close'])
            
            if symbol_data.empty:
                self._handle_empty_earnings_chart(ax, symbol)
                return
            
            # Get values for the title
            max_close = symbol_data['close'].max()
            min_close = symbol_data['close'].min()
            first_price = symbol_data.iloc[0]['close']
            last_price = symbol_data.iloc[-1]['close']
            total_return = ((last_price - first_price) / first_price) * 100
            
            # Get earnings info if available
            earnings_info = self._get_earnings_info_for_symbol(symbol, earnings_data)
            earnings_date_str = earnings_info.get('earnings_date_str', 'N/A')
            
            # Determine color for overall return
            return_color = 'green' if total_return >= 0 else 'red'
            
            # Plot Close as a line graph with markers (blue)
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
            self._add_price_annotations(ax, symbol_data)
            
            # Set title with bold symbol and earnings date
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
            
            # Set labels
            ax.set_xlabel("Date", fontsize=10, fontweight='bold')
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
    
    def _add_price_annotations(self, ax, symbol_data: pd.DataFrame) -> None:
        """Add price annotations with color coding."""
        try:
            prices = symbol_data['close'].tolist()
            dates = symbol_data['date'].tolist()
            
            for i, (date, close) in enumerate(zip(dates, prices)):
                # Determine color based on price change from previous point
                if i == 0:
                    text_color = "black"  # First point
                else:
                    prev_price = prices[i-1]
                    if close > prev_price:
                        text_color = "green"  # Price increased
                    elif close < prev_price:
                        text_color = "red"    # Price decreased
                    else:
                        text_color = "black"  # Price unchanged
                
                ax.text(
                    date, close, f"${close:.2f}",
                    fontsize=10,
                    fontweight='bold',
                    ha="right",
                    va="bottom",
                    color=text_color,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
                )
        except Exception as e:
            logger.error(f"Error adding price annotations: {e}")
    
    def _get_earnings_info_for_symbol(self, symbol: str, earnings_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Get earnings information for a specific symbol.
        
        Args:
            symbol: Stock symbol
            earnings_data: Optional earnings data DataFrame
            
        Returns:
            Dictionary with earnings information
        """
        try:
            earnings_date_str = "N/A"
            
            if earnings_data is not None and not earnings_data.empty:
                symbol_earnings = earnings_data[earnings_data['symbol'] == symbol]
                if not symbol_earnings.empty:
                    earnings_date = symbol_earnings['earnings_date'].iloc[0]
                    if pd.notna(earnings_date):
                        if hasattr(earnings_date, 'strftime'):
                            earnings_date_str = earnings_date.strftime('%m/%d')
                        else:
                            earnings_date_str = str(earnings_date)
            
            return {
                'earnings_date_str': earnings_date_str
            }
            
        except Exception as e:
            logger.error(f"Error getting earnings info for {symbol}: {e}")
            return {'earnings_date_str': 'N/A'}
    
    def _handle_empty_earnings_chart(self, ax, symbol: str) -> None:
        """Handle case where no data is available for an earnings symbol."""
        try:
            ax.text(0.5, 0.5, f'No data available\nfor {symbol}', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='gray')
            ax.set_title(f'{symbol} - No Data (Earnings)', fontsize=18, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as e:
            logger.error(f"Error handling empty chart for {symbol}: {e}")