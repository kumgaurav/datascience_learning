"""
Chart Utilities for Stock Analysis Dashboard

Centralized matplotlib chart functions to be shared across all tabs.
This module contains reusable chart creation functions to avoid code duplication
and make maintenance easier.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import logging
from datetime import datetime, timedelta

logger = logging.getLogger('StockApp')

class ChartUtils:
    """Utility class for creating standardized matplotlib charts"""
    
    @staticmethod
    def create_individual_stock_chart(ax, symbol_data, symbol, title_metrics=None):
        """
        Create a standardized individual stock chart
        
        Args:
            ax: matplotlib axis object
            symbol_data: DataFrame with columns ['date', 'close']
            symbol: str, stock symbol
            title_metrics: dict, optional additional metrics for title
                          e.g., {'volatility_score': 5.2, 'consistency_index': 0.85}
        """
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
            ChartUtils._add_price_annotations(ax, symbol_data)
            
            # Build title with metrics
            title_parts = [f"**{symbol}**"]
            
            # Add any additional metrics passed in
            if title_metrics:
                for key, value in title_metrics.items():
                    if key == 'volatility_score':
                        title_parts.append(f"Vol: {value:.1f}")
                    elif key == 'consistency_index':
                        title_parts.append(f"CI: {value:.3f}")
                    elif key == 'positive_months_ratio':
                        title_parts.append(f"+{value:.0%} pos months")
                    else:
                        title_parts.append(f"{key}: {value}")
            
            # Add standard metrics
            title_parts.extend([
                f"Max: ${max_close:.2f}",
                f"Min: ${min_close:.2f}",
                "Return: "
            ])
            
            # Set title (black text, no return value)
            title_text = " | ".join(title_parts)
            ax.set_title(
                title_text,
                fontsize=18,
                fontweight='bold',
                pad=10
            )
            
            # Add colored return percentage right after "Return: "
            ax.text(0.72, 1.02, f"{total_return:+.1f}%", 
                    transform=ax.transAxes, fontsize=16, fontweight='bold',
                    color=return_color, ha='left', va='bottom')
            
            # Set labels and formatting
            ChartUtils._format_chart_axes(ax)
            
        except Exception as e:
            logger.error(f"Error creating chart for {symbol}: {e}")
            ChartUtils._handle_empty_chart(ax, symbol)
    
    @staticmethod
    def create_grid_stock_chart(ax, stock_data, symbol, metrics_dict):
        """
        Create a standardized grid stock chart with enhanced metrics
        
        Args:
            ax: matplotlib axis object
            stock_data: DataFrame with columns ['Date', 'Close', 'max_close', 'min_close', etc.]
            symbol: str, stock symbol
            metrics_dict: dict containing metrics like volatility_score, total_return, etc.
        """
        try:
            # Get metrics from the dict
            max_close = metrics_dict.get('max_close', stock_data["Close"].max())
            min_close = metrics_dict.get('min_close', stock_data["Close"].min())
            total_return = metrics_dict.get('total_return', 0)
            
            # Determine color for overall return
            return_color = 'green' if total_return >= 0 else 'red'
            
            # Plot close price as line with markers (blue)
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
            
            # Annotate price values with color based on price change
            ChartUtils._add_price_annotations_grid(ax, stock_data)
            
            # Build title with all available metrics
            title_parts = [f"**{symbol}**"]
            
            # Add specific metrics based on what's available
            if 'volatility_score' in metrics_dict:
                title_parts.append(f"Vol: {metrics_dict['volatility_score']:.1f}")
            if 'consistency_index' in metrics_dict:
                title_parts.append(f"CI: {metrics_dict['consistency_index']:.3f}")
            if 'positive_months_ratio' in metrics_dict:
                title_parts.append(f"+{metrics_dict['positive_months_ratio']:.0%} pos")
            
            # Add standard metrics
            title_parts.extend([
                f"Max: ${max_close:.2f}",
                f"Min: ${min_close:.2f}",
                "Return: "
            ])
            
            # Set title (black text, no return value)
            title_text = " | ".join(title_parts)
            ax.set_title(
                title_text,
                fontsize=16,
                fontweight='bold',
                pad=10
            )
            
            # Add colored return percentage
            return_x_pos = ChartUtils._get_return_position(metrics_dict)
            ax.text(return_x_pos, 1.02, f"{total_return:+.1f}%", 
                    transform=ax.transAxes, fontsize=14, fontweight='bold',
                    color=return_color, ha='left', va='bottom')
            
            # Set labels and formatting for grid charts
            ChartUtils._format_grid_chart_axes(ax)
            
        except Exception as e:
            logger.error(f"Error creating grid chart for {symbol}: {e}")
            ChartUtils._handle_empty_chart(ax, symbol)
    
    @staticmethod
    def create_earnings_grid_chart(ax, stock_data, symbol, metrics_dict):
        """
        Create specialized chart for earnings stocks with momentum metrics
        
        Args:
            ax: matplotlib axis object
            stock_data: DataFrame with earnings-specific data
            symbol: str, stock symbol
            metrics_dict: dict containing earnings_date, momentum_score, etc.
        """
        try:
            # Get earnings-specific metrics
            earnings_date = metrics_dict.get('earnings_date', 'Unknown')
            momentum_score = metrics_dict.get('momentum_score', 0)
            total_return = metrics_dict.get('total_return', 0)
            max_close = metrics_dict.get('max_close', stock_data["Close"].max())
            min_close = metrics_dict.get('min_close', stock_data["Close"].min())
            
            # Determine colors
            return_color = 'green' if total_return >= 0 else 'red'
            momentum_color = 'green' if momentum_score >= 0 else 'red'
            
            # Plot close price
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
            
            # Add price annotations
            ChartUtils._add_price_annotations_grid(ax, stock_data)
            
            # Build title for earnings
            title_text = f"**{symbol}** | Earnings: {earnings_date} | Max: ${max_close:.2f}, Min: ${min_close:.2f}"
            ax.set_title(title_text, fontsize=16, fontweight='bold', pad=10)
            
            # Add Return and Momentum metrics with proper positioning
            ax.text(0.12, 1.02, f"Return: {total_return:+.1f}%", 
                    transform=ax.transAxes, fontsize=14, fontweight='bold',
                    color=return_color, ha='left', va='bottom')
            
            ax.text(0.20, 1.02, f"Momentum: {momentum_score:+.1f}%", 
                    transform=ax.transAxes, fontsize=14, fontweight='bold',
                    color=momentum_color, ha='left', va='bottom')
            
            # Format axes
            ChartUtils._format_grid_chart_axes(ax)
            
        except Exception as e:
            logger.error(f"Error creating earnings chart for {symbol}: {e}")
            ChartUtils._handle_empty_chart(ax, symbol)
    
    @staticmethod
    def _add_price_annotations(ax, symbol_data):
        """Add price annotations with color coding for individual charts"""
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
    
    @staticmethod
    def _add_price_annotations_grid(ax, stock_data):
        """Add price annotations with color coding for grid charts"""
        prices = stock_data["Close"].tolist()
        dates = stock_data["Date"].tolist()
        
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
                fontsize=9,
                fontweight='bold',
                ha="center",
                va="bottom",
                color=text_color,
                bbox=dict(facecolor='white', edgecolor='gray', 
                         boxstyle='round,pad=0.2', alpha=0.8)
            )
    
    @staticmethod
    def _format_chart_axes(ax):
        """Format axes for individual charts"""
        ax.set_xlabel("Date", fontsize=10, fontweight='bold')
        ax.set_ylabel("Close Price ($)", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Configure x-axis for proper date formatting
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis='x', rotation=45)
    
    @staticmethod
    def _format_grid_chart_axes(ax):
        """Format axes for grid charts"""
        ax.set_xlabel("Date", fontsize=10, fontweight='bold')
        ax.set_ylabel("Close Price ($)", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=8)
        
        # Configure x-axis for grid charts
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.margins(x=0.02, y=0.1)
    
    @staticmethod
    def _get_return_position(metrics_dict):
        """Get appropriate x-position for return percentage based on metrics"""
        # Adjust position based on how many metrics are in the title
        metric_count = sum(1 for key in ['volatility_score', 'consistency_index', 'positive_months_ratio'] 
                          if key in metrics_dict)
        
        if metric_count >= 2:
            return 0.65  # More metrics, position further right
        elif metric_count == 1:
            return 0.72  # One metric, standard position
        else:
            return 0.78  # No extra metrics, position further right
    
    @staticmethod
    def _handle_empty_chart(ax, symbol):
        """Handle case where no data is available for a symbol"""
        ax.text(0.5, 0.5, f'No data available\nfor {symbol}', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
        ax.set_title(f'{symbol} - No Data', fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def create_cached_matplotlib_figure(_chart_type: str, price_data_dict: dict, symbols: list, ncols: int, metrics_dict: dict = None):
        """
        Create cached matplotlib figure for any chart type
        
        Args:
            _chart_type: str, type of chart ('individual', 'grid', 'earnings')
            price_data_dict: dict, price data converted to dict for caching
            symbols: list, list of symbols to chart
            ncols: int, number of columns in grid
            metrics_dict: dict, optional metrics for each symbol
        """
        try:
            # Convert dict back to DataFrame
            price_data = pd.DataFrame(price_data_dict)
            if not price_data.empty:
                if 'date' in price_data.columns:
                    price_data['date'] = pd.to_datetime(price_data['date'])
                if 'Date' in price_data.columns:
                    price_data['Date'] = pd.to_datetime(price_data['Date'])
            
            logger.info(f"[CACHED CHART] Creating {_chart_type} matplotlib figure for {len(symbols)} symbols")
            
            # Calculate number of rows needed
            nrows = (len(symbols) + ncols - 1) // ncols
            
            # Create figure with proper sizing
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(18, 5 * nrows),
                sharex=False,
                sharey=False
            )
            
            # Set style
            plt.style.use('seaborn-v0_8')
            
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
                    
                    if _chart_type == 'individual':
                        symbol_data = price_data[price_data['symbol'] == symbol].copy()
                        if not symbol_data.empty:
                            symbol_metrics = metrics_dict.get(symbol, {}) if metrics_dict else {}
                            ChartUtils.create_individual_stock_chart(ax, symbol_data, symbol, symbol_metrics)
                        else:
                            ChartUtils._handle_empty_chart(ax, symbol)
                    
                    elif _chart_type == 'grid':
                        symbol_data = price_data[price_data['symbol'] == symbol].copy()
                        if not symbol_data.empty:
                            symbol_metrics = metrics_dict.get(symbol, {}) if metrics_dict else {}
                            ChartUtils.create_grid_stock_chart(ax, symbol_data, symbol, symbol_metrics)
                        else:
                            ChartUtils._handle_empty_chart(ax, symbol)
                    
                    elif _chart_type == 'earnings':
                        symbol_data = price_data[price_data['symbol'] == symbol].copy()
                        if not symbol_data.empty:
                            symbol_metrics = metrics_dict.get(symbol, {}) if metrics_dict else {}
                            ChartUtils.create_earnings_grid_chart(ax, symbol_data, symbol, symbol_metrics)
                        else:
                            ChartUtils._handle_empty_chart(ax, symbol)
            
            # Hide unused subplots
            for idx in range(len(symbols), len(axes)):
                axes[idx].axis('off')
            
            # Apply tight layout
            plt.tight_layout()
            
            logger.info(f"[CACHED CHART] {_chart_type.title()} matplotlib figure created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cached {_chart_type} matplotlib figure: {e}")
            return None 