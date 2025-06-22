"""
Volatile Stocks Tab Component - Top 25 volatile stocks for last 4 weeks with positive returns
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

class VolatileStocksTab:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tech_indicators = TechnicalIndicators()
    
    # Add caching for price data to avoid repeated DB queries
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def _get_cached_price_data_for_symbols(_self, symbols: list) -> pd.DataFrame:
        """Get cached price data for selected symbols over last 4 weeks"""
        try:
            # Create fresh db manager for cached call to avoid pickling issues
            from data import DatabaseManager
            db_manager = DatabaseManager()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=28)  # 4 weeks for volatile stocks
            
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
        logger.info(f"Rendering Volatile Stocks tab")
        

        
        try:
            # Get most volatile stocks with positive returns (use pre-calculated if available)
            if pre_calculated_data is not None and not pre_calculated_data.empty:
                logger.info("Using pre-calculated volatility data")
                volatile_stocks = pre_calculated_data
            else:
                volatile_stocks = self._get_volatile_stocks()
            
            if volatile_stocks.empty:
                st.warning("No volatile stocks data available for the last 4 weeks")
                return
            
            # Filter by selected symbols if provided
            if selected_symbols:
                filtered_stocks = volatile_stocks[volatile_stocks['symbol'].isin(selected_symbols)]
                if filtered_stocks.empty:
                    st.warning(f"No volatility data available for selected symbols: {', '.join(selected_symbols)}")
                    return
                volatile_stocks = filtered_stocks
                st.info(f"📊 Showing analysis for {len(selected_symbols)} selected symbols")
            
            # Display chart
            self._display_volatility_chart(volatile_stocks)
            
            # Display individual stock charts if symbols are selected
            if selected_symbols:
                st.markdown("---")
                self._display_individual_volatile_charts(selected_symbols, chart_columns, volatile_stocks)
            
            # Display analysis table
            self._display_volatility_table(volatile_stocks)
            
            # Display insights
            self._display_insights(volatile_stocks, symbol)
            
        except Exception as e:
            logger.error(f"Error in Volatile Stocks tab: {e}")
            st.error("Unable to load volatile stocks data. Please try again later.")
    
    def _get_volatile_stocks(self) -> pd.DataFrame:
        """Get most volatile stocks for the last 4 weeks with positive returns"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=28)
            
            logger.info(f"Fetching stock data from {start_date.date()} to {end_date.date()}")
            
            # Include minimum price filter from config
            min_price = self.db_manager.get_minimum_price_filter()
            
            query = """
            SELECT p.symbol, date, close, high, low, volume
            FROM stocksinfp as p
            LEFT JOIN stock_change_tracker as s ON s.symbol = p.symbol
            WHERE date >= %s AND date <= %s
            AND close IS NOT NULL AND close != '' AND close != '0'
            AND high IS NOT NULL AND high != '' AND high != '0'
            AND low IS NOT NULL AND low != '' AND low != '0'
            AND CAST(close AS DECIMAL(10,2)) >= %s
            AND s.is_active = 1
            ORDER BY symbol, date
            """
            
            all_data = self.db_manager.execute_query(query, (start_date, end_date, min_price))
            
            if all_data.empty:
                return pd.DataFrame()
            
            # Convert data types
            all_data['date'] = pd.to_datetime(all_data['date'])
            for col in ['close', 'high', 'low', 'volume']:
                all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
            
            # Clean data
            all_data = all_data.dropna(subset=['close', 'high', 'low'])
            all_data = all_data[(all_data['close'] > 0) & (all_data['high'] > 0) & (all_data['low'] > 0)]
            
            # Calculate volatility statistics
            volatility_results = self._calculate_volatility_statistics(all_data)
            
            if volatility_results.empty:
                return pd.DataFrame()
            
            # Filter for positive returns only and get top 25
            positive_returns = volatility_results[volatility_results['total_return'] > 0]
            top_25_volatile = positive_returns.nlargest(25, 'volatility_score')
            
            logger.info(f"Found {len(top_25_volatile)} volatile stocks with positive returns")
            return top_25_volatile
            
        except Exception as e:
            logger.error(f"Error getting volatile stocks: {e}")
            return pd.DataFrame()
    
    def _calculate_volatility_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility statistics for each stock"""
        symbol_groups = data.groupby('symbol')
        volatility_list = []
        
        for symbol, group_data in symbol_groups:
            try:
                group_data = group_data.sort_values('date')
                
                if len(group_data) < 10:
                    continue
                
                # Calculate daily returns
                group_data = group_data.copy()
                group_data['daily_return'] = group_data['close'].pct_change()
                group_data = group_data.dropna(subset=['daily_return'])
                
                if len(group_data) < 5:
                    continue
                
                # Volatility metrics
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
                
                result = {
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
                
                volatility_list.append(result)
                
            except Exception as e:
                logger.debug(f"Error processing symbol {symbol}: {e}")
                continue
        
        if not volatility_list:
            return pd.DataFrame()
        
        volatility_df = pd.DataFrame(volatility_list)
        # Filter outliers
        volatility_df = volatility_df[volatility_df['volatility_score'] <= 200]
        
        return volatility_df
    
    def _display_volatility_chart(self, data: pd.DataFrame):
        """Display volatility visualization chart"""
        st.subheader("⚡ Top 25 Most Volatile Stocks - Last 4 Weeks (Positive Returns Only)")
        
        if data.empty:
            st.warning("No data available for chart")
            return
        
        data = data.sort_values('volatility_score', ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Volatility Score',
            x=data['volatility_score'],
            y=data['symbol'],
            orientation='h',
            marker_color='#ff6b6b',
            text=[f"{val:.1f}" for val in data['volatility_score']],
            textposition='inside',
            textfont=dict(color='white', size=10)
        ))
        
        fig.update_layout(
            title='Volatility Score by Stock (Higher = More Volatile)',
            xaxis_title='Volatility Score',
            yaxis_title='Stock Symbol',
            height=800,
            showlegend=False,
            hovermode='y unified'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_stock_price_grid(self, data: pd.DataFrame):
        """Display individual stock price charts in a grid layout"""
        st.subheader("📈 Individual Stock Price Charts - Last 4 Weeks")
        
        if data.empty:
            st.warning("No data available for price charts")
            return
        
        try:
            # Create comprehensive stock data for grid
            stock_grid_data = self._prepare_stock_grid_data(data)
            
            if stock_grid_data.empty:
                st.warning("Unable to prepare data for price charts")
                return
            
            # Get top 25 stocks sorted by volatility
            top_25_stocks = data.head(25)['symbol'].tolist()
            
            # Create the grid visualization using the structured approach
            self._create_stock_price_grid(stock_grid_data, top_25_stocks, ncols=3)
            
        except Exception as e:
            logger.error(f"Error creating price grid: {e}")
            st.error("Unable to create price charts")
    
    def _prepare_stock_grid_data(self, volatility_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive stock data for grid visualization"""
        try:
            from datetime import datetime, timedelta
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=28)
            
            # Get symbols from top 25 volatile stocks
            top_symbols = volatility_data.head(25)['symbol'].tolist()
            
            # Create placeholders for symbols
            symbol_placeholders = ', '.join(['%s'] * len(top_symbols))
            
            # Include minimum price filter from config
            min_price = self.db_manager.get_minimum_price_filter()
            
            query = f"""
            SELECT symbol, date, close, high, low
            FROM stocksinfp
            WHERE symbol IN ({symbol_placeholders})
            AND date >= %s AND date <= %s
            AND close IS NOT NULL AND close != '' AND close != '0'
            AND CAST(close AS DECIMAL(10,2)) >= %s
            ORDER BY symbol, date
            """
            
            params = top_symbols + [start_date, end_date, min_price]
            price_data = self.db_manager.execute_query(query, params)
            
            if price_data.empty:
                return pd.DataFrame()
            
            # Convert data types
            price_data['date'] = pd.to_datetime(price_data['date'])
            for col in ['close', 'high', 'low']:
                price_data[col] = pd.to_numeric(price_data[col], errors='coerce')
            
            # Clean data
            price_data = price_data.dropna(subset=['close', 'high', 'low'])
            
            # Calculate additional metrics for each symbol
            enhanced_data = []
            
            for symbol in top_symbols:
                symbol_data = price_data[price_data['symbol'] == symbol].sort_values('date')
                
                if len(symbol_data) < 5:
                    continue
                
                # Get volatility info
                vol_info = volatility_data[volatility_data['symbol'] == symbol]
                if vol_info.empty:
                    continue
                
                vol_row = vol_info.iloc[0]
                
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
                        'volatility_score': vol_row['volatility_score'],
                        'total_return': vol_row['total_return'],
                        'first_price': vol_row['first_price'],
                        'last_price': vol_row['last_price']
                    }
                    enhanced_data.append(enhanced_row)
            
            if not enhanced_data:
                return pd.DataFrame()
            
            result_df = pd.DataFrame(enhanced_data)
            result_df['Date'] = pd.to_datetime(result_df['Date'])
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error preparing stock grid data: {e}")
            return pd.DataFrame()
    
    def _create_stock_price_grid(self, pandas_df: pd.DataFrame, top_25_stocks: list, ncols: int = 3):
        """
        Create a grid of stock price plots using Matplotlib and Seaborn.
        Adapted from your original matplotlib code for better visualization.
        
        Parameters:
        -----------
        pandas_df : pd.DataFrame
            DataFrame containing stock data with columns: Date, symbol, Close, volatility_score, max_close, min_close.
        top_25_stocks : list
            List of stock symbols to plot.
        ncols : int, optional
            Number of columns in the grid (default: 3).
        """
        try:
            # Convert Date column to datetime format
            pandas_df = pandas_df.copy()
            pandas_df["Date"] = pd.to_datetime(pandas_df["Date"], errors='coerce')

            # Filter and sort stocks by total_return (descending) - highest returns first
            # Note: top_25_stocks were already selected by volatility score in step 1
            sorted_df = pandas_df[pandas_df["symbol"].isin(top_25_stocks)].copy()
            
            # Get unique symbols with their return values, then sort by return
            symbol_returns = sorted_df.groupby('symbol')['total_return'].first().sort_values(ascending=False)
            symbols_by_return = symbol_returns.index.tolist()
            

            
            # Create a dictionary of stock data ordered by return (highest first)
            filtered_stocks = {
                symbol: sorted_df[sorted_df["symbol"] == symbol].sort_values("Date")
                for symbol in symbols_by_return
            }

            if not filtered_stocks:
                st.warning("No stock data available for grid visualization")
                return

            # Calculate layout
            nrows = int(np.ceil(len(filtered_stocks) / ncols))

            # Create figure and axes with better spacing
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(18, 6 * nrows),  # Increased height per row
                sharex=False,  # Disable shared x-axis so each plot gets its own label
                sharey=False   # Disable shared y-axis
            )
            
            # Handle single row case
            if nrows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()

            # Set style
            plt.style.use('default')  # Use default style for better Streamlit compatibility
            sns.set_palette("husl")

            # Iterate over stocks and create plots
            for i, (symbol, stock_data) in enumerate(filtered_stocks.items()):
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

            # Apply layout with extra spacing
            plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
            
            # Add extra spacing between subplots
            plt.subplots_adjust(hspace=0.4, wspace=0.3)

            # Display in Streamlit
            st.pyplot(fig)
            
            # Close the figure to free memory
            plt.close(fig)
            
            # Add explanation
            st.info("""
            **Chart Information (2-Step Process):**
            - **Step 1**: Selected top 25 stocks by volatility score (highest volatility first)
            - **Step 2**: Charts are sorted by total return (highest returns first)
            - Each chart shows the stock's close price movement over the last 4 weeks
            - Price values are annotated on each data point
            - Title includes Max/Min close prices, volatility score, and total return
            - Blue line with markers shows the price trend
            """)
            
        except Exception as e:
            logger.error(f"Error creating matplotlib grid: {e}")
            st.error(f"Unable to create price charts: {e}")

    def _handle_empty_plot(self, ax, symbol):
        """Handle cases where there's no data for a symbol."""
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
            stock_data["Close"] = pd.to_numeric(stock_data["Close"], errors='coerce')
            stock_data["max_close"] = pd.to_numeric(stock_data["max_close"], errors='coerce')
            stock_data["min_close"] = pd.to_numeric(stock_data["min_close"], errors='coerce')
            
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
            
            # Determine color for overall return
            return_color = 'green' if total_return >= 0 else 'red'
            
            # Plot close price as line with markers (back to blue)
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
            prices = stock_data["Close"].tolist()
            dates = stock_data["Date"].tolist()
            
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
                    fontsize=9,
                    fontweight='bold',
                    ha="center",
                    va="bottom",
                    color=text_color,
                    bbox=dict(facecolor='white', edgecolor='gray', 
                             boxstyle='round,pad=0.2', alpha=0.8)
                )
            
            # Set title with bold symbol (black text, no return value)
            ax.set_title(
                f"**{symbol}** | Max: ${max_close:.2f}, Min: ${min_close:.2f}, Return: ",
                fontsize=12,
                fontweight='bold',
                pad=10
            )
            
            # Add colored return percentage right after "Return: "
            ax.text(0.72, 1.02, f"{total_return:+.1f}%", 
                    transform=ax.transAxes, fontsize=12, fontweight='bold',
                    color=return_color, ha='left', va='bottom')
            
            # Set axis labels with bold text
            ax.set_xlabel("Date", fontsize=10, fontweight='bold')
            
            # Set y-axis label
            ax.set_ylabel("Close Price ($)", fontsize=10, fontweight='bold')

            # Add grid for better readability
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add legend
            ax.legend(loc='upper left', fontsize=8)

            # Configure x-axis for proper date formatting
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

            # Ensure date labels are readable
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            
            # Set tight margins
            ax.margins(x=0.02, y=0.1)
            
        except Exception as e:
            logger.error(f"Error creating plot for {symbol}: {e}")
            self._handle_empty_plot(ax, symbol)
    
    def _display_volatility_table(self, data: pd.DataFrame):
        """Display volatility data in table format"""
        st.subheader("📋 Volatility Analysis Table")
        
        if data.empty:
            st.warning("No data available for table")
            return
        
        display_data = data.sort_values('volatility_score', ascending=False)
        
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
            st.markdown("**Most Volatile:**")
            top_3 = data.nlargest(3, 'volatility_score')
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                st.write(f"{i}. **{row['symbol']}**: {row['volatility_score']:.1f}")
                st.write(f"   Return: +{row['total_return']:.1f}%")
        
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
                rank = (data['volatility_score'] > current_data['volatility_score']).sum() + 1
                
                st.success(f"⚡ **{current_symbol} is #{rank}!**")
                st.write(f"• Score: {current_data['volatility_score']:.1f}")
                st.write(f"• Return: +{current_data['total_return']:.1f}%")
            else:
                st.info(f"📊 {current_symbol} not in top 25")
        
        st.markdown("---")
        st.warning("""
        **⚠️ Risk Warning:** High volatility = High risk & High opportunity.
        Always use proper risk management and position sizing.
        """)
    
    def _display_individual_volatile_charts(self, selected_symbols: list, chart_columns: int, volatility_data: pd.DataFrame = None):
        """Display individual stock charts for selected symbols"""
        st.subheader("📈 Individual Stock Performance Charts")
        st.markdown("**4-Week Volatility Analysis**")
        
        try:
            # Get price data for selected symbols
            price_data = self._get_cached_price_data_for_symbols(selected_symbols)
            
            if price_data.empty:
                st.warning("No price data available for selected symbols")
                return
            
            # Use provided volatility data or get fresh data as fallback
            if volatility_data is None or volatility_data.empty:
                logger.warning("No volatility data provided, fetching fresh data")
                volatility_data = self._get_volatile_stocks()
            
            # Create individual charts with volatility data for sorting
            self._create_individual_price_charts_with_sorting(price_data, selected_symbols, chart_columns, volatility_data)
            
        except Exception as e:
            logger.error(f"Error displaying individual volatile charts: {e}")
            st.error("Failed to load individual stock charts")
    

    
    # Use centralized chart utils for matplotlib figure generation

    def _create_individual_price_charts_with_sorting(self, price_data: pd.DataFrame, symbols: list, ncols: int, volatility_data: pd.DataFrame):
        """Create individual matplotlib charts for each symbol with proper sorting by return"""
        try:
            # Sort symbols by total return (highest first) using volatility data
            if not volatility_data.empty:
                # Get return data for selected symbols
                symbol_returns = volatility_data[volatility_data['symbol'].isin(symbols)].set_index('symbol')['total_return'].sort_values(ascending=False)
                sorted_symbols = symbol_returns.index.tolist()
                
                # Add any missing symbols at the end (shouldn't happen but safety check)
                missing_symbols = [s for s in symbols if s not in sorted_symbols]
                sorted_symbols.extend(missing_symbols)
                

            else:
                sorted_symbols = symbols
                logger.warning("No volatility data available for sorting individual charts")
            
            # Convert DataFrame to dict for caching (Streamlit can't cache DataFrames with complex objects)
            price_data_dict = price_data.to_dict('records') if not price_data.empty else {}
            
            # Use centralized chart utils for figure generation with sorted symbols
            fig = ChartUtils.create_cached_matplotlib_figure(
                'individual', price_data_dict, sorted_symbols, ncols
            )
            
            if fig is not None:
                # Display the cached figure
                st.pyplot(fig, clear_figure=True)
            else:
                st.error("Failed to create individual stock charts")
            
        except Exception as e:
            logger.error(f"Error creating individual charts with sorting: {e}")
            st.error("Failed to create individual stock charts")
    
    def _create_individual_price_charts(self, price_data: pd.DataFrame, symbols: list, ncols: int):
        """Create individual matplotlib charts for each symbol using centralized chart utils (legacy method)"""
        try:
            # Convert DataFrame to dict for caching (Streamlit can't cache DataFrames with complex objects)
            price_data_dict = price_data.to_dict('records') if not price_data.empty else {}
            
            # Use centralized chart utils for figure generation
            fig = ChartUtils.create_cached_matplotlib_figure(
                'individual', price_data_dict, symbols, ncols
            )
            
            if fig is not None:
                # Display the cached figure
                st.pyplot(fig, clear_figure=True)
            else:
                st.error("Failed to create individual stock charts")
            
        except Exception as e:
            logger.error(f"Error creating individual charts: {e}")
            st.error("Failed to create individual stock charts")
    
    # Chart creation methods moved to centralized utils/chart_utils.py

    def _display_symbol_filter_section(self, data: pd.DataFrame):
        """Display symbol filter and individual stock price charts"""
        st.markdown("---")
        st.subheader("📈 Individual Stock Analysis")
        
        if data.empty:
            st.warning("No data available for individual analysis")
            return
        
        # Create symbol filter
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get top 25 symbols sorted by volatility score (Step 1: Best volatile symbols)
            top_symbols = data.nlargest(25, 'volatility_score')['symbol'].tolist()
            
            # Multi-select for symbols
            selected_symbols = st.multiselect(
                "⚡ Select Volatile Stocks to Analyze:",
                options=top_symbols,
                default=top_symbols[:6],  # Default to top 6
                help="Select up to 15 stocks to display detailed price charts. Symbols are from top 25 volatile stocks, but charts will be sorted by highest returns."
            )
        
        with col2:
            # Options for chart display
            chart_columns = st.selectbox(
                "Chart Columns:",
                options=[1, 2, 3, 4],
                index=0,  # Default to 1 column
                help="Number of columns in the chart grid"
            )
        
        if not selected_symbols:
            st.info("👆 Select one or more symbols above to view detailed price charts")
            return
        
        if len(selected_symbols) > 15:
            st.warning("⚠️ Please select maximum 15 symbols for better performance")
            selected_symbols = selected_symbols[:15]
        
        # Display individual stock price charts
        self._display_filtered_stock_charts(data, selected_symbols, chart_columns)
    
    def _display_filtered_stock_charts(self, volatility_data: pd.DataFrame, selected_symbols: list, ncols: int = 3):
        """Display individual stock price charts for selected volatile stocks"""
        st.subheader(f"📊 Price Charts for Selected Volatile Stocks - Last 4 Weeks")
        
        if not selected_symbols:
            st.warning("No symbols selected for chart display")
            return
        
        try:
            # Prepare stock data for selected symbols
            stock_chart_data = self._prepare_filtered_stock_data(volatility_data, selected_symbols)
            
            if stock_chart_data.empty:
                st.warning("Unable to prepare chart data for selected symbols")
                return
            
            # Create the matplotlib grid
            self._create_filtered_stock_grid(stock_chart_data, selected_symbols, ncols)
            
        except Exception as e:
            logger.error(f"Error creating filtered stock charts: {e}")
            st.error("Unable to create price charts")
    
    def _prepare_filtered_stock_data(self, volatility_data: pd.DataFrame, selected_symbols: list) -> pd.DataFrame:
        """Prepare stock price data for selected symbols"""
        try:
            # Calculate date range for last 4 weeks
            end_date = datetime.now()
            start_date = end_date - timedelta(days=28)
            
            # Create placeholders for symbols
            symbol_placeholders = ', '.join(['%s'] * len(selected_symbols))
            
            # Include minimum price filter from config
            min_price = self.db_manager.get_minimum_price_filter()
            
            query = f"""
            SELECT symbol, date, close, high, low
            FROM stocksinfp
            WHERE symbol IN ({symbol_placeholders})
            AND date >= %s AND date <= %s
            AND close IS NOT NULL AND close != '' AND close != '0'
            AND CAST(close AS DECIMAL(10,2)) >= %s
            ORDER BY symbol, date
            """
            
            params = selected_symbols + [start_date, end_date, min_price]
            price_data = self.db_manager.execute_query(query, params)
            
            if price_data.empty:
                logger.warning("No price data found for selected symbols")
                return pd.DataFrame()
            
            # Convert data types
            price_data['date'] = pd.to_datetime(price_data['date'])
            for col in ['close', 'high', 'low']:
                price_data[col] = pd.to_numeric(price_data[col], errors='coerce')
            
            # Clean data
            price_data = price_data.dropna(subset=['close', 'high', 'low'])
            
            # Enhance data with volatility metrics
            enhanced_data = []
            
            for symbol in selected_symbols:
                symbol_price_data = price_data[price_data['symbol'] == symbol].sort_values('date')
                
                if len(symbol_price_data) < 5:
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
            
            logger.info(f"Prepared chart data for {len(selected_symbols)} volatile stocks")
            return result_df
            
        except Exception as e:
            logger.error(f"Error preparing filtered stock data: {e}")
            return pd.DataFrame()
    
    def _create_filtered_stock_grid(self, pandas_df: pd.DataFrame, selected_symbols: list, ncols: int = 3):
        """Create matplotlib grid for selected volatile stock charts"""
        try:
            # Convert Date column to datetime
            pandas_df = pandas_df.copy()
            pandas_df["Date"] = pd.to_datetime(pandas_df["Date"], errors='coerce')
            
            # Filter and sort by total_return (descending) - Step 2: Sort by highest returns
            # Note: selected_symbols come from top 25 volatile stocks (Step 1)
            sorted_df = pandas_df[pandas_df["symbol"].isin(selected_symbols)].copy()
            
            # Get unique symbols with their return values, then sort by return
            symbol_returns = sorted_df.groupby('symbol')['total_return'].first().sort_values(ascending=False)
            symbols_by_return = symbol_returns.index.tolist()
            

            
            # Create dictionary of stock data ordered by return (highest first)
            filtered_stocks = {
                symbol: sorted_df[sorted_df["symbol"] == symbol].sort_values("Date")
                for symbol in symbols_by_return
            }
            
            if not filtered_stocks:
                st.warning("No stock data available for selected symbols")
                return
            
            # Calculate grid layout
            nrows = int(np.ceil(len(filtered_stocks) / ncols))
            
            # Create figure and axes with better spacing
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(18, 6 * nrows),  # Increased height per row
                sharex=False,
                sharey=False
            )
            
            # Handle single row case
            if nrows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()
            
            # Use default matplotlib style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create plots for each selected stock
            for i, (symbol, stock_data) in enumerate(filtered_stocks.items()):
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
            
            # Apply layout with extra spacing
            plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
            
            # Add extra spacing between subplots
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            
            # Display in Streamlit
            st.pyplot(fig)
            
            # Close figure to free memory
            plt.close(fig)
            
            # Add chart explanation
            st.info("""
            **📊 Volatile Stocks Chart Information (2-Step Process):**
            - **Step 1**: Selected from top 25 most volatile stocks (by volatility score)
            - **Step 2**: Charts are sorted by **Total Return** (highest returns first)
            - Each chart shows 4-week price movement for the selected volatile stock
            - Price values are annotated on each data point for precise tracking
            - Title includes: Max/Min prices, Volatility Score, and Total Return
            - Blue line shows price movement over the 4-week period
            - These stocks combine high volatility with positive returns
            """)
            
        except Exception as e:
            logger.error(f"Error creating filtered volatile stock grid: {e}")
            st.error(f"Unable to create volatile stock charts: {e}") 