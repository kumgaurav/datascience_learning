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

logger = logging.getLogger('StockApp')

class VolatileStocksTab:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tech_indicators = TechnicalIndicators()
    
    def render(self, symbol: str, stock_data: pd.DataFrame):
        logger.info(f"Rendering Volatile Stocks tab")
        
        try:
            # Get most volatile stocks with positive returns
            volatile_stocks = self._get_volatile_stocks()
            
            if volatile_stocks.empty:
                st.warning("No volatile stocks data available for the last 4 weeks")
                return
            
            # Display chart
            self._display_volatility_chart(volatile_stocks)
            
            # Display individual stock price charts
            self._display_stock_price_grid(volatile_stocks)
            
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
            
            query = """
            SELECT p.symbol, date, close, high, low, volume
            FROM stocksinfp as p
            LEFT JOIN stock_change_tracker as s ON s.symbol = p.symbol
            WHERE date >= %s AND date <= %s
            AND close IS NOT NULL AND close != '' AND close != '0'
            AND high IS NOT NULL AND high != '' AND high != '0'
            AND low IS NOT NULL AND low != '' AND low != '0'
            AND s.is_active = 1
            ORDER BY symbol, date
            """
            
            all_data = self.db_manager.execute_query(query, (start_date, end_date))
            
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
            
            query = f"""
            SELECT symbol, date, close, high, low
            FROM stocksinfp
            WHERE symbol IN ({symbol_placeholders})
            AND date >= %s AND date <= %s
            AND close IS NOT NULL AND close != '' AND close != '0'
            ORDER BY symbol, date
            """
            
            params = top_symbols + [start_date, end_date]
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

            # Filter and sort stocks by volatility_score (descending)
            sorted_df = pandas_df[pandas_df["symbol"].isin(top_25_stocks)].copy()
            sorted_df = sorted_df.sort_values(by="volatility_score", ascending=False)

            # Create a dictionary of sorted stock data
            filtered_stocks = {
                symbol: sorted_df[sorted_df["symbol"] == symbol].sort_values("Date")
                for symbol in sorted_df["symbol"].unique()
            }

            if not filtered_stocks:
                st.warning("No stock data available for grid visualization")
                return

            # Calculate layout
            nrows = int(np.ceil(len(filtered_stocks) / ncols))

            # Create figure and axes
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(18, 5 * nrows),
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

            # Apply tight layout to ensure labels don't overlap
            plt.tight_layout()

            # Display in Streamlit
            st.pyplot(fig)
            
            # Close the figure to free memory
            plt.close(fig)
            
            # Add explanation
            st.info("""
            **Chart Information:**
            - Each chart shows the stock's close price movement over the last 4 weeks
            - Price values are annotated on each data point
            - Charts are sorted by volatility score (highest volatility first)
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
        """Create individual stock plot on given axis."""
        try:
            # Ensure numeric conversion
            stock_data = stock_data.copy()
            stock_data["Close"] = pd.to_numeric(stock_data["Close"], errors='coerce')
            stock_data["max_close"] = pd.to_numeric(stock_data["max_close"], errors='coerce')
            stock_data["min_close"] = pd.to_numeric(stock_data["min_close"], errors='coerce')
            
            # Drop any rows with NaN values
            stock_data = stock_data.dropna(subset=["Close"])
            
            if stock_data.empty:
                self._handle_empty_plot(ax, symbol)
                return

            # Get values for the title
            max_close = stock_data["max_close"].iloc[0]
            min_close = stock_data["min_close"].iloc[0] 
            volatility_score = stock_data["volatility_score"].iloc[0]
            total_return = stock_data["total_return"].iloc[0]

            # Plot Close as a line graph
            ax.plot(
                stock_data["Date"],
                stock_data["Close"],
                marker='o',
                linestyle='-',
                label="Close Price",
                linewidth=2,
                color="blue",
                markersize=4
            )

            # Annotate each Close value on the graph
            for date, close in zip(stock_data["Date"], stock_data["Close"]):
                ax.text(
                    date, close, f"{close:.2f}",
                    fontsize=9,
                    ha="center",
                    va="bottom",
                    color="black",
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.8)
                )

            # Set title with max_close, min_close, volatility score, and return
            ax.set_title(
                f"{symbol} | Max: {max_close:.2f}, Min: {min_close:.2f}, Vol: {volatility_score:.2f}, Return: +{total_return:.1f}%",
                fontsize=11,
                pad=15,
                fontweight='bold'
            )

            # Set individual x-axis label for each chart
            ax.set_xlabel("Date", fontsize=10)

            # Set y-axis label
            ax.set_ylabel("Close Price ($)", fontsize=10)

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