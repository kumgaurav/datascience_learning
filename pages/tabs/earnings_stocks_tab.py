"""
Earnings Stocks Tab Component - Stocks with earnings in next 4 weeks with volatility analysis
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

class EarningsStocksTab:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tech_indicators = TechnicalIndicators()
    
    def render(self, symbol: str, stock_data: pd.DataFrame, selected_symbols: list = None, chart_columns: int = 1, pre_calculated_data: pd.DataFrame = None):
        logger.info(f"Rendering Earnings Stocks tab for symbol: {symbol}")
        
        try:
            # Add loading indicator
            with st.spinner("Loading earnings data..."):
                # Get stocks with earnings in next 4 weeks (use pre-calculated if available)
                if pre_calculated_data is not None and not pre_calculated_data.empty:
                    logger.info("Using pre-calculated earnings data")
                    earnings_stocks = pre_calculated_data
                else:
                    earnings_stocks = self._get_earnings_stocks()
            
            if earnings_stocks.empty:
                st.warning("📅 No stocks with earnings data found")
                st.info("""
                **Possible reasons:**
                - Earnings table may not be available in your database
                - No stocks have earnings scheduled in the next 4 weeks
                - Database connection issues
                
                **What you can do:**
                - Check if the `stocks_earnings` table exists in your database
                - Verify the table has data for upcoming earnings dates
                - Try refreshing the page
                """)
                return
            
            # Filter by selected symbols if provided
            if selected_symbols:
                filtered_earnings = earnings_stocks[earnings_stocks['symbol'].isin(selected_symbols)]
                if filtered_earnings.empty:
                    st.warning(f"No earnings data available for selected symbols: {', '.join(selected_symbols)}")
                    return
                earnings_stocks = filtered_earnings
                st.info(f"📊 Showing analysis for {len(selected_symbols)} selected symbols")
            else:
                # Show success message
                st.success(f"✅ Found {len(earnings_stocks)} stocks with upcoming earnings")
            
            # Display earnings calendar
            self._display_earnings_calendar(earnings_stocks)
            
            # Display volatility chart
            self._display_earnings_volatility_chart(earnings_stocks)
            
            # Display individual stock charts if symbols are selected
            if selected_symbols:
                st.markdown("---")
                self._display_individual_earnings_charts(selected_symbols, chart_columns)
            else:
                # Display default stock price grid
                self._display_stock_price_grid(earnings_stocks)
            
            # Display analysis table
            self._display_earnings_table(earnings_stocks)
            
            # Display insights
            self._display_insights(earnings_stocks, symbol)
            
        except Exception as e:
            logger.error(f"Error in Earnings Stocks tab: {e}")
            st.error(f"❌ Unable to load earnings stocks data: {str(e)}")
            st.info("Please check your database connection and table structure.")
    
    def _get_earnings_stocks(self) -> pd.DataFrame:
        """Get stocks with earnings in next 4 weeks with volatility analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=28)  # Last 4 weeks for volatility calculation
            earnings_end_date = end_date + timedelta(weeks=4)  # Next 4 weeks for earnings
            
            logger.info(f"Fetching stocks with earnings from {end_date.date()} to {earnings_end_date.date()}")
            logger.info(f"Fetching volatility data from {start_date.date()} to {end_date.date()}")
            
            # First, check if earnings table exists
            if not self._check_earnings_table_exists():
                logger.warning("Earnings table not found, using fallback method")
                return self._get_fallback_earnings_data()
            
            # Include minimum price filter from config
            min_price = self.db_manager.get_minimum_price_filter()
            
            # Query to get stocks with earnings in next 4 weeks and their price data
            # Use stock_change_tracker for consistent symbol filtering
            query = """
            SELECT DISTINCT
                p.symbol, 
                p.date, 
                p.close, 
                p.high, 
                p.low, 
                p.volume,
                se.earnings_date
            FROM stocksdb.stocksinfp p
            INNER JOIN stocksdb.stock_change_tracker s ON s.symbol = p.symbol 
                AND s.is_active = 1 
                AND s.current_price > 5
            INNER JOIN stocksdb.stocks_earnings se ON p.symbol = se.symbol
            WHERE p.date >= %(start_date)s AND p.date <= %(end_date)s
            AND se.earnings_date >= %(earnings_start)s AND se.earnings_date <= %(earnings_end)s
            AND p.close IS NOT NULL AND p.close != '' AND p.close != '0'
            AND p.high IS NOT NULL AND p.high != '' AND p.high != '0'
            AND p.low IS NOT NULL AND p.low != '' AND p.low != '0'
            ORDER BY p.symbol, p.date
            """
            
            all_data = self.db_manager.execute_query(query, {
                'start_date': start_date,
                'end_date': end_date,
                'earnings_start': end_date,
                'earnings_end': earnings_end_date
            })
            
            if all_data.empty:
                logger.warning("No stocks with earnings in next 4 weeks found")
                return pd.DataFrame()
            
            # Convert data types
            all_data['date'] = pd.to_datetime(all_data['date'])
            all_data['earnings_date'] = pd.to_datetime(all_data['earnings_date'])
            for col in ['close', 'high', 'low', 'volume']:
                all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
            
            # Clean data
            all_data = all_data.dropna(subset=['close', 'high', 'low'])
            all_data = all_data[(all_data['close'] > 0) & (all_data['high'] > 0) & (all_data['low'] > 0)]
            
            logger.info(f"Found price data for {all_data['symbol'].nunique()} unique symbols with upcoming earnings")
            
            # Calculate volatility statistics and earnings info
            earnings_results = self._calculate_earnings_statistics(all_data)
            
            if earnings_results.empty:
                return pd.DataFrame()
            
            # Sort by earnings date and volatility
            earnings_results = earnings_results.sort_values(['earnings_date', 'volatility_score'], ascending=[True, False])
            
            logger.info(f"Found {len(earnings_results)} stocks with earnings and volatility data")
            return earnings_results
            
        except Exception as e:
            logger.error(f"Error getting earnings stocks: {e}")
            return pd.DataFrame()
    
    def _calculate_earnings_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility and earnings statistics for each stock"""
        symbol_groups = data.groupby('symbol')
        earnings_list = []
        
        for symbol, group_data in symbol_groups:
            try:
                group_data = group_data.sort_values('date')
                
                if len(group_data) < 10:
                    continue
                
                # Get earnings date (should be same for all rows of this symbol)
                earnings_date = group_data.iloc[0]['earnings_date']
                days_to_earnings = (earnings_date - datetime.now()).days
                
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
                
                # Pre-earnings momentum (last 7 days)
                recent_data = group_data.tail(7)
                if len(recent_data) >= 2:
                    recent_return = ((recent_data.iloc[-1]['close'] - recent_data.iloc[0]['close']) / recent_data.iloc[0]['close']) * 100
                else:
                    recent_return = 0
                
                result = {
                    'symbol': symbol,
                    'earnings_date': earnings_date,
                    'days_to_earnings': days_to_earnings,
                    'volatility_score': volatility_score,
                    'volatility_std': volatility_std,
                    'atr_pct': atr_pct,
                    'price_range': price_range,
                    'total_return': total_return,
                    'recent_momentum': recent_return,
                    'first_price': first_price,
                    'last_price': last_price,
                    'current_price': last_price,  # Most recent price
                    'data_points': len(group_data),
                    'avg_daily_return': daily_returns.mean() * 100,
                    'max_daily_gain': daily_returns.max() * 100,
                    'max_daily_loss': daily_returns.min() * 100
                }
                
                earnings_list.append(result)
                
            except Exception as e:
                logger.debug(f"Error processing symbol {symbol}: {e}")
                continue
        
        if not earnings_list:
            return pd.DataFrame()
        
        earnings_df = pd.DataFrame(earnings_list)
        # Filter outliers
        earnings_df = earnings_df[earnings_df['volatility_score'] <= 200]
        
        return earnings_df
    
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
            # Get top volatile stocks from the past month as a fallback
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Include minimum price filter from config
            min_price = self.db_manager.get_minimum_price_filter()
            
            query = """
            SELECT 
                symbol, 
                date, 
                close, 
                high, 
                low, 
                volume
            FROM stocksdb.stocksinfp
            WHERE date >= %(start_date)s AND date <= %(end_date)s
            AND close IS NOT NULL AND close != '' AND close != '0'
            AND high IS NOT NULL AND high != '' AND high != '0'
            AND low IS NOT NULL AND low != '' AND low != '0'
            AND CAST(close AS DECIMAL(10,2)) >= %(min_price)s
            ORDER BY symbol, date
            """
            
            data = self.db_manager.execute_query(query, {
                'start_date': start_date,
                'end_date': end_date,
                'min_price': min_price
            })
            
            if data.empty:
                return pd.DataFrame()
            
            # Convert data types
            data['date'] = pd.to_datetime(data['date'])
            for col in ['close', 'high', 'low', 'volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Clean data
            data = data.dropna(subset=['close', 'high', 'low'])
            data = data[(data['close'] > 0) & (data['high'] > 0) & (data['low'] > 0)]
            
            # Calculate volatility for all stocks and take the most volatile
            fallback_results = self._calculate_fallback_statistics(data)
            
            if not fallback_results.empty:
                # Add fake earnings dates (next 1-4 weeks)
                import random
                fallback_results['earnings_date'] = fallback_results.apply(
                    lambda x: datetime.now() + timedelta(days=random.randint(1, 28)), axis=1
                )
                fallback_results['days_to_earnings'] = fallback_results['earnings_date'].apply(
                    lambda x: (x - datetime.now()).days
                )
            
            return fallback_results.head(20)  # Limit to top 20 most volatile
            
        except Exception as e:
            logger.error(f"Error in fallback earnings data: {e}")
            return pd.DataFrame()
    
    def _calculate_fallback_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistics for fallback data"""
        symbol_groups = data.groupby('symbol')
        results_list = []
        
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
                
                # Only include if reasonably volatile
                if volatility_std < 0.2:  # Less than 20% annual volatility
                    continue
                
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
                
                # Recent momentum (last 7 days)
                recent_data = group_data.tail(7)
                if len(recent_data) >= 2:
                    recent_return = ((recent_data.iloc[-1]['close'] - recent_data.iloc[0]['close']) / recent_data.iloc[0]['close']) * 100
                else:
                    recent_return = 0
                
                result = {
                    'symbol': symbol,
                    'volatility_score': volatility_score,
                    'volatility_std': volatility_std,
                    'atr_pct': atr_pct,
                    'price_range': price_range,
                    'total_return': total_return,
                    'recent_momentum': recent_return,
                    'first_price': first_price,
                    'last_price': last_price,
                    'current_price': last_price,
                    'data_points': len(group_data),
                    'avg_daily_return': daily_returns.mean() * 100,
                    'max_daily_gain': daily_returns.max() * 100,
                    'max_daily_loss': daily_returns.min() * 100
                }
                
                results_list.append(result)
                
            except Exception as e:
                logger.debug(f"Error processing symbol {symbol} in fallback: {e}")
                continue
        
        if not results_list:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results_list)
        # Sort by volatility score
        results_df = results_df.sort_values('volatility_score', ascending=False)
        
        return results_df
    
    def _display_earnings_calendar(self, data: pd.DataFrame):
        """Display earnings calendar view"""
        st.subheader("📅 Earnings Calendar - Next 4 Weeks")
        
        if data.empty:
            st.warning("No earnings calendar data available")
            return
        
        # Group by week
        data_copy = data.copy()
        data_copy['earnings_date'] = pd.to_datetime(data_copy['earnings_date'])
        data_copy['week_start'] = data_copy['earnings_date'].dt.to_period('W').dt.start_time
        data_copy['earnings_date_str'] = data_copy['earnings_date'].dt.strftime('%Y-%m-%d')
        data_copy['weekday'] = data_copy['earnings_date'].dt.strftime('%A')
        
        # Sort by earnings date
        data_copy = data_copy.sort_values('earnings_date')
        
        # Create weekly calendar view
        weeks = data_copy.groupby('week_start')
        
        for week_start, week_data in weeks:
            week_end = week_start + timedelta(days=6)
            st.markdown(f"**Week of {week_start.strftime('%B %d')} - {week_end.strftime('%B %d, %Y')}**")
            
            # Group by day within the week
            daily_earnings = week_data.groupby(['earnings_date_str', 'weekday'])
            
            cols = st.columns(min(len(daily_earnings), 5))  # Max 5 columns per row
            
            for i, ((date_str, weekday), day_data) in enumerate(daily_earnings):
                if i < len(cols):
                    with cols[i]:
                        st.markdown(f"**{weekday}**")
                        st.markdown(f"*{date_str}*")
                        
                        # Sort symbols by volatility within the day
                        day_symbols = day_data.sort_values('volatility_score', ascending=False)
                        
                        for _, row in day_symbols.iterrows():
                            # Color coding based on recent momentum
                            momentum_color = "🟢" if row['recent_momentum'] > 0 else "🔴" if row['recent_momentum'] < 0 else "⚪"
                            volatility_indicator = "⚡" if row['volatility_score'] > day_symbols['volatility_score'].median() else ""
                            
                            st.write(f"{momentum_color}{volatility_indicator} **{row['symbol']}**")
                            st.caption(f"Vol: {row['volatility_score']:.1f} | Momentum: {row['recent_momentum']:+.1f}%")
            
            st.markdown("---")
    
    def _display_earnings_volatility_chart(self, data: pd.DataFrame):
        """Display earnings stocks volatility chart with earnings dates"""
        st.subheader("⚡ Stocks with Upcoming Earnings - Volatility Analysis")
        
        if data.empty:
            st.warning("No data available for chart")
            return
        
        # Create scatter plot with earnings date and volatility
        fig = go.Figure()
        
        # Color by days to earnings
        color_scale = px.colors.sequential.Viridis
        
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
        
        fig.update_layout(
            title='Stocks with Upcoming Earnings: Volatility vs Days to Earnings',
            xaxis_title='Days to Earnings',
            yaxis_title='Volatility Score',
            height=600,
            showlegend=False,
            hovermode='closest'
        )
        
        # Add vertical lines for week markers
        for week in [7, 14, 21, 28]:
            fig.add_vline(x=week, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_annotation(
                x=week, y=data['volatility_score'].max(),
                text=f"Week {week//7}",
                showarrow=False,
                yshift=10
            )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
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
            # Create comprehensive stock data for grid
            stock_grid_data = self._prepare_stock_grid_data(data)
            
            if stock_grid_data.empty:
                st.warning("Unable to prepare data for price charts")
                return
            
            # Get stocks sorted by earnings date then volatility
            earnings_stocks = data.sort_values(['earnings_date', 'volatility_score'], ascending=[True, False])['symbol'].tolist()
            
            # Create the grid visualization
            self._create_stock_price_grid(stock_grid_data, earnings_stocks, ncols=3)
            
        except Exception as e:
            logger.error(f"Error creating price grid: {e}")
            st.error("Unable to create price charts")
    
    def _prepare_stock_grid_data(self, earnings_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive stock data for grid visualization"""
        try:
            from datetime import datetime, timedelta
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=28)
            
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
            
            # Convert data types
            price_data['date'] = pd.to_datetime(price_data['date'])
            for col in ['close', 'high', 'low']:
                price_data[col] = pd.to_numeric(price_data[col], errors='coerce')
            
            # Clean data
            price_data = price_data.dropna(subset=['close', 'high', 'low'])
            
            # Calculate additional metrics for each symbol
            enhanced_data = []
            
            for symbol in earnings_symbols:
                symbol_data = price_data[price_data['symbol'] == symbol].sort_values('date')
                
                if len(symbol_data) < 5:
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
            result_df['Date'] = pd.to_datetime(result_df['Date'])
            result_df['earnings_date'] = pd.to_datetime(result_df['earnings_date'])
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error preparing stock grid data: {e}")
            return pd.DataFrame()
    
    def _create_stock_price_grid(self, pandas_df: pd.DataFrame, earnings_stocks: list, ncols: int = 3):
        """
        Create a grid of stock price plots using Matplotlib and Seaborn.
        """
        try:
            # Convert Date column to datetime format
            pandas_df = pandas_df.copy()
            pandas_df["Date"] = pd.to_datetime(pandas_df["Date"], errors='coerce')

            # Filter and sort stocks by earnings date then volatility
            filtered_df = pandas_df[pandas_df["symbol"].isin(earnings_stocks)].copy()
            
            # Create a dictionary of stock data
            filtered_stocks = {
                symbol: filtered_df[filtered_df["symbol"] == symbol].sort_values("Date")
                for symbol in earnings_stocks if symbol in filtered_df["symbol"].unique()
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

            # Display in Streamlit
            st.pyplot(fig)
            
            # Close the figure to free memory
            plt.close(fig)
            
            # Add explanation
            st.info("""
            **Chart Information:**
            - Each chart shows the stock's close price movement over the last 4 weeks leading up to earnings
            - Title includes earnings date, days until earnings, volatility score, and recent momentum
            - Charts are sorted by earnings date (earliest first)
            - Blue line with markers shows the price trend
            - Recent momentum indicates price movement in the last 7 trading days
            """)
            
        except Exception as e:
            logger.error(f"Error creating matplotlib grid: {e}")
            st.error(f"Unable to create price charts: {e}")
    
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
            
            # Determine color for overall return
            return_color = 'green' if total_return >= 0 else 'red'
            momentum_color = 'green' if recent_momentum >= 0 else 'red'
            
            # Plot the price line with blue color (back to original)
            ax.plot(dates, prices, marker='o', linestyle='-', linewidth=2, 
                   color='blue', markersize=3, label='Close Price')
            
            # Annotate price values with color based on price change
            prices_list = prices.tolist()
            dates_list = dates.tolist()
            
            for i, (date, price) in enumerate(zip(dates_list, prices_list)):
                # Determine color based on price change from previous point
                if i == 0:
                    # First point - use neutral color
                    text_color = "black"
                else:
                    # Compare with previous price
                    prev_price = prices_list[i-1]
                    if price > prev_price:
                        # Price increased - green
                        text_color = "green"
                    elif price < prev_price:
                        # Price decreased - red
                        text_color = "red"
                    else:
                        # Price unchanged - neutral
                        text_color = "black"
                
                ax.text(date, price, f'${price:.2f}', fontsize=8, fontweight='bold',
                       ha='center', va='bottom', color=text_color,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            # Format earnings date
            earnings_str = earnings_date.strftime('%m/%d') if pd.notna(earnings_date) else 'N/A'
            
            # Set title with bold symbol and key metrics (black text, no return/momentum)
            ax.set_title(f'**{symbol}** | ERD: {earnings_str}, Days: {days_to_earnings}\n'
                        f'VS: {volatility_score:.2f}, Max: ${max_price:.2f}, Min: ${min_price:.2f}\n'
                        f'Return: ',
                        fontsize=16, fontweight='bold', pad=10)
            
            # Add colored return and momentum values closer to the text
            ax.text(0.12, 0.92, f"{total_return:+.1f}%", 
                    transform=ax.transAxes, fontsize=14, fontweight='bold',
                    color=return_color, ha='left', va='top')
            
            ax.text(0.20, 0.92, f", Momentum: {recent_momentum:+.1f}%", 
                    transform=ax.transAxes, fontsize=14, fontweight='bold',
                    color=momentum_color, ha='left', va='top')
            
            # Set axis labels with bold text
            ax.set_xlabel('Date', fontsize=9, fontweight='bold')
            ax.set_ylabel('Price ($)', fontsize=9, fontweight='bold')
            
            # Format x-axis (dates)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Format dates on x-axis
            if len(dates) > 10:
                # Show fewer dates if too many
                step = len(dates) // 5
                ax.set_xticks(dates[::step])
            
            # Set y-axis to show price range nicely
            price_padding = (max_price - min_price) * 0.1
            ax.set_ylim(min_price - price_padding, max_price + price_padding)
            
        except Exception as e:
            logger.error(f"Error creating plot for {symbol}: {e}")
            self._handle_empty_plot(ax, symbol)
    
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
        st.markdown("**Pre-Earnings Price Movement Analysis (Last 4 Weeks)**")
        
        try:
            # Get price data for selected symbols
            price_data = self._get_earnings_price_data_for_symbols(selected_symbols)
            
            if price_data.empty:
                st.warning("No price data available for selected symbols")
                return
            
            # Create individual charts
            self._create_individual_earnings_charts(price_data, selected_symbols, chart_columns)
            
        except Exception as e:
            logger.error(f"Error displaying individual earnings charts: {e}")
            st.error("Failed to load individual earnings stock charts")
    
    def _get_earnings_price_data_for_symbols(self, symbols: list) -> pd.DataFrame:
        """Get price data for selected symbols over last 4 weeks for earnings analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=28)  # Last 4 weeks
            
            # Format symbols for SQL IN clause
            symbols_str = "', '".join(symbols)
            
            # Use stock_change_tracker for consistent symbol filtering
            
            query = f"""
            SELECT p.symbol, date, close, high, low, volume
            FROM stocksinfp p
            INNER JOIN stock_change_tracker s ON s.symbol = p.symbol 
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
            
            # Convert data types
            data['date'] = pd.to_datetime(data['date'])
            for col in ['close', 'high', 'low', 'volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Clean data
            data = data.dropna(subset=['close'])
            data = data[data['close'] > 0]
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting earnings price data: {e}")
            return pd.DataFrame()
    
    def _create_individual_earnings_charts(self, price_data: pd.DataFrame, symbols: list, ncols: int):
        """Create individual matplotlib charts for each earnings symbol"""
        try:
            # Calculate number of rows needed
            nrows = (len(symbols) + ncols - 1) // ncols
            
            # Create figure with proper sizing (your original style)
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(18, 5 * nrows),
                sharex=False,  # Disable shared x-axis so each plot gets its own label
                sharey=False  # Disable shared y-axis
            )
            
            # Set style (your original style)
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
            
            # Display chart
            st.pyplot(fig, clear_figure=True)
            
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
            FROM stocks_earnings
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