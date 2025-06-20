"""
RSI Analysis Tab Component

Handles the RSI > 75 analysis and overbought periods analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import logging
from datetime import datetime, timedelta
from data import DatabaseManager
from analysis import RSICalculator
from utils.helpers import format_currency, format_percentage

logger = logging.getLogger('StockApp')


class RSIAnalysisTab:
    """
    RSI analysis tab for periods when RSI > threshold
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize RSI analysis tab
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.rsi_calculator = RSICalculator(period=14)
    
    def render(self, symbol: str, stock_data: pd.DataFrame):
        """
        Render the RSI analysis tab
        
        Args:
            symbol: Stock symbol
            stock_data: Historical stock data
        """
        logger.info(f"Rendering RSI analysis tab for {symbol}")
        
        if len(stock_data) < 15:
            st.warning("Insufficient data for RSI analysis (need at least 15 data points)")
            return
        
        # User controls
        threshold = self._render_controls()
        
        # Analyze RSI periods
        analysis_result = self._analyze_rsi_periods(stock_data, threshold)
        
        if 'error' in analysis_result:
            st.error(analysis_result['error'])
            return
        
        # Display analysis results
        self._display_analysis_summary(symbol, analysis_result, threshold)
        self._display_overbought_periods(symbol, analysis_result, stock_data, threshold)
        self._display_performance_analysis(symbol, analysis_result, stock_data)
        self._display_timing_analysis(analysis_result)
    
    def _render_controls(self) -> float:
        """
        Render user controls for RSI analysis
        
        Returns:
            Selected RSI threshold
        """
        st.subheader("🔧 Analysis Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            threshold = st.slider(
                "RSI Threshold",
                min_value=60.0,
                max_value=90.0,
                value=75.0,
                step=1.0,
                help="Analyze periods when RSI exceeds this threshold"
            )
        
        with col2:
            st.info(f"Analyzing periods when RSI > {threshold}")
            
            # Interpretation guide
            if threshold >= 80:
                st.caption("🔴 Very Overbought - Strong sell signals")
            elif threshold >= 70:
                st.caption("🟠 Overbought - Traditional sell threshold")
            else:
                st.caption("🟡 Moderately High - Early warning signals")
        
        return threshold
    
    def _analyze_rsi_periods(self, stock_data: pd.DataFrame, threshold: float) -> dict:
        """
        Analyze periods when RSI exceeds threshold
        
        Args:
            stock_data: Historical stock data
            threshold: RSI threshold value
        
        Returns:
            Analysis results dictionary
        """
        try:
            analysis = self.rsi_calculator.analyze_periods_above_threshold(
                stock_data, threshold
            )
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing RSI periods: {e}")
            return {'error': str(e)}
    
    def _display_analysis_summary(self, symbol: str, analysis: dict, threshold: float):
        """
        Display analysis summary metrics
        
        Args:
            symbol: Stock symbol
            analysis: Analysis results
            threshold: RSI threshold
        """
        st.subheader(f"📊 RSI > {threshold} Analysis Summary")
        
        # Create summary columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            count = analysis.get('count', 0)
            st.metric(
                label=f"Periods > {threshold}",
                value=count
            )
        
        with col2:
            percentage = analysis.get('percentage', 0)
            st.metric(
                label="% of Time",
                value=f"{percentage:.1f}%"
            )
        
        with col3:
            max_rsi = analysis.get('max_rsi', 0)
            st.metric(
                label="Maximum RSI",
                value=f"{float(max_rsi):.2f}" if pd.notna(max_rsi) else "N/A"
            )
        
        with col4:
            total_periods = analysis.get('total_periods', 0)
            st.metric(
                label="Total RSI Periods",
                value=total_periods
            )
        
        # Interpretation
        self._display_interpretation(analysis, threshold)
    
    def _display_interpretation(self, analysis: dict, threshold: float):
        """
        Display interpretation of the analysis
        
        Args:
            analysis: Analysis results
            threshold: RSI threshold
        """
        percentage = analysis.get('percentage', 0)
        count = analysis.get('count', 0)
        
        st.markdown("---")
        st.subheader("💡 Interpretation")
        
        if count == 0:
            st.info(f"📈 The stock has never reached RSI > {threshold} in the analyzed period. This suggests the stock has not experienced extreme overbought conditions.")
        elif percentage < 5:
            st.success(f"✅ The stock rarely reaches extreme overbought levels (RSI > {threshold}), occurring only {percentage:.1f}% of the time. This indicates relatively stable price movements.")
        elif percentage < 15:
            st.warning(f"⚠️ The stock occasionally reaches overbought levels (RSI > {threshold}) about {percentage:.1f}% of the time. Monitor these periods for potential selling opportunities.")
        else:
            st.error(f"🚨 The stock frequently reaches extreme overbought levels (RSI > {threshold}) {percentage:.1f}% of the time. This suggests high volatility and potential frequent reversal opportunities.")
    
    def _display_overbought_periods(self, symbol: str, analysis: dict, stock_data: pd.DataFrame, threshold: float):
        """
        Display detailed overbought periods
        
        Args:
            symbol: Stock symbol
            analysis: Analysis results
            stock_data: Historical stock data
            threshold: RSI threshold
        """
        periods = analysis.get('periods', [])
        
        if not periods:
            st.info(f"No periods found where RSI > {threshold}")
            return
        
        st.subheader(f"📅 Detailed Periods (RSI > {threshold})")
        
        # Convert to DataFrame for display
        periods_df = pd.DataFrame(periods)
        periods_df['date'] = pd.to_datetime(periods_df['date'])
        
        # Sort by date (most recent first)
        periods_df = periods_df.sort_values('date', ascending=False)
        
        # Format for display
        display_df = periods_df.copy()
        display_df['Date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df['RSI'] = display_df['rsi'].round(2)
        # Safe price formatting
        def safe_price_format(x):
            try:
                return f"${float(x):.2f}"
            except (ValueError, TypeError):
                return "N/A"
        
        display_df['Price'] = display_df['price'].apply(safe_price_format)
        
        # Add days ago column
        today = pd.Timestamp.now()
        display_df['Days Ago'] = (today - display_df['date']).dt.days
        
        # Color code by recency
        def color_recency(days):
            if days <= 7:
                return "🔴 Very Recent"
            elif days <= 30:
                return "🟠 Recent"
            elif days <= 90:
                return "🟡 Moderate"
            else:
                return "🟢 Old"
        
        display_df['Recency'] = display_df['Days Ago'].apply(color_recency)
        
        # Display table
        display_cols = ['Date', 'RSI', 'Price', 'Days Ago', 'Recency']
        st.dataframe(
            display_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Chart of overbought periods
        self._display_overbought_chart(symbol, periods_df, stock_data, threshold)
    
    def _display_overbought_chart(self, symbol: str, periods_df: pd.DataFrame, 
                                 stock_data: pd.DataFrame, threshold: float):
        """
        Display chart highlighting overbought periods
        
        Args:
            symbol: Stock symbol
            periods_df: DataFrame with overbought periods
            stock_data: Historical stock data
            threshold: RSI threshold
        """
        st.subheader(f"📈 Price Chart with RSI > {threshold} Periods")
        
        # Calculate RSI for the full dataset
        rsi_values = self.rsi_calculator.calculate(stock_data['close'].values)
        
        # Create chart data
        chart_data = stock_data.copy()
        if len(rsi_values) > 0:
            start_idx = 14  # RSI starts from index 14
            end_idx = start_idx + len(rsi_values)
            if end_idx <= len(chart_data):
                chart_data.loc[chart_data.index[start_idx:end_idx], 'rsi'] = rsi_values
        
        # Filter to recent data for better visualization
        recent_data = chart_data.tail(200)  # Last 200 data points
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=recent_data['date'],
            y=recent_data['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ))
        
        # Highlight overbought periods
        if not periods_df.empty:
            # Filter periods to match recent data range
            # Ensure both dates are pandas Timestamps for comparison
            periods_df_copy = periods_df.copy()
            periods_df_copy['date'] = pd.to_datetime(periods_df_copy['date'])
            recent_min_date = pd.to_datetime(recent_data['date'].min())
            
            recent_periods = periods_df_copy[
                periods_df_copy['date'] >= recent_min_date
            ]
            
            if not recent_periods.empty:
                fig.add_trace(go.Scatter(
                    x=recent_periods['date'],
                    y=recent_periods['price'],
                    mode='markers',
                    name=f'RSI > {threshold}',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='triangle-up'
                    )
                ))
        
        fig.update_layout(
            title=f"{symbol} - Price with Overbought Periods (RSI > {threshold})",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_performance_analysis(self, symbol: str, analysis: dict, stock_data: pd.DataFrame):
        """
        Display performance analysis after overbought periods
        
        Args:
            symbol: Stock symbol
            analysis: Analysis results
            stock_data: Historical stock data
        """
        periods = analysis.get('periods', [])
        
        if len(periods) < 2:
            st.info("Need at least 2 overbought periods for performance analysis")
            return
        
        st.subheader("📈 Performance After Overbought Periods")
        
        # Calculate returns after overbought periods
        returns_data = self._calculate_post_overbought_returns(periods, stock_data)
        
        if not returns_data:
            st.warning("Could not calculate performance data")
            return
        
        # Display performance metrics
        col1, col2, col3 = st.columns(3)
        
        returns_1w = [r['return_1w'] for r in returns_data if r['return_1w'] is not None]
        returns_1m = [r['return_1m'] for r in returns_data if r['return_1m'] is not None]
        
        with col1:
            if returns_1w:
                avg_return_1w = np.mean(returns_1w)
                st.metric(
                    label="Avg Return (1 Week)",
                    value=f"{avg_return_1w:.2f}%",
                    delta="After Overbought"
                )
        
        with col2:
            if returns_1m:
                avg_return_1m = np.mean(returns_1m)
                st.metric(
                    label="Avg Return (1 Month)",
                    value=f"{avg_return_1m:.2f}%",
                    delta="After Overbought"
                )
        
        with col3:
            if returns_1w:
                positive_pct = (np.array(returns_1w) > 0).mean() * 100
                st.metric(
                    label="Positive Returns",
                    value=f"{positive_pct:.0f}%",
                    delta="1 Week Success Rate"
                )
        
        # Enhanced explanation section
        self._display_detailed_explanation(returns_data, avg_return_1w if returns_1w else 0)
        
        # Detailed examples section
        self._display_entry_exit_examples(returns_data, symbol)
        
        # Performance distribution chart
        if returns_1w:
            self._display_returns_distribution(returns_1w, "1 Week Returns After Overbought")
    
    def _calculate_post_overbought_returns(self, periods: list, stock_data: pd.DataFrame) -> list:
        """
        Calculate returns after overbought periods
        
        Args:
            periods: List of overbought periods
            stock_data: Historical stock data
        
        Returns:
            List of return data
        """
        returns_data = []
        
        try:
            for period in periods:
                try:
                    period_date = pd.to_datetime(period['date'])
                    # Ensure period_price is numeric with better error handling
                    try:
                        period_price = float(period['price'])
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid price value for period {period}: {period['price']}")
                        continue
                    
                    # Find the index of this date - use different approach to avoid datetime comparison issues
                    stock_data_copy = stock_data.copy()
                    stock_data_copy['date'] = pd.to_datetime(stock_data_copy['date'])
                    
                    # Find matching date using string comparison instead
                    period_date_str = period_date.strftime('%Y-%m-%d')
                    stock_data_copy['date_str'] = stock_data_copy['date'].dt.strftime('%Y-%m-%d')
                    
                    matching_rows = stock_data_copy[stock_data_copy['date_str'] == period_date_str]
                    if matching_rows.empty:
                        logger.debug(f"No matching date found for {period_date_str}")
                        continue
                    
                    period_idx = matching_rows.index[0]
                    
                    # Calculate returns
                    return_data = {
                        'date': period_date,
                        'price': period_price,
                        'return_1w': None,
                        'return_1m': None
                    }
                    
                    # 1 week return
                    week_idx = period_idx + 7
                    if week_idx < len(stock_data):
                        try:
                            week_price = float(stock_data.iloc[week_idx]['close'])
                            return_data['return_1w'] = ((week_price - period_price) / period_price) * 100
                        except (ValueError, TypeError, ZeroDivisionError) as e:
                            logger.debug(f"Error calculating 1-week return: {e}")
                            return_data['return_1w'] = None
                    
                    # 1 month return
                    month_idx = period_idx + 30
                    if month_idx < len(stock_data):
                        try:
                            month_price = float(stock_data.iloc[month_idx]['close'])
                            return_data['return_1m'] = ((month_price - period_price) / period_price) * 100
                        except (ValueError, TypeError, ZeroDivisionError) as e:
                            logger.debug(f"Error calculating 1-month return: {e}")
                            return_data['return_1m'] = None
                    
                    returns_data.append(return_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing period {period}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error calculating post-overbought returns: {e}")
            return []
        
        return returns_data
    
    def _display_returns_distribution(self, returns: list, title: str):
        """
        Display returns distribution chart
        
        Args:
            returns: List of return values
            title: Chart title
        """
        fig = px.histogram(
            x=returns,
            nbins=10,
            title=title,
            labels={'x': 'Return (%)', 'y': 'Frequency'}
        )
        
        # Add reference line at 0%
        fig.add_vline(x=0, line_dash="dash", line_color="red", 
                     annotation_text="Break-even")
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_timing_analysis(self, analysis: dict):
        """
        Display timing analysis of overbought periods
        
        Args:
            analysis: Analysis results
        """
        periods = analysis.get('periods', [])
        
        if len(periods) < 3:
            return
        
        st.subheader("⏰ Timing Analysis")
        
        # Convert to DataFrame
        periods_df = pd.DataFrame(periods)
        periods_df['date'] = pd.to_datetime(periods_df['date'])
        
        # Add time-based features
        periods_df['month'] = periods_df['date'].dt.month
        periods_df['day_of_week'] = periods_df['date'].dt.day_name()
        periods_df['quarter'] = periods_df['date'].dt.quarter
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly distribution
            monthly_counts = periods_df['month'].value_counts().sort_index()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig = px.bar(
                x=[month_names[i-1] for i in monthly_counts.index],
                y=monthly_counts.values,
                title="Overbought Periods by Month"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day of week distribution
            dow_counts = periods_df['day_of_week'].value_counts()
            
            fig = px.bar(
                x=dow_counts.index,
                y=dow_counts.values,
                title="Overbought Periods by Day of Week"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _display_detailed_explanation(self, returns_data: list, avg_return_1w: float):
        """
        Display detailed explanation of the analysis
        
        Args:
            returns_data: List of return calculations
            avg_return_1w: Average 1-week return
        """
        st.markdown("---")
        st.subheader("💡 Understanding the Analysis")
        
        # Create explanation tabs
        tab1, tab2, tab3 = st.tabs(["📚 How It Works", "🎯 Trading Strategy", "⚠️ Risk Factors"])
        
        with tab1:
            st.markdown("""
            **What This Analysis Shows:**
            
            1. **Entry Point**: When RSI > 75 (overbought condition)
            2. **Exit Point**: 7 trading days later (1 week hold period)
            3. **Return Calculation**: `((Exit Price - Entry Price) / Entry Price) × 100`
            
            **Example Calculation:**
            - Entry Price: $100.00 (when RSI > 75)
            - Exit Price: $124.00 (after 7 trading days)
            - Return: ((124 - 100) / 100) × 100 = **24.0%**
            
            **Why This Matters:**
            - Shows historical performance of buying during "overbought" periods
            - Helps identify if overbought conditions lead to continued growth or reversals
            - Provides data-driven insights for timing decisions
            """)
        
        with tab2:
            if avg_return_1w > 0:
                strategy_color = "🟢"
                strategy_text = "**Momentum Strategy**: Historical data suggests buying during overbought periods has been profitable"
            else:
                strategy_color = "🔴"
                strategy_text = "**Contrarian Strategy**: Historical data suggests overbought periods often lead to price corrections"
            
            st.markdown(f"""
            **{strategy_color} Suggested Strategy Based on Historical Data:**
            
            {strategy_text}
            
            **If Average Return is Positive ({avg_return_1w:.2f}%):**
            - Consider that "overbought" might indicate strong momentum
            - Stock may continue rising despite high RSI
            - Traditional RSI signals may not apply to this stock
            
            **If Average Return is Negative:**
            - Traditional RSI interpretation holds
            - Overbought conditions often lead to pullbacks
            - Consider taking profits or waiting for better entry points
            
            **Recommended Actions:**
            - Use this data alongside other indicators
            - Consider position sizing based on historical volatility
            - Set stop-losses based on worst-case scenarios from the data
            """)
        
        with tab3:
            st.markdown("""
            **Important Risk Considerations:**
            
            ⚠️ **Past Performance Doesn't Guarantee Future Results**
            - Historical patterns may not repeat
            - Market conditions change over time
            - External factors can override technical indicators
            
            ⚠️ **Sample Size Matters**
            - Small number of overbought periods = less reliable data
            - Look for at least 10+ occurrences for statistical significance
            - Consider the time period of your analysis
            
            ⚠️ **Market Context**
            - Bull markets: Overbought conditions may persist longer
            - Bear markets: Reversals may be more severe
            - Sector rotation can affect individual stock behavior
            
            **Risk Management Tips:**
            - Never risk more than you can afford to lose
            - Diversify across multiple stocks and strategies
            - Use stop-losses to limit downside risk
            - Consider position sizing based on volatility
            """)
    
    def _display_entry_exit_examples(self, returns_data: list, symbol: str):
        """
        Display specific entry/exit examples from historical data
        
        Args:
            returns_data: List of return calculations
            symbol: Stock symbol
        """
        if not returns_data:
            return
        
        st.markdown("---")
        st.subheader(f"📋 Historical Entry/Exit Examples for {symbol}")
        
        # Sort by date (most recent first)
        sorted_data = sorted(returns_data, key=lambda x: x['date'], reverse=True)
        
        # Display top 10 most recent examples
        examples_to_show = min(10, len(sorted_data))
        
        # Create detailed examples table
        examples_data = []
        for i, data in enumerate(sorted_data[:examples_to_show]):
            entry_date = data['date'].strftime('%Y-%m-%d')
            entry_price = data['price']
            
            # Calculate exit details
            exit_date_1w = (data['date'] + timedelta(days=7)).strftime('%Y-%m-%d')
            return_1w = data.get('return_1w')
            
            if return_1w is not None:
                exit_price_1w = entry_price * (1 + return_1w / 100)
                profit_loss = exit_price_1w - entry_price
                
                # Determine outcome
                if return_1w > 5:
                    outcome = "🟢 Strong Gain"
                elif return_1w > 0:
                    outcome = "🟡 Small Gain"
                elif return_1w > -5:
                    outcome = "🟠 Small Loss"
                else:
                    outcome = "🔴 Significant Loss"
                
                examples_data.append({
                    'Entry Date': entry_date,
                    'Entry Price': f"${entry_price:.2f}",
                    'Exit Date (1W)': exit_date_1w,
                    'Exit Price': f"${exit_price_1w:.2f}",
                    'Profit/Loss': f"${profit_loss:.2f}",
                    'Return %': f"{return_1w:.2f}%",
                    'Outcome': outcome
                })
        
        if examples_data:
            st.dataframe(
                pd.DataFrame(examples_data),
                use_container_width=True,
                hide_index=True
            )
            
            # Summary statistics
            valid_returns = [data['return_1w'] for data in sorted_data if data['return_1w'] is not None]
            if valid_returns:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    best_return = max(valid_returns)
                    st.metric("Best Return", f"{best_return:.2f}%", delta="Single Trade")
                
                with col2:
                    worst_return = min(valid_returns)
                    st.metric("Worst Return", f"{worst_return:.2f}%", delta="Single Trade")
                
                with col3:
                    win_rate = (np.array(valid_returns) > 0).mean() * 100
                    st.metric("Win Rate", f"{win_rate:.0f}%", delta="Profitable Trades")
                
                with col4:
                    avg_return = np.mean(valid_returns)
                    st.metric("Average Return", f"{avg_return:.2f}%", delta="All Trades")
            
            # Practical example
            st.markdown("---")
            st.subheader("💰 Practical Investment Example")
            
            # Use the most recent example for practical calculation
            if examples_data:
                recent_example = examples_data[0]
                entry_price_str = recent_example['Entry Price'].replace('$', '')
                return_str = recent_example['Return %'].replace('%', '')
                
                try:
                    entry_price = float(entry_price_str)
                    return_pct = float(return_str)
                    
                    st.markdown(f"""
                    **Scenario**: You invested $10,000 on {recent_example['Entry Date']} when RSI > 75
                    
                    - **Entry Price**: {recent_example['Entry Price']} per share
                    - **Shares Purchased**: {10000/entry_price:.0f} shares
                    - **Exit Date**: {recent_example['Exit Date (1W)']}
                    - **Exit Price**: {recent_example['Exit Price']} per share
                    - **Total Return**: ${10000 * (return_pct/100):.2f}
                    - **Final Portfolio Value**: ${10000 * (1 + return_pct/100):.2f}
                    
                    **This represents a {return_pct:.2f}% return in just 1 week!**
                    """)
                    
                except (ValueError, ZeroDivisionError):
                    st.info("Could not calculate practical example with current data")
        else:
            st.info("No historical examples available for the selected period") 