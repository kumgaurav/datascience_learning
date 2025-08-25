"""
RSI Tab Component

Handles the 14-day RSI calculation and visualization tab.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import logging
from data import DatabaseManager
from analysis import RSICalculator, TechnicalIndicators
from utils.helpers import format_percentage

logger = logging.getLogger('StockApp')


class RSITab:
    """
    RSI analysis tab component
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize RSI tab
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.rsi_calculator = RSICalculator(period=14)
        self.tech_indicators = TechnicalIndicators()
    
    def render(self, symbol: str, stock_data: pd.DataFrame):
        """
        Render the RSI analysis tab
        
        Args:
            symbol: Stock symbol
            stock_data: Historical stock data
        """
        logger.info(f"Rendering RSI tab for {symbol}")
        
        if len(stock_data) < 15:
            st.warning("Insufficient data for RSI calculation (need at least 15 data points)")
            return
        
        # Calculate RSI and technical indicators
        enriched_data = self._calculate_indicators(stock_data)
        
        # Display RSI summary
        self._display_rsi_summary(symbol, enriched_data)
        
        # Display RSI chart
        self._display_rsi_chart(symbol, enriched_data)
        
        # Display RSI signals
        self._display_rsi_signals(symbol, stock_data)
        
        # Display RSI statistics
        self._display_rsi_statistics(enriched_data)
    
    def _calculate_indicators(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI and other technical indicators
        
        Args:
            stock_data: Raw stock data
        
        Returns:
            DataFrame with calculated indicators
        """
        try:
            enriched_data = self.tech_indicators.calculate_all(stock_data)
            logger.info("Technical indicators calculated successfully")
            return enriched_data
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            st.error(f"Error calculating indicators: {e}")
            return stock_data
    
    def _display_rsi_summary(self, symbol: str, data: pd.DataFrame):
        """
        Display RSI summary metrics
        
        Args:
            symbol: Stock symbol
            data: Data with RSI values
        """
        st.subheader("📊 RSI Summary")
        
        # Get latest RSI value
        rsi_values = data['rsi'].dropna()
        if len(rsi_values) == 0:
            st.warning("No RSI values available")
            return
        
        current_rsi = rsi_values.iloc[-1]
        
        # Create summary columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                current_rsi_val = f"{float(current_rsi):.2f}"
            except (ValueError, TypeError):
                current_rsi_val = "N/A"
            st.metric(
                label="Current RSI",
                value=current_rsi_val,
                delta=self._get_rsi_status(current_rsi) if isinstance(current_rsi, (int, float)) else None
            )
        
        with col2:
            avg_rsi = rsi_values.mean()
            try:
                avg_rsi_val = f"{float(avg_rsi):.2f}"
            except (ValueError, TypeError):
                avg_rsi_val = "N/A"
            st.metric(
                label="Average RSI",
                value=avg_rsi_val
            )
        
        with col3:
            max_rsi = rsi_values.max()
            try:
                max_rsi_val = f"{float(max_rsi):.2f}"
            except (ValueError, TypeError):
                max_rsi_val = "N/A"
            st.metric(
                label="Max RSI",
                value=max_rsi_val
            )
        
        with col4:
            min_rsi = rsi_values.min()
            try:
                min_rsi_val = f"{float(min_rsi):.2f}"
            except (ValueError, TypeError):
                min_rsi_val = "N/A"
            st.metric(
                label="Min RSI", 
                value=min_rsi_val
            )
    
    def _get_rsi_status(self, rsi_value: float) -> str:
        """
        Get RSI status indicator
        
        Args:
            rsi_value: Current RSI value
        
        Returns:
            Status string
        """
        if rsi_value >= 70:
            return "🔴 Overbought"
        elif rsi_value <= 30:
            return "🟢 Oversold"
        else:
            return "🟡 Neutral"
    
    def _display_rsi_chart(self, symbol: str, data: pd.DataFrame):
        """
        Display interactive RSI chart
        
        Args:
            symbol: Stock symbol
            data: Data with RSI and price values
        """
        st.subheader("📈 RSI Chart with Price")
        
        # Filter data with valid RSI values
        chart_data = data.dropna(subset=['rsi']).copy()
        
        if len(chart_data) == 0:
            st.warning("No data available for charting")
            return
        
        # Create subplot figure
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f'{symbol} Price', 'RSI (14-day)'],
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'],
                y=chart_data['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # RSI chart
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'],
                y=chart_data['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Overbought (70)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     annotation_text="Oversold (30)", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                     annotation_text="Neutral (50)", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Price and RSI Analysis",
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_rsi_signals(self, symbol: str, stock_data: pd.DataFrame):
        """
        Display RSI-based trading signals
        
        Args:
            symbol: Stock symbol
            stock_data: Historical stock data
        """
        st.subheader("🎯 RSI Trading Signals")
        
        # Generate RSI signals
        signals = self.rsi_calculator.get_rsi_signals(
            stock_data, 
            oversold_threshold=30, 
            overbought_threshold=70
        )
        
        if not signals:
            st.info("No RSI signals generated")
            return
        
        # Convert to DataFrame
        signals_df = pd.DataFrame(signals)
        
        # Filter recent signals (last 30 days)
        signals_df['date'] = pd.to_datetime(signals_df['date'])
        recent_signals = signals_df.tail(20)  # Show last 20 signals
        
        # Display signals table
        st.write("**Recent RSI Signals:**")
        
        # Format the display
        display_signals = recent_signals.copy()
        display_signals['RSI'] = display_signals['rsi'].round(2)
        
        # Safe price formatting
        def safe_price_format(x):
            try:
                return f"${float(x):.2f}"
            except (ValueError, TypeError):
                return "N/A"
        
        display_signals['Price'] = display_signals['close'].apply(safe_price_format)
        display_signals['Confidence'] = display_signals['confidence'].round(1)
        display_signals['Date'] = display_signals['date'].dt.strftime('%Y-%m-%d')
        
        # Color code signals
        def color_signal(signal):
            if signal == 'BUY':
                return '🟢 BUY'
            elif signal == 'SELL':
                return '🔴 SELL'
            else:
                return '🟡 HOLD'
        
        display_signals['Signal'] = display_signals['signal'].apply(color_signal)
        
        # Select columns for display
        display_cols = ['Date', 'Signal', 'Price', 'RSI', 'Confidence']
        st.dataframe(
            display_signals[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Signal summary
        signal_counts = signals_df['signal'].value_counts()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            buy_count = signal_counts.get('BUY', 0)
            st.metric("🟢 Buy Signals", buy_count)
        
        with col2:
            sell_count = signal_counts.get('SELL', 0)
            st.metric("🔴 Sell Signals", sell_count)
        
        with col3:
            hold_count = signal_counts.get('HOLD', 0)
            st.metric("🟡 Hold Signals", hold_count)
    
    def _display_rsi_statistics(self, data: pd.DataFrame):
        """
        Display RSI statistics and analysis
        
        Args:
            data: Data with RSI values
        """
        rsi_values = data['rsi'].dropna()
        
        if len(rsi_values) == 0:
            return
        
        st.subheader("📈 RSI Statistics & Analysis")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**RSI Distribution:**")
            
            # Count periods in different RSI ranges
            oversold_count = (rsi_values <= 30).sum()
            overbought_count = (rsi_values >= 70).sum()
            neutral_count = ((rsi_values > 30) & (rsi_values < 70)).sum()
            
            total_periods = len(rsi_values)
            
            st.metric("Oversold Periods (≤30)", oversold_count, f"{oversold_count/total_periods*100:.1f}%")
            st.metric("Overbought Periods (≥70)", overbought_count, f"{overbought_count/total_periods*100:.1f}%")
            st.metric("Neutral Periods (30-70)", neutral_count, f"{neutral_count/total_periods*100:.1f}%")
        
        with col2:
            # RSI histogram
            fig = px.histogram(
                x=rsi_values,
                nbins=20,
                title="RSI Distribution",
                labels={'x': 'RSI Value', 'y': 'Frequency'}
            )
            
            # Add reference lines
            fig.add_vline(x=30, line_dash="dash", line_color="green", 
                         annotation_text="Oversold (30)")
            fig.add_vline(x=70, line_dash="dash", line_color="red", 
                         annotation_text="Overbought (70)")
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced educational section
        self._display_rsi_education()
    
    def _display_rsi_education(self):
        """
        Display educational content about RSI
        """
        st.markdown("---")
        st.subheader("📚 Understanding RSI (Relative Strength Index)")
        
        # Create educational tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 What is RSI?", "📊 How to Read RSI", "💡 Trading Strategies", "⚠️ Limitations"])
        
        with tab1:
            st.markdown("""
            **What is RSI?**
            
            The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. 
            It oscillates between 0 and 100 and is primarily used to identify overbought or oversold conditions.
            
            **How is RSI Calculated?**
            
            1. **Calculate Average Gains and Losses** over 14 periods (default)
            2. **Relative Strength (RS)** = Average Gain / Average Loss  
            3. **RSI** = 100 - (100 / (1 + RS))
            
            **Example Calculation:**
            - If average gain = $2.00 and average loss = $1.00
            - RS = 2.00 / 1.00 = 2.0
            - RSI = 100 - (100 / (1 + 2.0)) = 100 - 33.33 = **66.67**
            
            **Key Characteristics:**
            - Range: 0 to 100
            - Period: Typically 14 days (can be adjusted)
            - Type: Momentum oscillator
            - Best for: Identifying potential reversal points
            """)
        
        with tab2:
            st.markdown("""
            **How to Interpret RSI Values:**
            
            🔴 **Overbought Zone (RSI ≥ 70)**
            - Stock may be overvalued
            - Potential selling opportunity
            - Price might decline soon
            - **Caution**: In strong uptrends, RSI can stay overbought for extended periods
            
            🟡 **Neutral Zone (RSI 30-70)**
            - Normal trading range
            - No clear overbought/oversold signal
            - Look for other indicators for confirmation
            - Most trading occurs in this range
            
            🟢 **Oversold Zone (RSI ≤ 30)**
            - Stock may be undervalued
            - Potential buying opportunity  
            - Price might increase soon
            - **Caution**: In strong downtrends, RSI can stay oversold for extended periods
            
            **RSI Divergence:**
            - **Bullish Divergence**: Price makes lower lows, but RSI makes higher lows
            - **Bearish Divergence**: Price makes higher highs, but RSI makes lower highs
            - Divergences often signal potential trend reversals
            
            **RSI Ranges by Market Condition:**
            - **Bull Market**: Overbought = 80, Oversold = 40
            - **Bear Market**: Overbought = 60, Oversold = 20
            - **Sideways Market**: Standard 70/30 levels work well
            """)
        
        with tab3:
            st.markdown("""
            **Common RSI Trading Strategies:**
            
            **1. Basic Overbought/Oversold Strategy**
            - **Buy Signal**: RSI crosses above 30 (leaving oversold)
            - **Sell Signal**: RSI crosses below 70 (leaving overbought)
            - **Best for**: Range-bound markets
            - **Risk**: Can generate false signals in trending markets
            
            **2. RSI Centerline Strategy**
            - **Buy Signal**: RSI crosses above 50
            - **Sell Signal**: RSI crosses below 50  
            - **Best for**: Trending markets
            - **Logic**: RSI above 50 indicates upward momentum
            
            **3. RSI Divergence Strategy**
            - Look for divergences between price and RSI
            - **Bullish**: Price falls but RSI rises → potential upward reversal
            - **Bearish**: Price rises but RSI falls → potential downward reversal
            - **Best for**: Identifying trend reversals
            
            **4. Multiple Timeframe RSI**
            - Use RSI on different timeframes (daily, weekly, hourly)
            - **Buy**: When multiple timeframes show oversold
            - **Sell**: When multiple timeframes show overbought
            - **Best for**: Confirmation and reducing false signals
            
            **Risk Management Tips:**
            - Never rely on RSI alone
            - Use stop-losses to limit downside
            - Consider volume confirmation
            - Be aware of market context (trending vs. ranging)
            """)
        
        with tab4:
            st.markdown("""
            **RSI Limitations and Important Considerations:**
            
            ⚠️ **False Signals in Trending Markets**
            - RSI can remain overbought/oversold for extended periods
            - Strong trends can override RSI signals
            - Example: In a strong bull market, "overbought" stocks often continue rising
            
            ⚠️ **Lagging Indicator**
            - RSI is based on past price data
            - May not predict sudden market changes
            - News events can override technical signals
            
            ⚠️ **Period Sensitivity**
            - Shorter periods (e.g., 7 days) = more sensitive, more signals
            - Longer periods (e.g., 21 days) = less sensitive, fewer signals
            - Default 14-day period is a compromise
            
            ⚠️ **Market Context Matters**
            - Bull markets: Adjust thresholds higher (80/40 instead of 70/30)
            - Bear markets: Adjust thresholds lower (60/20 instead of 70/30)
            - Sideways markets: Standard 70/30 works well
            
            **Best Practices:**
            - ✅ Combine RSI with other indicators (moving averages, volume, etc.)
            - ✅ Consider the overall market trend
            - ✅ Use proper risk management (stop-losses, position sizing)
            - ✅ Practice with paper trading before using real money
            - ✅ Understand that no indicator is 100% accurate
            
            **When RSI Works Best:**
            - Range-bound markets
            - When combined with support/resistance levels
            - For identifying potential reversal points
            - As confirmation with other technical indicators
            
            **When to Be Cautious:**
            - Strong trending markets
            - During major news events
            - Low volume trading periods
            - When used as the only decision factor
            """)
        
        # Practical example section
        st.markdown("---")
        st.subheader("💼 Practical RSI Example")
        
        st.markdown("""
        **Real-World RSI Trading Scenario:**
        
        **Setup**: You're watching a stock that has been trading between $90-$110 for months
        
        **Scenario 1 - Buy Signal**:
        - Stock price drops to $92 
        - RSI falls to 28 (oversold)
        - Volume is normal (not panic selling)
        - **Action**: Consider buying as RSI suggests oversold condition
        - **Target**: Previous resistance around $108-110
        - **Stop Loss**: Below $88 (if RSI was wrong)
        
        **Scenario 2 - Sell Signal**:
        - Stock price rises to $108
        - RSI reaches 74 (overbought)  
        - Volume is high (possible distribution)
        - **Action**: Consider selling or taking profits
        - **Target**: Previous support around $92-95
        - **Stop Loss**: Above $112 (in case of breakout)
        
        **Result Analysis**:
        - If the range holds: RSI signals work well
        - If the stock breaks out: RSI signals may fail
        - **Key**: Always have a plan for both scenarios!
        """)
        
        # Interactive RSI calculator
        st.markdown("---")
        st.subheader("🧮 RSI Quick Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_gain = st.number_input("Average Gain ($)", min_value=0.0, value=2.0, step=0.1)
            avg_loss = st.number_input("Average Loss ($)", min_value=0.01, value=1.0, step=0.1)
        
        with col2:
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                calculated_rsi = 100 - (100 / (1 + rs))
                
                st.metric("Relative Strength (RS)", f"{rs:.2f}")
                st.metric("Calculated RSI", f"{calculated_rsi:.2f}")
                
                # Interpretation
                if calculated_rsi >= 70:
                    st.error("🔴 Overbought - Consider selling")
                elif calculated_rsi <= 30:
                    st.success("🟢 Oversold - Consider buying")
                else:
                    st.info("🟡 Neutral - No clear signal")
            else:
                st.warning("Average loss must be greater than 0") 