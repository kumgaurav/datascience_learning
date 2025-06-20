"""
Price with Moving Averages and Bollinger Bands Tab Component

Handles price visualization with moving averages and Bollinger Bands.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import logging
from data import DatabaseManager
from analysis import TechnicalIndicators
from utils.helpers import format_currency, format_percentage

logger = logging.getLogger('StockApp')


class PriceMABBTab:
    """
    Price with Moving Averages and Bollinger Bands analysis tab
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize Price MA BB tab
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.tech_indicators = TechnicalIndicators()
    
    def render(self, symbol: str, stock_data: pd.DataFrame):
        """
        Render the Price MA BB analysis tab
        
        Args:
            symbol: Stock symbol
            stock_data: Historical stock data
        """
        logger.info(f"Rendering Price MA BB tab for {symbol}")
        
        if len(stock_data) < 50:
            st.warning("Insufficient data for moving averages and Bollinger Bands")
            return
        
        # Calculate indicators
        enriched_data = self._calculate_indicators(stock_data)
        
        # Display chart
        self._display_price_chart(symbol, enriched_data)
        
        # Display analysis
        self._display_analysis(symbol, enriched_data)
    
    def _calculate_indicators(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            stock_data: Raw stock data
        
        Returns:
            DataFrame with calculated indicators
        """
        try:
            enriched_data = self.tech_indicators.calculate_all(stock_data)
            
            # Moving averages
            enriched_data['ma_20'] = enriched_data['close'].rolling(window=20).mean()
            enriched_data['ma_50'] = enriched_data['close'].rolling(window=50).mean()
            
            # Bollinger Bands
            bb_ma = enriched_data['close'].rolling(window=20).mean()
            bb_std = enriched_data['close'].rolling(window=20).std()
            enriched_data['bb_upper'] = bb_ma + (bb_std * 2)
            enriched_data['bb_lower'] = bb_ma - (bb_std * 2)
            
            return enriched_data
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return stock_data
    
    def _display_price_chart(self, symbol: str, data: pd.DataFrame):
        """
        Display the main price chart with MAs and BBs
        
        Args:
            symbol: Stock symbol
            data: Data with indicators
        """
        st.subheader("📈 Price with Moving Averages & Bollinger Bands")
        
        chart_data = data.tail(100).copy()
        fig = go.Figure()
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=chart_data['date'], y=chart_data['bb_upper'],
            mode='lines', name='BB Upper', line=dict(color='lightblue', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=chart_data['date'], y=chart_data['bb_lower'],
            mode='lines', name='BB Lower', line=dict(color='lightblue', width=1),
            fill='tonexty', fillcolor='rgba(173,216,230,0.2)'
        ))
        
        # Price
        fig.add_trace(go.Scatter(
            x=chart_data['date'], y=chart_data['close'],
            mode='lines', name='Close Price', line=dict(color='black', width=2)
        ))
        
        # Moving Averages
        fig.add_trace(go.Scatter(
            x=chart_data['date'], y=chart_data['ma_20'],
            mode='lines', name='MA 20', line=dict(color='orange', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=chart_data['date'], y=chart_data['ma_50'],
            mode='lines', name='MA 50', line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f'{symbol} - Price with MAs & Bollinger Bands',
            xaxis_title='Date', yaxis_title='Price ($)', height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_analysis(self, symbol: str, data: pd.DataFrame):
        """
        Display analysis of the price chart
        
        Args:
            symbol: Stock symbol
            data: Data with indicators
        """
        st.subheader("📊 Analysis")
        
        latest = data.iloc[-1]
        current_price = latest['close']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Moving Averages:**")
            ma_20 = latest['ma_20']
            ma_50 = latest['ma_50']
            
            st.write(f"• MA 20: ${ma_20:.2f}")
            st.write(f"• MA 50: ${ma_50:.2f}")
            
            if current_price > ma_20 > ma_50:
                st.success("🟢 Bullish trend (Price > MA20 > MA50)")
            elif current_price < ma_20 < ma_50:
                st.error("🔴 Bearish trend (Price < MA20 < MA50)")
            else:
                st.info("🟡 Mixed signals")
        
        with col2:
            st.markdown("**Bollinger Bands:**")
            bb_upper = latest['bb_upper']
            bb_lower = latest['bb_lower']
            bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower)) * 100
            
            st.write(f"• Upper: ${bb_upper:.2f}")
            st.write(f"• Lower: ${bb_lower:.2f}")
            st.write(f"• Position: {bb_position:.1f}%")
            
            if bb_position > 80:
                st.error("🔴 Overbought (near upper band)")
            elif bb_position < 20:
                st.success("🟢 Oversold (near lower band)")
            else:
                st.info("🟡 Neutral zone")
    
    def _display_educational_content(self):
        """
        Display educational content about MAs and BBs
        """
        st.markdown("---")
        st.subheader("📚 Understanding Moving Averages & Bollinger Bands")
        
        tab1, tab2, tab3 = st.tabs(["📈 Moving Averages", "📊 Bollinger Bands", "💡 Trading Strategies"])
        
        with tab1:
            st.markdown("""
            **What are Moving Averages?**
            
            Moving Averages (MAs) smooth out price data to identify trends by filtering out short-term fluctuations.
            
            **Types of Moving Averages:**
            - **Simple MA (SMA)**: Average of closing prices over N periods
            - **Exponential MA (EMA)**: Gives more weight to recent prices
            
            **Common Periods:**
            - **Short-term**: 5, 10, 20 days (trend following)
            - **Medium-term**: 50 days (support/resistance)
            - **Long-term**: 100, 200 days (major trend)
            
            **Key Signals:**
            - **Golden Cross**: Short MA crosses above Long MA (Bullish)
            - **Death Cross**: Short MA crosses below Long MA (Bearish)
            - **Price vs MA**: Price above MA = Bullish, below = Bearish
            
            **MA as Support/Resistance:**
            - In uptrends: MA acts as support
            - In downtrends: MA acts as resistance
            - Bounces off MA confirm trend strength
            """)
        
        with tab2:
            st.markdown("""
            **What are Bollinger Bands?**
            
            Bollinger Bands consist of three lines:
            1. **Middle Band**: 20-period Simple Moving Average
            2. **Upper Band**: Middle Band + (2 × Standard Deviation)
            3. **Lower Band**: Middle Band - (2 × Standard Deviation)
            
            **Key Concepts:**
            
            **Band Width:**
            - Wide bands = High volatility
            - Narrow bands = Low volatility (squeeze)
            - Squeeze often precedes significant moves
            
            **Price Position:**
            - 0-20%: Near lower band (oversold)
            - 20-80%: Normal range
            - 80-100%: Near upper band (overbought)
            
            **The Squeeze:**
            - When bands contract (low volatility)
            - Often followed by explosive moves
            - Direction determined by breakout
            
            **Band Walking:**
            - Price hugging upper band = strong uptrend
            - Price hugging lower band = strong downtrend
            """)
        
        with tab3:
            st.markdown("""
            **Combined Trading Strategies:**
            
            **1. MA + BB Confirmation Strategy**
            - **Buy**: Price above MA + BB oversold (< 20%)
            - **Sell**: Price below MA + BB overbought (> 80%)
            - **Logic**: Trend + mean reversion
            
            **2. BB Squeeze Breakout**
            - **Setup**: BB width in bottom 20% percentile
            - **Entry**: Price breaks above/below bands with volume
            - **Target**: Previous high/low or next resistance/support
            
            **3. MA Crossover + BB Position**
            - **Golden Cross + BB > 50%**: Strong bullish signal
            - **Death Cross + BB < 50%**: Strong bearish signal
            - **Confirmation**: Wait for price to hold above/below MA
            
            **4. Dynamic Support/Resistance**
            - **In Uptrend**: Buy dips to middle BB or MA
            - **In Downtrend**: Sell rallies to middle BB or MA
            - **Stop Loss**: Below lower BB (uptrend) or above upper BB (downtrend)
            
            **Risk Management:**
            - Never rely on single indicator
            - Use volume for confirmation
            - Set stop-losses based on volatility
            - Consider overall market conditions
            """) 