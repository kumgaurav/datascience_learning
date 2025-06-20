"""
RSI and Stochastic Oscillator Tab Component
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import logging
from data import DatabaseManager
from analysis import TechnicalIndicators

logger = logging.getLogger('StockApp')

class RSIStochTab:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tech_indicators = TechnicalIndicators()
    
    def render(self, symbol: str, stock_data: pd.DataFrame):
        logger.info(f"Rendering RSI Stochastic tab for {symbol}")
        
        if len(stock_data) < 30:
            st.warning("Insufficient data for RSI and Stochastic calculations")
            return
        
        # Calculate indicators
        enriched_data = self._calculate_indicators(stock_data)
        
        # Display chart
        self._display_oscillators_chart(symbol, enriched_data)
        
        # Display analysis
        self._display_analysis(symbol, enriched_data)
    
    def _calculate_indicators(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        try:
            enriched_data = self.tech_indicators.calculate_all(stock_data)
            
            # Stochastic Oscillator
            low_14 = enriched_data['low'].rolling(window=14).min()
            high_14 = enriched_data['high'].rolling(window=14).max()
            enriched_data['stoch_k'] = ((enriched_data['close'] - low_14) / (high_14 - low_14)) * 100
            enriched_data['stoch_d'] = enriched_data['stoch_k'].rolling(window=3).mean()
            
            return enriched_data
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return stock_data
    
    def _display_oscillators_chart(self, symbol: str, data: pd.DataFrame):
        st.subheader("📊 RSI and Stochastic Oscillator")
        
        chart_data = data.tail(100).copy()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['RSI (14-day)', 'Stochastic Oscillator (14,3)'],
            vertical_spacing=0.1,
            row_heights=[0.5, 0.5]
        )
        
        # RSI chart
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'], y=chart_data['rsi'],
                mode='lines', name='RSI', line=dict(color='blue', width=2)
            ), row=1, col=1
        )
        
        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=1)
        
        # Stochastic chart
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'], y=chart_data['stoch_k'],
                mode='lines', name='%K', line=dict(color='orange', width=2)
            ), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'], y=chart_data['stoch_d'],
                mode='lines', name='%D', line=dict(color='red', width=2)
            ), row=2, col=1
        )
        
        # Stochastic reference lines
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        fig.update_layout(
            title=f'{symbol} - RSI and Stochastic Oscillator',
            height=600,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="RSI Value", row=1, col=1, range=[0, 100])
        fig.update_yaxes(title_text="Stochastic Value", row=2, col=1, range=[0, 100])
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_analysis(self, symbol: str, data: pd.DataFrame):
        st.subheader("📊 Oscillator Analysis")
        
        latest = data.iloc[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**RSI Analysis:**")
            rsi_val = latest['rsi']
            st.write(f"• Current RSI: {rsi_val:.2f}")
            
            if rsi_val >= 70:
                st.error("🔴 Overbought (RSI ≥ 70)")
                st.write("• Consider selling")
                st.write("• Watch for reversal")
            elif rsi_val <= 30:
                st.success("🟢 Oversold (RSI ≤ 30)")
                st.write("• Potential buying opportunity")
                st.write("• Wait for confirmation")
            else:
                st.info("🟡 Neutral zone (30-70)")
                st.write("• No clear signal")
                st.write("• Monitor for breakouts")
        
        with col2:
            st.markdown("**Stochastic Analysis:**")
            stoch_k = latest['stoch_k']
            stoch_d = latest['stoch_d']
            
            st.write(f"• %K: {stoch_k:.2f}")
            st.write(f"• %D: {stoch_d:.2f}")
            
            if stoch_k >= 80 and stoch_d >= 80:
                st.error("🔴 Overbought (>80)")
                st.write("• Strong sell signal")
            elif stoch_k <= 20 and stoch_d <= 20:
                st.success("🟢 Oversold (<20)")
                st.write("• Strong buy signal")
            else:
                st.info("🟡 Neutral zone")
            
            # Crossover analysis
            if stoch_k > stoch_d:
                st.info("📈 %K above %D (Bullish)")
            else:
                st.info("📉 %K below %D (Bearish)")
        
        # Combined analysis
        st.markdown("---")
        st.markdown("**Combined Signal:**")
        
        rsi_signal = "bullish" if rsi_val <= 30 else "bearish" if rsi_val >= 70 else "neutral"
        stoch_signal = "bullish" if (stoch_k <= 20 and stoch_d <= 20) else "bearish" if (stoch_k >= 80 and stoch_d >= 80) else "neutral"
        
        if rsi_signal == "bullish" and stoch_signal == "bullish":
            st.success("🟢 **Strong Buy Signal** - Both oscillators oversold")
        elif rsi_signal == "bearish" and stoch_signal == "bearish":
            st.error("🔴 **Strong Sell Signal** - Both oscillators overbought")
        elif rsi_signal == stoch_signal:
            st.info(f"🟡 **{rsi_signal.title()} Signal** - Both oscillators agree")
        else:
            st.warning("⚠️ **Mixed Signals** - Oscillators disagree") 