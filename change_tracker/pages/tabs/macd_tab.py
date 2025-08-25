"""
MACD Tab Component
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

class MACDTab:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tech_indicators = TechnicalIndicators()
    
    def render(self, symbol: str, stock_data: pd.DataFrame):
        logger.info(f"Rendering MACD tab for {symbol}")
        
        if len(stock_data) < 50:
            st.warning("Insufficient data for MACD calculations")
            return
        
        # Calculate indicators
        enriched_data = self._calculate_indicators(stock_data)
        
        # Display chart
        self._display_macd_chart(symbol, enriched_data)
        
        # Display analysis
        self._display_analysis(symbol, enriched_data)
    
    def _calculate_indicators(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        try:
            enriched_data = self.tech_indicators.calculate_all(stock_data)
            
            # MACD calculation
            exp1 = enriched_data['close'].ewm(span=12).mean()
            exp2 = enriched_data['close'].ewm(span=26).mean()
            enriched_data['macd'] = exp1 - exp2
            enriched_data['macd_signal'] = enriched_data['macd'].ewm(span=9).mean()
            enriched_data['macd_histogram'] = enriched_data['macd'] - enriched_data['macd_signal']
            
            return enriched_data
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return stock_data
    
    def _display_macd_chart(self, symbol: str, data: pd.DataFrame):
        st.subheader("📊 MACD (Moving Average Convergence Divergence)")
        
        chart_data = data.tail(100).copy()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f'{symbol} Price', 'MACD'],
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'], y=chart_data['close'],
                mode='lines', name='Close Price', line=dict(color='black', width=2)
            ), row=1, col=1
        )
        
        # MACD line
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'], y=chart_data['macd'],
                mode='lines', name='MACD', line=dict(color='blue', width=2)
            ), row=2, col=1
        )
        
        # Signal line
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'], y=chart_data['macd_signal'],
                mode='lines', name='Signal', line=dict(color='red', width=2)
            ), row=2, col=1
        )
        
        # Histogram
        colors = ['green' if val >= 0 else 'red' for val in chart_data['macd_histogram']]
        fig.add_trace(
            go.Bar(
                x=chart_data['date'], y=chart_data['macd_histogram'],
                name='Histogram', marker_color=colors, opacity=0.6
            ), row=2, col=1
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig.update_layout(
            title=f'{symbol} - Price and MACD',
            height=600,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="MACD Value", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_analysis(self, symbol: str, data: pd.DataFrame):
        st.subheader("📊 MACD Analysis")
        
        latest = data.iloc[-1]
        previous = data.iloc[-2] if len(data) > 1 else latest
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current MACD Values:**")
            macd_val = latest['macd']
            signal_val = latest['macd_signal']
            histogram_val = latest['macd_histogram']
            
            st.write(f"• MACD: {macd_val:.4f}")
            st.write(f"• Signal: {signal_val:.4f}")
            st.write(f"• Histogram: {histogram_val:.4f}")
            
            # MACD position
            if macd_val > 0:
                st.success("🟢 MACD above zero (Bullish territory)")
            else:
                st.error("🔴 MACD below zero (Bearish territory)")
        
        with col2:
            st.markdown("**MACD Signals:**")
            
            # Signal line crossover
            if macd_val > signal_val and previous['macd'] <= previous['macd_signal']:
                st.success("🟢 **Bullish Crossover!**")
                st.write("• MACD crossed above Signal")
                st.write("• Potential buy signal")
            elif macd_val < signal_val and previous['macd'] >= previous['macd_signal']:
                st.error("🔴 **Bearish Crossover!**")
                st.write("• MACD crossed below Signal")
                st.write("• Potential sell signal")
            elif macd_val > signal_val:
                st.info("📈 MACD above Signal (Bullish)")
            else:
                st.info("📉 MACD below Signal (Bearish)")
        
        # Histogram analysis
        st.markdown("---")
        st.markdown("**Histogram Analysis:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if histogram_val > 0:
                st.success("🟢 Positive Histogram")
                st.write("Bullish momentum")
            else:
                st.error("🔴 Negative Histogram")
                st.write("Bearish momentum")
        
        with col2:
            # Histogram trend
            prev_histogram = previous['macd_histogram']
            if histogram_val > prev_histogram:
                st.info("📈 Histogram Increasing")
                st.write("Momentum strengthening")
            else:
                st.info("📉 Histogram Decreasing")
                st.write("Momentum weakening")
        
        with col3:
            # Zero line crossover
            if macd_val > 0 and previous['macd'] <= 0:
                st.success("🚀 **Zero Line Crossover!**")
                st.write("Strong bullish signal")
            elif macd_val < 0 and previous['macd'] >= 0:
                st.error("💥 **Zero Line Crossover!**")
                st.write("Strong bearish signal")
            else:
                st.info("➡️ No Zero Crossover")
        
        # Overall signal
        st.markdown("---")
        st.markdown("**Overall MACD Signal:**")
        
        signals = []
        if macd_val > signal_val:
            signals.append("bullish")
        else:
            signals.append("bearish")
        
        if histogram_val > 0:
            signals.append("bullish")
        else:
            signals.append("bearish")
        
        if macd_val > 0:
            signals.append("bullish")
        else:
            signals.append("bearish")
        
        bullish_count = signals.count("bullish")
        
        if bullish_count >= 2:
            st.success("🟢 **Overall Bullish Signal**")
            st.write(f"• {bullish_count}/3 indicators are bullish")
        else:
            st.error("🔴 **Overall Bearish Signal**")
            st.write(f"• {3-bullish_count}/3 indicators are bearish") 