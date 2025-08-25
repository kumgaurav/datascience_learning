"""
ATR (Average True Range) Tab Component
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

class ATRTab:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tech_indicators = TechnicalIndicators()
    
    def render(self, symbol: str, stock_data: pd.DataFrame):
        logger.info(f"Rendering ATR tab for {symbol}")
        
        if len(stock_data) < 30:
            st.warning("Insufficient data for ATR calculations")
            return
        
        # Calculate indicators
        enriched_data = self._calculate_indicators(stock_data)
        
        # Display chart
        self._display_atr_chart(symbol, enriched_data)
        
        # Display analysis
        self._display_analysis(symbol, enriched_data)
    
    def _calculate_indicators(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        try:
            enriched_data = self.tech_indicators.calculate_all(stock_data)
            
            # True Range calculation
            enriched_data['prev_close'] = enriched_data['close'].shift(1)
            enriched_data['tr1'] = enriched_data['high'] - enriched_data['low']
            enriched_data['tr2'] = abs(enriched_data['high'] - enriched_data['prev_close'])
            enriched_data['tr3'] = abs(enriched_data['low'] - enriched_data['prev_close'])
            enriched_data['true_range'] = enriched_data[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # ATR calculation (14-period)
            enriched_data['atr'] = enriched_data['true_range'].rolling(window=14).mean()
            
            # ATR as percentage of price
            enriched_data['atr_percent'] = (enriched_data['atr'] / enriched_data['close']) * 100
            
            # Volatility bands
            enriched_data['upper_band'] = enriched_data['close'] + enriched_data['atr']
            enriched_data['lower_band'] = enriched_data['close'] - enriched_data['atr']
            
            return enriched_data
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return stock_data
    
    def _display_atr_chart(self, symbol: str, data: pd.DataFrame):
        st.subheader("📊 Average True Range (ATR)")
        
        chart_data = data.tail(100).copy()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[f'{symbol} Price with ATR Bands', 'ATR Value', 'ATR Percentage'],
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price chart with ATR bands
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'], y=chart_data['close'],
                mode='lines', name='Close Price', line=dict(color='black', width=2)
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'], y=chart_data['upper_band'],
                mode='lines', name='Upper ATR Band', 
                line=dict(color='red', width=1, dash='dash'), opacity=0.7
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'], y=chart_data['lower_band'],
                mode='lines', name='Lower ATR Band', 
                line=dict(color='green', width=1, dash='dash'), opacity=0.7
            ), row=1, col=1
        )
        
        # ATR value chart
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'], y=chart_data['atr'],
                mode='lines', name='ATR', line=dict(color='blue', width=2)
            ), row=2, col=1
        )
        
        # ATR percentage chart
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'], y=chart_data['atr_percent'],
                mode='lines', name='ATR %', line=dict(color='purple', width=2)
            ), row=3, col=1
        )
        
        fig.update_layout(
            title=f'{symbol} - Average True Range Analysis',
            height=700,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="ATR Value", row=2, col=1)
        fig.update_yaxes(title_text="ATR %", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_analysis(self, symbol: str, data: pd.DataFrame):
        st.subheader("📊 ATR Analysis")
        
        latest = data.iloc[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current ATR Metrics:**")
            current_atr = latest['atr']
            current_atr_pct = latest['atr_percent']
            current_price = latest['close']
            
            st.write(f"• Current ATR: ${current_atr:.2f}")
            st.write(f"• ATR Percentage: {current_atr_pct:.2f}%")
            st.write(f"• Upper Band: ${latest['upper_band']:.2f}")
            st.write(f"• Lower Band: ${latest['lower_band']:.2f}")
            
            # Volatility assessment
            if current_atr_pct > 5:
                st.error("🔴 **High Volatility**")
                st.write("• ATR > 5% of price")
                st.write("• Increased risk")
            elif current_atr_pct > 3:
                st.warning("🟡 **Moderate Volatility**")
                st.write("• ATR 3-5% of price")
                st.write("• Normal market conditions")
            else:
                st.success("🟢 **Low Volatility**")
                st.write("• ATR < 3% of price")
                st.write("• Stable market conditions")
        
        with col2:
            st.markdown("**ATR Trend Analysis:**")
            
            # Compare with historical ATR
            if len(data) >= 20:
                recent_atr = data.tail(20)['atr_percent']
                avg_atr = recent_atr.mean()
                atr_trend = ((current_atr_pct - avg_atr) / avg_atr) * 100
                
                st.write(f"• 20-day Avg ATR%: {avg_atr:.2f}%")
                st.write(f"• ATR Trend: {atr_trend:+.1f}%")
                
                if atr_trend > 20:
                    st.error("📈 **Volatility Increasing**")
                    st.write("• ATR rising significantly")
                    st.write("• Expect larger price moves")
                elif atr_trend < -20:
                    st.success("📉 **Volatility Decreasing**")
                    st.write("• ATR falling significantly")
                    st.write("• Market stabilizing")
                else:
                    st.info("➡️ **Volatility Stable**")
                    st.write("• ATR relatively unchanged")
        
        # Trading applications
        st.markdown("---")
        st.markdown("**Trading Applications:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Stop Loss Levels:**")
            stop_loss_long = current_price - (current_atr * 2)
            stop_loss_short = current_price + (current_atr * 2)
            
            st.write(f"• Long Stop: ${stop_loss_long:.2f}")
            st.write(f"• Short Stop: ${stop_loss_short:.2f}")
            st.write("• Based on 2x ATR")
        
        with col2:
            st.markdown("**Position Sizing:**")
            risk_1_percent = current_atr * 2  # 2x ATR risk
            position_size_1pct = 1000 / risk_1_percent if risk_1_percent > 0 else 0
            
            st.write(f"• Risk per share: ${risk_1_percent:.2f}")
            st.write(f"• Shares for $1000 risk: {position_size_1pct:.0f}")
            st.write("• Based on 2x ATR stop")
        
        with col3:
            st.markdown("**Profit Targets:**")
            target_1 = current_price + current_atr
            target_2 = current_price + (current_atr * 2)
            
            st.write(f"• Target 1: ${target_1:.2f}")
            st.write(f"• Target 2: ${target_2:.2f}")
            st.write("• Based on 1x and 2x ATR")
        
        # ATR-based signals
        st.markdown("---")
        st.markdown("**ATR-Based Signals:**")
        
        # Breakout potential
        price_position = (current_price - latest['lower_band']) / (latest['upper_band'] - latest['lower_band'])
        
        if price_position > 0.8:
            st.warning("⚠️ **Near Upper ATR Band**")
            st.write("• Price approaching resistance")
            st.write("• Potential breakout or reversal")
        elif price_position < 0.2:
            st.warning("⚠️ **Near Lower ATR Band**")
            st.write("• Price approaching support")
            st.write("• Potential bounce or breakdown")
        else:
            st.info("🟡 **Within Normal Range**")
            st.write("• Price in middle of ATR bands")
        
        # Volatility expansion/contraction
        if len(data) >= 5:
            recent_atr_values = data.tail(5)['atr_percent']
            atr_expansion = (recent_atr_values.iloc[-1] - recent_atr_values.iloc[0]) / recent_atr_values.iloc[0] * 100
            
            if atr_expansion > 50:
                st.error("💥 **Volatility Expansion**")
                st.write("• ATR increased >50% in 5 days")
                st.write("• Major move likely continuing")
            elif atr_expansion < -30:
                st.success("😴 **Volatility Contraction**")
                st.write("• ATR decreased >30% in 5 days")
                st.write("• Potential for breakout soon")
            else:
                st.info("📊 **Normal Volatility Pattern**") 