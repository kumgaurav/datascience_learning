"""
Volume and OBV Tab Component
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

class VolumeOBVTab:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.tech_indicators = TechnicalIndicators()
    
    def render(self, symbol: str, stock_data: pd.DataFrame):
        logger.info(f"Rendering Volume OBV tab for {symbol}")
        
        if len(stock_data) < 20:
            st.warning("Insufficient data for volume analysis")
            return
        
        # Calculate indicators
        enriched_data = self._calculate_indicators(stock_data)
        
        # Display chart
        self._display_volume_chart(symbol, enriched_data)
        
        # Display analysis
        self._display_analysis(symbol, enriched_data)
    
    def _calculate_indicators(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        try:
            enriched_data = self.tech_indicators.calculate_all(stock_data)
            
            # On-Balance Volume (OBV)
            obv = []
            obv_val = 0
            
            for i in range(len(enriched_data)):
                if i == 0:
                    obv_val = enriched_data.iloc[i]['volume']
                else:
                    if enriched_data.iloc[i]['close'] > enriched_data.iloc[i-1]['close']:
                        obv_val += enriched_data.iloc[i]['volume']
                    elif enriched_data.iloc[i]['close'] < enriched_data.iloc[i-1]['close']:
                        obv_val -= enriched_data.iloc[i]['volume']
                obv.append(obv_val)
            
            enriched_data['obv'] = obv
            
            # Volume moving average
            enriched_data['volume_ma'] = enriched_data['volume'].rolling(window=20).mean()
            
            return enriched_data
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return stock_data
    
    def _display_volume_chart(self, symbol: str, data: pd.DataFrame):
        st.subheader("📊 Volume and On-Balance Volume (OBV)")
        
        chart_data = data.tail(100).copy()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[f'{symbol} Price', 'Volume', 'On-Balance Volume (OBV)'],
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'], y=chart_data['close'],
                mode='lines', name='Close Price', line=dict(color='black', width=2)
            ), row=1, col=1
        )
        
        # Volume bars with color coding
        colors = ['green' if close >= open_price else 'red' 
                 for close, open_price in zip(chart_data['close'], chart_data['open'])]
        
        fig.add_trace(
            go.Bar(
                x=chart_data['date'], y=chart_data['volume'],
                name='Volume', marker_color=colors, opacity=0.7
            ), row=2, col=1
        )
        
        # Volume moving average
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'], y=chart_data['volume_ma'],
                mode='lines', name='Volume MA (20)', line=dict(color='orange', width=2)
            ), row=2, col=1
        )
        
        # OBV line
        fig.add_trace(
            go.Scatter(
                x=chart_data['date'], y=chart_data['obv'],
                mode='lines', name='OBV', line=dict(color='purple', width=2)
            ), row=3, col=1
        )
        
        fig.update_layout(
            title=f'{symbol} - Price, Volume, and OBV',
            height=700,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="OBV", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_analysis(self, symbol: str, data: pd.DataFrame):
        st.subheader("📊 Volume Analysis")
        
        latest = data.iloc[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Volume Metrics:**")
            current_volume = latest['volume']
            avg_volume = latest['volume_ma']
            
            st.write(f"• Current Volume: {current_volume:,.0f}")
            st.write(f"• 20-day Avg Volume: {avg_volume:,.0f}")
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            st.write(f"• Volume Ratio: {volume_ratio:.2f}x")
            
            if volume_ratio > 2:
                st.error("🔴 **Extremely High Volume**")
                st.write("• Unusual activity")
                st.write("• Potential breakout/breakdown")
            elif volume_ratio > 1.5:
                st.warning("🟡 **High Volume**")
                st.write("• Above average interest")
                st.write("• Confirm price moves")
            elif volume_ratio < 0.5:
                st.info("🔵 **Low Volume**")
                st.write("• Below average interest")
                st.write("• Weak price moves")
            else:
                st.success("🟢 **Normal Volume**")
                st.write("• Average trading activity")
        
        with col2:
            st.markdown("**OBV Analysis:**")
            current_obv = latest['obv']
            
            # OBV trend (compare with 5 days ago)
            if len(data) >= 5:
                past_obv = data.iloc[-6]['obv']
                obv_change = ((current_obv - past_obv) / abs(past_obv)) * 100 if past_obv != 0 else 0
                
                st.write(f"• Current OBV: {current_obv:,.0f}")
                st.write(f"• 5-day Change: {obv_change:+.2f}%")
                
                if obv_change > 5:
                    st.success("🟢 **OBV Rising**")
                    st.write("• Buying pressure")
                    st.write("• Bullish divergence potential")
                elif obv_change < -5:
                    st.error("🔴 **OBV Falling**")
                    st.write("• Selling pressure")
                    st.write("• Bearish divergence potential")
                else:
                    st.info("🟡 **OBV Stable**")
                    st.write("• Balanced buying/selling")
        
        # Price-Volume relationship
        st.markdown("---")
        st.markdown("**Price-Volume Relationship:**")
        
        # Check last 5 days for price-volume correlation
        recent_data = data.tail(5)
        
        price_up_days = (recent_data['close'] > recent_data['open']).sum()
        high_volume_days = (recent_data['volume'] > recent_data['volume_ma']).sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Up Days (5d)", f"{price_up_days}/5")
        
        with col2:
            st.metric("High Volume Days (5d)", f"{high_volume_days}/5")
        
        with col3:
            if price_up_days >= 3 and high_volume_days >= 3:
                st.success("🟢 **Bullish Pattern**")
                st.write("Price up + High volume")
            elif price_up_days <= 2 and high_volume_days >= 3:
                st.error("🔴 **Bearish Pattern**")
                st.write("Price down + High volume")
            else:
                st.info("🟡 **Mixed Pattern**")
                st.write("No clear trend")
        
        # Volume-based signals
        st.markdown("---")
        st.markdown("**Volume-Based Signals:**")
        
        # Price and volume confirmation
        latest_price_change = ((latest['close'] - data.iloc[-2]['close']) / data.iloc[-2]['close']) * 100
        volume_confirmation = current_volume > avg_volume
        
        if latest_price_change > 2 and volume_confirmation:
            st.success("🚀 **Strong Bullish Signal**")
            st.write("• Price up > 2% with high volume")
            st.write("• Volume confirms price move")
        elif latest_price_change < -2 and volume_confirmation:
            st.error("💥 **Strong Bearish Signal**")
            st.write("• Price down > 2% with high volume")
            st.write("• Volume confirms price move")
        elif abs(latest_price_change) > 2 and not volume_confirmation:
            st.warning("⚠️ **Weak Signal**")
            st.write("• Significant price move but low volume")
            st.write("• Move may not be sustainable")
        else:
            st.info("➡️ **No Clear Signal**")
            st.write("• Normal price and volume activity") 