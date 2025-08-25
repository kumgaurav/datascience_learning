import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

def display_technical_analysis_summary(stock_featured_data):
    """
    Display technical analysis summary with key metrics.
    """
    st.subheader(f"📊 Technical Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"${stock_featured_data['close']:.2f}")
        if not pd.isna(stock_featured_data.get('resistance_20d')):
            st.metric("Resistance Level", f"${stock_featured_data['resistance_20d']:.2f}")
        if not pd.isna(stock_featured_data.get('support_20d')):
            st.metric("Support Level", f"${stock_featured_data['support_20d']:.2f}")
    
    with col2:
        rsi_value = stock_featured_data.get('rsi_14d')
        rsi_display = f"{rsi_value:.1f}" if rsi_value is not None and not pd.isna(rsi_value) else "N/A"
        st.metric("RSI (14d)", rsi_display)
        
        momentum_5d = stock_featured_data.get('momentum_5d')
        momentum_5d_display = f"{momentum_5d:.2f}" if momentum_5d is not None and not pd.isna(momentum_5d) else "N/A"
        st.metric("Momentum (5d)", momentum_5d_display)
        
        momentum_10d = stock_featured_data.get('momentum_10d')
        momentum_10d_display = f"{momentum_10d:.2f}" if momentum_10d is not None and not pd.isna(momentum_10d) else "N/A"
        st.metric("Momentum (10d)", momentum_10d_display)
    
    with col3:
        resistance_status = "🟢 BROKEN" if stock_featured_data.get('broke_resistance', False) else "🔴 HOLDING"
        st.metric("Resistance Status", resistance_status)
        
        if 'predicted_return_pct' in stock_featured_data:
            st.metric("Predicted Return", f"{stock_featured_data['predicted_return_pct']:.2f}%")
        else:
            st.metric("Predicted Return", "N/A")
            
        if 'confidence_score' in stock_featured_data:
            st.metric("Confidence Score", f"{stock_featured_data['confidence_score']:.1f}/100")
        else:
            st.metric("Confidence Score", "N/A")

def display_key_price_levels(stock_featured_data):
    """
    Display key price levels and validation metrics.
    """
    st.subheader("🎯 Key Price Levels & Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Resistance break price
        if stock_featured_data.get('broke_resistance', False):
            resistance_level = stock_featured_data.get('resistance_20d')
            current_price = stock_featured_data['close']
            if resistance_level is not None and not pd.isna(resistance_level):
                price_above_resistance = current_price - resistance_level
                st.metric(
                    "🟢 Resistance Break Price", 
                    f"${resistance_level:.2f}",
                    delta=f"+${price_above_resistance:.2f} above resistance"
                )
                st.info(f"**Resistance Level**: ${resistance_level:.2f} was broken. Current price ${current_price:.2f} is ${price_above_resistance:.2f} above resistance.")
        else:
            resistance_level = stock_featured_data.get('resistance_20d')
            current_price = stock_featured_data['close']
            if resistance_level is not None and not pd.isna(resistance_level):
                distance_to_resistance = resistance_level - current_price
                st.metric(
                    "🔴 Resistance Level", 
                    f"${resistance_level:.2f}",
                    delta=f"${distance_to_resistance:.2f} to break"
                )
                st.info(f"**Resistance Level**: ${resistance_level:.2f}. Current price ${current_price:.2f} needs to rise ${distance_to_resistance:.2f} to break resistance.")
            else:
                st.metric("🔴 Resistance Break Price", "Not broken yet")
    
    with col2:
        # Support level
        support_level = stock_featured_data.get('support_20d')
        if support_level is not None and not pd.isna(support_level):
            current_price = stock_featured_data['close']
            distance_from_support = current_price - support_level
            st.metric(
                "🟢 Support Level", 
                f"${support_level:.2f}",
                delta=f"+${distance_from_support:.2f} above support"
            )
            st.info(f"**Support Level**: ${support_level:.2f}. Current price ${current_price:.2f} is ${distance_from_support:.2f} above support.")
        else:
            st.metric("🔴 Support Level", "Not available")

def display_resistance_analysis(stock_featured_data):
    """
    Display detailed resistance price analysis.
    """
    st.subheader("🎯 Resistance Price Analysis")
    
    resistance_level = stock_featured_data.get('resistance_20d')
    current_price = stock_featured_data['close']
    broke_resistance = stock_featured_data.get('broke_resistance', False)
    
    if resistance_level is not None and not pd.isna(resistance_level):
        col1, col2 = st.columns(2)
        
        with col1:
            if broke_resistance:
                price_above_resistance = current_price - resistance_level
                st.success(f"**🟢 RESISTANCE BROKEN!**")
                st.metric(
                    "Resistance Price", 
                    f"${resistance_level:.2f}",
                    delta=f"BROKEN by ${price_above_resistance:.2f}"
                )
                st.write(f"**Current Price**: ${current_price:.2f}")
                st.write(f"**Resistance Level**: ${resistance_level:.2f}")
                st.write(f"**Price Above Resistance**: ${price_above_resistance:.2f}")
            else:
                distance_to_resistance = resistance_level - current_price
                st.warning(f"**🔴 RESISTANCE NOT BROKEN**")
                st.metric(
                    "Resistance Price", 
                    f"${resistance_level:.2f}",
                    delta=f"${distance_to_resistance:.2f} to break"
                )
                st.write(f"**Current Price**: ${current_price:.2f}")
                st.write(f"**Resistance Level**: ${resistance_level:.2f}")
                st.write(f"**Distance to Break**: ${distance_to_resistance:.2f}")
        
        with col2:
            if broke_resistance:
                st.info("**📅 Resistance Break Details**")
                st.write("✅ **Status**: Resistance level has been successfully broken")
                st.write("📈 **Signal**: Bullish - price is now above resistance")
                st.write("🎯 **Next Target**: Look for continued upward momentum")
            else:
                st.info("**📅 Resistance Status**")
                st.write("⏳ **Status**: Price is below resistance level")
                st.write("📊 **Signal**: Waiting for breakout confirmation")
                st.write("🎯 **Watch For**: Price breaking above resistance with volume")
    else:
        st.warning("Resistance level data not available for this stock.")

def display_support_analysis(stock_featured_data):
    """
    Display detailed support level analysis.
    """
    st.subheader("🎯 Support Level Analysis")
    
    support_level = stock_featured_data.get('support_20d')
    current_price = stock_featured_data['close']
    
    if support_level is not None and not pd.isna(support_level):
        col1, col2 = st.columns(2)
        
        with col1:
            distance_from_support = current_price - support_level
            if distance_from_support > 0:
                st.success(f"**🟢 ABOVE SUPPORT**")
                st.metric(
                    "Support Price", 
                    f"${support_level:.2f}",
                    delta=f"${distance_from_support:.2f} above support"
                )
                st.write(f"**Current Price**: ${current_price:.2f}")
                st.write(f"**Support Level**: ${support_level:.2f}")
                st.write(f"**Distance Above Support**: ${distance_from_support:.2f}")
            else:
                st.error(f"**🔴 BELOW SUPPORT**")
                st.metric(
                    "Support Price", 
                    f"${support_level:.2f}",
                    delta=f"${abs(distance_from_support):.2f} below support"
                )
                st.write(f"**Current Price**: ${current_price:.2f}")
                st.write(f"**Support Level**: ${support_level:.2f}")
                st.write(f"**Distance Below Support**: ${abs(distance_from_support):.2f}")
        
        with col2:
            if distance_from_support > 0:
                st.info("**📅 Support Status**")
                st.write("✅ **Status**: Price is above support level")
                st.write("📈 **Signal**: Support is holding - bullish")
                st.write("🎯 **Watch For**: Price staying above support")
            else:
                st.info("**📅 Support Status**")
                st.write("⚠️ **Status**: Price is below support level")
                st.write("📉 **Signal**: Support broken - bearish")
                st.write("🎯 **Watch For**: Price recovery above support")
    else:
        st.warning("Support level data not available for this stock.")

def display_technical_insights(stock_featured_data):
    """
    Display technical insights and analysis.
    """
    st.subheader("🔍 Technical Insights")
    
    insights = []
    
    # Resistance analysis
    if stock_featured_data.get('broke_resistance', False):
        insights.append("🟢 **Resistance Break**: Price has successfully broken above the 20-day resistance level, indicating bullish momentum")
    else:
        insights.append("🔴 **Resistance Test**: Price is testing the resistance level. A break above could signal upward momentum")
    
    # RSI analysis
    rsi_value = stock_featured_data.get('rsi_14d')
    if rsi_value is not None and not pd.isna(rsi_value):
        if rsi_value > 70:
            insights.append("⚠️ **Overbought**: RSI above 70 suggests the stock may be overbought and could face resistance")
        elif rsi_value < 30:
            insights.append("🟢 **Oversold**: RSI below 30 suggests the stock may be oversold and could bounce back")
        else:
            insights.append("📊 **Neutral RSI**: RSI in neutral territory, no extreme overbought/oversold conditions")
    
    # Momentum analysis
    momentum_5d = stock_featured_data.get('momentum_5d')
    momentum_10d = stock_featured_data.get('momentum_10d')
    
    if momentum_5d is not None and momentum_10d is not None and not pd.isna(momentum_5d) and not pd.isna(momentum_10d):
        if momentum_5d > 0 and momentum_10d > 0:
            insights.append("🚀 **Positive Momentum**: Both 5-day and 10-day momentum are positive, indicating upward price movement")
        elif momentum_5d < 0 and momentum_10d < 0:
            insights.append("📉 **Negative Momentum**: Both 5-day and 10-day momentum are negative, indicating downward pressure")
        else:
            insights.append("📊 **Mixed Momentum**: Short-term and medium-term momentum are mixed, suggesting consolidation")
    
    # Display insights
    for insight in insights:
        st.write(insight)

def display_chart_interpretation_guide():
    """
    Display comprehensive chart interpretation guide.
    """
    st.subheader("📚 How to Interpret the Chart for Investment Decisions")
    
    # Create expandable sections for different signal types
    with st.expander("🟢 Bullish Signals (Good to Buy)", expanded=False):
        st.markdown("""
        **Resistance Break**: Price closes above the red resistance line
        - Look for the 🟢 Resistance Break annotation on the chart
        - This indicates strong buying pressure and potential upward movement
        
        **Green Candlesticks**: Multiple consecutive green days
        - Shows consistent buying pressure
        - Each green candle means the day closed higher than it opened
        
        **High Volume**: Volume increases with price gains
        - Check the gray volume bars at the bottom
        - Higher bars with price increases confirm strong buying interest
        
        **RSI 30-70**: Stock is not overbought
        - RSI in the neutral range allows for continued upside
        - Avoid stocks with RSI > 70 (overbought)
        
        **Momentum Point**: Recent strong upward movement
        - Look for the 🚀 Momentum annotation on the chart
        - Shows the peak of recent buying pressure
        """)
    
    with st.expander("🔴 Bearish Signals (Avoid or Sell)", expanded=False):
        st.markdown("""
        **Support Break**: Price closes below the green support line
        - This indicates potential further downside
        - Support level acts as a floor for the stock price
        
        **Red Candlesticks**: Multiple consecutive red days
        - Shows consistent selling pressure
        - Each red candle means the day closed lower than it opened
        
        **High Volume Down**: High volume with price declines
        - High volume bars with price drops indicate strong selling
        - This confirms bearish sentiment
        
        **RSI >70**: Stock is overbought and may pull back
        - RSI above 70 suggests the stock may be due for a correction
        - Consider taking profits or waiting for a pullback
        
        **Low Volume**: Lack of buying interest
        - Small volume bars indicate lack of conviction
        - Price movements without volume are less reliable
        """)
    
    with st.expander("📊 Neutral/Consolidation", expanded=False):
        st.markdown("""
        **Small Candlesticks**: Price moving sideways
        - Small candle bodies indicate indecision
        - Stock is consolidating before next move
        
        **Low Volume**: Lack of strong directional movement
        - Low volume suggests lack of conviction
        - Wait for volume confirmation before making decisions
        
        **RSI 40-60**: Neutral momentum
        - RSI in middle range indicates balanced buying/selling
        - Stock is not overbought or oversold
        """)
    
    with st.expander("💡 Practical Investment Strategy", expanded=False):
        st.markdown("""
        **For Buying (Entry Points)**
        1. **Wait for resistance break** with high volume
        2. **Buy on pullbacks** to support level
        3. **Look for oversold RSI** (<30) for bounce-back opportunities
        4. **Confirm with positive momentum** (5d and 10d both positive)
        
        **For Selling (Exit Points)**
        1. **Sell on resistance rejection** (price fails to break above resistance)
        2. **Exit on support break** (price closes below support)
        3. **Take profits on overbought RSI** (>70)
        4. **Watch for negative momentum** (both 5d and 10d negative)
        
        **Risk Management**
        - **Set stop-loss** below support level
        - **Take partial profits** at resistance levels
        - **Don't chase overbought stocks** (RSI >70)
        - **Use volume confirmation** for major moves
        """)
