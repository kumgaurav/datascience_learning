import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

def create_stock_price_chart(ticker, stock_data, featured_data):
    """
    Create a comprehensive candlestick chart with technical indicators and annotations.
    """
    if stock_data is None or stock_data.empty:
        return None
    
    try:
        # Create candlestick chart
        fig = go.Figure()
        
        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=stock_data['date'],
            open=stock_data['open'],
            high=stock_data['high'],
            low=stock_data['low'],
            close=stock_data['close'],
            name='Price',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        ))
        
        # Add moving averages if available
        if 'ma_20' in stock_data.columns and not stock_data['ma_20'].isna().all():
            fig.add_trace(go.Scatter(
                x=stock_data['date'],
                y=stock_data['ma_20'],
                mode='lines',
                name='MA 20',
                line=dict(color='orange', width=1)
            ))
        
        if 'ma_50' in stock_data.columns and not stock_data['ma_50'].isna().all():
            fig.add_trace(go.Scatter(
                x=stock_data['date'],
                y=stock_data['ma_50'],
                mode='lines',
                name='MA 50',
                line=dict(color='blue', width=1)
            ))
        
        # Add resistance and support levels
        resistance_level = featured_data.get('resistance_20d')
        support_level = featured_data.get('support_20d')
        
        if resistance_level is not None and not pd.isna(resistance_level):
            fig.add_hline(
                y=resistance_level,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Resistance: ${resistance_level:.2f}",
                annotation_position="top right"
            )
        
        if support_level is not None and not pd.isna(support_level):
            fig.add_hline(
                y=support_level,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Support: ${support_level:.2f}",
                annotation_position="bottom right"
            )
        
        # Add RSI annotation if available
        rsi_value = featured_data.get('rsi_14d')
        if rsi_value is not None and not pd.isna(rsi_value):
            current_price = featured_data['close']
            fig.add_annotation(
                x=stock_data['date'].iloc[-1],
                y=current_price,
                text=f"RSI: {rsi_value:.1f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="purple",
                bgcolor="white",
                bordercolor="purple",
                borderwidth=1
            )
        
        # Add momentum annotation if resistance was broken
        if featured_data.get('broke_resistance', False):
            resistance_level = featured_data.get('resistance_20d')
            if resistance_level is not None and not pd.isna(resistance_level):
                # Find the date when resistance was broken
                break_date = stock_data[stock_data['close'] > resistance_level]['date'].iloc[0] if len(stock_data[stock_data['close'] > resistance_level]) > 0 else stock_data['date'].iloc[-1]
                break_price = resistance_level
                
                fig.add_annotation(
                    x=break_date,
                    y=break_price,
                    text="🟢 Resistance Break",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="green",
                    bgcolor="lightgreen",
                    bordercolor="green",
                    borderwidth=1
                )
        
        # Add momentum annotation
        momentum_5d = featured_data.get('momentum_5d')
        momentum_10d = featured_data.get('momentum_10d')
        
        if momentum_5d is not None and momentum_10d is not None and not pd.isna(momentum_5d) and not pd.isna(momentum_10d):
            if momentum_5d > 0 and momentum_10d > 0:
                # Find recent high point for momentum annotation
                recent_high = stock_data['high'].tail(5).max()
                recent_date = stock_data[stock_data['high'] == recent_high]['date'].iloc[0] if len(stock_data[stock_data['high'] == recent_high]) > 0 else stock_data['date'].iloc[-1]
                
                fig.add_annotation(
                    x=recent_date,
                    y=recent_high,
                    text="🚀 Momentum",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="orange",
                    bgcolor="lightyellow",
                    bordercolor="orange",
                    borderwidth=1
                )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} - 3-Week Price Chart',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def create_price_line_chart(ticker, stock_data):
    """
    Create a simple line chart showing price movement over time.
    """
    if stock_data is None or stock_data.empty:
        return None
    
    try:
        fig = px.line(
            stock_data,
            x='date',
            y='close',
            title=f'{ticker} - 3-Week Price Movement',
            labels={'close': 'Price ($)', 'date': 'Date'}
        )
        
        # Add moving averages
        if 'ma_20' in stock_data.columns and not stock_data['ma_20'].isna().all():
            fig.add_scatter(
                x=stock_data['date'],
                y=stock_data['ma_20'],
                mode='lines',
                name='MA 20',
                line=dict(color='orange', width=2)
            )
        
        if 'ma_50' in stock_data.columns and not stock_data['ma_50'].isna().all():
            fig.add_scatter(
                x=stock_data['date'],
                y=stock_data['ma_50'],
                mode='lines',
                name='MA 50',
                line=dict(color='blue', width=2)
            )
        
        fig.update_layout(height=400)
        return fig
        
    except Exception as e:
        st.error(f"Error creating line chart: {str(e)}")
        return None

def create_earnings_chart(earnings_data, ticker):
    """
    Create earnings surprise trend chart.
    """
    if earnings_data is None or earnings_data.empty:
        return None
    
    try:
        # Create earnings surprise trend
        fig = px.bar(
            earnings_data,
            x='earnings_date',
            y='surprise_percentage',
            title=f'{ticker} - Earnings Surprise Trend',
            labels={'surprise_percentage': 'Surprise (%)', 'earnings_date': 'Earnings Date'},
            color='surprise_percentage',
            color_continuous_scale='RdYlGn'
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig.update_layout(height=400)
        return fig
        
    except Exception as e:
        st.error(f"Error creating earnings chart: {str(e)}")
        return None
