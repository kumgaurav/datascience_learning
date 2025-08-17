import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta

def load_stock_data(ticker, days=21):
    """
    Load historical stock data for a given ticker.
    """
    try:
        # Load stock prices data
        stock_prices_file = os.getenv('STOCK_PRICES_CSV', 'data/stock_prices.csv')
        if os.path.exists(stock_prices_file):
            stock_data = pd.read_csv(stock_prices_file)
            
            # Filter for the specific ticker
            ticker_data = stock_data[stock_data['ticker'] == ticker.upper()].copy()
            
            if not ticker_data.empty:
                # Convert date column
                ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                
                # Sort by date and get last N days
                ticker_data = ticker_data.sort_values('date')
                ticker_data = ticker_data.tail(days)
                
                return ticker_data
            else:
                st.warning(f"No data found for ticker {ticker}")
                return None
        else:
            st.error("Stock prices data file not found")
            return None
            
    except Exception as e:
        st.error(f"Error loading stock data: {str(e)}")
        return None

def load_featured_stocks_data():
    """
    Load featured stocks data with all technical indicators.
    """
    try:
        featured_file = os.getenv('FEATURED_STOCKS_CSV', 'data/featured_stocks.csv')
        if os.path.exists(featured_file):
            featured_data = pd.read_csv(featured_file)
            return featured_data
        else:
            st.error("Featured stocks data file not found")
            return None
    except Exception as e:
        st.error(f"Error loading featured stocks data: {str(e)}")
        return None

def load_earnings_history_data():
    """
    Load earnings history data.
    """
    try:
        earnings_file = os.getenv('EARNINGS_HISTORY_CSV', 'data/earnings_history.csv')
        if os.path.exists(earnings_file):
            earnings_data = pd.read_csv(earnings_file)
            earnings_data['earnings_date'] = pd.to_datetime(earnings_data['earnings_date'])
            return earnings_data
        else:
            st.warning("Earnings history data file not found")
            return None
    except Exception as e:
        st.error(f"Error loading earnings data: {str(e)}")
        return None

def get_stock_featured_data(featured_data, ticker, top_stocks_df=None):
    """
    Get featured data for a specific stock, combining with top stocks data if available.
    """
    if featured_data is None:
        return None
    
    # Get data for the specific ticker
    stock_data = featured_data[featured_data['ticker'] == ticker.upper()]
    
    if stock_data.empty:
        return None
    
    stock_featured_data = stock_data.iloc[0].copy()
    
    # If we have top stocks data, merge prediction-related columns
    if top_stocks_df is not None:
        top_stock_data = top_stocks_df[top_stocks_df['ticker'] == ticker.upper()]
        if not top_stock_data.empty:
            prediction_cols = ['predicted_return_pct', 'confidence_score', 'predicted_change', 
                             'risk_score', 'composite_score']
            for col in prediction_cols:
                if col in top_stock_data.columns:
                    stock_featured_data[col] = top_stock_data[col].iloc[0]
    
    return stock_featured_data

def get_earnings_data_for_ticker(earnings_data, ticker):
    """
    Get earnings data for a specific ticker.
    """
    if earnings_data is None:
        return None
    
    ticker_earnings = earnings_data[earnings_data['ticker'] == ticker.upper()].copy()
    
    if ticker_earnings.empty:
        return None
    
    # Sort by earnings date and get last 4 quarters
    ticker_earnings = ticker_earnings.sort_values('earnings_date', ascending=False)
    ticker_earnings = ticker_earnings.head(4)
    
    return ticker_earnings

def format_price_stats(stock_data):
    """
    Format price statistics for display.
    """
    if stock_data is None or stock_data.empty:
        return None
    
    stats = {}
    
    # Basic price stats
    stats['current_price'] = stock_data['close'].iloc[-1]
    stats['price_change'] = stock_data['close'].iloc[-1] - stock_data['close'].iloc[0]
    stats['price_change_pct'] = (stats['price_change'] / stock_data['close'].iloc[0]) * 100
    
    # High and low
    stats['period_high'] = stock_data['high'].max()
    stats['period_low'] = stock_data['low'].min()
    
    # Volume stats
    stats['avg_volume'] = stock_data['volume'].mean()
    stats['current_volume'] = stock_data['volume'].iloc[-1]
    stats['volume_change'] = ((stats['current_volume'] - stats['avg_volume']) / stats['avg_volume']) * 100
    
    return stats

def calculate_price_movement_summary(stock_data):
    """
    Calculate price movement summary for the period.
    """
    if stock_data is None or stock_data.empty:
        return None
    
    summary = {}
    
    # Calculate daily returns
    stock_data['daily_return'] = stock_data['close'].pct_change()
    
    # Summary statistics
    summary['total_return'] = ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[0]) - 1) * 100
    summary['volatility'] = stock_data['daily_return'].std() * 100
    summary['positive_days'] = len(stock_data[stock_data['daily_return'] > 0])
    summary['negative_days'] = len(stock_data[stock_data['daily_return'] < 0])
    summary['total_days'] = len(stock_data)
    
    return summary
