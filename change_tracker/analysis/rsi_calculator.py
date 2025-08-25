"""
RSI Calculator Module

Handles all RSI (Relative Strength Index) calculations and analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class RSICalculator:
    """
    Dedicated class for RSI calculations and analysis
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize RSI calculator
        
        Args:
            period: RSI calculation period (default: 14)
        """
        self.period = period
    
    def calculate(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Array of closing prices
        
        Returns:
            Array of RSI values
        """
        # Convert to pandas Series for easier data cleaning
        if isinstance(prices, np.ndarray):
            prices_series = pd.Series(prices)
        else:
            prices_series = pd.Series(prices)
        
        # Clean the data: remove NaN values and convert to numeric
        prices_series = pd.to_numeric(prices_series, errors='coerce')
        prices_series = prices_series.dropna()
        
        # Convert back to numpy array
        prices_clean = prices_series.values
        
        if len(prices_clean) < self.period + 1:
            return np.array([])
        
        # Calculate price changes
        deltas = np.diff(prices_clean)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:self.period])
        avg_loss = np.mean(losses[:self.period])
        
        # Initialize RSI array - make sure it matches the expected output length
        rsi = np.zeros(len(prices_clean) - self.period)
        
        # Calculate RSI for each period
        for i in range(self.period, len(prices_clean)):
            if i == self.period:
                rsi[i - self.period] = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 100
            else:
                # Exponential moving average
                avg_gain = (avg_gain * (self.period - 1) + gains[i - 1]) / self.period
                avg_loss = (avg_loss * (self.period - 1) + losses[i - 1]) / self.period
                
                if avg_loss == 0:
                    rsi[i - self.period] = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi[i - self.period] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def analyze_periods_above_threshold(self, stock_data: pd.DataFrame, threshold: float = 75) -> Dict:
        """
        Analyze periods when RSI exceeds a threshold
        
        Args:
            stock_data: DataFrame with stock data including 'date' and 'close' columns
            threshold: RSI threshold value (default: 75)
        
        Returns:
            Dictionary with analysis results
        """
        if len(stock_data) < self.period + 1:  # Need at least period+1 data points
            return {
                'periods': [],
                'count': 0,
                'percentage': 0,
                'max_rsi': 0,
                'error': f'Insufficient data: need at least {self.period + 1} data points'
            }
        
        rsi_values = self.calculate(stock_data['close'].values)
        
        if len(rsi_values) == 0:
            return {
                'periods': [],
                'count': 0,
                'percentage': 0,
                'max_rsi': 0,
                'error': 'No RSI values calculated'
            }
        
        # RSI starts from index period (after the period)
        start_idx = self.period
        dates = stock_data['date'].iloc[start_idx:start_idx + len(rsi_values)]
        
        high_rsi_mask = rsi_values > threshold
        high_rsi_periods = []
        
        # Ensure we don't go out of bounds
        min_len = min(len(dates), len(rsi_values), len(high_rsi_mask))
        
        for i in range(min_len):
            if high_rsi_mask[i]:
                high_rsi_periods.append({
                    'date': dates.iloc[i],
                    'rsi': rsi_values[i],
                    'price': stock_data.iloc[start_idx + i]['close']
                })
        
        return {
            'periods': high_rsi_periods,
            'count': len(high_rsi_periods),
            'percentage': (len(high_rsi_periods) / len(rsi_values)) * 100 if len(rsi_values) > 0 else 0,
            'max_rsi': max(rsi_values) if len(rsi_values) > 0 else 0,
            'total_periods': len(rsi_values)
        }
    
    def get_current_rsi(self, stock_data: pd.DataFrame) -> float:
        """
        Get the most recent RSI value
        
        Args:
            stock_data: DataFrame with stock data
        
        Returns:
            Most recent RSI value or 0 if calculation fails
        """
        rsi_values = self.calculate(stock_data['close'].values)
        return rsi_values[-1] if len(rsi_values) > 0 else 0.0
    
    def get_rsi_signals(self, stock_data: pd.DataFrame, 
                       oversold_threshold: float = 30, 
                       overbought_threshold: float = 70) -> List[Dict]:
        """
        Generate buy/sell signals based on RSI levels
        
        Args:
            stock_data: DataFrame with stock data
            oversold_threshold: RSI level considered oversold (buy signal)
            overbought_threshold: RSI level considered overbought (sell signal)
        
        Returns:
            List of signal dictionaries
        """
        rsi_values = self.calculate(stock_data['close'].values)
        
        if len(rsi_values) == 0:
            return []
        
        signals = []
        start_idx = self.period
        
        for i, rsi_val in enumerate(rsi_values):
            data_idx = start_idx + i
            if data_idx < len(stock_data):
                signal_type = 'HOLD'
                
                if rsi_val <= oversold_threshold:
                    signal_type = 'BUY'
                elif rsi_val >= overbought_threshold:
                    signal_type = 'SELL'
                
                signals.append({
                    'date': stock_data.iloc[data_idx]['date'],
                    'close': stock_data.iloc[data_idx]['close'],
                    'rsi': rsi_val,
                    'signal': signal_type,
                    'confidence': abs(rsi_val - 50) * 2  # Distance from neutral (50)
                })
        
        return signals 