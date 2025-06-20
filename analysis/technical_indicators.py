"""
Technical Indicators Module

Handles calculation of various technical indicators like MACD, Bollinger Bands, etc.
"""

import numpy as np
import pandas as pd
from .rsi_calculator import RSICalculator


class TechnicalIndicators:
    """
    Class for calculating various technical indicators
    """
    
    def __init__(self):
        """Initialize technical indicators calculator"""
        self.rsi_calculator = RSICalculator()
    
    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given data
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with all technical indicators added
        """
        df = data.copy()
        
        # Clean input data first
        df = self._clean_data(df)
        
        # Calculate all indicators
        df = self._calculate_moving_averages(df)
        df = self._calculate_rsi(df)
        df = self._calculate_macd(df)
        df = self._calculate_bollinger_bands(df)
        df = self._calculate_volume_indicators(df)
        df = self._calculate_price_indicators(df)
        
        # Final cleanup
        df = self._clean_indicators(df)
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean input data"""
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Replace infinite values
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages"""
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI using the dedicated RSI calculator"""
        if len(df) >= 14:
            rsi_values = self.rsi_calculator.calculate(df['close'].values)
            # Ensure proper alignment - RSI starts from index 14 (period + 1)
            if len(rsi_values) > 0:
                # Create a full RSI column with NaN for initial values
                df['rsi'] = np.nan
                start_idx = 14  # period + 1
                end_idx = start_idx + len(rsi_values)
                if end_idx <= len(df):
                    df.iloc[start_idx:end_idx, df.columns.get_loc('rsi')] = rsi_values
        else:
            df['rsi'] = np.nan
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators"""
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        if 'volume' in df.columns and not df['volume'].isna().all():
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            # Avoid division by zero
            volume_ma_safe = df['volume_ma'].replace(0, np.nan)
            df['volume_ratio'] = df['volume'] / volume_ma_safe
            # Fill infinite ratios
            df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], np.nan)
            
            # Volume-weighted average price (VWAP)
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        else:
            df['volume_ma'] = 1.0
            df['volume_ratio'] = 1.0
            df['vwap'] = df['close']
        
        return df
    
    def _calculate_price_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based indicators"""
        # Price changes and volatility
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        # True Range and Average True Range
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # Drop intermediate columns
        df = df.drop(['high_low', 'high_close', 'low_close'], axis=1)
        
        return df
    
    def _clean_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean all calculated indicators"""
        indicator_columns = [
            'ma_5', 'ma_20', 'ma_50', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'volume_ma', 'volume_ratio', 'vwap', 'price_change', 'volatility',
            'true_range', 'atr'
        ]
        
        for col in indicator_columns:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def get_latest_indicators(self, data: pd.DataFrame) -> dict:
        """
        Get the latest values of all indicators
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            Dictionary with latest indicator values
        """
        df = self.calculate_all(data)
        
        if len(df) == 0:
            return {}
        
        latest = df.iloc[-1]
        
        return {
            'price': latest.get('close', 0),
            'rsi': latest.get('rsi', 0),
            'macd': latest.get('macd', 0),
            'macd_signal': latest.get('macd_signal', 0),
            'bb_position': latest.get('bb_position', 0),
            'volume_ratio': latest.get('volume_ratio', 1),
            'volatility': latest.get('volatility', 0),
            'atr': latest.get('atr', 0)
        } 