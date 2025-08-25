"""
Earnings Calculator Utility

Handles all calculations and data processing for earnings analysis.
Extracted from EarningsStocksTab for better separation of concerns.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Custom Exception Classes
class EarningsCalculationError(Exception):
    """Raised when earnings calculations fail"""
    pass

logger = logging.getLogger('StockApp')


class EarningsCalculator:
    """
    Handles all calculations and data processing for earnings analysis.
    
    Responsibilities:
    - Earnings week categorization
    - Volatility calculations
    - Performance metrics
    - Data aggregation and sorting
    """
    
    def __init__(self):
        """Initialize the earnings calculator."""
        logger.info("[EarningsCalculator] Initialized")
    
    def categorize_earnings_by_week(self, earnings_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Categorize earnings data by week relative to current date.
        
        Args:
            earnings_data: DataFrame with earnings data including earnings_date
            
        Returns:
            Dictionary with week categories as keys and DataFrames as values
        """
        try:
            if earnings_data.empty:
                return {
                    'current_week': pd.DataFrame(),
                    'next_week': pd.DataFrame(),
                    'week_after': pd.DataFrame(),
                    'later': pd.DataFrame()
                }
            
            logger.info(f"[categorize_earnings_by_week] Categorizing {len(earnings_data)} earnings records")
            
            # Ensure earnings_date is datetime
            earnings_data = earnings_data.copy()
            earnings_data['earnings_date'] = pd.to_datetime(earnings_data['earnings_date'])
            
            # Get current date and week boundaries
            today = datetime.now().date()
            current_week_start = today - timedelta(days=today.weekday())
            current_week_end = current_week_start + timedelta(days=6)
            
            next_week_start = current_week_end + timedelta(days=1)
            next_week_end = next_week_start + timedelta(days=6)
            
            week_after_start = next_week_end + timedelta(days=1)
            week_after_end = week_after_start + timedelta(days=6)
            
            # Categorize earnings
            categorized = {}
            
            # Current week (Monday to Sunday of current week)
            current_week_mask = (
                (earnings_data['earnings_date'].dt.date >= current_week_start) &
                (earnings_data['earnings_date'].dt.date <= current_week_end)
            )
            categorized['current_week'] = earnings_data[current_week_mask].copy()
            
            # Next week
            next_week_mask = (
                (earnings_data['earnings_date'].dt.date >= next_week_start) &
                (earnings_data['earnings_date'].dt.date <= next_week_end)
            )
            categorized['next_week'] = earnings_data[next_week_mask].copy()
            
            # Week after next
            week_after_mask = (
                (earnings_data['earnings_date'].dt.date >= week_after_start) &
                (earnings_data['earnings_date'].dt.date <= week_after_end)
            )
            categorized['week_after'] = earnings_data[week_after_mask].copy()
            
            # Later (beyond week after next)
            later_mask = earnings_data['earnings_date'].dt.date > week_after_end
            categorized['later'] = earnings_data[later_mask].copy()
            
            # Log categorization results
            for week, data in categorized.items():
                if not data.empty:
                    logger.debug(f"[categorize_earnings_by_week] {week}: {len(data)} earnings")
            
            return categorized
            
        except Exception as e:
            logger.error(f"Error categorizing earnings by week: {e}")
            raise EarningsCalculationError(f"Failed to categorize earnings: {e}") from e
    
    def calculate_pre_earnings_metrics(self, price_data: pd.DataFrame, earnings_symbols: List[str]) -> pd.DataFrame:
        """
        Calculate pre-earnings performance metrics for given symbols.
        
        Args:
            price_data: DataFrame with price data
            earnings_symbols: List of symbols with upcoming earnings
            
        Returns:
            DataFrame with pre-earnings metrics
        """
        try:
            if price_data.empty or not earnings_symbols:
                return pd.DataFrame()
            
            logger.debug(f"[calculate_pre_earnings_metrics] Calculating metrics for {len(earnings_symbols)} symbols")
            
            metrics_list = []
            
            for symbol in earnings_symbols:
                symbol_data = price_data[price_data['symbol'] == symbol].copy()
                
                # Reduced minimum data requirement to 3 points specifically for earnings analysis
                if len(symbol_data) < 3:  
                    logger.debug(f"[calculate_pre_earnings_metrics] Skipping {symbol}: only {len(symbol_data)} data points (need 3+)")
                    continue
                
                # Sort by date and ensure numeric prices
                symbol_data = symbol_data.sort_values('date')
                symbol_data['close'] = pd.to_numeric(symbol_data['close'], errors='coerce')
                symbol_data = symbol_data.dropna(subset=['close'])
                
                # Re-check after cleaning
                if len(symbol_data) < 3:
                    logger.debug(f"[calculate_pre_earnings_metrics] Skipping {symbol}: only {len(symbol_data)} clean data points")
                    continue
                
                # Calculate metrics
                first_price = symbol_data['close'].iloc[0]
                last_price = symbol_data['close'].iloc[-1]
                
                total_return = ((last_price - first_price) / first_price) * 100
                
                # Calculate daily returns
                symbol_data['daily_return'] = symbol_data['close'].pct_change() * 100
                daily_returns = symbol_data['daily_return'].dropna()
                
                if len(daily_returns) == 0:
                    logger.debug(f"[calculate_pre_earnings_metrics] No daily returns for {symbol}")
                    continue
                
                # Calculate volatility metrics with better error handling
                volatility_std = daily_returns.std() if len(daily_returns) > 1 else abs(daily_returns.iloc[0])
                max_daily_gain = daily_returns.max()
                max_daily_loss = daily_returns.min()
                
                # Calculate volatility score
                volatility_score = volatility_std + (max_daily_gain - max_daily_loss) * 0.1
                
                # Price statistics
                max_price = symbol_data['close'].max()
                min_price = symbol_data['close'].min()
                
                metrics_list.append({
                    'symbol': symbol,
                    'total_return': total_return,
                    'volatility_score': volatility_score,
                    'volatility_std': volatility_std,
                    'max_daily_gain': max_daily_gain,
                    'max_daily_loss': max_daily_loss,
                    'max_price': max_price,
                    'min_price': min_price,
                    'first_price': first_price,
                    'last_price': last_price,
                    'data_points': len(symbol_data)
                })
            
            if not metrics_list:
                logger.warning("[calculate_pre_earnings_metrics] No metrics calculated for any symbols")
                return pd.DataFrame()
            
            metrics_df = pd.DataFrame(metrics_list)
            
            # Sort by volatility score (highest first)
            metrics_df = metrics_df.sort_values('volatility_score', ascending=False)
            
            logger.info(f"[calculate_pre_earnings_metrics] Calculated metrics for {len(metrics_df)} symbols")
            return metrics_df
            
        except Exception as e:
            logger.error(f"Error calculating pre-earnings metrics: {e}")
            raise EarningsCalculationError(f"Failed to calculate pre-earnings metrics: {e}") from e
    
    def merge_earnings_with_metrics(self, earnings_data: pd.DataFrame, metrics_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge earnings data with calculated metrics.
        
        Args:
            earnings_data: DataFrame with earnings information
            metrics_data: DataFrame with calculated metrics
            
        Returns:
            DataFrame with merged data
        """
        try:
            if earnings_data.empty:
                return pd.DataFrame()
            
            if metrics_data.empty:
                # Return earnings data with default metric columns
                default_columns = {
                    'total_return': 0.0,
                    'volatility_score': 0.0,
                    'volatility_std': 0.0,
                    'max_daily_gain': 0.0,
                    'max_daily_loss': 0.0,
                    'max_price': 0.0,
                    'min_price': 0.0,
                    'first_price': 0.0,
                    'last_price': 0.0,
                    'data_points': 0
                }
                
                for col, default_val in default_columns.items():
                    if col not in earnings_data.columns:
                        earnings_data[col] = default_val
                
                return earnings_data
            
            logger.info(f"[merge_earnings_with_metrics] Merging {len(earnings_data)} earnings with {len(metrics_data)} metrics")
            
            # Merge on symbol
            merged_data = earnings_data.merge(metrics_data, on='symbol', how='left')
            
            # Fill missing metric values with defaults
            metric_columns = ['total_return', 'volatility_score', 'volatility_std', 'max_daily_gain', 
                            'max_daily_loss', 'max_price', 'min_price', 'first_price', 'last_price', 'data_points']
            
            for col in metric_columns:
                if col in merged_data.columns:
                    merged_data[col] = merged_data[col].fillna(0.0)
                else:
                    merged_data[col] = 0.0
            
            logger.info(f"[merge_earnings_with_metrics] Merged data: {len(merged_data)} records")
            return merged_data
            
        except Exception as e:
            logger.error(f"Error merging earnings with metrics: {e}")
            raise EarningsCalculationError(f"Failed to merge earnings with metrics: {e}") from e
    
    def sort_by_returns(self, data: pd.DataFrame, ascending: bool = False) -> List[str]:
        """
        Sort symbols by total returns.
        
        Args:
            data: DataFrame with total_return column
            ascending: Whether to sort in ascending order
            
        Returns:
            List of symbols sorted by returns
        """
        try:
            if data.empty or 'total_return' not in data.columns:
                return []
            
            # Sort by total returns
            sorted_data = data.sort_values('total_return', ascending=ascending)
            
            return sorted_data['symbol'].tolist()
            
        except Exception as e:
            logger.error(f"Error sorting by returns: {e}")
            return data['symbol'].tolist() if 'symbol' in data.columns else []
    
    def calculate_week_summary_stats(self, week_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate summary statistics for a week's earnings data.
        
        Args:
            week_data: DataFrame with week's earnings data
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            if week_data.empty:
                return {
                    'total_stocks': 0,
                    'avg_volatility': 0.0,
                    'avg_return': 0.0,
                    'max_return': 0.0,
                    'min_return': 0.0,
                    'positive_return_count': 0,
                    'positive_return_pct': 0.0
                }
            
            # Calculate statistics
            total_stocks = len(week_data)
            
            # Volatility and return statistics
            if 'volatility_score' in week_data.columns:
                avg_volatility = week_data['volatility_score'].mean()
            else:
                avg_volatility = 0.0
            
            if 'total_return' in week_data.columns:
                avg_return = week_data['total_return'].mean()
                max_return = week_data['total_return'].max()
                min_return = week_data['total_return'].min()
                positive_return_count = len(week_data[week_data['total_return'] > 0])
                positive_return_pct = (positive_return_count / total_stocks) * 100 if total_stocks > 0 else 0.0
            else:
                avg_return = max_return = min_return = 0.0
                positive_return_count = 0
                positive_return_pct = 0.0
            
            return {
                'total_stocks': total_stocks,
                'avg_volatility': avg_volatility,
                'avg_return': avg_return,
                'max_return': max_return,
                'min_return': min_return,
                'positive_return_count': positive_return_count,
                'positive_return_pct': positive_return_pct
            }
            
        except Exception as e:
            logger.error(f"Error calculating week summary stats: {e}")
            return {
                'total_stocks': 0,
                'avg_volatility': 0.0,
                'avg_return': 0.0,
                'max_return': 0.0,
                'min_return': 0.0,
                'positive_return_count': 0,
                'positive_return_pct': 0.0
            }
    
    def filter_top_performers(self, data: pd.DataFrame, top_n: int = 25, sort_by: str = 'volatility_score') -> pd.DataFrame:
        """
        Filter to top N performers based on specified metric.
        
        Args:
            data: DataFrame with performance data
            top_n: Number of top performers to return
            sort_by: Column to sort by ('volatility_score', 'total_return', etc.)
            
        Returns:
            DataFrame with top N performers
        """
        try:
            if data.empty or sort_by not in data.columns:
                return data
            
            # Sort by specified column (highest first)
            top_performers = data.nlargest(top_n, sort_by)
            
            logger.info(f"[filter_top_performers] Filtered to top {len(top_performers)} performers by {sort_by}")
            return top_performers
            
        except Exception as e:
            logger.error(f"Error filtering top performers: {e}")
            return data
    
    def calculate_earnings_calendar_stats(self, categorized_earnings: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for all earnings calendar categories.
        
        Args:
            categorized_earnings: Dictionary with categorized earnings data
            
        Returns:
            Dictionary with statistics for each category
        """
        try:
            calendar_stats = {}
            
            for category, data in categorized_earnings.items():
                calendar_stats[category] = self.calculate_week_summary_stats(data)
            
            logger.info("[calculate_earnings_calendar_stats] Calculated stats for all earnings categories")
            return calendar_stats
            
        except Exception as e:
            logger.error(f"Error calculating earnings calendar stats: {e}")
            return {category: {} for category in categorized_earnings.keys()} 