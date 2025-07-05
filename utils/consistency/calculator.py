"""
Consistency Calculator Utility

Handles all statistical calculations and consistency analysis.
Extracted from ConsistentPerformersTab for better separation of concerns.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np

# Configuration constants
TOP_PERFORMERS_COUNT = 25
MIN_DATA_POINTS_SHORT_TERM = 3  # For 1-month analysis
MIN_DATA_POINTS_LONG_TERM = 5   # For 2+ month analysis
MAX_REASONABLE_RETURN = 1000  # Cap for data error filtering

# Consistency index weights
CONSISTENCY_WEIGHTS = {
    'positive_months': 0.4,
    'low_volatility': 0.3,
    'geometric_mean': 0.3
}

# Custom Exception Classes
class CalculationError(Exception):
    """Raised when statistical calculations fail"""
    pass

logger = logging.getLogger('StockApp')


class ConsistencyCalculator:
    """
    Handles all statistical calculations for consistency analysis.
    
    Responsibilities:
    - Consistency statistics calculation
    - Monthly performance metrics
    - Statistical filtering and ranking
    - Geometric mean and coefficient of variation calculations
    """
    
    def calculate_consistency_statistics(self, data: pd.DataFrame, months: int = 3) -> pd.DataFrame:
        """
        Calculate consistency statistics for stock data.
        
        Args:
            data: Stock price data grouped by symbol
            months: Number of months to analyze (1, 2, or 3)
            
        Returns:
            DataFrame with consistency statistics for all qualifying symbols
        """
        try:
            # Calculate date range for the analysis
            end_date = datetime.now()
            if months == 1:
                start_date = datetime(end_date.year, end_date.month, 1)
            else:
                start_date = end_date - timedelta(days=30*months)
            
            logger.info(f"[calculate_consistency_statistics] Calculating statistics for {months} months: {start_date.date()} to {end_date.date()}")
            
            symbol_groups = data.groupby('symbol')
            min_data_points = self._get_min_data_points(months)
            
            consistency_list = []
            for symbol, group_data in symbol_groups:
                metrics = self._calculate_symbol_consistency(group_data, symbol, start_date, end_date, months, min_data_points)
                if metrics:
                    consistency_list.append(metrics)
            
            if not consistency_list:
                logger.warning(f"[calculate_consistency_statistics] No stocks passed filtering for {months}-month analysis")
                return pd.DataFrame()
            
            consistency_df = pd.DataFrame(consistency_list)
            return self._filter_and_rank_results(consistency_df, months)
            
        except Exception as e:
            logger.error(f"Error calculating consistency statistics: {e}")
            return pd.DataFrame()
    
    def _get_min_data_points(self, months: int) -> int:
        """
        Determine minimum data points required based on analysis period.
        
        Args:
            months: Number of months to analyze (1, 2, or 3)
            
        Returns:
            Minimum number of data points required for analysis
        """
        if months == 1:
            min_data_points = 3  # Relaxed for 1-month
            logger.info(f"[_get_min_data_points] Using relaxed requirement of {min_data_points} minimum data points for 1-month analysis")
        else:
            min_data_points = max(5, months * 5)
            logger.info(f"[_get_min_data_points] Using standard requirement of {min_data_points} minimum data points for {months}-month analysis")
        
        return min_data_points
    
    def _calculate_symbol_consistency(self, group_data: pd.DataFrame, symbol: str, start_date: datetime, 
                                    end_date: datetime, months: int, min_data_points: int) -> Optional[Dict[str, Any]]:
        """
        Calculate consistency metrics for a single symbol.
        
        Args:
            group_data: Price data for a single symbol
            symbol: Stock symbol
            start_date: Analysis start date
            end_date: Analysis end date
            months: Number of months to analyze
            min_data_points: Minimum required data points
            
        Returns:
            Dictionary with consistency metrics or None if insufficient data
        """
        try:
            # Ensure data is sorted by date
            group_data = group_data.sort_values('date')
            
            # Need sufficient data points for meaningful consistency analysis
            if len(group_data) < min_data_points:
                logger.debug(f"[_calculate_symbol_consistency] Skipping {symbol}: only {len(group_data)} data points, need {min_data_points}")
                return None
            
            # Calculate monthly consistency metrics
            monthly_stats = self._calculate_monthly_consistency(group_data, start_date, end_date, months)
            
            if not monthly_stats or len(monthly_stats['monthly_returns']) < months:
                logger.debug(f"[_calculate_symbol_consistency] Skipping {symbol}: insufficient monthly data")
                return None
            
            return self._build_symbol_metrics(group_data, symbol, monthly_stats)
            
        except Exception as e:
            logger.debug(f"Error processing symbol {symbol}: {e}")
            return None
    
    def _build_symbol_metrics(self, group_data: pd.DataFrame, symbol: str, monthly_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build comprehensive metrics dictionary for a symbol.
        
        Args:
            group_data: Price data for the symbol
            symbol: Stock symbol
            monthly_stats: Dictionary containing monthly statistics
            
        Returns:
            Dictionary with comprehensive metrics for the symbol
        """
        monthly_returns = monthly_stats['monthly_returns']
        
        # Statistical consistency calculations
        geometric_mean = self._geometric_mean(monthly_returns)
        cv = self._calculate_coefficient_variation(monthly_returns)
        positive_ratio = sum(1 for r in monthly_returns if r > 0) / len(monthly_returns)
        consistency_index = self._calculate_consistency_index(positive_ratio, cv, geometric_mean)
        
        # Total return calculation
        first_price = group_data.iloc[0]['close']
        last_price = group_data.iloc[-1]['close']
        total_return = ((last_price - first_price) / first_price) * 100
        
        # Create result record
        result = {
            'symbol': symbol,
            'consistency_index': consistency_index,
            'geometric_mean_return': geometric_mean,
            'volatility_cv': cv,
            'positive_months_ratio': positive_ratio,
            'min_monthly_return': min(monthly_returns),
            'total_return': total_return,
            'first_price': first_price,
            'last_price': last_price,
            'data_points': len(group_data),
            'first_date': group_data.iloc[0]['date'].strftime('%Y-%m-%d'),
            'last_date': group_data.iloc[-1]['date'].strftime('%Y-%m-%d')
        }
        
        # Add monthly performance for visualization
        result.update(monthly_stats['monthly_display'])
        return result
    
    def _calculate_coefficient_variation(self, monthly_returns: List[float]) -> float:
        """
        Calculate coefficient of variation (volatility relative to mean).
        
        Args:
            monthly_returns: List of monthly return percentages
            
        Returns:
            Coefficient of variation
        """
        if not monthly_returns:
            return float('inf')
        
        mean_return = np.mean(monthly_returns)
        std_return = np.std(monthly_returns)
        return std_return / abs(mean_return) if mean_return != 0 else float('inf')
    
    def _calculate_consistency_index(self, positive_ratio: float, cv: float, geometric_mean: float) -> float:
        """
        Calculate consistency index using weighted formula.
        
        Args:
            positive_ratio: Ratio of positive months
            cv: Coefficient of variation
            geometric_mean: Geometric mean return
            
        Returns:
            Weighted consistency index (0-1 scale)
        """
        # Normalize CV (cap at 2.0 for extreme cases)
        normalized_cv = min(cv, 2.0) / 2.0
        # Normalize geometric mean (assume max reasonable monthly return is 50%)
        normalized_geometric_mean = max(0, min(geometric_mean / 50.0, 1.0))
        
        return (
            positive_ratio * CONSISTENCY_WEIGHTS['positive_months'] +
            (1 - normalized_cv) * CONSISTENCY_WEIGHTS['low_volatility'] +
            normalized_geometric_mean * CONSISTENCY_WEIGHTS['geometric_mean']
        )
    
    def _filter_and_rank_results(self, consistency_df: pd.DataFrame, months: int) -> pd.DataFrame:
        """
        Filter and rank consistency results based on analysis period.
        
        Args:
            consistency_df: DataFrame with consistency calculations
            months: Number of months analyzed
            
        Returns:
            Filtered and ranked DataFrame with top performers
        """
        original_count = len(consistency_df)
        
        # Basic filtering
        consistency_df = consistency_df[consistency_df['consistency_index'] > 0]
        consistency_df = consistency_df[consistency_df['total_return'] <= MAX_REASONABLE_RETURN]
        
        # Period-specific filtering
        if months == 1:
            consistency_df = self._apply_relaxed_filtering(consistency_df, original_count)
        else:
            consistency_df = self._apply_strict_filtering(consistency_df, original_count, months)
        
        # Return top performers sorted by consistency index
        return consistency_df.nlargest(TOP_PERFORMERS_COUNT, 'consistency_index')
    
    def _apply_relaxed_filtering(self, consistency_df: pd.DataFrame, original_count: int) -> pd.DataFrame:
        """
        Apply relaxed filtering for 1-month analysis.
        
        Args:
            consistency_df: DataFrame to filter
            original_count: Original number of candidates
            
        Returns:
            Filtered DataFrame
        """
        logger.info("[_apply_relaxed_filtering] Using relaxed filtering criteria for 1-month analysis")
        
        # For positive months ratio, only require it to be defined (>= 0)
        before_months_filter = len(consistency_df)
        consistency_df = consistency_df[consistency_df['positive_months_ratio'] >= 0]
        after_months_filter = len(consistency_df)
        
        logger.info(f"[_apply_relaxed_filtering] Filtering results:")
        logger.info(f"[_apply_relaxed_filtering]   Original candidates: {original_count}")
        logger.info(f"[_apply_relaxed_filtering]   After months filter: {after_months_filter} (removed {before_months_filter - after_months_filter})")
        logger.info(f"[_apply_relaxed_filtering]   Final consistent performers: {len(consistency_df)}")
        
        return consistency_df
    
    def _apply_strict_filtering(self, consistency_df: pd.DataFrame, original_count: int, months: int) -> pd.DataFrame:
        """
        Apply strict filtering for 2+ month analysis.
        
        Args:
            consistency_df: DataFrame to filter
            original_count: Original number of candidates
            months: Number of months analyzed
            
        Returns:
            Filtered DataFrame
        """
        # Only include stocks with positive total returns
        before_positive_filter = len(consistency_df)
        consistency_df = consistency_df[consistency_df['total_return'] > 0]
        after_positive_filter = len(consistency_df)
        
        # Only include stocks with at least 1 positive month
        before_months_filter = len(consistency_df)
        consistency_df = consistency_df[consistency_df['positive_months_ratio'] > 0]
        after_months_filter = len(consistency_df)
        
        logger.info(f"[_apply_strict_filtering] Filtering results for {months} months:")
        logger.info(f"[_apply_strict_filtering]   Original candidates: {original_count}")
        logger.info(f"[_apply_strict_filtering]   After positive return filter: {after_positive_filter} (removed {before_positive_filter - after_positive_filter})")
        logger.info(f"[_apply_strict_filtering]   After positive months filter: {after_months_filter} (removed {before_months_filter - after_months_filter})")
        logger.info(f"[_apply_strict_filtering]   Final consistent performers: {len(consistency_df)}")
        
        return consistency_df
    
    def _geometric_mean(self, returns: List[float]) -> float:
        """
        Calculate geometric mean of returns (compound growth rate).
        
        Args:
            returns: List of return percentages
            
        Returns:
            Geometric mean return percentage
        """
        if len(returns) == 0:
            return 0
        
        # Convert percentages to growth factors, calculate geometric mean, convert back
        growth_factors = [(r/100 + 1) for r in returns if r/100 + 1 > 0]
        if len(growth_factors) == 0:
            return 0
        
        geometric_mean_factor = np.prod(growth_factors) ** (1/len(growth_factors))
        return (geometric_mean_factor - 1) * 100
    
    def _calculate_monthly_consistency(self, group_data: pd.DataFrame, start_date: datetime, end_date: datetime, months: int = 3) -> Dict[str, Any]:
        """
        Calculate monthly consistency metrics for a single symbol.
        
        Args:
            group_data: DataFrame for a single symbol
            start_date: Start date
            end_date: End date
            months: Number of months to analyze (1, 2, or 3)
        
        Returns:
            Dictionary with monthly consistency data including returns and display info
        """
        try:
            # Add month-year column
            group_data = group_data.copy()
            group_data['month_year'] = group_data['date'].dt.to_period('M')
            
            # Group by month and get first/last prices
            monthly_stats = group_data.groupby('month_year')['close'].agg(['first', 'last', 'count']).reset_index()
            
            # Calculate monthly returns (including negative ones for consistency analysis)
            monthly_stats['monthly_return'] = ((monthly_stats['last'] - monthly_stats['first']) / monthly_stats['first']) * 100
            
            # Get the last N months of data based on the specified period
            monthly_stats = monthly_stats.tail(months)
            
            # Prepare return data
            monthly_returns = monthly_stats['monthly_return'].tolist()
            
            # Initialize default values for display based on analysis period
            display_data = {}
            
            # Initialize all possible months
            for i in range(1, 4):  # Always support up to 3 months
                display_data[f'month_{i}_pct'] = 0.0
                display_data[f'month_{i}_name'] = 'N/A'
                display_data[f'month_{i}_consistent'] = False
            
            # Fill in actual values for the specified period
            for i, (_, row) in enumerate(monthly_stats.iterrows()):
                month_idx = min(i + 1, 3)  # Cap at 3 months
                month_key = f'month_{month_idx}_pct'
                month_name_key = f'month_{month_idx}_name'
                month_consistent_key = f'month_{month_idx}_consistent'
                
                return_val = row['monthly_return']
                display_data[month_key] = return_val
                display_data[month_name_key] = str(row['month_year'])
                display_data[month_consistent_key] = return_val > 0  # Positive return = consistent
            
            return {
                'monthly_returns': monthly_returns,
                'monthly_display': display_data
            }
            
        except Exception as e:
            logger.debug(f"Error calculating monthly consistency: {e}")
            return {
                'monthly_returns': [],
                'monthly_display': {
                    'month_1_pct': 0.0,
                    'month_2_pct': 0.0,
                    'month_3_pct': 0.0,
                    'month_1_name': 'N/A',
                    'month_2_name': 'N/A',
                    'month_3_name': 'N/A',
                    'month_1_consistent': False,
                    'month_2_consistent': False,
                    'month_3_consistent': False
                }
            }
    
    def process_precalculated_data(self, pre_calculated_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and filter pre-calculated consistency data.
        
        Args:
            pre_calculated_data: Pre-calculated consistency analysis results
            
        Returns:
            Filtered DataFrame with valid consistency data
        """
        logger.info(f"[process_precalculated_data] Processing pre-calculated data: {len(pre_calculated_data)} stocks")
        
        # Apply same filtering as fresh data
        filtered_data = pre_calculated_data[pre_calculated_data['total_return'] > 0]
        filtered_data = filtered_data[filtered_data['positive_months_ratio'] > 0]
        
        logger.info(f"[process_precalculated_data] After filtering: {len(filtered_data)} stocks")
        return filtered_data
    
    def calculate_cross_period_rankings(self, all_time_performers: pd.DataFrame, consistent_1m: pd.DataFrame, 
                                      consistent_2m: pd.DataFrame, consistent_3m: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-period rankings for all-time performers.
        
        Args:
            all_time_performers: DataFrame with all-time performers
            consistent_1m: 1-month consistency data
            consistent_2m: 2-month consistency data
            consistent_3m: 3-month consistency data
            
        Returns:
            DataFrame with cross-period ranking comparison
        """
        try:
            comparison_data = []
            
            for _, row in all_time_performers.iterrows():
                symbol = row['symbol']
                
                # Find ranking in each period
                rank_1m = (consistent_1m['consistency_index'] > 
                          consistent_1m[consistent_1m['symbol'] == symbol]['consistency_index'].iloc[0]).sum() + 1
                rank_2m = (consistent_2m['consistency_index'] > 
                          consistent_2m[consistent_2m['symbol'] == symbol]['consistency_index'].iloc[0]).sum() + 1  
                rank_3m = (consistent_3m['consistency_index'] > 
                          consistent_3m[consistent_3m['symbol'] == symbol]['consistency_index'].iloc[0]).sum() + 1
                
                comparison_data.append({
                    'Symbol': symbol,
                    '1M Rank': f"#{rank_1m}",
                    '2M Rank': f"#{rank_2m}",
                    '3M Rank': f"#{rank_3m}",
                    'Avg Rank': f"#{(rank_1m + rank_2m + rank_3m) / 3:.1f}",
                    'Consistency Stability': "🟢 High" if max(rank_1m, rank_2m, rank_3m) - min(rank_1m, rank_2m, rank_3m) <= 10 else "🟡 Moderate" if max(rank_1m, rank_2m, rank_3m) - min(rank_1m, rank_2m, rank_3m) <= 20 else "🔴 Variable"
                })
            
            return pd.DataFrame(comparison_data)
            
        except Exception as e:
            logger.error(f"Error calculating cross-period rankings: {e}")
            return pd.DataFrame()
    
    def find_all_time_performers(self, data_1m: pd.DataFrame, data_2m: pd.DataFrame, data_3m: pd.DataFrame) -> pd.DataFrame:
        """
        Find stocks that appear in all 3 time periods - the ultimate consistency champions.
        
        Args:
            data_1m: 1-month consistency data
            data_2m: 2-month consistency data  
            data_3m: 3-month consistency data
            
        Returns:
            DataFrame with stocks appearing in all periods, with aggregated metrics
        """
        logger.info("[find_all_time_performers] Finding stocks appearing in all 3 time periods")
        
        if any(df.empty for df in [data_1m, data_2m, data_3m]):
            logger.warning("[find_all_time_performers] One or more period datasets are empty")
            return pd.DataFrame()
        
        # Get symbols that appear in all 3 periods
        symbols_1m = set(data_1m['symbol'].tolist())
        symbols_2m = set(data_2m['symbol'].tolist()) 
        symbols_3m = set(data_3m['symbol'].tolist())
        
        all_time_symbols = symbols_1m.intersection(symbols_2m).intersection(symbols_3m)
        
        if not all_time_symbols:
            logger.info("[find_all_time_performers] No stocks found in all 3 time periods")
            return pd.DataFrame()
        
        logger.info(f"[find_all_time_performers] Found {len(all_time_symbols)} all-time performers: {sorted(list(all_time_symbols))}")
        
        # Create aggregated data for all-time performers
        all_time_data = []
        
        for symbol in all_time_symbols:
            # Get data for this symbol from each period
            data_1m_symbol = data_1m[data_1m['symbol'] == symbol].iloc[0]
            data_2m_symbol = data_2m[data_2m['symbol'] == symbol].iloc[0]
            data_3m_symbol = data_3m[data_3m['symbol'] == symbol].iloc[0]
            
            # Calculate aggregated metrics
            avg_consistency = (data_1m_symbol['consistency_index'] + 
                             data_2m_symbol['consistency_index'] + 
                             data_3m_symbol['consistency_index']) / 3
            
            avg_positive_ratio = (data_1m_symbol['positive_months_ratio'] + 
                                data_2m_symbol['positive_months_ratio'] + 
                                data_3m_symbol['positive_months_ratio']) / 3
            
            avg_volatility = (data_1m_symbol['volatility_cv'] + 
                            data_2m_symbol['volatility_cv'] + 
                            data_3m_symbol['volatility_cv']) / 3
            
            # Use 3-month data as base for comprehensive metrics
            all_time_record = {
                'symbol': symbol,
                'consistency_index': avg_consistency,
                'consistency_1m': data_1m_symbol['consistency_index'],
                'consistency_2m': data_2m_symbol['consistency_index'],
                'consistency_3m': data_3m_symbol['consistency_index'],
                'avg_positive_months_ratio': avg_positive_ratio,
                'positive_ratio_1m': data_1m_symbol['positive_months_ratio'],
                'positive_ratio_2m': data_2m_symbol['positive_months_ratio'],
                'positive_ratio_3m': data_3m_symbol['positive_months_ratio'],
                'avg_volatility_cv': avg_volatility,
                'total_return_1m': data_1m_symbol['total_return'],
                'total_return_2m': data_2m_symbol['total_return'],
                'total_return_3m': data_3m_symbol['total_return'],
                'geometric_mean_return': data_3m_symbol['geometric_mean_return'],
                'volatility_cv': avg_volatility,
                'positive_months_ratio': avg_positive_ratio,
                'total_return': data_3m_symbol['total_return'],  # Use 3-month for primary sorting
                'min_monthly_return': data_3m_symbol['min_monthly_return'],
                'first_price': data_3m_symbol['first_price'],
                'last_price': data_3m_symbol['last_price'],
                'data_points': data_3m_symbol['data_points'],
                'first_date': data_3m_symbol['first_date'],
                'last_date': data_3m_symbol['last_date'],
                # Add monthly display data from 3-month analysis
                'month_1_pct': data_3m_symbol.get('month_1_pct', 0.0),
                'month_2_pct': data_3m_symbol.get('month_2_pct', 0.0),
                'month_3_pct': data_3m_symbol.get('month_3_pct', 0.0),
                'month_1_name': data_3m_symbol.get('month_1_name', 'N/A'),
                'month_2_name': data_3m_symbol.get('month_2_name', 'N/A'),
                'month_3_name': data_3m_symbol.get('month_3_name', 'N/A'),
                'month_1_consistent': data_3m_symbol.get('month_1_consistent', False),
                'month_2_consistent': data_3m_symbol.get('month_2_consistent', False),
                'month_3_consistent': data_3m_symbol.get('month_3_consistent', False)
            }
            
            all_time_data.append(all_time_record)
        
        # Create DataFrame and sort by average consistency index
        result_df = pd.DataFrame(all_time_data)
        result_df = result_df.sort_values('consistency_index', ascending=False)
        
        logger.info(f"[find_all_time_performers] Returning {len(result_df)} all-time performers sorted by average consistency")
        return result_df 