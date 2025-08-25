"""
Earnings Data Fetcher Utility

Handles all data retrieval and caching for earnings analysis.
Extracted from EarningsStocksTab for better separation of concerns.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import time

import pandas as pd
import streamlit as st

# Configuration constants
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds

# Custom Exception Classes
class EarningsDataFetchError(Exception):
    """Raised when earnings data cannot be fetched from database"""
    pass

class EarningsDataValidationError(Exception):
    """Raised when earnings data fails validation checks"""
    pass

logger = logging.getLogger('StockApp')


class EarningsDataFetcher:
    """
    Handles all data retrieval and caching for earnings analysis.
    
    Responsibilities:
    - Database queries for earnings data
    - Price data retrieval for earnings stocks
    - Data cleaning and validation
    - Caching for performance
    """
    
    def __init__(self, db_manager):
        """
        Initialize with database manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        logger.info("[EarningsDataFetcher] Initialized")
    
    @st.cache_data(ttl=DEFAULT_CACHE_TTL)
    def get_cached_earnings_data(_self) -> pd.DataFrame:
        """
        Get cached earnings data for all upcoming earnings.
        
        Returns:
            DataFrame with earnings data including volatility analysis
        """
        try:
            # Create fresh data fetcher for cached call to avoid pickling issues
            from data import DatabaseManager
            from utils.earnings import EarningsDataFetcher
            
            db_manager = DatabaseManager()
            data_fetcher = EarningsDataFetcher(db_manager)
            
            logger.info("[get_cached_earnings_data] Fetching earnings data")
            return data_fetcher.fetch_earnings_data()
            
        except Exception as e:
            logger.error(f"Error in cached earnings data fetch: {e}")
            st.error(f"Unable to fetch earnings data: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=DEFAULT_CACHE_TTL)
    def get_cached_price_data_for_symbols(_self, symbols: List[str], weeks: int = 3) -> pd.DataFrame:
        """
        Get cached price data for earnings symbols.
        
        Args:
            symbols: List of stock symbols
            weeks: Number of weeks of price data to retrieve
            
        Returns:
            DataFrame with price data
        """
        try:
            # Create fresh data fetcher for cached call to avoid pickling issues
            from data import DatabaseManager
            from utils.earnings import EarningsDataFetcher
            
            db_manager = DatabaseManager()
            data_fetcher = EarningsDataFetcher(db_manager)
            
            logger.info(f"[get_cached_price_data_for_symbols] Fetching data for {len(symbols)} symbols over {weeks} weeks")
            return data_fetcher.get_price_data_for_symbols(symbols, weeks)
            
        except Exception as e:
            logger.error(f"Error in cached price data fetch: {e}")
            st.error(f"Unable to fetch price data: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=DEFAULT_CACHE_TTL)
    def get_cached_earnings_price_data_for_symbols(_self, symbols: List[str], weeks: int = 8) -> pd.DataFrame:
        """
        Get cached price data specifically for earnings analysis with extended time period.
        This is separate from the general cached method to avoid affecting other visualizations.
        
        Args:
            symbols: List of stock symbols
            weeks: Number of weeks of price data to retrieve (default 8 for earnings)
            
        Returns:
            DataFrame with price data for earnings analysis
        """
        try:
            # Create fresh data fetcher for cached call to avoid pickling issues
            from data import DatabaseManager
            from utils.earnings import EarningsDataFetcher
            
            db_manager = DatabaseManager()
            data_fetcher = EarningsDataFetcher(db_manager)
            
            logger.info(f"[get_cached_earnings_price_data_for_symbols] Fetching earnings data for {len(symbols)} symbols over {weeks} weeks")
            return data_fetcher.get_earnings_price_data_for_symbols(symbols, weeks)
            
        except Exception as e:
            logger.error(f"Error in cached earnings price data fetch: {e}")
            st.error(f"Unable to fetch earnings price data: {e}")
            return pd.DataFrame()
    
    def fetch_earnings_data(self) -> pd.DataFrame:
        """
        Fetch earnings data for upcoming earnings announcements.
        
        Returns:
            DataFrame with earnings data and volatility analysis
        """
        try:
            logger.info("[fetch_earnings_data] Starting earnings data fetch")
            
            # Get current date for earnings filtering
            current_date = datetime.now()
            end_date = current_date + timedelta(days=30)  # Next 30 days
            
            # Query for upcoming earnings
            earnings_query = """
            SELECT DISTINCT CAST(e.ticker AS CHAR(32) CHARACTER SET utf8mb4) AS symbol, e.earnings_date,
                   NULL AS estimated_eps, NULL AS actual_eps,
                   s.current_price,
                   COALESCE(d.market_cap, 0) AS market_cap,
                   COALESCE(d.sector, 'Unknown') AS sector
            FROM stocks_earnings e
            INNER JOIN stock_change_tracker s ON (s.symbol COLLATE utf8mb4_0900_ai_ci) = (e.ticker COLLATE utf8mb4_0900_ai_ci)
                AND s.is_active = 1
                AND s.current_price > 5
            LEFT JOIN stock_details d ON (d.symbol COLLATE utf8mb4_0900_ai_ci) = (e.ticker COLLATE utf8mb4_0900_ai_ci)
            WHERE e.earnings_date >= %(start_date)s
                AND e.earnings_date <= %(end_date)s
                AND e.earnings_date IS NOT NULL
            ORDER BY e.earnings_date, symbol
            """
            
            # Execute query
            start_time = time.time()
            try:
                earnings_data = self.db_manager.execute_query(earnings_query, {
                    'start_date': current_date,
                    'end_date': end_date
                })
            except Exception as db_error:
                logger.error(f"Earnings database query execution failed: {db_error}")
                raise EarningsDataFetchError(f"Failed to fetch earnings data: {db_error}") from db_error
            end_time = time.time()
            
            logger.info(f"[fetch_earnings_data] Database query completed in {end_time - start_time:.2f} seconds")
            
            if earnings_data.empty:
                logger.warning("[fetch_earnings_data] No upcoming earnings found")
                return pd.DataFrame()
            
            logger.info(f"[fetch_earnings_data] Found {len(earnings_data)} earnings announcements")
            
            # Get volatility analysis for these stocks
            earnings_data = self._add_volatility_analysis(earnings_data)
            
            # Clean and validate data
            earnings_data = self._clean_earnings_data(earnings_data)
            
            logger.info(f"[fetch_earnings_data] Returning {len(earnings_data)} earnings records with volatility analysis")
            return earnings_data
            
        except Exception as e:
            logger.error(f"Error fetching earnings data: {e}")
            raise EarningsDataFetchError(f"Failed to fetch earnings data: {e}") from e
    
    def get_price_data_for_symbols(self, symbols: List[str], weeks: int = 3) -> pd.DataFrame:
        """
        Get price data for specific symbols over specified weeks.
        
        Args:
            symbols: List of stock symbols
            weeks: Number of weeks of data to retrieve
            
        Returns:
            DataFrame with price data
        """
        try:
            if not symbols:
                return pd.DataFrame()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks)
            
            logger.info(f"[get_price_data_for_symbols] Fetching {weeks}-week price data for {len(symbols)} symbols")
            
            # Create parameterized query for symbols
            symbols_placeholder = ', '.join(['%s'] * len(symbols))
            
            query = f"""
            SELECT p.symbol, date, close, high, low, volume
            FROM stocksinfp p
            WHERE p.symbol IN ({symbols_placeholder})
                AND date >= %s AND date <= %s
                AND close IS NOT NULL AND close != '' AND close != '0'
                AND high IS NOT NULL AND high != '' AND high != '0'
                AND low IS NOT NULL AND low != '' AND low != '0'
            ORDER BY symbol, date
            """
            
            # Execute query
            start_time = time.time()
            try:
                # Combine symbols with date parameters
                query_params = list(symbols) + [start_date, end_date]
                data = self.db_manager.execute_query(query, query_params)
            except Exception as db_error:
                logger.error(f"Price data database query execution failed: {db_error}")
                raise EarningsDataFetchError(f"Failed to fetch price data: {db_error}") from db_error
            end_time = time.time()
            
            logger.info(f"[get_price_data_for_symbols] Database query completed in {end_time - start_time:.2f} seconds")
            
            if data.empty:
                logger.warning(f"[get_price_data_for_symbols] No price data found for {weeks}-week period")
                return pd.DataFrame()
            
            # Clean and process data
            data = self._clean_price_data(data)
            
            logger.info(f"[get_price_data_for_symbols] Returning {len(data)} rows of price data for {weeks}-week period")
            return data
            
        except Exception as e:
            logger.error(f"Error getting price data for symbols: {e}")
            raise EarningsDataFetchError(f"Failed to fetch price data for symbols: {e}") from e

    def get_earnings_price_data_for_symbols(self, symbols: List[str], weeks: int = 8) -> pd.DataFrame:
        """
        Get price data specifically for earnings analysis with extended time period.
        This method is separate from the general price data method to avoid affecting other visualizations.
        
        Args:
            symbols: List of stock symbols
            weeks: Number of weeks of data to retrieve (default 8 for earnings analysis)
            
        Returns:
            DataFrame with price data for earnings analysis
        """
        try:
            if not symbols:
                return pd.DataFrame()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks)
            
            logger.info(f"[get_earnings_price_data_for_symbols] Fetching {weeks}-week price data for {len(symbols)} symbols for earnings analysis")
            
            # Create parameterized query for symbols
            symbols_placeholder = ', '.join(['%s'] * len(symbols))
            
            query = f"""
            SELECT p.symbol, date, close, high, low, volume
            FROM stocksinfp p
            WHERE p.symbol IN ({symbols_placeholder})
                AND date >= %s AND date <= %s
                AND close IS NOT NULL AND close != '' AND close != '0'
                AND high IS NOT NULL AND high != '' AND high != '0'
                AND low IS NOT NULL AND low != '' AND low != '0'
            ORDER BY symbol, date
            """
            
            # Execute query
            start_time = time.time()
            try:
                # Combine symbols with date parameters
                query_params = list(symbols) + [start_date, end_date]
                data = self.db_manager.execute_query(query, query_params)
            except Exception as db_error:
                logger.error(f"Earnings price data database query execution failed: {db_error}")
                raise EarningsDataFetchError(f"Failed to fetch earnings price data: {db_error}") from db_error
            end_time = time.time()
            
            logger.info(f"[get_earnings_price_data_for_symbols] Database query completed in {end_time - start_time:.2f} seconds")
            
            if data.empty:
                logger.warning(f"[get_earnings_price_data_for_symbols] No price data found for {weeks}-week period")
                return pd.DataFrame()
            
            # Clean and process data
            data = self._clean_price_data(data)
            
            logger.info(f"[get_earnings_price_data_for_symbols] Returning {len(data)} rows of price data for earnings analysis")
            return data
            
        except Exception as e:
            logger.error(f"Error getting earnings price data for symbols: {e}")
            raise EarningsDataFetchError(f"Failed to fetch earnings price data for symbols: {e}") from e
    
    def _add_volatility_analysis(self, earnings_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility analysis to earnings data.
        
        Args:
            earnings_data: DataFrame with earnings information
            
        Returns:
            DataFrame with added volatility metrics
        """
        try:
            if earnings_data.empty:
                return earnings_data
            
            logger.info("[_add_volatility_analysis] Adding volatility analysis to earnings data")
            
            # Get symbols for volatility analysis
            symbols = earnings_data['symbol'].unique().tolist()
            logger.info(f"[_add_volatility_analysis] Analyzing volatility for {len(symbols)} symbols: {symbols}")
            
            # Use earnings-specific 8-week price data for better volatility calculation
            price_data = self.get_earnings_price_data_for_symbols(symbols, weeks=8)
            
            if price_data.empty:
                logger.warning("[_add_volatility_analysis] No price data available for volatility analysis")
                # Add default volatility columns
                earnings_data['volatility_score'] = 0.0
                earnings_data['total_return'] = 0.0
                earnings_data['max_daily_gain'] = 0.0
                earnings_data['max_daily_loss'] = 0.0
                earnings_data['volatility_std'] = 0.0
                return earnings_data
            
            logger.debug(f"[_add_volatility_analysis] Retrieved {len(price_data)} price data points for {len(price_data['symbol'].unique())} symbols")
            
            # Calculate volatility metrics for each symbol
            volatility_metrics = []
            symbols_with_data = []
            symbols_without_data = []
            
            for symbol in symbols:
                symbol_data = price_data[price_data['symbol'] == symbol].copy()
                
                # Reduced minimum data requirement from 5 to 3 points for earnings analysis
                if len(symbol_data) < 3:  
                    logger.warning(f"[_add_volatility_analysis] Insufficient data for {symbol}: {len(symbol_data)} points (need 3+)")
                    symbols_without_data.append(symbol)
                    # Default values for insufficient data
                    volatility_metrics.append({
                        'symbol': symbol,
                        'volatility_score': 0.0,
                        'total_return': 0.0,
                        'max_daily_gain': 0.0,
                        'max_daily_loss': 0.0,
                        'volatility_std': 0.0
                    })
                    continue
                
                # Sort by date and clean data
                symbol_data = symbol_data.sort_values('date')
                symbol_data['close'] = pd.to_numeric(symbol_data['close'], errors='coerce')
                symbol_data = symbol_data.dropna(subset=['close'])
                
                # Re-check data after cleaning
                if len(symbol_data) < 3:
                    logger.warning(f"[_add_volatility_analysis] Insufficient clean data for {symbol}: {len(symbol_data)} points after cleaning")
                    symbols_without_data.append(symbol)
                    volatility_metrics.append({
                        'symbol': symbol,
                        'volatility_score': 0.0,
                        'total_return': 0.0,
                        'max_daily_gain': 0.0,
                        'max_daily_loss': 0.0,
                        'volatility_std': 0.0
                    })
                    continue
                
                symbols_with_data.append(symbol)
                
                # Calculate total return
                first_price = symbol_data['close'].iloc[0]
                last_price = symbol_data['close'].iloc[-1]
                total_return = ((last_price - first_price) / first_price) * 100
                
                # Calculate daily returns
                symbol_data['daily_return'] = symbol_data['close'].pct_change() * 100
                daily_returns = symbol_data['daily_return'].dropna()
                
                # Calculate volatility metrics with better error handling
                if len(daily_returns) == 0:
                    logger.warning(f"[_add_volatility_analysis] No daily returns calculated for {symbol}")
                    volatility_std = 0.0
                    max_daily_gain = 0.0
                    max_daily_loss = 0.0
                    volatility_score = 0.0
                else:
                    volatility_std = daily_returns.std() if len(daily_returns) > 1 else abs(daily_returns.iloc[0])
                    max_daily_gain = daily_returns.max()
                    max_daily_loss = daily_returns.min()
                    
                    # Calculate volatility score (combination of std dev and range)
                    volatility_score = volatility_std + (max_daily_gain - max_daily_loss) * 0.1
                
                logger.debug(f"[_add_volatility_analysis] {symbol}: return={total_return:.2f}%, volatility={volatility_score:.2f}, data_points={len(symbol_data)}")
                
                volatility_metrics.append({
                    'symbol': symbol,
                    'volatility_score': volatility_score,
                    'total_return': total_return,
                    'max_daily_gain': max_daily_gain,
                    'max_daily_loss': max_daily_loss,
                    'volatility_std': volatility_std
                })
            
            # Log summary
            logger.debug(f"[_add_volatility_analysis] Successfully calculated metrics for {len(symbols_with_data)} symbols")
            if symbols_without_data:
                logger.warning(f"[_add_volatility_analysis] Failed to calculate metrics for {len(symbols_without_data)} symbols: {symbols_without_data}")
            
            # Convert to DataFrame and merge with earnings data
            volatility_df = pd.DataFrame(volatility_metrics)
            earnings_with_volatility = earnings_data.merge(volatility_df, on='symbol', how='left')
            
            # Fill any missing volatility values
            volatility_columns = ['volatility_score', 'total_return', 'max_daily_gain', 'max_daily_loss', 'volatility_std']
            for col in volatility_columns:
                if col not in earnings_with_volatility.columns:
                    earnings_with_volatility[col] = 0.0
                else:
                    earnings_with_volatility[col] = earnings_with_volatility[col].fillna(0.0)
            
            logger.info(f"[_add_volatility_analysis] Added volatility analysis for {len(earnings_with_volatility)} earnings records")
            return earnings_with_volatility
            
        except Exception as e:
            logger.error(f"Error adding volatility analysis: {e}")
            # Return original data with default volatility columns
            earnings_data['volatility_score'] = 0.0
            earnings_data['total_return'] = 0.0
            earnings_data['max_daily_gain'] = 0.0
            earnings_data['max_daily_loss'] = 0.0
            earnings_data['volatility_std'] = 0.0
            return earnings_data
    
    def _clean_earnings_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate earnings data.
        
        Args:
            data: Raw earnings data
            
        Returns:
            Cleaned earnings data
        """
        try:
            if data.empty:
                return data
            
            logger.debug(f"[_clean_earnings_data] Cleaning {len(data)} earnings records")
            
            # Convert data types
            if 'earnings_date' in data.columns:
                data['earnings_date'] = pd.to_datetime(data['earnings_date'], errors='coerce')
            
            numeric_columns = ['estimated_eps', 'actual_eps', 'current_price', 'market_cap',
                             'volatility_score', 'total_return', 'max_daily_gain', 'max_daily_loss', 'volatility_std']
            
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove rows with invalid dates
            if 'earnings_date' in data.columns:
                data = data.dropna(subset=['earnings_date'])
            
            # Remove rows with missing symbols
            data = data.dropna(subset=['symbol'])

            # Ensure symbol column is plain Python string (object dtype) to avoid Arrow conversion issues
            if 'symbol' in data.columns:
                def _to_plain_str(v):
                    if pd.isna(v):
                        return ''
                    try:
                        # Handle bytes
                        if isinstance(v, (bytes, bytearray)):
                            return v.decode('utf-8', 'ignore')
                        # JSON-encode list/dict/tuple/set
                        if isinstance(v, (list, dict, set, tuple)):
                            return str(v)
                        # Fallback to string
                        return str(v)
                    except Exception:
                        return ''
                data['symbol'] = data['symbol'].map(_to_plain_str).astype(object)
            
            # Remove duplicates
            data = data.drop_duplicates(subset=['symbol', 'earnings_date'])
            
            logger.info(f"[_clean_earnings_data] Cleaned data: {len(data)} records remaining")
            return data
            
        except Exception as e:
            logger.error(f"Error cleaning earnings data: {e}")
            return data
    
    def _clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate price data.
        
        Args:
            data: Raw price data
            
        Returns:
            Cleaned price data
        """
        try:
            if data.empty:
                return data
            
            logger.debug(f"[_clean_price_data] Cleaning {len(data)} price records")
            
            # Convert data types
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
            
            numeric_columns = ['close', 'high', 'low', 'volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove rows with invalid data
            data = data.dropna(subset=['date', 'symbol'])
            
            # Remove rows with non-positive prices
            if 'close' in data.columns:
                data = data[data['close'] > 0]
            
            # Remove duplicates
            data = data.drop_duplicates(subset=['symbol', 'date'])
            
            logger.debug(f"[_clean_price_data] Cleaned data: {len(data)} records remaining")
            return data
            
        except Exception as e:
            logger.error(f"Error cleaning price data: {e}")
            return data 