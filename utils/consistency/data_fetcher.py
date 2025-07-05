"""
Consistency Data Fetcher Utility

Handles all database operations and data fetching for consistency analysis.
Extracted from ConsistentPerformersTab for better separation of concerns.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st

from data import DatabaseManager

# Configuration constants
DEFAULT_CACHE_TTL = 3600  # 1 hour in seconds
MIN_PRICE_FILTER = 5  # Minimum stock price filter

# Custom Exception Classes
class DataFetchError(Exception):
    """Raised when data cannot be fetched from database"""
    pass

class DataValidationError(Exception):
    """Raised when data fails validation checks"""
    pass

logger = logging.getLogger('StockApp')


class ConsistencyDataFetcher:
    """
    Handles all data fetching operations for consistency analysis.
    
    Responsibilities:
    - Database query execution
    - Data caching management  
    - Data cleaning and validation
    - Date range calculations
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the data fetcher.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
    
    @st.cache_data(ttl=DEFAULT_CACHE_TTL)
    def get_cached_price_data_for_symbols(_self, symbols: List[str], months: int = 3) -> pd.DataFrame:
        """
        Get cached price data for selected symbols over specified period.
        
        Args:
            symbols: List of stock symbols to fetch
            months: Number of months to analyze (1, 2, or 3)
            
        Returns:
            DataFrame with price data for the symbols
            
        Raises:
            DataFetchError: If database query fails
            DataValidationError: If invalid parameters provided
        """
        try:
            # Create fresh db manager for cached call to avoid pickling issues
            from data import DatabaseManager
            db_manager = DatabaseManager()
            
            end_date = datetime.now()
            start_date = _self._calculate_date_range(end_date, months)
            
            logger.info(f"[get_cached_price_data_for_symbols] Fetching price data for {len(symbols)} symbols over {months} months")
            
            # Format symbols for SQL IN clause
            symbols_str = "', '".join(symbols)
            
            query = f"""
            SELECT p.symbol, date, close, high, low, volume
            FROM stocksinfp p
            INNER JOIN stock_change_tracker s ON s.symbol = p.symbol 
                AND s.is_active = 1 
                AND s.current_price > {MIN_PRICE_FILTER}
            WHERE p.symbol IN ('{symbols_str}')
            AND date >= %s AND date <= %s
            AND close IS NOT NULL AND close != '' AND close != '0'
            ORDER BY symbol, date
            """
            
            query_params = (start_date, end_date)
            logger.info(f"[get_cached_price_data_for_symbols] Executing database query with params: {query_params}")
            logger.debug(f"[get_cached_price_data_for_symbols] SQL query:\n{query}")
            
            # Execute query with timing and error handling
            start_time = time.time()
            try:
                data = db_manager.execute_query(query, query_params)
            except Exception as db_error:
                logger.error(f"Database query execution failed: {db_error}")
                raise DataFetchError(f"Failed to fetch cached price data: {db_error}") from db_error
            end_time = time.time()
            
            logger.info(f"[get_cached_price_data_for_symbols] Database query completed in {end_time - start_time:.2f} seconds")
            
            if data.empty:
                logger.warning(f"[get_cached_price_data_for_symbols] No price data found for {months}-month period")
                return pd.DataFrame()
            
            # Clean and process data
            data = _self._clean_price_data(data)
            
            logger.info(f"[get_cached_price_data_for_symbols] Returning {len(data)} rows of price data for {months}-month period")
            return data
            
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid parameters for price data fetch: {e}")
            st.error("Invalid parameters provided for data retrieval.")
            return pd.DataFrame()
        except DataFetchError as e:
            logger.error(f"Database query failed: {e}")
            st.error("Unable to fetch stock data. Please check your database connection.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error getting cached price data: {e}")
            st.error("An unexpected error occurred while fetching data. Please try again.")
            return pd.DataFrame()
    
    def fetch_consistency_data(self, months: int = 3, include_raw_data: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Fetch all stock data for consistency analysis.
        
        Args:
            months: Number of months to analyze (1, 2, or 3)
            include_raw_data: If True, returns dict with both consistency data and raw data
            
        Returns:
            Union[pd.DataFrame, Dict[str, pd.DataFrame]]: 
                If include_raw_data=False: Just the raw data DataFrame
                If include_raw_data=True: Dict with 'raw_data' key
                
        Raises:
            DataFetchError: If database connection fails
            DataValidationError: If invalid parameters provided
        """
        try:
            end_date = datetime.now()
            start_date = self._calculate_date_range(end_date, months)
            
            logger.info(f"[fetch_consistency_data] Fetching all stock data from {start_date.date()} to {end_date.date()}")
            logger.info(f"[fetch_consistency_data] Using stock_change_tracker filter: is_active=1 AND current_price>{MIN_PRICE_FILTER}")
            
            query = f"""
            SELECT p.symbol, date, close, high, low, volume
            FROM stocksinfp p
            INNER JOIN stock_change_tracker s ON s.symbol = p.symbol 
                AND s.is_active = 1 
                AND s.current_price > {MIN_PRICE_FILTER}
            WHERE date >= %(start_date)s AND date <= %(end_date)s
            AND close IS NOT NULL AND close != '' AND close != '0'
            AND high IS NOT NULL AND high != '' AND high != '0'
            AND low IS NOT NULL AND low != '' AND low != '0'
            ORDER BY symbol, date
            """
            
            query_params = {
                'start_date': start_date,
                'end_date': end_date
            }
            
            logger.info(f"[fetch_consistency_data] Executing database query with params: {query_params}")
            logger.debug(f"[fetch_consistency_data] SQL query:\n{query}")
            
            # Execute query with timing and error handling
            start_time = time.time()
            try:
                all_data = self.db_manager.execute_query(query, query_params)
            except Exception as db_error:
                logger.error(f"Database query execution failed: {db_error}")
                raise DataFetchError(f"Failed to fetch consistency data: {db_error}") from db_error
            end_time = time.time()
            
            logger.info(f"[fetch_consistency_data] Database query completed in {end_time - start_time:.2f} seconds")
            
            if all_data.empty:
                logger.warning(f"[fetch_consistency_data] No data found for {months}-month date range")
                if include_raw_data:
                    return {'raw_data': pd.DataFrame()}
                else:
                    return pd.DataFrame()
            
            logger.info(f"[fetch_consistency_data] Loaded {len(all_data)} rows of stock data")
            
            # Log unique symbols and date ranges for debugging
            unique_symbols = all_data['symbol'].nunique()
            if len(all_data) > 0:
                date_range = f"{all_data['date'].min()} to {all_data['date'].max()}"
                logger.info(f"[fetch_consistency_data] Processing {unique_symbols} unique symbols, date range: {date_range}")
            
            # Clean and process data
            all_data = self._clean_price_data(all_data)
            
            logger.info(f"[fetch_consistency_data] After data cleaning: {len(all_data)} rows remaining")
            
            if include_raw_data:
                return {'raw_data': all_data}
            else:
                return all_data
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Database connection failed: {e}")
            error_msg = "Database connection failed. Please check your connection and try again."
            raise DataFetchError(error_msg) from e
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid parameters for consistency analysis: {e}")
            error_msg = f"Invalid parameters provided for {months}-month analysis."
            raise DataValidationError(error_msg) from e
        except pd.errors.EmptyDataError as e:
            logger.error(f"No data returned from database: {e}")
            if include_raw_data:
                return {'raw_data': pd.DataFrame()}
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error fetching consistency data: {e}")
            error_msg = f"Unexpected error during {months}-month data fetch."
            if include_raw_data:
                return {'raw_data': pd.DataFrame()}
            else:
                return pd.DataFrame()
    
    def fetch_stock_data_for_charts(self, selected_symbols: List[str], months: int = 3) -> pd.DataFrame:
        """
        Fetch stock data specifically for chart creation.
        
        Args:
            selected_symbols: List of symbols to fetch data for
            months: Number of months to analyze
            
        Returns:
            DataFrame with stock data for charts
            
        Raises:
            DataFetchError: If database query fails
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = self._calculate_date_range(end_date, months)
            
            # Create placeholders for symbols
            symbol_placeholders = ', '.join(['%s'] * len(selected_symbols))
            
            query = f"""
            SELECT p.symbol, date, close, high, low
            FROM stocksinfp p
            WHERE p.symbol IN ({symbol_placeholders})
            AND date >= %s AND date <= %s
            AND close IS NOT NULL AND close != '' AND close != '0'
            ORDER BY symbol, date
            """
            
            params = selected_symbols + [start_date, end_date]
            
            logger.info(f"[fetch_stock_data_for_charts] Executing database query with params: {params}")
            logger.debug(f"[fetch_stock_data_for_charts] SQL query:\n{query}")
            
            # Execute query with timing and error handling
            start_time = time.time()
            try:
                price_data = self.db_manager.execute_query(query, params)
            except Exception as db_error:
                logger.error(f"Database query execution failed: {db_error}")
                raise DataFetchError(f"Failed to fetch stock data for charts: {db_error}") from db_error
            end_time = time.time()
            
            logger.info(f"[fetch_stock_data_for_charts] Database query completed in {end_time - start_time:.2f} seconds")
            
            if price_data.empty:
                logger.warning("[fetch_stock_data_for_charts] No price data found for selected symbols")
                return pd.DataFrame()
            
            # Clean and process data
            price_data = self._clean_price_data(price_data)
            
            logger.info(f"[fetch_stock_data_for_charts] Prepared chart data for {len(selected_symbols)} symbols")
            return price_data
            
        except Exception as e:
            logger.error(f"Unexpected error fetching stock data for charts: {e}")
            raise DataFetchError(f"Failed to fetch chart data: {e}") from e
    
    def _calculate_date_range(self, end_date: datetime, months: int) -> datetime:
        """
        Calculate start date based on end date and number of months.
        
        Args:
            end_date: End date for analysis
            months: Number of months to analyze
            
        Returns:
            Start date for the analysis period
        """
        if months == 1:
            # Use current month from start to now
            start_date = datetime(end_date.year, end_date.month, 1)
            logger.info(f"[_calculate_date_range] Using current month boundaries for 1-month analysis: {start_date.date()} to {end_date.date()}")
        else:
            # For 2+ months, use the previous approach but more accurate
            start_date = end_date - timedelta(days=30*months)
            logger.info(f"[_calculate_date_range] Using {months*30} days for {months}-month analysis: {start_date.date()} to {end_date.date()}")
        
        return start_date
    
    def _clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate price data.
        
        Args:
            data: Raw price data from database
            
        Returns:
            Cleaned DataFrame with proper data types
            
        Raises:
            DataValidationError: If data validation fails
        """
        try:
            if data.empty:
                return data
            
            # Convert data types
            data['date'] = pd.to_datetime(data['date'])
            for col in ['close', 'high', 'low', 'volume']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Clean data
            data = data.dropna(subset=['close'])
            data = data[data['close'] > 0]
            
            # Additional cleaning for high/low if present
            if 'high' in data.columns and 'low' in data.columns:
                data = data.dropna(subset=['high', 'low'])
                data = data[data['high'] > 0]
                data = data[data['low'] > 0]
            
            return data
            
        except Exception as e:
            logger.error(f"Error cleaning price data: {e}")
            raise DataValidationError(f"Failed to clean price data: {e}") from e
    
    def has_data_quality_issues(self, data: pd.DataFrame) -> bool:
        """
        Check for basic data quality issues.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data quality issues are detected, False otherwise
        """
        if data.empty:
            return False
        
        # Check for required columns
        required_columns = ['symbol', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"[has_data_quality_issues] Missing required columns: {missing_columns}")
            return True
        
        # Check for negative or zero prices
        if 'close' in data.columns:
            invalid_prices = len(data[data['close'] <= 0])
            if invalid_prices > 0:
                logger.warning(f"[has_data_quality_issues] Found {invalid_prices} rows with invalid prices")
                return True
        
        # Check for missing dates
        if 'date' in data.columns:
            missing_dates = data['date'].isna().sum()
            if missing_dates > 0:
                logger.warning(f"[has_data_quality_issues] Found {missing_dates} rows with missing dates")
                return True
        
        return False
    
    def get_price_data_for_symbols(self, symbols: List[str], months: int = 3) -> pd.DataFrame:
        """
        Get price data for specific symbols over specified period (non-cached version).
        
        Args:
            symbols: List of stock symbols to fetch
            months: Number of months to analyze
            
        Returns:
            DataFrame with price data for the symbols
        """
        try:
            end_date = datetime.now()
            start_date = self._calculate_date_range(end_date, months)
            
            logger.info(f"[get_price_data_for_symbols] Fetching price data for {len(symbols)} symbols over {months} months")
            
            # Format symbols for SQL IN clause
            symbols_str = "', '".join(symbols)
            
            query = f"""
            SELECT p.symbol, date, close, high, low, volume
            FROM stocksinfp p
            INNER JOIN stock_change_tracker s ON s.symbol = p.symbol 
                AND s.is_active = 1 
                AND s.current_price > {MIN_PRICE_FILTER}
            WHERE p.symbol IN ('{symbols_str}')
            AND date >= %s AND date <= %s
            AND close IS NOT NULL AND close != '' AND close != '0'
            ORDER BY symbol, date
            """
            
            query_params = (start_date, end_date)
            logger.info(f"[get_price_data_for_symbols] Executing database query with params: {query_params}")
            logger.debug(f"[get_price_data_for_symbols] SQL query:\n{query}")
            
            # Execute query with timing and error handling
            start_time = time.time()
            try:
                data = self.db_manager.execute_query(query, query_params)
            except Exception as db_error:
                logger.error(f"Database query execution failed: {db_error}")
                raise DataFetchError(f"Failed to fetch price data: {db_error}") from db_error
            end_time = time.time()
            
            logger.info(f"[get_price_data_for_symbols] Database query completed in {end_time - start_time:.2f} seconds")
            
            if data.empty:
                logger.warning(f"[get_price_data_for_symbols] No price data found for {months}-month period")
                return pd.DataFrame()
            
            # Clean and process data
            data = self._clean_price_data(data)
            
            logger.info(f"[get_price_data_for_symbols] Returning {len(data)} rows of price data for {months}-month period")
            return data
            
        except DataFetchError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting price data for symbols: {e}")
            raise DataFetchError(f"Failed to fetch price data for symbols: {e}") from e
    
    def get_all_stock_data(self, months: int) -> pd.DataFrame:
        """
        Get all stock data for consistency analysis.
        
        Args:
            months: Number of months to analyze
            
        Returns:
            DataFrame with all stock data (raw_data only)
        """
        try:
            result = self.fetch_consistency_data(months, include_raw_data=True)
            if isinstance(result, dict):
                return result.get('raw_data', pd.DataFrame())
            else:
                return result
        except Exception as e:
            logger.error(f"Error getting all stock data: {e}")
            return pd.DataFrame() 