"""
Database Manager Module

Handles all database connections and operations.
"""

import pandas as pd
import mysql.connector
from mysql.connector import Error
import configparser
import os
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

logger = logging.getLogger('StockApp')


class DatabaseManager:
    """
    Manages database connections and operations
    """
    
    def __init__(self, config_file: str = 'conf/config.ini'):
        """
        Initialize database manager
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.connection = None
        self.config = self._load_config()
        self._test_connection()
    
    def _load_config(self) -> configparser.ConfigParser:
        """Load database configuration"""
        config = configparser.ConfigParser()
        
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        config.read(self.config_file)
        
        if 'mysql' not in config:
            raise ValueError("Database configuration section not found in config file")
        
        return config
    
    def _test_connection(self) -> bool:
        """Test database connection"""
        try:
            connection = self._get_connection()
            if connection and connection.is_connected():
                logger.info("Database connection test successful")
                connection.close()
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
        
        return False
    
    def _get_connection(self):
        """Get database connection"""
        try:
            db_config = self.config['mysql']
            connection = mysql.connector.connect(
                host=db_config.get('url', 'localhost'),
                port=db_config.getint('port', 3306),
                database=db_config.get('database', 'stocksdb'),
                user=db_config.get('username', 'root'),
                password=db_config.get('password', ''),
                charset=db_config.get('charset', 'utf8mb4'),
                use_unicode=True,
                autocommit=True
            )
            return connection
        except Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute a SELECT query and return results as DataFrame
        
        Args:
            query: SQL query string
            params: Query parameters
        
        Returns:
            DataFrame with query results
        """
        try:
            connection = self._get_connection()
            
            if params:
                df = pd.read_sql(query, connection, params=params)
            else:
                df = pd.read_sql(query, connection)
            
            connection.close()
            logger.debug(f"Query executed successfully, returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
    
    def get_stock_change_tracker(self, page: int = 1, page_size: int = 50, 
                                search: str = '') -> pd.DataFrame:
        """
        Get stock change tracker data with pagination including company details
        
        Args:
            page: Page number (1-based)
            page_size: Number of records per page
            search: Search term for symbol filtering
        
        Returns:
            DataFrame with stock change tracker data including company details
        """
        try:
            offset = (page - 1) * page_size
            
            base_query = """
                SELECT 
                    sct.symbol,
                    sct.change_in_percent,
                    sct.current_price,
                    sct.price_when_added,
                    se.earnings_date,
                    sd.company_name,
                    sd.sector,
                    sd.industry,
                    sd.market_cap
                FROM stocksdb.stock_change_tracker sct
                LEFT JOIN stocksdb.stocks_earnings se ON sct.symbol = se.symbol
                LEFT JOIN stocksdb.stock_details sd ON sct.symbol = sd.symbol
            """
            
            if search:
                base_query += " WHERE (sct.symbol LIKE %(search)s OR sd.company_name LIKE %(search)s)"
                params = {'search': f'%{search}%'}
            else:
                params = None
            
            base_query += " ORDER BY sct.change_in_percent DESC"
            base_query += f" LIMIT {page_size} OFFSET {offset}"
            
            df = self.execute_query(base_query, params)
            logger.info(f"Loaded {len(df)} rows from stock_change_tracker with company details")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching stock change tracker data: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['symbol', 'change_in_percent', 'current_price', 
                                       'price_when_added', 'market_cap', 'earnings_date',
                                       'company_name', 'sector', 'industry'])
    
    def get_stock_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """
        Get historical stock data for a symbol
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of records to return
        
        Returns:
            DataFrame with historical stock data
        """
        try:
            query = """
                SELECT date, open, high, low, close, volume
                FROM stocksdb.stocksinfp
                WHERE symbol = %(symbol)s
                ORDER BY date DESC
                LIMIT %(limit)s
            """
            
            df = self.execute_query(query, {'symbol': symbol, 'limit': limit})
            
            if not df.empty:
                # Sort by date ascending for analysis
                df = df.sort_values('date').reset_index(drop=True)
                logger.info(f"Loaded {len(df)} rows of stock data for {symbol}")
            else:
                logger.warning(f"No stock data found for symbol: {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get basic stock information including company details
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Dictionary with stock information including company details
        """
        try:
            # Get basic info from stock_change_tracker with company details
            query = """
                SELECT 
                    sct.symbol,
                    sct.change_in_percent,
                    sct.current_price,
                    sct.price_when_added,
                    se.earnings_date,
                    sd.company_name,
                    sd.sector,
                    sd.industry,
                    sd.market_cap
                FROM stocksdb.stock_change_tracker sct
                LEFT JOIN stocksdb.stocks_earnings se ON sct.symbol = se.symbol
                LEFT JOIN stocksdb.stock_details sd ON sct.symbol = sd.symbol
                WHERE sct.symbol = %(symbol)s
            """
            
            df = self.execute_query(query, {'symbol': symbol})
            
            if not df.empty:
                row = df.iloc[0]
                return {
                    'symbol': row['symbol'],
                    'change_in_percent': row['change_in_percent'],
                    'current_price': row['current_price'],
                    'price_when_added': row['price_when_added'],
                    'market_cap': row['market_cap'],
                    'earnings_date': row['earnings_date'],
                    'company_name': row['company_name'],
                    'sector': row['sector'],
                    'industry': row['industry']
                }
            else:
                return {'symbol': symbol, 'error': 'Stock not found'}
                
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def is_earnings_within_weeks(self, earnings_date, weeks: int = 3) -> bool:
        """
        Check if earnings date is within specified weeks from now
        
        Args:
            earnings_date: Earnings date (can be string or datetime)
            weeks: Number of weeks to check
        
        Returns:
            True if earnings date is within the specified weeks
        """
        if pd.isna(earnings_date) or earnings_date is None:
            return False
        
        try:
            # Convert earnings_date to datetime first, then to date
            if isinstance(earnings_date, str):
                earnings_dt = pd.to_datetime(earnings_date)
            else:
                earnings_dt = pd.to_datetime(earnings_date)
            
            # Convert to date object for comparison
            earnings_date_obj = earnings_dt.date()
            
            # Get today's date
            today = datetime.now().date()
            weeks_from_now = today + timedelta(weeks=weeks)
            
            return today <= earnings_date_obj <= weeks_from_now
            
        except Exception as e:
            logger.error(f"Error checking earnings date: {e}")
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current database connection status"""
        try:
            connection = self._get_connection()
            if connection and connection.is_connected():
                connection.close()
                return {
                    'status': 'Connected', 
                    'timestamp': datetime.now(),
                    'host': self.config['mysql']['url'],
                    'port': self.config['mysql'].get('port', '3306'),
                    'database': self.config['mysql']['database']
                }
            else:
                return {
                    'status': 'Disconnected', 
                    'error': 'Failed to establish connection',
                    'host': self.config['mysql']['url'],
                    'port': self.config['mysql'].get('port', '3306'),
                    'database': self.config['mysql']['database']
                }
        except Exception as e:
            return {
                'status': 'Error', 
                'error': str(e),
                'host': self.config['mysql'].get('url', 'unknown'),
                'port': self.config['mysql'].get('port', '3306'),
                'database': self.config['mysql'].get('database', 'unknown')
            }
    
    def get_minimum_price_filter(self) -> float:
        """
        Get the minimum price filter from configuration
        
        Returns:
            float: Minimum price threshold (default: 1.0)
        """
        try:
            if 'stocks' in self.config and 'minimum_price_filter' in self.config['stocks']:
                return self.config['stocks'].getfloat('minimum_price_filter', 1.0)
            else:
                return 1.0  # Default to $1.00 if not configured
        except Exception as e:
            logger.warning(f"Error reading minimum price filter from config: {e}, using default of 1.0")
            return 1.0 