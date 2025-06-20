import configparser
import mysql.connector
import pymysql
from sqlalchemy import create_engine
import pandas as pd
import streamlit as st
import os
from logger_config import logger

class ConfigManager:
    """Manages database configuration from config.ini file"""
    
    def __init__(self, config_path='conf/config.ini'):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """Load configuration from ini file"""
        if not os.path.exists(self.config_path):
            st.error(f"Configuration file not found: {self.config_path}")
            return
        
        try:
            self.config.read(self.config_path)
        except Exception as e:
            st.error(f"Error reading configuration: {e}")
    
    def get_mysql_config(self):
        """Get MySQL configuration parameters"""
        try:
            return {
                'host': self.config.get('mysql', 'url'),
                'user': self.config.get('mysql', 'username'),
                'password': self.config.get('mysql', 'password'),
                'database': self.config.get('mysql', 'database'),
                'port': 3306,  # Default MySQL port
                'charset': 'utf8mb4'
            }
        except Exception as e:
            st.error(f"Error getting MySQL config: {e}")
            return {}
    
    def get_table_names(self):
        """Get table names from configuration"""
        try:
            return {
                'stocksinfp': self.config.get('mysql', 'table'),
                'stock_change_tracker': self.config.get('mysql', 'stock_change_tracker_table'),
                'stocks_earnings': self.config.get('mysql', 'earning_table'),
                'stocks_revenue': self.config.get('mysql', 'revenue_table')
            }
        except Exception as e:
            st.error(f"Error getting table names: {e}")
            return {
                'stocksinfp': 'stocksinfp',
                'stock_change_tracker': 'stock_change_tracker',
                'stocks_earnings': 'stocks_earnings',
                'stocks_revenue': 'stocks_revenue'
            }
    
    def get_symbols(self):
        """Get stock symbols from configuration"""
        try:
            symbols_str = self.config.get('stocks', 'symbols', fallback='')
            if symbols_str.strip():
                return symbols_str.strip().split()
            else:
                return []
        except Exception as e:
            st.error(f"Error getting symbols: {e}")
            return []

class DatabaseManager:
    """Manages database connections and queries"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.mysql_config = config_manager.get_mysql_config()
        self.table_names = config_manager.get_table_names()
        self.connection = None
        self.engine = None
    
    @st.cache_resource
    def get_mysql_connection(_self):
        """Get MySQL connection with caching"""
        try:
            connection = mysql.connector.connect(
                host=_self.mysql_config['host'],
                user=_self.mysql_config['user'],
                password=_self.mysql_config['password'],
                database=_self.mysql_config['database'],
                port=_self.mysql_config['port'],
                charset=_self.mysql_config['charset']
            )
            return connection
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return None
    
    @st.cache_resource
    def get_sqlalchemy_engine(_self):
        """Get SQLAlchemy engine for pandas operations"""
        try:
            connection_string = (
                f"mysql+pymysql://{_self.mysql_config['user']}:"
                f"{_self.mysql_config['password']}@{_self.mysql_config['host']}:"
                f"{_self.mysql_config['port']}/{_self.mysql_config['database']}"
            )
            engine = create_engine(connection_string, echo=False)
            return engine
        except Exception as e:
            st.error(f"SQLAlchemy engine creation failed: {e}")
            return None
    @st.cache_data
    def get_stock_count(_self, search_term=""):
        """Get total count of stocks in stock_change_tracker table"""
        engine = _self.get_sqlalchemy_engine()
        if engine is None:
            return 0

        try:
            database_name = _self.mysql_config['database']
            tracker_table = _self.table_names['stock_change_tracker']
            full_tracker_table = f"{database_name}.{tracker_table}"
            
            if search_term:
                query = f"""
                SELECT COUNT(*) as total_count
                FROM {full_tracker_table} 
                WHERE is_active = true AND symbol LIKE %(search_term)s
                """
                df = pd.read_sql_query(query, engine, params={'search_term': f'%{search_term}%'})
            else:
                query = f"""
                SELECT COUNT(*) as total_count
                FROM {full_tracker_table}
                WHERE is_active = true
                """
                df = pd.read_sql_query(query, engine)
            
            return df.iloc[0]['total_count']
        except Exception as e:
            st.error(f"Error getting stock count: {e}")
            return 0

    @st.cache_data
    def get_gainers_losers_count(_self, search_term=""):
        """Get count of gainers and losers from stock_change_tracker table"""
        engine = _self.get_sqlalchemy_engine()
        if engine is None:
            return 0, 0, 0.0

        try:
            database_name = _self.mysql_config['database']
            tracker_table = _self.table_names['stock_change_tracker']
            full_tracker_table = f"{database_name}.{tracker_table}"
            
            if search_term:
                query = f"""
                SELECT 
                    SUM(CASE WHEN change_in_percent > 0 THEN 1 ELSE 0 END) as gainers,
                    SUM(CASE WHEN change_in_percent < 0 THEN 1 ELSE 0 END) as losers,
                    AVG(change_in_percent) as avg_change
                FROM {full_tracker_table} 
                WHERE is_active = true AND symbol LIKE %(search_term)s
                """
                df = pd.read_sql_query(query, engine, params={'search_term': f'%{search_term}%'})
            else:
                query = f"""
                SELECT 
                    SUM(CASE WHEN change_in_percent > 0 THEN 1 ELSE 0 END) as gainers,
                    SUM(CASE WHEN change_in_percent < 0 THEN 1 ELSE 0 END) as losers,
                    AVG(change_in_percent) as avg_change
                FROM {full_tracker_table}
                WHERE is_active = true
                """
                df = pd.read_sql_query(query, engine)
            
            row = df.iloc[0]
            gainers = int(row['gainers']) if pd.notna(row['gainers']) else 0
            losers = int(row['losers']) if pd.notna(row['losers']) else 0
            avg_change = float(row['avg_change']) if pd.notna(row['avg_change']) else 0.0
            
            return gainers, losers, avg_change
        except Exception as e:
            st.error(f"Error getting gainers/losers count: {e}")
            return 0, 0, 0.0

    @st.cache_data
    def load_stock_change_tracker(_self, page=1, page_size=50, search_term=""):
        """Load stock change tracker data with earning dates from stocks_earnings table with pagination and search"""
        engine = _self.get_sqlalchemy_engine()
        if engine is None:
            return pd.DataFrame()

        try:
            database_name = _self.mysql_config['database']
            tracker_table = _self.table_names['stock_change_tracker']
            earnings_table = _self.table_names['stocks_earnings']
            
            full_tracker_table = f"{database_name}.{tracker_table}"
            full_earnings_table = f"{database_name}.{earnings_table}"
            
            offset = (page - 1) * page_size
            
            # Build WHERE clause for search
            where_clause = "WHERE sct.is_active = true"
            params = {}
            if search_term:
                where_clause = "WHERE sct.is_active = true AND sct.symbol LIKE %(search_term)s"
                params['search_term'] = f'%{search_term}%'
            
            query = f"""
            SELECT 
                sct.symbol, 
                sct.price_when_added,
                sct.current_price, 
                sct.change_in_percent, 
                se.earnings_date as earning_date
            FROM {full_tracker_table} sct
            LEFT JOIN {full_earnings_table} se ON sct.symbol = se.symbol
            {where_clause}
            ORDER BY sct.change_in_percent DESC
            LIMIT {page_size} OFFSET {offset}
            """
            df = pd.read_sql_query(query, engine, params=params)
            return df
        except Exception as e:
            st.error(f"Error loading stock change tracker: {e}")
            # Fallback query without earning dates if join fails
            try:
                database_name = _self.mysql_config['database']
                tracker_table = _self.table_names['stock_change_tracker']
                full_tracker_table = f"{database_name}.{tracker_table}"
                
                offset = (page - 1) * page_size
                
                # Build WHERE clause for search
                where_clause = "WHERE is_active = true"
                params = {}
                if search_term:
                    where_clause = "WHERE is_active = true AND symbol LIKE %(search_term)s"
                    params['search_term'] = f'%{search_term}%'
                
                fallback_query = f"""
                SELECT symbol, price_when_added, current_price, change_in_percent, 
                       NULL as earning_date
                FROM {full_tracker_table} 
                {where_clause}
                ORDER BY change_in_percent DESC
                LIMIT {page_size} OFFSET {offset}
                """
                df = pd.read_sql_query(fallback_query, engine, params=params)
                st.warning("Could not join with stocks_earnings table. Earning dates will not be displayed.")
                return df
            except Exception as e2:
                st.error(f"Fallback query also failed: {e2}")
                return pd.DataFrame()
    
    @st.cache_data
    def load_stock_data(_self, symbol):
        """Load daily stock data for a specific symbol"""
        engine = _self.get_sqlalchemy_engine()
        if engine is None:
            return pd.DataFrame()
        
        try:
            database_name = _self.mysql_config['database']
            table_name = _self.table_names['stocksinfp']
            full_table_name = f"{database_name}.{table_name}"
            
            query = f"""
            SELECT date, open, high, low, close, volume, symbol
            FROM {full_table_name} 
            WHERE symbol = %(symbol)s
            ORDER BY date DESC
            LIMIT 100
            """
            df = pd.read_sql_query(query, engine, params={'symbol': symbol})
            
            # Ensure numeric columns are properly typed
            if not df.empty:
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df.sort_values('date') if 'date' in df.columns else df
        except Exception as e:
            st.error(f"Error loading stock data for {symbol}: {e}")
            return pd.DataFrame()
    
    @st.cache_data
    def get_available_symbols(_self):
        """Get all available symbols from the database"""
        engine = _self.get_sqlalchemy_engine()
        if engine is None:
            return []
        
        try:
            database_name = _self.mysql_config['database']
            table_name = _self.table_names['stock_change_tracker']
            full_table_name = f"{database_name}.{table_name}"
            
            query = f"SELECT DISTINCT symbol FROM {full_table_name} WHERE is_active = true ORDER BY symbol"
            df = pd.read_sql_query(query, engine)
            return df['symbol'].tolist()
        except Exception as e:
            st.error(f"Error getting available symbols: {e}")
            return []
    
    @st.cache_data
    def get_stocksinfp_symbols(_self):
        """Get all available symbols from the stocksinfp table"""
        engine = _self.get_sqlalchemy_engine()
        if engine is None:
            return []
        
        try:
            database_name = _self.mysql_config['database']
            table_name = _self.table_names['stocksinfp']
            full_table_name = f"{database_name}.{table_name}"
            
            query = f"SELECT DISTINCT symbol FROM {full_table_name} ORDER BY symbol"
            df = pd.read_sql_query(query, engine)
            return df['symbol'].tolist()
        except Exception as e:
            st.error(f"Error getting stocksinfp symbols: {e}")
            return []
    
    @st.cache_data
    def check_symbol_data(_self, symbol):
        """Check if a symbol has data in stocksinfp table"""
        engine = _self.get_sqlalchemy_engine()
        if engine is None:
            return False, "No database connection"
        
        try:
            database_name = _self.mysql_config['database']
            table_name = _self.table_names['stocksinfp']
            full_table_name = f"{database_name}.{table_name}"
            
            query = f"SELECT COUNT(*) as count FROM {full_table_name} WHERE symbol = %(symbol)s"
            result = pd.read_sql_query(query, engine, params={'symbol': symbol})
            count = result.iloc[0]['count']
            return count > 0, f"Found {count} records for {symbol}"
        except Exception as e:
            return False, f"Error checking symbol data: {e}"
    
    def test_connection(self):
        """Test database connection"""
        try:
            connection = self.get_mysql_connection()
            if connection is None:
                return False, "Failed to establish connection"
            
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            
            if result and result[0] == 1:
                return True, "Connection successful"
            else:
                return False, "Connection test failed"
                
        except Exception as e:
            return False, f"Connection error: {e}"
    
    def get_table_info(self):
        """Get information about available tables"""
        try:
            connection = self.get_mysql_connection()
            if connection is None:
                return {}
            
            cursor = connection.cursor()
            table_info = {}
            
            database_name = self.mysql_config['database']
            
            for table_key, table_name in self.table_names.items():
                try:
                    full_table_name = f"{database_name}.{table_name}"
                    cursor.execute(f"SELECT COUNT(*) FROM {full_table_name}")
                    count = cursor.fetchone()[0]
                    table_info[table_key] = {'name': full_table_name, 'count': count}
                except Exception as e:
                    table_info[table_key] = {'name': f"{database_name}.{table_name}", 'error': str(e)}
            
            cursor.close()
            return table_info
            
        except Exception as e:
            st.error(f"Error getting table info: {e}")
            return {}
    
    def test_earnings_join(self):
        """Test if the earnings table join works correctly"""
        try:
            engine = self.get_sqlalchemy_engine()
            if engine is None:
                return False, "No database connection"
            
            database_name = self.mysql_config['database']
            tracker_table = self.table_names['stock_change_tracker']
            earnings_table = self.table_names['stocks_earnings']
            
            full_tracker_table = f"{database_name}.{tracker_table}"
            full_earnings_table = f"{database_name}.{earnings_table}"
            
            test_query = f"""
            SELECT COUNT(*) as total_records, 
                   COUNT(se.earnings_date) as records_with_earnings
            FROM {full_tracker_table} sct
            LEFT JOIN {full_earnings_table} se ON sct.symbol = se.symbol
            """
            
            result = pd.read_sql_query(test_query, engine)
            total = result.iloc[0]['total_records']
            with_earnings = result.iloc[0]['records_with_earnings']
            
            return True, f"Join successful: {total} total records, {with_earnings} with earning dates"
            
        except Exception as e:
            return False, f"Join test failed: {e}"
    
    def get_table_schema(self, table_key='stocksinfp'):
        """Get the schema of a table"""
        try:
            connection = self.get_mysql_connection()
            if connection is None:
                return "No database connection"
            
            cursor = connection.cursor()
            database_name = self.mysql_config['database']
            table_name = self.table_names[table_key]
            full_table_name = f"{database_name}.{table_name}"
            
            cursor.execute(f"DESCRIBE {full_table_name}")
            schema = cursor.fetchall()
            cursor.close()
            
            return schema
            
        except Exception as e:
            return f"Error getting schema: {e}" 