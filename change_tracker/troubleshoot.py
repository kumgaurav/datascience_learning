#!/usr/bin/env python3
"""
Troubleshooting script for Stock Change Tracker App
This script helps identify issues step by step
"""

import sys
import os
import traceback
from datetime import datetime

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def test_imports():
    """Test all required imports"""
    print_header("TESTING IMPORTS")
    
    imports_to_test = [
        ("streamlit", "st"),
        ("pandas", "pd"), 
        ("numpy", "np"),
        ("plotly.graph_objects", "go"),
        ("plotly.express", "px"),
        ("mysql.connector", "mysql"),
        ("sqlalchemy", "sqlalchemy"),
        ("pymysql", "pymysql"),
        ("configparser", "configparser"),
        ("datetime", "datetime"),
    ]
    
    failed_imports = []
    
    for module_name, alias in imports_to_test:
        try:
            exec(f"import {module_name} as {alias}")
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            failed_imports.append((module_name, e))
        except Exception as e:
            print(f"⚠️  {module_name}: Unexpected error - {e}")
            failed_imports.append((module_name, e))
    
    return failed_imports

def test_config_file():
    """Test configuration file"""
    print_header("TESTING CONFIGURATION FILE")
    
    config_path = "conf/config.ini"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print("   Expected location: conf/config.ini")
        print("   Please create the configuration file with MySQL settings")
        return False
    
    print(f"✅ Configuration file exists: {config_path}")
    
    try:
        import configparser
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Check required sections
        required_sections = ['mysql']
        for section in required_sections:
            if section in config:
                print(f"✅ Section [{section}] found")
            else:
                print(f"❌ Section [{section}] missing")
                return False
        
        # Check required MySQL keys
        required_mysql_keys = ['url', 'username', 'password', 'database', 'table']
        mysql_section = config['mysql']
        
        for key in required_mysql_keys:
            if key in mysql_section:
                value = mysql_section[key]
                print(f"✅ {key}: {'*' * len(value) if key == 'password' else value}")
            else:
                print(f"❌ Missing key: {key}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False

def test_database_connection():
    """Test database connectivity"""
    print_header("TESTING DATABASE CONNECTION")
    
    try:
        from config_manager import ConfigManager
        config_manager = ConfigManager()
        mysql_config = config_manager.get_mysql_config()
        
        print("Configuration loaded:")
        for key, value in mysql_config.items():
            if key == 'password':
                print(f"  {key}: {'*' * len(value)}")
            else:
                print(f"  {key}: {value}")
        
        # Test MySQL connection
        import mysql.connector
        print("\nTesting MySQL connection...")
        
        connection = mysql.connector.connect(
            host=mysql_config['host'],
            user=mysql_config['user'],
            password=mysql_config['password'],
            database=mysql_config['database'],
            port=mysql_config['port'],
            charset=mysql_config['charset']
        )
        
        if connection.is_connected():
            print("✅ MySQL connection successful")
            
            # Test basic query
            cursor = connection.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"✅ MySQL Version: {version[0]}")
            
            cursor.close()
            connection.close()
            return True
        else:
            print("❌ MySQL connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Full traceback:")
        traceback.print_exc()
        return False

def test_table_access():
    """Test access to required tables"""
    print_header("TESTING TABLE ACCESS")
    
    try:
        from config_manager import ConfigManager, DatabaseManager
        
        config_manager = ConfigManager()
        db_manager = DatabaseManager(config_manager)
        
        # Test table access
        tables_to_test = [
            'stock_change_tracker',
            'stocks_earnings', 
            'stocksinfp'
        ]
        
        engine = db_manager.get_sqlalchemy_engine()
        if engine is None:
            print("❌ Could not create SQLAlchemy engine")
            return False
        
        import pandas as pd
        
        for table_key in tables_to_test:
            try:
                table_name = db_manager.table_names.get(table_key, table_key)
                database_name = db_manager.mysql_config['database']
                full_table_name = f"{database_name}.{table_name}"
                
                query = f"SELECT COUNT(*) as count FROM {full_table_name}"
                df = pd.read_sql_query(query, engine)
                count = df.iloc[0]['count']
                print(f"✅ {table_key} ({full_table_name}): {count} rows")
                
            except Exception as e:
                print(f"❌ {table_key}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Table access error: {e}")
        traceback.print_exc()
        return False

def test_sample_data_query():
    """Test loading sample data"""
    print_header("TESTING SAMPLE DATA QUERY")
    
    try:
        from config_manager import ConfigManager, DatabaseManager
        
        config_manager = ConfigManager()
        db_manager = DatabaseManager(config_manager)
        
        print("Loading stock change tracker data...")
        df = db_manager.load_stock_change_tracker()
        
        if df.empty:
            print("❌ No data returned from stock_change_tracker query")
            return False
        
        print(f"✅ Loaded {len(df)} rows")
        print("Columns:", list(df.columns))
        print("First few rows:")
        print(df.head())
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data query error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all troubleshooting tests"""
    print(f"Stock Change Tracker - Troubleshooting Script")
    print(f"Started at: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config_file),
        ("Database Connection Test", test_database_connection),
        ("Table Access Test", test_table_access),
        ("Sample Data Query Test", test_sample_data_query),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = "CRASH"
    
    # Summary
    print_header("TROUBLESHOOTING SUMMARY")
    for test_name, result in results.items():
        status_icon = "✅" if result == "PASS" else "❌" if result == "FAIL" else "💥"
        print(f"{status_icon} {test_name}: {result}")
    
    # Recommendations
    print_header("TROUBLESHOOTING RECOMMENDATIONS")
    
    if results.get("Import Test") != "PASS":
        print("🔧 Install missing packages:")
        print("   conda install streamlit pandas plotly mysql-connector-python sqlalchemy pymysql")
    
    if results.get("Configuration Test") != "PASS":
        print("🔧 Fix configuration file:")
        print("   1. Create conf/config.ini")
        print("   2. Add [mysql] section with url, username, password, database, table")
    
    if results.get("Database Connection Test") != "PASS":
        print("🔧 Fix database connection:")
        print("   1. Check MySQL server is running")
        print("   2. Verify credentials in conf/config.ini")
        print("   3. Check firewall/network connectivity")
    
    if results.get("Table Access Test") != "PASS":
        print("🔧 Fix table access:")
        print("   1. Verify table names in config.ini")
        print("   2. Check database permissions")
        print("   3. Ensure tables exist and have data")
    
    print("\nFor detailed logs, check the logs/ directory when running the main app.")

if __name__ == "__main__":
    main() 