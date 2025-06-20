"""
Application Settings

Contains all application-wide settings and constants.
"""

# Database settings
DATABASE_CONFIG = {
    'default_page_size': 50,
    'max_page_size': 200,
    'connection_timeout': 30,
    'query_timeout': 60
}

# RSI settings
RSI_CONFIG = {
    'default_period': 14,
    'overbought_threshold': 70,
    'oversold_threshold': 30,
    'default_analysis_threshold': 75
}

# Technical indicators settings
TECHNICAL_INDICATORS_CONFIG = {
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_period': 20,
    'bollinger_std': 2,
    'sma_periods': [20, 50, 200],
    'ema_periods': [12, 26]
}

# Neuro-evolution settings
NEURO_EVOLUTION_CONFIG = {
    'population_size': 50,
    'generations': 20,
    'mutation_rate': 0.1,
    'crossover_rate': 0.8,
    'tournament_size': 3,
    'lookback_period': 20,
    'risk_tolerance': 0.02,
    'confidence_threshold': 0.7
}

# UI settings
UI_CONFIG = {
    'page_title': "Stock Analysis Dashboard",
    'page_icon': "📊",
    'layout': "wide",
    'sidebar_state': "expanded",
    'chart_height': 500,
    'max_chart_points': 200
}

# Color scheme
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'buy_signal': '#2ca02c',
    'sell_signal': '#d62728',
    'hold_signal': '#ffc107',
    'earnings_highlight': '#28a745'
}

# Formatting settings
FORMAT_CONFIG = {
    'currency_decimals': 2,
    'percentage_decimals': 2,
    'rsi_decimals': 2,
    'price_decimals': 2,
    'volume_format': 'K',  # K for thousands, M for millions
    'date_format': '%Y-%m-%d'
}

# Performance settings
PERFORMANCE_CONFIG = {
    'cache_ttl': 300,  # 5 minutes
    'max_concurrent_queries': 5,
    'chunk_size': 1000,
    'enable_caching': True
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_max_bytes': 10 * 1024 * 1024,  # 10MB
    'file_backup_count': 5
}

# Analysis settings
ANALYSIS_CONFIG = {
    'min_data_points_rsi': 15,
    'min_data_points_ai': 50,
    'default_analysis_period': 200,  # days
    'earnings_warning_weeks': 3,
    'signal_confidence_threshold': 60  # percentage
}

# Export commonly used values
DEFAULT_PAGE_SIZE = DATABASE_CONFIG['default_page_size']
DEFAULT_RSI_PERIOD = RSI_CONFIG['default_period']
OVERBOUGHT_THRESHOLD = RSI_CONFIG['overbought_threshold']
OVERSOLD_THRESHOLD = RSI_CONFIG['oversold_threshold'] 