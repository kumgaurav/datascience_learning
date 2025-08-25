"""
Helper Utilities

Common utility functions used across the application.
"""

import pandas as pd
from typing import Union


def format_currency(value: Union[float, int, None]) -> str:
    """
    Format a numeric value as currency
    
    Args:
        value: Numeric value to format
    
    Returns:
        Formatted currency string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    try:
        return f"${float(value):,.2f}"
    except (ValueError, TypeError):
        return "N/A"


def format_percentage(value: Union[float, int, None]) -> str:
    """
    Format a numeric value as percentage
    
    Args:
        value: Numeric value to format
    
    Returns:
        Formatted percentage string
    """
    if pd.isna(value) or value is None:
        return "N/A"
    
    try:
        return f"{float(value):.2f}%"
    except (ValueError, TypeError):
        return "N/A"


def get_color_for_change(change_str: str) -> str:
    """
    Get CSS color styling for percentage change
    
    Args:
        change_str: Formatted percentage change string
    
    Returns:
        CSS color styling
    """
    if change_str == "N/A":
        return "color: gray;"
    
    try:
        # Extract numeric value from percentage string
        value = float(change_str.replace('%', ''))
        
        if value > 0:
            return "color: green; font-weight: bold;"
        elif value < 0:
            return "color: red; font-weight: bold;"
        else:
            return "color: gray;"
    except (ValueError, AttributeError):
        return "color: gray;"


def safe_float_conversion(value: Union[str, float, int, None], default: float = 0.0) -> float:
    """
    Safely convert a value to float
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Float value or default
    """
    if pd.isna(value) or value is None:
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Truncate text to specified length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
    
    Returns:
        Truncated text with ellipsis if needed
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."


def get_trend_indicator(current: float, previous: float) -> str:
    """
    Get trend indicator (↑, ↓, →)
    
    Args:
        current: Current value
        previous: Previous value
    
    Returns:
        Trend indicator string
    """
    if pd.isna(current) or pd.isna(previous):
        return "→"
    
    if current > previous:
        return "↑"
    elif current < previous:
        return "↓"
    else:
        return "→" 