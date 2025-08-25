"""
Earnings UI Renderer Utility

Handles all UI components and layout for earnings analysis.
Extracted from EarningsStocksTab for better separation of concerns.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st
from pandas.api.types import is_extension_array_dtype, is_string_dtype
import numpy as np
import pyarrow as pa
import os
import tempfile


def _force_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    # Coerce any extension dtypes to NumPy or Python object equivalents
    for col in df.columns:
        try:
            series = df[col]
            if is_extension_array_dtype(series):
                # Numeric-like extension dtypes to numpy float64 to preserve NaN
                if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
                    df[col] = pd.to_numeric(series, errors="coerce").astype("float64")
                elif pd.api.types.is_bool_dtype(series):
                    df[col] = series.astype(bool).astype(object)
                else:
                    # Strings/others → plain Python str in object dtype
                    df[col] = series.astype(str).astype(object)
        except Exception:
            pass

    # Convert pandas "string" extension to plain Python str (object)
    for col in df.select_dtypes(include="string").columns:
        try:
            df[col] = df[col].astype(str).astype(object)
        except Exception:
            df[col] = df[col].astype(object)

    # Ensure object columns are Python scalars (no bytes/np scalar)
    for col in df.select_dtypes(include="object").columns:
        try:
            def _to_py(x):
                if isinstance(x, (bytes, bytearray)):
                    return x.decode("utf-8", errors="ignore")
                if isinstance(x, np.generic):
                    return x.item()
                return "" if x is None else x
            df[col] = df[col].map(_to_py).astype(object)
        except Exception:
            pass

    # Standardize numeric dtypes
    for col in df.columns:
        try:
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].astype("float64")  # keep float64 for NaN safety
            elif pd.api.types.is_float_dtype(df[col]) and str(df[col].dtype) != "float64":
                df[col] = df[col].astype("float64")
        except Exception:
            pass
    return df


def _get_ui_arrow_converter():
    """Create a local Arrow converter (no dependency on test.py).

    Rules:
    - Columns whose name contains "date" -> convert to pa.date32(day)
    - Columns that parse as numeric -> pa.float64
    - Everything else -> pa.string
    """
    def convert_to_arrow_table(df: pd.DataFrame, debug: bool = True) -> pa.Table:
        arrays = {}
        # Identify date-like columns by name
        date_cols = [c for c in df.columns if "date" in c.lower()]

        for col in df.columns:
            series = df[col]

            # Date handling -> date32
            if col in date_cols:
                parsed = pd.to_datetime(series, errors="coerce")
                if parsed.notna().any():
                    arrays[col] = pa.array([x.date() if pd.notna(x) else None for x in parsed], type=pa.date32())
                else:
                    arrays[col] = pa.array(series.fillna("").astype(str).tolist(), type=pa.string())
                continue

            # Try numeric conversion
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                arrays[col] = pa.array(numeric.tolist(), type=pa.float64())
                continue

            # Fallback to string
            arrays[col] = pa.array(series.fillna("").astype(str).tolist(), type=pa.string())

        table = pa.table(arrays)
        if debug:
            logger.debug(f"Arrow schema: {table.schema}")
        return table

    return convert_to_arrow_table

# ensure_arrow_compatible_df removed

# Custom Exception Classes
class EarningsUIRenderError(Exception):
    """Raised when earnings UI rendering fails"""
    pass

logger = logging.getLogger('StockApp')


class EarningsUIRenderer:
    """
    Handles all UI components and layout for earnings analysis.
    
    Responsibilities:
    - Control panels and settings
    - Data tables and metrics display
    - Layout management
    - User feedback and information
    """
    
    def __init__(self):
        """Initialize the earnings UI renderer."""
        logger.info("[EarningsUIRenderer] Initialized")
    
    def render_control_panel(self) -> Tuple[bool, int]:
        """
        Render the control panel for earnings analysis.
        
        Returns:
            Tuple of (debug_mode, chart_columns)
        """
        try:
            st.markdown("### 🎛️ Earnings Analysis Controls")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                debug_mode = st.checkbox(
                    "🔍 Debug Mode",
                    value=False,
                    help="Show detailed debug information and processing steps"
                )
            
            with col2:
                chart_columns = st.selectbox(
                    "📊 Chart Columns:",
                    options=[1, 2, 3, 4],
                    index=0,  # Default to 1 column
                    help="Number of columns in the chart grid",
                    key="earnings_chart_columns"
                )
            
            return debug_mode, chart_columns
            
        except Exception as e:
            logger.error(f"Error rendering control panel: {e}")
            return False, 2
    
    def display_earnings_summary(self, categorized_earnings: Dict[str, pd.DataFrame],
                                calendar_stats: Dict[str, Dict[str, float]]) -> None:
        """
        Display earnings summary with key metrics.
        
        Args:
            categorized_earnings: Dictionary with categorized earnings data
            calendar_stats: Dictionary with calculated statistics
        """
        try:
            logger.info("[display_earnings_summary] Displaying earnings summary")
            
            # Calculate totals
            total_earnings = sum(len(data) for data in categorized_earnings.values())
            
            if total_earnings == 0:
                st.info("📅 No upcoming earnings found in the next 30 days")
                return
            
            # Display summary metrics
            st.markdown("### 📊 Earnings Calendar Summary")
            
            # Create metric columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Upcoming Earnings",
                    value=total_earnings,
                    help="Total number of earnings announcements in the next 30 days"
                )
            
            with col2:
                current_week_count = len(categorized_earnings.get('current_week', pd.DataFrame()))
                st.metric(
                    label="This Week",
                    value=current_week_count,
                    help="Earnings announcements this week"
                )
            
            with col3:
                next_week_count = len(categorized_earnings.get('next_week', pd.DataFrame()))
                st.metric(
                    label="Next Week",
                    value=next_week_count,
                    help="Earnings announcements next week"
                )
            
            with col4:
                # Calculate average volatility across all stocks
                all_volatilities = []
                for data in categorized_earnings.values():
                    if 'volatility_score' in data.columns:
                        all_volatilities.extend(data['volatility_score'].dropna().tolist())
                
                avg_volatility = sum(all_volatilities) / len(all_volatilities) if all_volatilities else 0
                st.metric(
                    label="Avg Volatility",
                    value=f"{avg_volatility:.1f}",
                    help="Average volatility score across all earnings stocks"
                )
            
        except Exception as e:
            logger.error(f"Error displaying earnings summary: {e}")
            st.error("Failed to display earnings summary")
    
    def display_earnings_week_tabs(self, categorized_earnings: Dict[str, pd.DataFrame]) -> str:
        """
        Display earnings week tabs and return selected week.
        
        Args:
            categorized_earnings: Dictionary with categorized earnings data
            
        Returns:
            Selected week key
        """
        try:
            # Create tabs for each week
            tab_labels = []
            tab_keys = []
            
            week_info = {
                'current_week': ('📅 This Week', 'current_week'),
                'next_week': ('📅 Next Week', 'next_week'),
                'week_after': ('📅 Week After', 'week_after'),
                'later': ('📅 Later', 'later')
            }
            
            for week_key, (label, key) in week_info.items():
                count = len(categorized_earnings.get(week_key, pd.DataFrame()))
                if count > 0:
                    tab_labels.append(f"{label} ({count})")
                    tab_keys.append(key)
            
            if not tab_labels:
                st.info("📅 No upcoming earnings to display")
                return None
            
            # Create tabs
            tabs = st.tabs(tab_labels)
            
            # Return the first available tab's key for default selection
            return tab_keys[0] if tab_keys else None
            
        except Exception as e:
            logger.error(f"Error displaying earnings week tabs: {e}")
            return None
    
    def display_earnings_table(self, earnings_data: pd.DataFrame, week_title: str) -> None:
        """
        Display earnings data in a formatted table.
        
        Args:
            earnings_data: DataFrame with earnings data
            week_title: Title for the week section
        """
        try:
            if earnings_data.empty:
                st.info(f"📅 No earnings scheduled for {week_title}")
                return
            
            logger.info(f"[display_earnings_table] Displaying table for {week_title} with {len(earnings_data)} stocks")
            
            st.markdown(f"### {week_title}")
            st.markdown(f"**{len(earnings_data)} earnings announcements**")
            
            # Prepare table data
            display_data = earnings_data.copy()
            # Normalize dtypes proactively per request
            try:
                display_data = display_data.convert_dtypes()
            except Exception:
                pass
            
            # Format columns for display
            if 'earnings_date' in display_data.columns:
                # Cast to string first to safely handle mixed types, then coerce to datetime
                display_data['earnings_date'] = pd.to_datetime(
                    display_data['earnings_date'].astype(str), errors='coerce'
                ).dt.strftime('%Y-%m-%d')
                # Replace NaT results from strftime with empty string for cleaner display
                display_data['earnings_date'] = display_data['earnings_date'].fillna('')
            
            # Format numeric columns
            numeric_columns = ['estimated_eps', 'actual_eps', 'current_price', 'market_cap']
            for col in numeric_columns:
                if col in display_data.columns:
                    # Cast to string first to avoid errors with non-numeric mixed types
                    display_data[col] = pd.to_numeric(display_data[col].astype(str), errors='coerce')
            
            # Format specific columns
            if 'current_price' in display_data.columns:
                display_data['current_price'] = display_data['current_price'].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                )
            
            if 'estimated_eps' in display_data.columns:
                display_data['estimated_eps'] = display_data['estimated_eps'].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                )
            
            if 'market_cap' in display_data.columns:
                display_data['market_cap'] = display_data['market_cap'].apply(
                    lambda x: f"${x/1e9:.2f}B" if pd.notna(x) and x > 0 else "N/A"
                )
            
            # Add performance metrics if available
            if 'total_return' in display_data.columns:
                display_data['total_return'] = display_data['total_return'].apply(
                    lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
                )
            
            if 'volatility_score' in display_data.columns:
                display_data['volatility_score'] = display_data['volatility_score'].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                )
            
            # Select columns to display
            display_columns = ['symbol', 'earnings_date', 'current_price', 'estimated_eps', 'sector']
            
            # Add performance columns if available
            if 'total_return' in display_data.columns:
                display_columns.append('total_return')
            if 'volatility_score' in display_data.columns:
                display_columns.append('volatility_score')
            
            # Filter to available columns
            available_columns = [col for col in display_columns if col in display_data.columns]
            
            # Display table with fallback
            try:
                # Helper: (disabled) Arrow serialization per column
                def _check_arrow_serialization(df: pd.DataFrame, title: str = "") -> None:
                    return

                # Helper: CSV roundtrip to normalize dtypes, write only once per key
                def _csv_roundtrip(df: pd.DataFrame, key: str) -> pd.DataFrame:
                    try:
                        tmp_root = os.path.join(tempfile.gettempdir(), "stockapp_tmp")
                        os.makedirs(tmp_root, exist_ok=True)
                        safe_key = "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in key)
                        path = os.path.join(tmp_root, f"{safe_key}.csv")
                        if not os.path.exists(path):
                            df.to_csv(path, index=False, encoding="utf-8")
                        return pd.read_csv(path, low_memory=False)
                    except Exception as e:
                        logger.warning(f"CSV roundtrip failed ({key}): {e}")
                        return df

                display_subset = display_data[available_columns].copy()
                # CSV roundtrip (write once) to avoid Arrow dtype issues
                display_subset = _csv_roundtrip(display_subset, key=f"earnings_full_{week_title}")
                # Run Arrow compatibility check before rendering (disabled)
                _check_arrow_serialization(display_subset, title="(full table)")

                # Log column dtypes and sample value types before render (no UI prints)
                try:
                    type_lines = []
                    for col in display_subset.columns:
                        series = display_subset[col]
                        non_null = series[series.notna()] if hasattr(series, 'notna') else series
                        sample_type = type(non_null.iloc[0]).__name__ if hasattr(non_null, 'iloc') and len(non_null) > 0 else 'None'
                        type_lines.append(f"{col}: dtype={getattr(series, 'dtype', type(series))}, sample={sample_type}")
                    logger.info("[display_earnings_table] Column types before render: " + "; ".join(type_lines))
                except Exception as _e:
                    logger.debug(f"Failed to assemble column types: {_e}")

                # Use production Arrow converter to normalize then render with st.dataframe
                try:
                    converter = _get_ui_arrow_converter()
                    arrow_table = converter(display_subset, debug=True)
                    try:
                        display_subset = arrow_table.to_pandas(split_blocks=True, ignore_metadata=True)
                    except Exception as to_pd_err:
                        logger.warning(f"Arrow to_pandas failed for full table: {to_pd_err}; falling back to pylist reconstruction")
                        try:
                            fallback_data = {}
                            for name, col in zip(arrow_table.schema.names, arrow_table.columns):
                                fallback_data[name] = col.to_pylist()
                            display_subset = pd.DataFrame(fallback_data)
                        except Exception as fb_err:
                            logger.warning(f"Pylist reconstruction failed for full table: {fb_err}")
                    # Final coercion to Arrow-safe numpy types
                    display_subset = _force_arrow_safe(display_subset)
                except Exception as conv_e:
                    logger.warning(f"Arrow normalization failed for full table: {conv_e}")
                try:
                    final_types = ", ".join([f"{c}:{str(display_subset[c].dtype)}" for c in display_subset.columns])
                    logger.info(f"[display_earnings_table] Final dtypes before render (full): {final_types}")
                except Exception:
                    pass
                st.dataframe(
                    display_subset.reset_index(drop=True),
                    use_container_width=True
                )
            except Exception as inner_e:
                logger.error(f"Fallback display due to table render error: {inner_e.__class__.__name__}: {inner_e}")
                # Minimal fallback: symbol and earnings_date only
                minimal_cols = [col for col in ['symbol', 'earnings_date'] if col in display_data.columns]
                if minimal_cols:
                    # Ensure earnings_date is string for display
                    if 'earnings_date' in minimal_cols:
                        display_data['earnings_date'] = pd.to_datetime(
                            display_data['earnings_date'].astype(str), errors='coerce'
                        ).dt.strftime('%Y-%m-%d')
                        display_data['earnings_date'] = display_data['earnings_date'].fillna('')
                    minimal_subset = display_data[minimal_cols].copy()
                    # Avoid convert_dtypes to prevent introducing ExtensionArray dtypes
                    # CSV roundtrip to normalize like full table
                    minimal_subset = _csv_roundtrip(minimal_subset, key=f"earnings_min_{week_title}")
                    # Run Arrow compatibility check for minimal subset (disabled)
                    _check_arrow_serialization(minimal_subset, title="(minimal subset)")
                    try:
                        # Log column types for minimal subset as well (no UI prints)
                        try:
                            type_lines_min = []
                            for col in minimal_subset.columns:
                                series = minimal_subset[col]
                                non_null = series[series.notna()] if hasattr(series, 'notna') else series
                                sample_type = type(non_null.iloc[0]).__name__ if hasattr(non_null, 'iloc') and len(non_null) > 0 else 'None'
                                type_lines_min.append(f"{col}: dtype={getattr(series, 'dtype', type(series))}, sample={sample_type}")
                            logger.info("[display_earnings_table] Minimal subset column types before render: " + "; ".join(type_lines_min))
                        except Exception as _e2:
                            logger.debug(f"Failed to assemble minimal column types: {_e2}")

                        # Normalize with Arrow converter and render with st.dataframe
                        try:
                            converter = _get_ui_arrow_converter()
                            arrow_table = converter(minimal_subset, debug=True)
                            try:
                                minimal_subset = arrow_table.to_pandas(split_blocks=True, ignore_metadata=True)
                            except Exception as to_pd_err2:
                                logger.warning(f"Arrow to_pandas failed for minimal table: {to_pd_err2}; falling back to pylist reconstruction")
                                try:
                                    fallback_data_min = {}
                                    for name, col in zip(arrow_table.schema.names, arrow_table.columns):
                                        fallback_data_min[name] = col.to_pylist()
                                    minimal_subset = pd.DataFrame(fallback_data_min)
                                except Exception as fb_err2:
                                    logger.warning(f"Pylist reconstruction failed for minimal table: {fb_err2}")
                            # Final coercion to Arrow-safe numpy types
                            minimal_subset = _force_arrow_safe(minimal_subset)
                        except Exception as conv_e2:
                            logger.warning(f"Arrow normalization failed for minimal table: {conv_e2}")
                        try:
                            final_types_min = ", ".join([f"{c}:{str(minimal_subset[c].dtype)}" for c in minimal_subset.columns])
                            logger.info(f"[display_earnings_table] Final dtypes before render (minimal): {final_types_min}")
                        except Exception:
                            pass
                        st.dataframe(
                            minimal_subset.reset_index(drop=True),
                            use_container_width=True
                        )
                    except Exception as minimal_e:
                        logger.error(f"HTML fallback due to minimal table render error: {minimal_e.__class__.__name__}: {minimal_e}")
                        # Final fallback: render as static HTML table to bypass Arrow serialization
                        st.markdown(minimal_subset.to_html(index=False), unsafe_allow_html=True)
                else:
                    st.info("No displayable columns available.")
                    # As a last resort, try rendering the full subset as HTML if available
                    try:
                        st.markdown(display_subset.to_html(index=False), unsafe_allow_html=True)
                    except Exception:
                        pass
            
        except Exception as e:
            logger.error(f"Error displaying earnings table: {e}")
            st.error(f"Failed to display earnings table for {week_title}")
    
    def display_week_statistics(self, week_stats: Dict[str, float], week_title: str) -> None:
        """
        Display statistics for a specific week.
        
        Args:
            week_stats: Dictionary with week statistics
            week_title: Title for the week
        """
        try:
            if not week_stats or week_stats.get('total_stocks', 0) == 0:
                return
            
            logger.info(f"[display_week_statistics] Displaying statistics for {week_title}")
            
            st.markdown(f"#### 📊 {week_title} Statistics")
            
            # Create columns for metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Stocks",
                    value=int(week_stats.get('total_stocks', 0)),
                    help="Number of stocks with earnings this week"
                )
            
            with col2:
                avg_return = week_stats.get('avg_return', 0)
                st.metric(
                    label="Avg Return",
                    value=f"{avg_return:.1f}%",
                    delta=f"{avg_return:.1f}%",
                    help="Average return over the analysis period"
                )
            
            with col3:
                avg_volatility = week_stats.get('avg_volatility', 0)
                st.metric(
                    label="Avg Volatility",
                    value=f"{avg_volatility:.1f}",
                    help="Average volatility score for the week"
                )
            
            with col4:
                positive_pct = week_stats.get('positive_return_pct', 0)
                st.metric(
                    label="Positive Returns",
                    value=f"{positive_pct:.0f}%",
                    help="Percentage of stocks with positive returns"
                )
            
        except Exception as e:
            logger.error(f"Error displaying week statistics: {e}")
            st.error(f"Failed to display statistics for {week_title}")
    
    def display_symbol_selection_interface_with_defaults(self, available_symbols: List[str], 
                                                       default_symbols: List[str], unique_id: str = "") -> Tuple[List[str], int]:
        """
        Display symbol selection interface with custom default symbols for chart creation.
        
        Args:
            available_symbols: List of all available symbols (should be pre-sorted)
            default_symbols: List of symbols to select by default
            unique_id: Unique identifier to avoid widget key conflicts
            
        Returns:
            Tuple of (selected_symbols, chart_columns)
        """
        try:
            if not available_symbols:
                st.warning("⚠️ No symbols available for selection")
                return [], 2
            
            st.markdown("---")
            st.markdown("### 📈 Individual Stock Analysis")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Symbol selection with custom defaults
                selected_symbols = st.multiselect(
                    "🎯 Select Earnings Stocks to Analyze:",
                    options=available_symbols,
                    default=default_symbols,
                    help=f"Select up to 20 stocks to display detailed price charts. Pre-selected: top {len(default_symbols)} performers by returns",
                    key=f"earnings_symbol_selection_{unique_id}"
                )
                
                # Show information about the sorting
                if default_symbols:
                    st.info(f"💡 **Pre-selected top {len(default_symbols)} performers by returns** - you can modify this selection above")
            
            with col2:
                # Chart configuration
                chart_columns = st.selectbox(
                    "Chart Columns:",
                    options=[1, 2, 3, 4],
                    index=0,  # Default to 1 column
                    help="Number of columns in the chart grid",
                    key=f"earnings_individual_chart_columns_{unique_id}"
                )
            
            return selected_symbols, chart_columns
            
        except Exception as e:
            logger.error(f"Error displaying symbol selection interface with defaults: {e}")
            return [], 2

    def display_symbol_selection_interface(self, available_symbols: List[str], 
                                         default_count: int = 10, unique_id: str = "") -> Tuple[List[str], int]:
        """
        Display symbol selection interface for chart creation.
        
        Args:
            available_symbols: List of available symbols
            default_count: Default number of symbols to select
            unique_id: Unique identifier to avoid widget key conflicts
            
        Returns:
            Tuple of (selected_symbols, chart_columns)
        """
        try:
            if not available_symbols:
                st.warning("⚠️ No symbols available for selection")
                return [], 2
            
            st.markdown("---")
            st.markdown("### 📈 Individual Stock Analysis")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Symbol selection
                default_symbols = available_symbols[:min(default_count, len(available_symbols))]
                
                selected_symbols = st.multiselect(
                    "🎯 Select Earnings Stocks to Analyze:",
                    options=available_symbols,
                    default=default_symbols,
                    help="Select up to 20 stocks to display detailed price charts",
                    key=f"earnings_symbol_selection_{unique_id}"
                )
            
            with col2:
                # Chart configuration
                chart_columns = st.selectbox(
                    "Chart Columns:",
                    options=[1, 2, 3, 4],
                    index=0,  # Default to 1 column
                    help="Number of columns in the chart grid",
                    key=f"earnings_individual_chart_columns_{unique_id}"
                )
            
            return selected_symbols, chart_columns
            
        except Exception as e:
            logger.error(f"Error displaying symbol selection interface: {e}")
            return [], 2
    
    def display_no_earnings_message(self) -> None:
        """Display message when no earnings data is available."""
        try:
            st.info("📅 No upcoming earnings found in the next 30 days.")
            st.markdown("""
            **Possible reasons:**
            - No earnings scheduled for the current period
            - Database might need updating
            - All earnings stocks are filtered out (price < $5)
            """)
            
        except Exception as e:
            logger.error(f"Error displaying no earnings message: {e}")
    
    def display_debug_info(self, earnings_data: pd.DataFrame, 
                          categorized_earnings: Dict[str, pd.DataFrame],
                          calendar_stats: Dict[str, Dict[str, float]]) -> None:
        """
        Display debug information for troubleshooting.
        
        Args:
            earnings_data: Raw earnings data
            categorized_earnings: Categorized earnings data
            calendar_stats: Calendar statistics
        """
        try:
            logger.info("[display_debug_info] Displaying debug information")
            
            with st.expander("🔍 Debug Information"):
                st.markdown("#### Raw Data Information")
                st.write(f"Total earnings records: {len(earnings_data)}")
                
                if not earnings_data.empty:
                    st.write(f"Date range: {earnings_data['earnings_date'].min()} to {earnings_data['earnings_date'].max()}")
                    st.write(f"Unique symbols: {earnings_data['symbol'].nunique()}")
                    st.write(f"Columns: {list(earnings_data.columns)}")
                
                st.markdown("#### Categorized Data")
                for week, data in categorized_earnings.items():
                    st.write(f"{week}: {len(data)} records")
                
                st.markdown("#### Statistics")
                st.json(calendar_stats)
                
                if not earnings_data.empty:
                    st.markdown("#### Sample Data")
                    st.dataframe(earnings_data.head(10))
                
        except Exception as e:
            logger.error(f"Error displaying debug info: {e}")
            st.error("Failed to display debug information")
    
    def display_performance_insights(self, categorized_earnings: Dict[str, pd.DataFrame],
                                   calendar_stats: Dict[str, Dict[str, float]]) -> None:
        """
        Display performance insights and recommendations.
        
        Args:
            categorized_earnings: Categorized earnings data
            calendar_stats: Calendar statistics
        """
        try:
            logger.info("[display_performance_insights] Displaying performance insights")
            
            st.markdown("### 💡 Earnings Performance Insights")
            
            # Find best performing week
            best_week = None
            best_return = float('-inf')
            
            for week, stats in calendar_stats.items():
                if stats.get('total_stocks', 0) > 0:
                    avg_return = stats.get('avg_return', 0)
                    if avg_return > best_return:
                        best_return = avg_return
                        best_week = week
            
            if best_week:
                st.success(f"🎯 **Best Performance**: {best_week.replace('_', ' ').title()} with {best_return:.1f}% average return")
            
            # Volatility insights
            high_volatility_weeks = []
            for week, stats in calendar_stats.items():
                if stats.get('avg_volatility', 0) > 5.0:  # Threshold for high volatility
                    high_volatility_weeks.append(week.replace('_', ' ').title())
            
            if high_volatility_weeks:
                st.warning(f"⚠️ **High Volatility Expected**: {', '.join(high_volatility_weeks)}")
            
            # Stock count insights
            total_stocks = sum(stats.get('total_stocks', 0) for stats in calendar_stats.values())
            if total_stocks > 50:
                st.info(f"📊 **Heavy Earnings Week**: {total_stocks} total earnings announcements - expect increased market volatility")
            
        except Exception as e:
            logger.error(f"Error displaying performance insights: {e}")
            st.error("Failed to display performance insights") 