"""
Consistency UI Renderer Utility

Handles all Streamlit UI rendering and display components for consistency analysis.
Extracted from ConsistentPerformersTab for better separation of concerns.
"""

import logging
from typing import List, Optional

import pandas as pd
import streamlit as st

# Configuration constants
TOP_PERFORMERS_COUNT = 25

logger = logging.getLogger('StockApp')


class ConsistencyUIRenderer:
    """
    Handles all UI rendering and display components for consistency analysis.
    
    Responsibilities:
    - Streamlit component rendering
    - Table display and formatting
    - Insights and metrics display
    - User interface layout
    """
    
    def render_debug_info(self, months: int, consistent_performers: pd.DataFrame, pre_calculated_data: Optional[pd.DataFrame] = None) -> None:
        """
        Render debug information for the analysis.
        
        Args:
            months: Number of months analyzed
            consistent_performers: Consistency analysis results
            pre_calculated_data: Optional pre-calculated data
        """
        period_text = "months" if months > 1 else "month"
        debug_info = f"""
        **🔍 Debug Information ({months} {period_text}):**
        - Data method: {'Pre-calculated' if months == 3 and pre_calculated_data is not None and not pre_calculated_data.empty else 'Fresh database fetch'}
        - Analysis period: {months} {period_text}
        - Total stocks found: {len(consistent_performers)}
        - Date range: {consistent_performers['first_date'].min()} to {consistent_performers['last_date'].max()}
        - Min total return: {consistent_performers['total_return'].min():.2f}%
        - Max total return: {consistent_performers['total_return'].max():.2f}%
        - Min positive periods: {consistent_performers['positive_months_ratio'].min():.0%}
        - SQL fix applied: ✅ Direct JOIN with stock_change_tracker (avoids historical data loss)
        - Symbol filter: ✅ is_active=1 AND current_price>5 (simple and fast)
        - Query optimization: ✅ No complex subqueries, direct filtering (much faster execution)
        """
        
        # Add specific 1-month debugging
        if months == 1:
            from datetime import datetime
            current_month = datetime.now().strftime('%Y-%m')
            debug_info += f"""
            
            **📅 1-Month Analysis Details:**
            - Current month: {current_month}
            - Date range method: Using current month boundaries instead of last 30 days
            - Minimum data points: 3 (relaxed from 5 for current month analysis)
            - Filtering: Relaxed - allows negative returns, only requires defined positive ratio
            - Why limited results: Early in month = fewer trading days = fewer qualifying stocks
            """
            
            # Show breakdown of the results
            if not consistent_performers.empty:
                positive_returns = len(consistent_performers[consistent_performers['total_return'] > 0])
                negative_returns = len(consistent_performers[consistent_performers['total_return'] < 0])
                zero_returns = len(consistent_performers[consistent_performers['total_return'] == 0])
                
                debug_info += f"""
                
                **📊 Current Month Results Breakdown:**
                - Stocks with positive returns: {positive_returns}
                - Stocks with negative returns: {negative_returns} 
                - Stocks with zero returns: {zero_returns}
                - Average data points per stock: {consistent_performers['data_points'].mean():.1f}
                - Date range span: {(pd.to_datetime(consistent_performers['last_date'].max()) - pd.to_datetime(consistent_performers['first_date'].min())).days} days
                """
        
        st.info(debug_info)
    
    def display_consistency_table(self, data: pd.DataFrame, months: int = 3) -> None:
        """
        Display consistency data in table format.
        
        Args:
            data: Consistency data
            months: Number of months analyzed (1, 2, or 3)
        """
        period_text = "Month" if months == 1 else "Months"
        st.subheader(f"📋 Consistency Analysis Summary - {months} {period_text}")
        
        if data.empty:
            st.warning("No data available for table")
            return
        
        # Prepare display data - sort by total return (highest first)
        display_data = data.copy()
        display_data = display_data.sort_values('total_return', ascending=False)
        
        # Format the data for display
        formatted_data = {
            'Rank': range(1, len(display_data) + 1),
            'Symbol': display_data['symbol'].tolist(),
            'Consistency Index': [f"{val:.3f}" for val in display_data['consistency_index']],
            'Positive Months': [f"{val:.0%}" for val in display_data['positive_months_ratio']],
            'Volatility (CV)': [f"{val:.2f}" for val in display_data['volatility_cv']],
            'Avg Monthly Return': [f"{val:.2f}%" for val in display_data['geometric_mean_return']],
            'Worst Month': [f"{val:.2f}%" for val in display_data['min_monthly_return']],
            'Total Return': [f"{val:.2f}%" for val in display_data['total_return']],
            'Period': [f"{row['first_date']} to {row['last_date']}" for _, row in display_data.iterrows()]
        }
        
        df_display = pd.DataFrame(formatted_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Add methodology explanation
        self._display_methodology_explanation(months)
    
    def display_insights(self, data: pd.DataFrame, current_symbol: str, months: int = 3) -> None:
        """
        Display insights about consistency performance.
        
        Args:
            data: Consistency data
            current_symbol: Currently selected symbol
            months: Number of months analyzed (1, 2, or 3)
        """
        period_text = "Month" if months == 1 else "Months"
        st.subheader(f"🔍 Consistency Insights - {months} {period_text}")
        
        if data.empty:
            st.warning("No data available for insights")
            return
        
        # Calculate insights
        total_stocks = len(data)
        all_positive_months = len(data[data['positive_months_ratio'] == 1.0])
        high_consistency = len(data[data['consistency_index'] >= 0.7])
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Total Analyzed",
                f"{total_stocks}",
                help="Total number of stocks with sufficient data for consistency analysis"
            )
        
        with col2:
            positive_text = f"ALL {months} {period_text.lower()}" if months > 1 else "the current month"
            st.metric(
                "✅ All Positive Months",
                f"{all_positive_months}",
                f"{all_positive_months/total_stocks:.1%}",
                help=f"Stocks that had positive returns in {positive_text}"
            )
        
        with col3:
            st.metric(
                "🏆 High Consistency",
                f"{high_consistency}",
                f"{high_consistency/total_stocks:.1%}",
                help="Stocks with Consistency Index >= 0.7"
            )
        
        with col4:
            avg_consistency = data['consistency_index'].mean()
            st.metric(
                "📈 Avg Consistency",
                f"{avg_consistency:.3f}",
                help="Average Consistency Index across all stocks"
            )
        
        # Most consistent stocks
        self._display_top_performers(data)
        
        # Current symbol analysis
        self._display_current_symbol_analysis(data, current_symbol, total_stocks)
    
    def display_symbol_filter_section(self, data: pd.DataFrame) -> tuple:
        """
        Display symbol filter section and return selected symbols and chart columns.
        
        Args:
            data: Consistency data for symbol selection
            
        Returns:
            Tuple of (selected_symbols, chart_columns)
        """
        st.markdown("---")
        st.subheader("📈 Individual Stock Analysis")
        
        if data.empty:
            st.warning("No data available for individual analysis")
            return [], 1
        
        # Create symbol filter
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get top 25 symbols sorted by consistency index
            top_symbols = data.nlargest(25, 'consistency_index')['symbol'].tolist()
            
            # Multi-select for symbols
            selected_symbols = st.multiselect(
                "🎯 Select Consistent Performers to Analyze:",
                options=top_symbols,
                default=top_symbols[:6],  # Default to top 6
                help="Select up to 15 stocks to display detailed price charts. Stocks are ordered by Consistency Index."
            )
        
        with col2:
            # Options for chart display
            chart_columns = st.selectbox(
                "Chart Columns:",
                options=[1, 2, 3, 4],
                index=0,  # Default to 1 column
                help="Number of columns in the chart grid"
            )
        
        if not selected_symbols:
            st.info("👆 Select one or more symbols above to view detailed price charts")
            return [], chart_columns
        
        if len(selected_symbols) > 15:
            st.warning("⚠️ Please select maximum 15 symbols for better performance")
            selected_symbols = selected_symbols[:15]
        
        return selected_symbols, chart_columns
    
    def display_filter_info(self, selected_symbols: List[str]) -> None:
        """
        Display information about filtered symbols.
        
        Args:
            selected_symbols: List of selected symbols
        """
        if selected_symbols:
            st.info(f"📊 Showing analysis for {len(selected_symbols)} selected symbols")
    
    def display_no_data_warning(self, selected_symbols: List[str]) -> None:
        """
        Display warning when no data is available for selected symbols.
        
        Args:
            selected_symbols: List of symbols that had no data
        """
        st.warning(f"No consistency data available for selected symbols: {', '.join(selected_symbols)}")
    
    def _display_methodology_explanation(self, months: int) -> None:
        """Display methodology explanation for the analysis."""
        period_text = "months" if months > 1 else "month"
        
        with st.expander("📊 Methodology Explanation"):
            st.markdown(f"""
            **Consistency Index Formula:**
            ```
            Consistency Index = (Positive Months Ratio × 0.4) + 
                              ((1 - Normalized Volatility) × 0.3) + 
                              (Normalized Geometric Mean × 0.3)
            ```
            
            **Components:**
            - **Positive Months Ratio (40%)**: Percentage of months with positive returns
            - **Volatility Score (30%)**: Lower coefficient of variation = higher score
            - **Geometric Mean Return (30%)**: Compound monthly growth rate
            
            **Why This Works:**
            - Rewards stocks that go up consistently over the {months} {period_text} period
            - Penalizes high volatility even with good total returns  
            - Balances growth with stability
            
            **Filters Applied:**
            - ✅ Must have positive total return over {months} {period_text}
            - ✅ Must have at least 1 positive month out of {months}
            - ✅ Must have sufficient data points ({max(5, months * 5)}+ days)
            
            **Data Processing:**
            - 🚀 Single database fetch for all {months}-{period_text} data
            - 📊 All calculations done with pandas operations
            - ✅ Guaranteed data consistency across all charts
            """)
    
    def _display_top_performers(self, data: pd.DataFrame) -> None:
        """Display most consistent performers."""
        st.markdown("**🥇 Most Consistent Performers:**")
        top_3 = data.nlargest(3, 'consistency_index')
        
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            st.write(f"{i}. **{row['symbol']}** - Consistency Index: {row['consistency_index']:.3f}")
            st.write(f"   • {row['positive_months_ratio']:.0%} positive months, {row['geometric_mean_return']:.2f}% avg monthly return, {row['volatility_cv']:.2f} volatility")
    
    def _display_current_symbol_analysis(self, data: pd.DataFrame, current_symbol: str, total_stocks: int) -> None:
        """Display analysis for the currently selected symbol."""
        if current_symbol and current_symbol in data['symbol'].values:
            current_data = data[data['symbol'] == current_symbol].iloc[0]
            rank = (data['consistency_index'] > current_data['consistency_index']).sum() + 1
            
            st.markdown(f"**📍 Current Symbol ({current_symbol}) Analysis:**")
            st.write(f"• Consistency Rank: #{rank} out of {total_stocks}")
            st.write(f"• Consistency Index: {current_data['consistency_index']:.3f}")
            st.write(f"• {current_data['positive_months_ratio']:.0%} positive months")
            st.write(f"• Volatility: {current_data['volatility_cv']:.2f}")
    
    def render_tab_controls(self) -> tuple:
        """
        Render tab controls and return debug mode state.
        
        Returns:
            Tuple of (debug_mode, refresh_requested)
        """
        # Add cache clearing option if seeing stale data
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col2:
            debug_mode = st.checkbox("🔍 Debug Mode", help="Show detailed data processing information")
        
        with col3:
            refresh_requested = st.button("🔄 Refresh Data", help="Clear cache and reload fresh data")
            if refresh_requested:
                st.cache_data.clear()
                st.rerun()
        
        return debug_mode, refresh_requested
    
    def render_period_tabs(self) -> None:
        """Render the period analysis tabs structure (returns tab objects)."""
        # This method would be used by the main class to create the tab structure
        # The actual tab rendering would be handled by the main orchestrator
        pass
    
    def display_all_time_performers_table(self, data: pd.DataFrame) -> None:
        """
        Display all-time performers data in specialized table format.
        
        Args:
            data: All-time performers data with cross-period metrics
        """
        st.subheader("📋 All-Time Performers Analysis Summary")
        st.caption("Stocks appearing in ALL 3 time periods - the ultimate consistency champions")
        
        if data.empty:
            st.warning("No stocks found that appear in all 3 time periods")
            return
        
        # Prepare display data - sort by average consistency (highest first)
        display_data = data.copy()
        display_data = display_data.sort_values('consistency_index', ascending=False)
        
        # Format the data for display with cross-period metrics
        formatted_data = {
            'Rank': range(1, len(display_data) + 1),
            'Symbol': display_data['symbol'].tolist(),
            'Avg Consistency': [f"{val:.3f}" for val in display_data['consistency_index']],
            '1M Consistency': [f"{val:.3f}" for val in display_data['consistency_1m']],
            '2M Consistency': [f"{val:.3f}" for val in display_data['consistency_2m']],
            '3M Consistency': [f"{val:.3f}" for val in display_data['consistency_3m']],
            'Avg Positive %': [f"{val:.0%}" for val in display_data['avg_positive_months_ratio']],
            '1M Return': [f"{val:.1f}%" for val in display_data['total_return_1m']],
            '2M Return': [f"{val:.1f}%" for val in display_data['total_return_2m']],
            '3M Return': [f"{val:.1f}%" for val in display_data['total_return_3m']],
            'Avg Volatility': [f"{val:.2f}" for val in display_data['avg_volatility_cv']],
        }
        
        df_display = pd.DataFrame(formatted_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Add all-time performers explanation
        self._display_all_time_methodology_explanation()
    
    def display_all_time_insights(self, data: pd.DataFrame, current_symbol: str) -> None:
        """
        Display insights about all-time performance.
        
        Args:
            data: All-time performers data
            current_symbol: Currently selected symbol
        """
        st.subheader("🔍 All-Time Performers Insights")
        
        if data.empty:
            st.warning("No all-time performers found for insights")
            return
        
        # Calculate insights
        total_all_time = len(data)
        perfect_consistency = len(data[data['consistency_index'] >= 0.8])
        high_returns_all_periods = len(data[
            (data['total_return_1m'] > 5) & 
            (data['total_return_2m'] > 5) & 
            (data['total_return_3m'] > 5)
        ])
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🏆 All-Time Performers",
                f"{total_all_time}",
                help="Stocks appearing in ALL 3 time periods (1M, 2M, 3M)"
            )
        
        with col2:
            st.metric(
                "⭐ Elite Consistency",
                f"{perfect_consistency}",
                f"{perfect_consistency/total_all_time:.1%}" if total_all_time > 0 else "0%",
                help="Stocks with average consistency index >= 0.8"
            )
        
        with col3:
            st.metric(
                "💰 High Returns All Periods",
                f"{high_returns_all_periods}",
                f"{high_returns_all_periods/total_all_time:.1%}" if total_all_time > 0 else "0%",
                help="Stocks with >5% returns in ALL periods"
            )
        
        with col4:
            avg_consistency = data['consistency_index'].mean()
            st.metric(
                "📈 Avg All-Time Consistency",
                f"{avg_consistency:.3f}",
                help="Average consistency index across all all-time performers"
            )
        
        # Top all-time performers
        st.markdown("**🥇 Top All-Time Champions:**")
        top_3 = data.nlargest(3, 'consistency_index')
        
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            st.write(f"{i}. **{row['symbol']}** - Avg Consistency: {row['consistency_index']:.3f}")
            st.write(f"   • Consistency: 1M({row['consistency_1m']:.3f}), 2M({row['consistency_2m']:.3f}), 3M({row['consistency_3m']:.3f})")
            st.write(f"   • Returns: 1M({row['total_return_1m']:.1f}%), 2M({row['total_return_2m']:.1f}%), 3M({row['total_return_3m']:.1f}%)")
        
        # Current symbol analysis
        if current_symbol and current_symbol in data['symbol'].values:
            current_data = data[data['symbol'] == current_symbol].iloc[0]
            rank = (data['consistency_index'] > current_data['consistency_index']).sum() + 1
            
            st.markdown(f"**📍 Current Symbol ({current_symbol}) All-Time Analysis:**")
            st.write(f"• All-Time Rank: #{rank} out of {total_all_time}")
            st.write(f"• Average Consistency: {current_data['consistency_index']:.3f}")
            st.write(f"• Period Consistency: 1M({current_data['consistency_1m']:.3f}), 2M({current_data['consistency_2m']:.3f}), 3M({current_data['consistency_3m']:.3f})")
            st.write(f"• Period Returns: 1M({current_data['total_return_1m']:.1f}%), 2M({current_data['total_return_2m']:.1f}%), 3M({current_data['total_return_3m']:.1f}%)")
    
    def _display_all_time_methodology_explanation(self) -> None:
        """Display methodology explanation for all-time performers analysis."""
        with st.expander("📊 All-Time Performers Methodology"):
            st.markdown("""
            **🏆 All-Time Performers Selection Criteria:**
            
            **Step 1: Cross-Period Filtering**
            - Stock must appear in **ALL 3 time periods**: Current Month, 2-Month, and 3-Month
            - Must pass individual consistency analysis for each period
            - Only the most consistent stocks across all timeframes qualify
            
            **Step 2: Aggregated Metrics Calculation**
            - **Avg Consistency**: Average of 1M, 2M, and 3M consistency indices
            - **Avg Positive %**: Average positive months ratio across all periods  
            - **Avg Volatility**: Average coefficient of variation across periods
            - **Period Returns**: Individual returns for each time period
            
            **Step 3: Ranking and Display**
            - Ranked by **Average Consistency Index** (highest first)
            - Shows individual period performance for comparison
            - Highlights stocks that excel across ALL timeframes
            
            **Why This Analysis Matters:**
            - 🎯 **Ultimate Filter**: Only the most consistent stocks qualify
            - 📈 **Trend Validation**: Confirms consistency isn't just short-term luck
            - 💼 **Investment Grade**: These are your premium, low-risk growth candidates
            - 🔍 **Quality Focus**: Small list of highest-quality opportunities
            
            **Interpretation Guide:**
            - **Avg Consistency ≥ 0.8**: Elite all-time performers
            - **Consistent Period Returns**: Look for positive returns across 1M, 2M, 3M
            - **Low Avg Volatility**: More stable, predictable performance
            - **High Rank**: Top candidates for conservative growth portfolios
            
            **Investment Strategy:**
            - Use all-time performers as **core holdings**
            - Combine with individual period analysis for **entry timing**
            - Monitor for any period where stock drops out of top performers
            """)
    
    def display_all_time_debug_info(self, all_time_data: pd.DataFrame, data_1m: pd.DataFrame, 
                                   data_2m: pd.DataFrame, data_3m: pd.DataFrame) -> None:
        """
        Display debug information for all-time performers analysis.
        
        Args:
            all_time_data: Final all-time performers dataset
            data_1m: 1-month consistency data
            data_2m: 2-month consistency data
            data_3m: 3-month consistency data
        """
        debug_info = f"""
        **🔍 All-Time Performers Debug Information:**
        
        **Period Analysis Results:**
        - 1-Month performers: {len(data_1m)} stocks
        - 2-Month performers: {len(data_2m)} stocks  
        - 3-Month performers: {len(data_3m)} stocks
        - **All-Time performers: {len(all_time_data)} stocks**
        
        **Cross-Period Filtering:**
        - Intersection method: Set intersection of all 3 periods
        - Qualification rate: {len(all_time_data)/max(len(data_3m), 1)*100:.1f}% of 3-month performers
        - Quality threshold: Must appear in ALL periods (strictest filter)
        
        **Data Quality:**
        - All-time symbols: {sorted(all_time_data['symbol'].tolist()) if not all_time_data.empty else 'None'}
        - Avg consistency range: {all_time_data['consistency_index'].min():.3f} to {all_time_data['consistency_index'].max():.3f}
        - Cross-period validation: ✅ All stocks verified in individual periods
        """
        
        if not all_time_data.empty:
            # Show top 3 with detailed breakdown
            top_3 = all_time_data.nlargest(3, 'consistency_index')
            debug_info += f"""
            
            **Top 3 All-Time Performers Breakdown:**
            """
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                debug_info += f"""
            {i}. **{row['symbol']}**: Avg CI={row['consistency_index']:.3f}
               - 1M: CI={row['consistency_1m']:.3f}, Return={row['total_return_1m']:.1f}%
               - 2M: CI={row['consistency_2m']:.3f}, Return={row['total_return_2m']:.1f}%  
               - 3M: CI={row['consistency_3m']:.3f}, Return={row['total_return_3m']:.1f}%
                """
        
        st.info(debug_info)
    
    def render_control_panel(self) -> bool:
        """
        Render the control panel with debug mode and refresh options.
        
        Returns:
            bool: True if debug mode is enabled, False otherwise
        """
        # Add cache clearing option if seeing stale data
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            debug_mode = st.checkbox("🔍 Debug Mode", help="Show detailed data processing information")
        with col3:
            if st.button("🔄 Refresh Data", help="Clear cache and reload fresh data"):
                st.cache_data.clear()
                st.rerun()
        
        return debug_mode
    
    def display_period_debug_info(self, consistent_performers: pd.DataFrame, months: int, raw_data: pd.DataFrame) -> None:
        """
        Display debug information for a specific period analysis.
        
        Args:
            consistent_performers: Consistency analysis results
            months: Number of months analyzed
            raw_data: Raw stock data used for analysis
        """
        if consistent_performers.empty:
            return
            
        period_text = "months" if months > 1 else "month"
        debug_info = f"""
        **🔍 Debug Information ({months} {period_text}):**
        - Data method: {'Raw data from database' if raw_data is not None and not raw_data.empty else 'Pre-calculated data'}
        - Analysis period: {months} {period_text}
        - Total stocks found: {len(consistent_performers)}
        - Date range: {consistent_performers['first_date'].min()} to {consistent_performers['last_date'].max()}
        - Min total return: {consistent_performers['total_return'].min():.2f}%
        - Max total return: {consistent_performers['total_return'].max():.2f}%
        - Min positive periods: {consistent_performers['positive_months_ratio'].min():.0%}
        - SQL fix applied: ✅ Direct JOIN with stock_change_tracker (avoids historical data loss)
        - Symbol filter: ✅ is_active=1 AND current_price>5 (simple and fast)
        - Query optimization: ✅ No complex subqueries, direct filtering (much faster execution)
        """
        
        # Add specific 1-month debugging
        if months == 1:
            from datetime import datetime
            # Show current month details
            current_month = datetime.now().strftime('%Y-%m')
            debug_info += f"""
            
            **📅 1-Month Analysis Details:**
            - Current month: {current_month}
            - Date range method: Using current month boundaries instead of last 30 days
            - Minimum data points: 3 (relaxed from 5 for current month analysis)
            - Filtering: Relaxed - allows negative returns, only requires defined positive ratio
            - Why limited results: Early in month = fewer trading days = fewer qualifying stocks
            """
            
            # Show breakdown of the results
            if not consistent_performers.empty:
                positive_returns = len(consistent_performers[consistent_performers['total_return'] > 0])
                negative_returns = len(consistent_performers[consistent_performers['total_return'] < 0])
                zero_returns = len(consistent_performers[consistent_performers['total_return'] == 0])
                
                debug_info += f"""
                
                **📊 Current Month Results Breakdown:**
                - Stocks with positive returns: {positive_returns}
                - Stocks with negative returns: {negative_returns} 
                - Stocks with zero returns: {zero_returns}
                - Average data points per stock: {consistent_performers['data_points'].mean():.1f}
                - Date range span: {(pd.to_datetime(consistent_performers['last_date'].max()) - pd.to_datetime(consistent_performers['first_date'].min())).days} days
                """
        
        st.info(debug_info)
    
    def display_insufficient_data_warning(self, consistent_1m: pd.DataFrame, consistent_2m: pd.DataFrame, consistent_3m: pd.DataFrame) -> None:
        """
        Display warning when insufficient data is available for all-time performers analysis.
        
        Args:
            consistent_1m: 1-month consistency data
            consistent_2m: 2-month consistency data  
            consistent_3m: 3-month consistency data
        """
        st.warning("⚠️ Insufficient data for all-time performers analysis. Need data for all 3 time periods.")
        
        # Show what data we have
        st.write("**Data availability:**")
        st.write(f"• 1-Month performers: {len(consistent_1m)} stocks")
        st.write(f"• 2-Month performers: {len(consistent_2m)} stocks") 
        st.write(f"• 3-Month performers: {len(consistent_3m)} stocks")
    
    def display_no_all_time_performers_found(self, consistent_1m: pd.DataFrame, consistent_2m: pd.DataFrame, consistent_3m: pd.DataFrame) -> None:
        """
        Display message when no all-time performers are found.
        
        Args:
            consistent_1m: 1-month consistency data
            consistent_2m: 2-month consistency data
            consistent_3m: 3-month consistency data
        """
        st.warning("🚫 No stocks found that appear in ALL 3 time periods.")
        st.write("**Analysis Results:**")
        st.write(f"• 1-Month performers: {len(consistent_1m)} stocks")
        st.write(f"• 2-Month performers: {len(consistent_2m)} stocks") 
        st.write(f"• 3-Month performers: {len(consistent_3m)} stocks")
        st.write(f"• **All-Time champions: 0 stocks**")
        
        # Show top performers from each period for comparison
        st.markdown("**Top 5 from each period:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**1-Month Top 5:**")
            if not consistent_1m.empty:
                top_1m = consistent_1m.nlargest(5, 'consistency_index')['symbol'].tolist()
                for i, sym in enumerate(top_1m, 1):
                    st.write(f"{i}. {sym}")
        
        with col2:
            st.write("**2-Month Top 5:**")
            if not consistent_2m.empty:
                top_2m = consistent_2m.nlargest(5, 'consistency_index')['symbol'].tolist()
                for i, sym in enumerate(top_2m, 1):
                    st.write(f"{i}. {sym}")
        
        with col3:
            st.write("**3-Month Top 5:**")
            if not consistent_3m.empty:
                top_3m = consistent_3m.nlargest(5, 'consistency_index')['symbol'].tolist()
                for i, sym in enumerate(top_3m, 1):
                    st.write(f"{i}. {sym}")
    
    def display_cross_period_comparison_table(self, comparison_data: pd.DataFrame) -> None:
        """
        Display cross-period comparison table for all-time performers.
        
        Args:
            comparison_data: DataFrame with cross-period ranking comparison
        """
        st.markdown("**Individual Period Rankings of All-Time Champions:**")
        
        if not comparison_data.empty:
            st.dataframe(comparison_data, use_container_width=True, hide_index=True)
            
            st.info("""
            **📊 Rankings Interpretation:**
            - **Consistency Stability**: How similar the rankings are across periods
            - **🟢 High**: Rank difference ≤ 10 positions (very stable)
            - **🟡 Moderate**: Rank difference 11-20 positions (somewhat stable)  
            - **🔴 Variable**: Rank difference > 20 positions (volatile ranking)
            """)
        else:
            st.warning("No cross-period comparison data available.")
    
    def display_all_time_debug_info(self, all_time_performers: pd.DataFrame, consistent_1m: pd.DataFrame, 
                                   consistent_2m: pd.DataFrame, consistent_3m: pd.DataFrame) -> None:
        """
        Display debug information for all-time performers analysis.
        
        Args:
            all_time_performers: All-time performers data
            consistent_1m: 1-month consistency data
            consistent_2m: 2-month consistency data
            consistent_3m: 3-month consistency data
        """
        debug_info = f"""
        **🔍 All-Time Performers Debug Information:**
        - Total all-time champions: {len(all_time_performers)}
        - 1-Month performers: {len(consistent_1m)} stocks
        - 2-Month performers: {len(consistent_2m)} stocks
        - 3-Month performers: {len(consistent_3m)} stocks
        - Intersection rate: {len(all_time_performers) / min(len(consistent_1m), len(consistent_2m), len(consistent_3m)) * 100:.1f}% of smallest period
        - Cross-period consistency: Ultra-high (appears in ALL periods)
        - Analysis method: Set intersection of all 3 periods
        """
        
        if not all_time_performers.empty:
            debug_info += f"""
            
            **📊 All-Time Champions List:**
            - Symbols: {', '.join(all_time_performers['symbol'].tolist())}
            - Avg Consistency Index: {all_time_performers['consistency_index'].mean():.3f}
            - Min Consistency Index: {all_time_performers['consistency_index'].min():.3f}
            - Max Consistency Index: {all_time_performers['consistency_index'].max():.3f}
            """
        
        st.info(debug_info) 