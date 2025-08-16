import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def display_stock_rankings(top_stocks_df):
    """
    Display stock rankings chart.
    """
    fig = px.bar(
        top_stocks_df.head(10),
        x='ticker',
        y='predicted_return_pct',
        color='confidence_score',
        title="Top 10 Stocks by Predicted Return",
        labels={'predicted_return_pct': 'Predicted Return (%)', 'confidence_score': 'Confidence Score'},
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def display_risk_return_analysis(top_stocks_df):
    """
    Display risk vs return scatter plot.
    """
    fig = px.scatter(
        top_stocks_df,
        x='risk_score',
        y='predicted_return_pct',
        size='confidence_score',
        color='composite_score',
        hover_data=['ticker'],
        title="Risk vs Return Analysis",
        labels={'risk_score': 'Risk Score', 'predicted_return_pct': 'Predicted Return (%)', 
               'confidence_score': 'Confidence Score', 'composite_score': 'Composite Score'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def display_technical_signals_heatmap(top_stocks_df):
    """
    Display technical signals heatmap.
    """
    if len(top_stocks_df) > 0:
        tech_signals = ['broke_resistance', 'post_earnings_dip_rally', 'strong_momentum', 
                       'breakout_confirmed', 'earnings_in_3_weeks', 'last_2q_positive_surprises']
        
        signal_data = []
        for _, stock in top_stocks_df.iterrows():
            for signal in tech_signals:
                if signal in stock:
                    signal_data.append({
                        'ticker': stock['ticker'],
                        'signal': signal.replace('_', ' ').title(),
                        'value': 1 if stock[signal] else 0
                    })
        
        if signal_data:
            signal_df = pd.DataFrame(signal_data)
            
            # Handle duplicate entries by aggregating values
            try:
                # Group by ticker and signal, then take the max value (in case of duplicates)
                signal_df_agg = signal_df.groupby(['ticker', 'signal'])['value'].max().reset_index()
                
                # Create pivot table
                pivot_df = signal_df_agg.pivot(index='ticker', columns='signal', values='value')
                
                # Fill NaN values with 0
                pivot_df = pivot_df.fillna(0)
                
                fig = px.imshow(
                    pivot_df,
                    title="Technical Signals Heatmap",
                    color_continuous_scale='RdYlGn',
                    aspect='auto'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not create heatmap due to data structure: {str(e)}")
                # Fallback: show a simple table
                st.write("**Technical Signals Summary:**")
                signal_summary = top_stocks_df[tech_signals].sum()
                st.write(signal_summary)

def display_sector_analysis(top_stocks_df):
    """
    Display sector analysis pie chart.
    """
    if 'sector' in top_stocks_df.columns:
        sector_counts = top_stocks_df['sector'].value_counts()
        fig = px.pie(
            values=sector_counts.values,
            names=sector_counts.index,
            title="Portfolio Sector Distribution"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display sector breakdown table
        st.subheader("Sector Breakdown")
        sector_summary = top_stocks_df.groupby('sector').agg({
            'ticker': 'count',
            'predicted_return_pct': 'mean',
            'confidence_score': 'mean',
            'risk_score': 'mean'
        }).round(2)
        sector_summary.columns = ['Count', 'Avg Return (%)', 'Avg Confidence', 'Avg Risk']
        st.dataframe(sector_summary)
    else:
        st.warning("Sector information not available in the data.")

def display_portfolio_metrics(top_stocks_df):
    """
    Display portfolio summary metrics.
    """
    st.subheader("📊 Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stocks", len(top_stocks_df))
        avg_return = top_stocks_df['predicted_return_pct'].mean()
        st.metric("Average Return", f"{avg_return:.2f}%")
    
    with col2:
        avg_confidence = top_stocks_df['confidence_score'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.1f}/100")
        avg_risk = top_stocks_df['risk_score'].mean()
        st.metric("Average Risk", f"{avg_risk:.1f}/100")
    
    with col3:
        positive_stocks = len(top_stocks_df[top_stocks_df['predicted_return_pct'] > 0])
        st.metric("Positive Predictions", positive_stocks)
        resistance_breaks = len(top_stocks_df[top_stocks_df['broke_resistance'] == True])
        st.metric("Resistance Breaks", resistance_breaks)
    
    with col4:
        high_confidence = len(top_stocks_df[top_stocks_df['confidence_score'] > 50])
        st.metric("High Confidence (>50)", high_confidence)
        low_risk = len(top_stocks_df[top_stocks_df['risk_score'] < 50])
        st.metric("Low Risk (<50)", low_risk)

def display_top_stocks_table(top_stocks_df):
    """
    Display top stocks in a formatted table.
    """
    st.subheader("🏆 Top Stock Picks")
    
    # Select columns to display
    display_cols = ['ticker', 'close', 'predicted_return_pct', 'confidence_score', 
                   'risk_score', 'composite_score', 'momentum_5d', 'momentum_10d', 
                   'rsi_14d', 'broke_resistance']
    
    # Filter columns that exist in the dataframe
    available_cols = [col for col in display_cols if col in top_stocks_df.columns]
    
    if available_cols:
        display_df = top_stocks_df[available_cols].copy()
        
        # Format numeric columns
        if 'predicted_return_pct' in display_df.columns:
            display_df['predicted_return_pct'] = display_df['predicted_return_pct'].round(2)
        if 'confidence_score' in display_df.columns:
            display_df['confidence_score'] = display_df['confidence_score'].round(1)
        if 'risk_score' in display_df.columns:
            display_df['risk_score'] = display_df['risk_score'].round(1)
        if 'composite_score' in display_df.columns:
            display_df['composite_score'] = display_df['composite_score'].round(2)
        if 'momentum_5d' in display_df.columns:
            display_df['momentum_5d'] = display_df['momentum_5d'].round(2)
        if 'momentum_10d' in display_df.columns:
            display_df['momentum_10d'] = display_df['momentum_10d'].round(2)
        if 'rsi_14d' in display_df.columns:
            display_df['rsi_14d'] = display_df['rsi_14d'].round(1)
        if 'close' in display_df.columns:
            display_df['close'] = display_df['close'].round(2)
        
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("No displayable columns found in the data.")

def display_export_options(top_stocks_df):
    """
    Display export options for the portfolio data.
    """
    st.subheader("📤 Export Portfolio Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export as CSV
        csv_data = top_stocks_df.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv_data,
            file_name=f"stock_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export as Excel
        try:
            import io
            from openpyxl import Workbook
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                top_stocks_df.to_excel(writer, sheet_name='Portfolio', index=False)
            
            excel_data = output.getvalue()
            st.download_button(
                label="📊 Download as Excel",
                data=excel_data,
                file_name=f"stock_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except ImportError:
            st.warning("Excel export requires openpyxl package. Install with: pip install openpyxl")
