import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
from stock_analysis import calculate_rsi
from config_manager import ConfigManager, DatabaseManager

# Page configuration
st.set_page_config(
    page_title="Stock Debug App",
    page_icon="📈",
    layout="wide"
)

# Initialize configuration and database managers
@st.cache_resource
def get_managers():
    """Initialize configuration and database managers"""
    config_manager = ConfigManager()
    db_manager = DatabaseManager(config_manager)
    return config_manager, db_manager

def is_earning_within_3_weeks(earning_date):
    """Check if earning date is within next 3 weeks"""
    if pd.isna(earning_date):
        return False
    
    try:
        earning_dt = pd.to_datetime(earning_date)
        today = datetime.now()
        three_weeks_later = today + timedelta(weeks=3)
        return today <= earning_dt <= three_weeks_later
    except:
        return False

def main():
    # Initialize managers
    config_manager, db_manager = get_managers()
    
    st.title("📈 Stock Debug Dashboard")
    
    # Test connection
    is_connected, message = db_manager.test_connection()
    if not is_connected:
        st.error(f"Database connection failed: {message}")
        return
    
    st.success(f"Database connected: {message}")
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = None
    
    st.write(f"Current page: {st.session_state.page}")
    st.write(f"Selected symbol: {st.session_state.selected_symbol}")
    
    if st.session_state.page == 'home':
        # Load data
        df = db_manager.load_stock_change_tracker()
        
        if df.empty:
            st.warning("No data available")
            return
        
        st.success(f"Loaded {len(df)} stocks")
        st.dataframe(df.head())
        
        # Display simplified table
        st.subheader("Click a symbol to test:")
        
        for idx, row in df.head(5).iterrows():  # Only show first 5 for testing
            cols = st.columns([2, 2, 2, 2])
            
            with cols[0]:
                symbol_color = "🟢" if is_earning_within_3_weeks(row['earning_date']) else "⚪"
                button_key = f"btn_{row['symbol']}_{idx}"
                st.write(f"Button key: {button_key}")
                
                if st.button(f"{symbol_color} {row['symbol']}", key=button_key):
                    st.write(f"Button clicked for {row['symbol']}")
                    st.session_state.selected_symbol = row['symbol']
                    st.session_state.page = 'stock_detail'
                    st.write("Session state updated")
                    st.rerun()
            
            with cols[1]:
                st.write(f"${row['price_when_added']:.2f}")
            
            with cols[2]:
                st.write(f"${row['current_price']:.2f}")
            
            with cols[3]:
                change_color = "🟢" if row['change_in_percent'] > 0 else "🔴"
                st.write(f"{change_color} {row['change_in_percent']:.2f}%")
    
    elif st.session_state.page == 'stock_detail':
        symbol = st.session_state.selected_symbol
        st.title(f"📊 {symbol} - Test Analysis")
        
        # Back button
        if st.button("← Back to Home"):
            st.session_state.page = 'home'
            st.session_state.selected_symbol = None
            st.rerun()
        
        # Load stock data
        stock_data = db_manager.load_stock_data(symbol)
        
        if stock_data.empty:
            st.warning(f"No historical data available for {symbol}")
            return
        
        st.success(f"Loaded {len(stock_data)} days of historical data for {symbol}")
        
        # Simple RSI test
        if len(stock_data) >= 14:
            st.subheader("RSI Test")
            
            # Debug the data first
            st.write("Close prices sample:", stock_data['close'].head())
            st.write("Close prices data type:", stock_data['close'].dtype)
            st.write("Close prices values type:", type(stock_data['close'].values))
            
            try:
                rsi_values = calculate_rsi(stock_data['close'].values, period=14)
                st.write(f"RSI calculated for {len(rsi_values)} periods")
                if len(rsi_values) > 0:
                    st.write(f"Current RSI: {rsi_values[-1]:.2f}")
                else:
                    st.error("No RSI values calculated")
            except Exception as e:
                st.error(f"Error calculating RSI: {e}")
                st.write("Trying with data conversion...")
                
                # Try converting to numeric
                close_prices = pd.to_numeric(stock_data['close'], errors='coerce')
                st.write("After conversion:", close_prices.head())
                st.write("Converted data type:", close_prices.dtype)
                
                try:
                    rsi_values = calculate_rsi(close_prices.values, period=14)
                    st.success(f"RSI calculated successfully: {len(rsi_values)} periods")
                    if len(rsi_values) > 0:
                        st.write(f"Current RSI: {rsi_values[-1]:.2f}")
                except Exception as e2:
                    st.error(f"Still failed after conversion: {e2}")
            
            # Simple plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(rsi_values))),
                y=rsi_values,
                mode='lines',
                name='RSI'
            ))
            fig.update_layout(title=f'{symbol} - RSI Test', height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Not enough data for RSI calculation")

if __name__ == "__main__":
    main() 