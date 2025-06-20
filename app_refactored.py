"""
🏠 Stock Analysis Dashboard - Home

Main Streamlit application serving as the home page in a multipage app.
Other pages are automatically detected from the pages/ directory.
"""

import streamlit as st
import logging
from logger_config import setup_logger
from data import DatabaseManager
from components.home_page import HomePage

# Setup logging
setup_logger()
logger = logging.getLogger('StockApp')

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_database_manager():
    """Initialize and cache the database manager"""
    try:
        return DatabaseManager()
    except Exception as e:
        st.error(f"Failed to initialize database: {e}")
        st.stop()

def display_connection_status(db_manager):
    """Display database connection status in sidebar"""
    try:
        status = db_manager.get_connection_status()
        
        if status['status'] == 'Connected':
            st.sidebar.success(f"🟢 Database Connected")
            st.sidebar.caption(f"Server: {status['host']}:{status['port']}")
            st.sidebar.caption(f"Database: {status['database']}")
        else:
            st.sidebar.error(f"🔴 Database Error: {status.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error displaying connection status: {e}")
        st.sidebar.error("🔴 Connection Status Unknown")

def display_navigation_info():
    """Display navigation information in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 Available Pages")
    st.sidebar.markdown("""
    - **🏠 Home** (Current page)
    - **🏆 Top 25 Performers** 
    - **⚡ Volatile Stocks**
    - **📅 Earnings Calendar**
    """)
    
    st.sidebar.info("""
    💡 **Navigation Tip**: 
    Streamlit automatically detects pages in the `pages/` directory. 
    Use the page selector above to navigate between different analyses.
    """)

def main():
    """Main application function"""
    logger.info("=== STOCK ANALYSIS APP STARTED ===")
    
    try:
        # Initialize database manager
        with st.spinner("Initializing database connection..."):
            db_manager = get_database_manager()
        
        # Display connection status
        display_connection_status(db_manager)
        
        # Display navigation info
        display_navigation_info()
        
        # Initialize home page component
        home_page = HomePage(db_manager)
        
        # Render the home page
        logger.info("Rendering home page...")
        home_page.render()
        logger.info("Home page rendered successfully")
        
        # Additional info about multipage structure
        st.markdown("---")
        st.info("""
        🎯 **Welcome to the Stock Analysis Dashboard!**
        
        This is a **Streamlit Multipage Application**. Here's how it works:
        
        - **🏠 Home**: Stock change tracker and main dashboard (this page)
        - **🏆 Top Performers**: Analysis of best performing stocks over 3 months
        - **⚡ Volatile Stocks**: Most volatile stocks with positive returns (4 weeks)  
        - **📅 Earnings Calendar**: Stocks with upcoming earnings (next 4 weeks)
        
        **Navigation**: Use the page selector in the sidebar to switch between analyses.
        Each page has its own URL and runs independently!
        """)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"An error occurred: {e}")
    
    logger.info("=== STOCK ANALYSIS APP COMPLETED ===")

# Run the main function
if __name__ == "__main__":
    main()
else:
    # This runs when the page is loaded by Streamlit's multipage system
    main() 