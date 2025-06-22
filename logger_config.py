import logging
import os
from datetime import datetime

# Global flag to prevent duplicate logger setup
_logger_initialized = False

def setup_logger():
    """Set up comprehensive logging configuration"""
    global _logger_initialized
    
    # Prevent duplicate initialization
    if _logger_initialized:
        return logging.getLogger('StockApp')
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log filename with timestamp
    log_filename = f"{log_dir}/stock_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    # Create logger
    logger = logging.getLogger('StockApp')
    logger.info("="*50)
    logger.info("STOCK APPLICATION STARTED")
    logger.info(f"Log file: {log_filename}")
    logger.info("="*50)
    
    _logger_initialized = True
    return logger

# Create global logger instance only once
logger = setup_logger() 