# Stock Investment Data Analysis

This project downloads and analyzes stock market data using the Yahoo Finance API through the `yfinance` library.

## Files

- `stocks/stk_batch_download.py` - Original script for downloading real stock data
- `stocks/stk_batch_download_offline.py` - Sample data generator for demonstration/testing
- `requirements.txt` - Python dependencies
- `data/` - Folder containing all generated CSV files

## Setup

1. **Activate your conda environment:**
   ```bash
   conda activate uiapp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### For Real Data (requires proper network access):
```bash
python stocks/stk_batch_download.py
```

### For Sample Data (works offline):
```bash
python stocks/stk_batch_download_offline.py
```

## SSL Certificate Issues

If you encounter SSL certificate errors like:
```
curl: (60) SSL certificate problem: self signed certificate in certificate chain
```

This is common in corporate environments. Here are solutions:

### Solution 1: Configure Corporate Proxy
If you're behind a corporate firewall, configure your proxy settings:

```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### Solution 2: Update SSL Certificates
```bash
# Update certificates on macOS
sudo /usr/bin/security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain > /tmp/certs.pem
export SSL_CERT_FILE=/tmp/certs.pem
```

### Solution 3: Use Sample Data
For development and testing, use the offline version:
```bash
python stocks/stk_batch_download_offline.py
```

## Output Files

The scripts generate the following CSV files in the `data/` folder:
- `data/stock_prices_[date].csv` - Daily stock price data
- `data/stock_earnings_[date].csv` - Company earnings information
- `data/earnings_estimates_[date].csv` - Analyst earnings estimates
- `data/quarterly_revenue_[date].csv` - Quarterly revenue data
- `data/growth_estimates_[date].csv` - Growth projections
- `data/revenue_estimates_[date].csv` - Revenue forecasts
- `data/quarterly_income_[date].csv` - Quarterly income statements

## Stock Tickers

Currently configured for:
- AAPL (Apple)
- GOOGL (Google)
- MSFT (Microsoft)
- AI (C3.ai)
- OPFI (OppFi)

## Troubleshooting

1. **ModuleNotFoundError: No module named 'yfinance'**
   - Ensure you're in the correct conda environment
   - Run: `pip install yfinance`

2. **SSL Certificate Errors**
   - Use the offline version for testing
   - Contact your IT department for proxy configuration
   - Try running from a different network (home vs office)

3. **Pandas API Changes**
   - The script has been updated to handle pandas version changes
   - If you encounter API errors, update pandas: `pip install --upgrade pandas`

## Data Structure

Each CSV file contains structured financial data that can be used for:
- Investment analysis
- Portfolio management
- Financial modeling
- Data visualization

## Next Steps

Consider building a Streamlit dashboard to visualize this data:
```bash
pip install streamlit
streamlit run your_dashboard.py
```
