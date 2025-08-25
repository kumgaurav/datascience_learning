import yfinance as yf
import pandas as pd
import ssl
import urllib3
import os
import requests

# Disable SSL warnings and configure for corporate environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Set environment variables to handle SSL issues
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Configure curl_cffi to skip SSL verification
os.environ['CURL_CFG_SSL_VERIFY'] = '0'
os.environ['CURL_CFG_SSL_VERIFYHOST'] = '0'

# Note: yfinance handles its own session management with curl_cffi
# We don't need to create a custom session
# Create a mapping for period codes to meaningful names
period_mapping = {
    "0q": "Current Quarter",
    "+1q": "Next Quarter",
    "0y": "Current Year",
    "+1y": "Next Year",
    "LTG": "Long-Term Growth"
}

tickers = ["AAPL", "GOOGL", "MSFT","AI", "OPFI","NBIS", "ALAB", "MP", "CRWV", "AMZN", "POWL","PLTR","AVGO","CLS","NVDA","SOUN","LMND","QUBT","ALAB"]
start_date = "2024-06-01"
end_date = "2025-08-08"

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# 📌 Download stock price data
try:
    # Download stock data - let yfinance handle its own session management
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, ignore_tz=True)
    
    if data.empty:
        print("⚠️ No stock price data downloaded. Check your internet connection and ticker symbols.")
    else:
        # ✅ Convert MultiIndex columns to normal DataFrame with tickers as a column
        data = data.stack(level=1).reset_index()
        data.rename(columns={"level_1": "ticker"}, inplace=True)
        
        # ✅ Convert all column names to lowercase
        data.columns = data.columns.str.lower()

        # ✅ Save stock prices
        price_filename = f"data/stock_prices_{start_date}.csv"
        data.to_csv(price_filename, index=False)
        print(f"📁 Stock prices saved as {price_filename}")
except Exception as e:
    print(f"⚠️ Error downloading stock prices: {e}")
    print("Continuing with other data downloads...")

# 📌 Initialize lists for storing financial data
earnings_list = []
earnings_estimate_list = []
earnings_history_list = []
quarterly_revenue_list = []
growth_estimates_list = []
revenue_estimates_list = []
quarterly_income_stmt_list = []

for ticker in tickers:
    stock = yf.Ticker(ticker)
    info = stock.info  # Get stock information

    # ✅ Extract financial summary (Revenue, Market Cap, etc.)
    earnings_list.append({
        "ticker": ticker,
        "short_name": info.get("shortName", ""),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "market_cap": info.get("marketCap", ""),
        "revenue": info.get("totalRevenue", ""),
        "gross_profit": info.get("grossProfits", ""),
        "ebitda": info.get("ebitda", ""),
        "earnings_date": info.get("earningsDate", ""),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh", ""),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow", ""),
        "dividend_yield": info.get("dividendYield", ""),
        "pe_ratio": info.get("trailingPE", ""),
        "forward_pe": info.get("forwardPE", ""),
    })

    # ✅ Extract quarterly revenue and gross profit
    try:
        quarterly_data = stock.quarterly_financials
        print("\n📌 Quarterly Schema:")
        if quarterly_data is not None and not quarterly_data.empty:
            # For each quarter (column) in the quarterly data
            for quarter in quarterly_data.columns:
                revenue = quarterly_data.loc["Total Revenue", quarter] if "Total Revenue" in quarterly_data.index else None # Keep in millions
                if pd.isna(revenue):  # pd.isna() checks for both None and NaN
                    # print(f"⚠️ Skipping {ticker} for {quarter_date}: Revenue is empty")
                    continue
                gross_profit = quarterly_data.loc["Gross Profit", quarter] if "Gross Profit" in quarterly_data.index else None
                # Create a dictionary with only the fields you want
                quarter_dict = {
                    "ticker": ticker,
                    "report_date": quarter.strftime("%Y-%m-%d"),
                    "revenue": revenue,
                    "gross_profit": gross_profit,
                }
                # Add this quarter's data to our list
                quarterly_revenue_list.append(quarter_dict)
    except Exception as e:
        print(f"⚠️ Failed to get quarterly revenue for {ticker}: {e}")

    # ✅ Extract earnings estimates
    try:
        earnings_estimates = stock.earnings_estimate
        # Check if we have data
        if earnings_estimates is not None and not earnings_estimates.empty:
            # Process each row in earnings estimates
            for index, row in earnings_estimates.iterrows():
                # Convert period code to meaningful name
                period_label = period_mapping.get(index, index)  # Use original if not in mapping
                # Create dictionary based on the actual structure you're seeing
                estimate_dict = {
                    "ticker": ticker,
                    "period": period_label,
                    "earnings_date": start_date,  # New field for when estimate was retrieved
                    "average_estimate": row.get('avg', None),
                    "low_estimate": row.get('low', None),
                    "high_estimate": row.get('high', None),
                    "number_of_analysts": row.get('numberOfAnalysts', None),
                    "year_ago_eps": row.get('yearAgoEps', None),
                    "growth": row.get('growth', None)
                }
                # Add to our list
                earnings_estimate_list.append(estimate_dict)
        else:
            print(f"No earnings estimate data available for {ticker}")
    except Exception as e:
        print(f"⚠️ Failed to get earnings estimates for {ticker}: {e}")

    # ✅ Extract earnings history
    try:
        earnings_history = stock.earnings_history  # Ensure this is a DataFrame
        for quarter, row in earnings_history.iterrows():
            earnings_history_list.append({
                "ticker": ticker,
                "earnings_date": quarter,  # Using index as the date
                "reported_eps": row["epsActual"],
                "estimate_eps": row["epsEstimate"],
                "surprise_percentage": row["surprisePercent"],
            })
    except AttributeError:
        print(f"⚠️ {ticker} does not have earnings history.")
    except Exception as e:
        print(f"⚠️ Failed to get earnings history for {ticker}: {e}")

    # ✅ Extract growth estimates
    try:
        growth_estimates = stock.growth_estimates
        if growth_estimates is not None and not growth_estimates.empty:
            for index, row in growth_estimates.iterrows():
                stock_growth = row.get("stockTrend", None)
                index_growth = row.get("indexTrend", None)
                # Map the period code to a readable name, default to original if not in mapping
                period_label = period_mapping.get(index, index)
                if stock_growth is not None or index_growth is not None:
                    growth_estimates_list.append({
                        "estimate_fetch_date": start_date,  # Added fetch date
                        "ticker": ticker,
                        "period": period_label,
                        "stock_growth": stock_growth,
                        "index_growth": index_growth,
                    })
            print(f"✅ Processed {ticker}")
        else:
            print(f"⚠️ {ticker} does not have growth estimates.")
    except Exception as e:
        print(f"⚠️ Failed to get growth estimates for {ticker}: {e}")

    # ✅ Extract revenue estimates
    try:
        revenue_estimates = stock.revenue_estimate
        # Check if we have data
        if revenue_estimates is not None and not revenue_estimates.empty:
            # Process each row in revenue forecasts
            for index, row in revenue_estimates.iterrows():
                period_label = period_mapping.get(index, index)  # Use original if not in mapping

                # Create dictionary with revenue forecast data including fetch date
                revenue_estimate_dict = {
                    "ticker": ticker,
                    "period": period_label,
                    "next_earnings_date": start_date,  # New field for when estimate was retrieved
                    "average_forecast": row.get('avg', None),
                    "low_forecast": row.get('low', None),
                    "high_forecast": row.get('high', None),
                    "number_of_analysts": row.get('numberOfAnalysts', None),
                    "year_ago_revenue": row.get('yearAgoRevenue', None),
                    "growth": row.get('growth', None)
                }

                # Add to our list
                revenue_estimates_list.append(revenue_estimate_dict)
        else:
            print(f"No revenue forecast data available for {ticker}")
    except Exception as e:
        print(f"⚠️ Failed to get revenue estimates for {ticker}: {e}")

    # ✅ Extract quarterly income statement
    try:
        # Fetch quarterly income statement
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is not None and not income_stmt.empty:
            # Iterate over all available quarters (columns)
            for quarter_date in income_stmt.columns:
                try:
                    revenue = income_stmt.loc["Total Revenue", quarter_date]  # Keep in millions
                    # Skip if revenue is None or NaN
                    if pd.isna(revenue):  # pd.isna() checks for both None and NaN
                        #print(f"⚠️ Skipping {ticker} for {quarter_date}: Revenue is empty")
                        continue
                    revenue_millions = revenue / 1e6
                    earnings = income_stmt.loc["Net Income", quarter_date]  # Keep in millions
                    earnings_millions = earnings / 1e6  # Convert to millions
                    # profits = income_stmt.loc["Gross Profit", quarter_date]  # Keep in millions
                    quarter_label = quarter_date.strftime('%Y-%m-%d')  # Quarter end date

                    quarterly_income_stmt_list.append({
                        "ticker": ticker,
                        "report_date": quarter_label,
                        "revenue_m": revenue_millions,
                        "earnings_m": earnings_millions,
                        # "gross_profit_m": earnings,
                    })
                except KeyError as e:
                    print(f"⚠️ Missing data for {ticker} on {quarter_date}: {e}")
            print(f"✅ Processed {ticker}")
        else:
            print(f"⚠️ {ticker} does not have financial data.")
    except Exception as e:
        print(f"⚠️ Failed to get financials for {ticker}: {e}")

# ✅ Convert extracted data to DataFrames
earnings_df = pd.DataFrame(earnings_list)
earnings_estimate_df = pd.DataFrame(earnings_estimate_list)
earnings_history_df = pd.DataFrame(earnings_history_list)
quarterly_revenue_df = pd.DataFrame(quarterly_revenue_list)
growth_estimates_df = pd.DataFrame(growth_estimates_list)
revenue_estimates_df = pd.DataFrame(revenue_estimates_list)
quarterly_income_df = pd.DataFrame(quarterly_income_stmt_list)
quarterly_income_df["revenue_m"] = quarterly_income_df["revenue_m"].round(2)
quarterly_income_df["earnings_m"] = quarterly_income_df["earnings_m"].round(2)

# ✅ Save data to CSV files
earnings_filename = f"data/stock_earnings_{start_date}.csv"
earnings_estimate_filename = f"data/earnings_estimates_{start_date}.csv"
earnings_history_filename = f"data/earnings_history_{start_date}.csv"
quarterly_revenue_filename = f"data/quarterly_revenue_{start_date}.csv"
growth_estimates_filename = f"data/growth_estimates_{start_date}.csv"
revenue_estimates_filename = f"data/revenue_estimates_{start_date}.csv"
quarterly_income_filename = f"data/quarterly_income_{start_date}.csv"

earnings_df.to_csv(earnings_filename, index=False)
earnings_estimate_df.to_csv(earnings_estimate_filename, index=False)
earnings_history_df.to_csv(earnings_history_filename, index=False)
quarterly_revenue_df.to_csv(quarterly_revenue_filename, index=False)
growth_estimates_df.to_csv(growth_estimates_filename, index=False)
revenue_estimates_df.to_csv(revenue_estimates_filename, index=False)
quarterly_income_df.to_csv(quarterly_income_filename, index=False)

print(f"📁 Earnings summary saved as {earnings_filename}")
print(f"📁 Earnings estimates saved as {earnings_estimate_filename}")
print(f"📁 Earnings history saved as {earnings_history_filename}")
print(f"📁 Quarterly revenue saved as {quarterly_revenue_filename}")
print(f"📁 Growth estimates saved as {growth_estimates_filename}")
print(f"📁 Revenue estimates saved as {revenue_estimates_filename}")
print(f"📁 Quarterly income saved as {quarterly_income_filename}")