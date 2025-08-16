import pandas as pd
from datetime import date

def load_all_data(file_date=date(2024, 6, 1)):
    """
    Loads and merges all data files for a specific date.

    It intelligently merges the latest quarterly report for each stock.

    Args:
        file_date (datetime.date): The date suffix of the files to load.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): 
            - A master DataFrame with one row per ticker containing the latest data.
            - A timeseries DataFrame with all historical prices.
    """
    suffix = file_date.strftime('%Y-%m-%d')
    correct_date_column = 'date' 
    try:
        # --- Load all data sources ---
        prices = pd.read_csv(f'data/stock_prices_{suffix}.csv', parse_dates=[correct_date_column])
        earnings = pd.read_csv(f'data/stock_earnings_{suffix}.csv', parse_dates=['earnings_date'])
        earn_est = pd.read_csv(f'data/earnings_estimates_{suffix}.csv', parse_dates=['earnings_date'])
        income = pd.read_csv(f'data/quarterly_income_{suffix}.csv', parse_dates=['report_date'])
        revenue = pd.read_csv(f'data/quarterly_revenue_{suffix}.csv', parse_dates=['report_date'])
        rev_est = pd.read_csv(f'data/revenue_estimates_{suffix}.csv', parse_dates=['next_earnings_date'])
        growth_est = pd.read_csv(f'data/growth_estimates_{suffix}.csv')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Ensure all data files for {suffix} exist in the 'data/' folder.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Combine latest quarterly fundamental data ---
    # Merge income and revenue on the same report date
    fundamentals = pd.merge(income, revenue, on=['ticker', 'report_date'], how='inner')
    
    # Also merge historical earnings data for surprise calculation
    fundamentals = pd.merge(fundamentals, earnings, left_on=['ticker', 'report_date'], right_on=['ticker', 'earnings_date'], how='left')

    # Get the MOST RECENT full report for each ticker
    fundamentals = fundamentals.sort_values('report_date', ascending=False).drop_duplicates(subset='ticker', keep='first')

    # --- Create a master DataFrame ---
    master_df = pd.DataFrame(prices['ticker'].unique(), columns=['ticker'])
    
    # Merge the latest fundamental data
    master_df = pd.merge(master_df, fundamentals, on='ticker', how='left')
    
    # Merge future estimates
    master_df = pd.merge(master_df, earn_est, on='ticker', how='left')
    master_df = pd.merge(master_df, rev_est, on='ticker', how='left')
    master_df = pd.merge(master_df, growth_est, on='ticker', how='left')
    
    print("All data files loaded and merged successfully.")
    
    return master_df, prices