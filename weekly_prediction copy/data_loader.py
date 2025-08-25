import pandas as pd
from datetime import date
from typing import Optional

def load_all_data(file_date: Optional[date] = None):
    """
    Loads and merges all data files for a specific date.

    It intelligently merges the latest quarterly report for each stock.

    Args:
        file_date (datetime.date | None): The date suffix of the files to load. If None, uses undated files.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): 
            - A master DataFrame with one row per ticker containing the latest data.
            - A timeseries DataFrame with all historical prices.
    """
    suffix = file_date.strftime('%Y-%m-%d') if file_date is not None else None
    correct_date_column = 'date' 
    try:
        # --- Load all data sources ---
        if suffix:
            prices = pd.read_csv(f'data/stock_prices_{suffix}.csv', parse_dates=[correct_date_column])
            earnings = pd.read_csv(f'data/stock_earnings_{suffix}.csv', parse_dates=['earnings_date'])
            earn_est = pd.read_csv(f'data/earnings_estimates_{suffix}.csv', parse_dates=['earnings_date'])
            income = pd.read_csv(f'data/quarterly_income_{suffix}.csv', parse_dates=['report_date'])
            revenue = pd.read_csv(f'data/quarterly_revenue_{suffix}.csv', parse_dates=['report_date'])
            rev_est = pd.read_csv(f'data/revenue_estimates_{suffix}.csv', parse_dates=['next_earnings_date'])
            growth_est = pd.read_csv(f'data/growth_estimates_{suffix}.csv')
        else:
            prices = pd.read_csv('data/stock_prices.csv', parse_dates=[correct_date_column])
            earnings = pd.read_csv('data/stock_earnings.csv', parse_dates=['earnings_date'])
            earn_est = pd.read_csv('data/earnings_estimates.csv', parse_dates=['earnings_date'])
            income = pd.read_csv('data/quarterly_income.csv', parse_dates=['report_date'])
            revenue = pd.read_csv('data/quarterly_revenue.csv', parse_dates=['report_date'])
            rev_est = pd.read_csv('data/revenue_estimates.csv', parse_dates=['next_earnings_date'])
            growth_est = pd.read_csv('data/growth_estimates.csv')
    except FileNotFoundError as e:
        if suffix:
            print(f"Error loading files: {e}. Ensure all data files for {suffix} exist in the 'data/' folder.")
        else:
            print(f"Error loading files: {e}. Ensure all undated data files exist in the 'data/' folder.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Normalize dtypes for prices (handle non-numeric strings safely) ---
    try:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in prices.columns:
                prices[col] = pd.to_numeric(prices[col], errors='coerce')
    except Exception:
        pass

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

    # Override/derive next_earnings_date from stock_earnings.csv (earnings_date)
    try:
        # Use latest price date as the "today" anchor to avoid system clock drift
        try:
            today_ts = prices[correct_date_column].max().normalize()
        except Exception:
            today_ts = pd.Timestamp.today().normalize()
        # Choose the next upcoming earnings_date per ticker (>= today)
        next_earn = (
            earnings[earnings['earnings_date'] >= today_ts]
            .sort_values(['ticker', 'earnings_date'])
            .groupby('ticker', as_index=False)
            .first()[['ticker', 'earnings_date']]
            .rename(columns={'earnings_date': 'next_earnings_date'})
        )
        master_df = pd.merge(master_df, next_earn, on='ticker', how='left', suffixes=('', '_from_earnings'))
        # If an estimate also provided next_earnings_date, prefer the explicit upcoming date from earnings file
        if 'next_earnings_date_from_earnings' in master_df.columns:
            master_df['next_earnings_date'] = master_df['next_earnings_date_from_earnings'].combine_first(master_df.get('next_earnings_date'))
            master_df.drop(columns=['next_earnings_date_from_earnings'], inplace=True)
    except Exception:
        # If anything fails, keep existing estimate-based next_earnings_date
        pass
    
    print("All data files loaded and merged successfully.")
    
    return master_df, prices