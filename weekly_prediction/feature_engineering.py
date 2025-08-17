import pandas as pd
import numpy as np
from datetime import date, timedelta


def _calculate_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calculate_bollinger_bands(series, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band


def _calculate_volume_indicators(df):
    """Calculate volume-based indicators."""
    # Volume moving average
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    
    # Volume ratio (current volume vs average)
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # Price-volume trend
    df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
    
    # On-balance volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    
    return df


def _calculate_risk_metrics(df):
    """Calculate risk and volatility metrics."""
    # Historical volatility (30-day)
    df['volatility_30d'] = df['close'].pct_change().rolling(window=30).std() * np.sqrt(252)
    
    # Maximum drawdown (30-day rolling)
    df['rolling_max'] = df['close'].rolling(window=30).max()
    df['drawdown_30d'] = (df['close'] - df['rolling_max']) / df['rolling_max']
    
    # Value at Risk (95% confidence, 30-day)
    returns = df['close'].pct_change()
    df['var_95_30d'] = returns.rolling(window=30).quantile(0.05)
    
    # Sharpe ratio (assuming risk-free rate of 2%)
    risk_free_rate = 0.02
    df['sharpe_ratio'] = (returns.rolling(window=30).mean() - risk_free_rate/252) / returns.rolling(window=30).std()
    
    return df


def _calculate_advanced_trend_indicators(df):
    """Calculate advanced trend and momentum indicators."""
    # Multiple timeframe trend analysis
    for period in [5, 10, 20, 50]:
        df[f'trend_{period}d'] = df['close'].rolling(window=period).apply(
            lambda x: 1 if x.iloc[-1] > x.iloc[0] else (-1 if x.iloc[-1] < x.iloc[0] else 0)
        )
    
    # Trend strength (ADX-like)
    df['trend_strength'] = abs(df['trend_20d'])
    
    # Price momentum across timeframes
    for period in [5, 10, 20, 30, 60]:
        df[f'momentum_{period}d'] = df['close'].pct_change(periods=period)
    
    # Golden/Death cross detection
    df['golden_cross'] = (df['ma_20'] > df['ma_50']) & (df['ma_20'].shift(1) <= df['ma_50'].shift(1))
    df['death_cross'] = (df['ma_20'] < df['ma_50']) & (df['ma_20'].shift(1) >= df['ma_50'].shift(1))
    
    return df


def _calculate_support_resistance_levels(df):
    """Calculate dynamic support and resistance levels."""
    # Dynamic support (20-day low)
    df['support_20d'] = df['close'].rolling(window=20).min()
    
    # Dynamic resistance (20-day high)
    df['resistance_20d'] = df['close'].rolling(window=20).max()
    
    # Distance from support/resistance
    df['distance_from_support'] = (df['close'] - df['support_20d']) / df['close']
    df['distance_from_resistance'] = (df['resistance_20d'] - df['close']) / df['close']
    
    # Breakout strength
    df['breakout_strength'] = df['close'] / df['resistance_20d'].shift(1)
    
    return df


def _calculate_post_earnings_rally(master_df, prices_df, today):
    """
    Calculates if a stock is in a rally phase after a post-earnings dip.
    This captures the bullish moment: good quarter → price up → dip → rally up again.
    """
    rally_tickers = []
    for index, row in master_df.iterrows():
        ticker = row['ticker']
        
        # Check if required columns exist
        if 'earnings_date' not in row or pd.isna(row['earnings_date']):
            continue
            
        if 'last_eps_surprise_pct' not in row or pd.isna(row['last_eps_surprise_pct']):
            continue
            
        earnings_date = row['earnings_date']
        surprise_pct = row['last_eps_surprise_pct']
        
        # Condition 1: Was there a recent positive earnings surprise?
        is_recent = (today - earnings_date.date()) < timedelta(days=90)  # Extended to 90 days
        is_positive_surprise = surprise_pct > 0.02 # Surprise must be > 2%
        
        if is_recent and is_positive_surprise:
            # Get prices since the earnings announcement
            post_earnings_prices = prices_df[
                (prices_df['ticker'] == ticker) & 
                (prices_df['date'] >= earnings_date)
            ].sort_values('date')
            
            if len(post_earnings_prices) >= 10:  # Need at least 10 days of data
                # Condition 2: Check for the pattern: up → dip → rally
                initial_price = post_earnings_prices.iloc[0]['close']
                max_price = post_earnings_prices['close'].max()
                dip_price = post_earnings_prices['close'].min()
                current_price = post_earnings_prices.iloc[-1]['close']
                
                # Pattern: Initial jump (at least 3%), then dip (at least 5% from peak), then rally (at least 3% from dip)
                initial_jump = (max_price - initial_price) / initial_price > 0.03
                significant_dip = (max_price - dip_price) / max_price > 0.05
                rally_from_dip = (current_price - dip_price) / dip_price > 0.03
                
                if initial_jump and significant_dip and rally_from_dip:
                    rally_tickers.append(ticker)

    # Create the new feature column
    master_df['post_earnings_dip_rally'] = master_df['ticker'].isin(rally_tickers)
    return master_df


def _calculate_momentum(df):
    """Calculate technical indicators for each stock."""
    df_out = pd.DataFrame()
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        ticker_df.sort_values('date', inplace=True)
        
        # Use 'close' column for price data
        price_col = 'close'

        # --- Standard Technical Indicators ---
        ticker_df['ma_20'] = ticker_df[price_col].rolling(window=20).mean()
        ticker_df['ma_50'] = ticker_df[price_col].rolling(window=50).mean()
        ticker_df['volatility_30d'] = ticker_df[price_col].pct_change().rolling(window=30).std()
        ticker_df['rsi_14d'] = _calculate_rsi(ticker_df[price_col], period=14)
        
        # --- MACD ---
        macd_line, signal_line, histogram = _calculate_macd(ticker_df[price_col])
        ticker_df['macd'] = macd_line
        ticker_df['macd_signal'] = signal_line
        ticker_df['macd_histogram'] = histogram
        
        # --- Bollinger Bands ---
        upper_band, middle_band, lower_band = _calculate_bollinger_bands(ticker_df[price_col])
        ticker_df['bb_upper'] = upper_band
        ticker_df['bb_middle'] = middle_band
        ticker_df['bb_lower'] = lower_band
        ticker_df['bb_position'] = (ticker_df[price_col] - lower_band) / (upper_band - lower_band)
        
        # --- Volume Indicators ---
        ticker_df = _calculate_volume_indicators(ticker_df)
        
        # --- Risk Metrics ---
        ticker_df = _calculate_risk_metrics(ticker_df)
        
        # --- Advanced Trend Indicators ---
        ticker_df = _calculate_advanced_trend_indicators(ticker_df)
        
        # --- Support/Resistance ---
        ticker_df = _calculate_support_resistance_levels(ticker_df)
        
        # --- Trend Slope (remains useful for momentum) ---
        window = 15
        x = np.arange(window)
        slopes = [np.polyfit(x, y, 1)[0] if len(y.dropna()) == window else np.nan 
                  for y in ticker_df[price_col].rolling(window=window)]
        ticker_df['trend_slope_15d'] = slopes

        # 30d trend slope
        window_30 = 30
        x_30 = np.arange(window_30)
        slopes_30 = [np.polyfit(x_30, y, 1)[0] if len(y.dropna()) == window_30 else np.nan 
                     for y in ticker_df[price_col].rolling(window=window_30)]
        ticker_df['trend_slope_30d'] = slopes_30

        # Up-day ratio over last 20 trading days
        daily_returns = ticker_df[price_col].pct_change()
        ticker_df['return_1d'] = daily_returns
        ticker_df['up_day_ratio_20d'] = daily_returns.rolling(window=20).apply(lambda r: np.mean(r > 0), raw=False)

        # 1. Define resistance as the highest price over the last 50 days, excluding today.
        # We use .shift(1) to ensure we're comparing today's price to PAST resistance.
        resistance_50d = ticker_df[price_col].rolling(window=50).max().shift(1)

        # 2. The 'broke_resistance' flag is True if the current price is above that past high.
        ticker_df['broke_resistance'] = ticker_df[price_col] > resistance_50d

        # Days since last broke_resistance
        days_since = []
        last_true_idx = None
        for idx, val in ticker_df['broke_resistance'].reset_index(drop=True).items():
            if bool(val):
                last_true_idx = idx
                days_since.append(0)
            else:
                days_since.append(idx - last_true_idx if last_true_idx is not None else np.nan)
        ticker_df['days_since_last_broke_resistance'] = days_since

        # --- ATR (Average True Range) 14d ---
        prev_close = ticker_df[price_col].shift(1)
        tr1 = ticker_df['high'] - ticker_df['low']
        tr2 = (ticker_df['high'] - prev_close).abs()
        tr3 = (ticker_df['low'] - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        ticker_df['atr_14d'] = true_range.rolling(window=14).mean()
        ticker_df['atr_pct'] = ticker_df['atr_14d'] / ticker_df[price_col]

        # --- Volume spike (z-score over 20d) ---
        vol_mean_20 = ticker_df['volume'].rolling(window=20).mean()
        vol_std_20 = ticker_df['volume'].rolling(window=20).std()
        ticker_df['volume_zscore_20'] = (ticker_df['volume'] - vol_mean_20) / vol_std_20
        ticker_df['volume_spike'] = ticker_df['volume_zscore_20'] > 2.0

        # --- Gap analysis ---
        gap_pct = (ticker_df['open'] - prev_close) / prev_close
        ticker_df['gap_pct'] = gap_pct
        ticker_df['gap_up_2pct'] = gap_pct > 0.02
        ticker_df['gap_down_2pct'] = gap_pct < -0.02

        df_out = pd.concat([df_out, ticker_df])
        
    # Cross-sectional relative strength vs median momentum on each date
    if not df_out.empty and 'momentum_20d' in df_out.columns:
        date_median = df_out.groupby('date')['momentum_20d'].median().rename('momentum_20d_median')
        df_out = df_out.merge(date_median, on='date', how='left')
        df_out['relative_strength_20d'] = df_out['momentum_20d'] - df_out['momentum_20d_median']
        df_out.drop(columns=['momentum_20d_median'], inplace=True)
    else:
        df_out['relative_strength_20d'] = 0

    # Volatility-adjusted momentum
    if 'momentum_20d' in df_out.columns and 'volatility_30d' in df_out.columns:
        df_out['vol_adj_momentum_20d'] = df_out['momentum_20d'] / (df_out['volatility_30d'] + 1e-6)
    else:
        df_out['vol_adj_momentum_20d'] = 0

    return df_out


def _calculate_earnings_surprise_enhanced(master_df):
    """
    Enhanced earnings surprise calculation using actual vs estimated data.
    """
    # Try to find actual vs estimated EPS data
    if 'reported_eps' in master_df.columns and 'estimate_eps' in master_df.columns:
        master_df['earnings_surprise_enhanced'] = (
            (master_df['reported_eps'] - master_df['estimate_eps']) / abs(master_df['estimate_eps'])
        )
    elif 'actual_eps' in master_df.columns and 'estimated_eps' in master_df.columns:
        master_df['earnings_surprise_enhanced'] = (
            (master_df['actual_eps'] - master_df['estimated_eps']) / abs(master_df['estimated_eps'])
        )
    else:
        # Fallback to existing calculation
        master_df['earnings_surprise_enhanced'] = master_df.get('last_eps_surprise_pct', 0)
    
    return master_df


def create_all_features(master_df, prices_df, today=date.today()):
    """
    Creates a rich set of features for each stock.

    Args:
        master_df (pd.DataFrame): DataFrame with latest static data per ticker.
        prices_df (pd.DataFrame): DataFrame with all historical price data.
        today (datetime.date): The current date for calculations.

    Returns:
        pd.DataFrame: A DataFrame with all features, one row per ticker.
    """
    # Convert today to datetime for timezone-aware comparison if necessary
    today_dt = pd.to_datetime(today)
    
    # --- Call the new feature creation function ---
    master_df_with_rally = _calculate_post_earnings_rally(master_df, prices_df, today)
    
    # --- Enhanced earnings surprise calculation ---
    master_df_with_rally = _calculate_earnings_surprise_enhanced(master_df_with_rally)
    
    # --- Calculate technical indicators ---
    prices_with_technicals = _calculate_momentum(prices_df)
    latest_technicals = prices_with_technicals.sort_values('date').drop_duplicates('ticker', keep='last')
    
    # --- Merge technicals into master frame ---
    featured_df = pd.merge(master_df_with_rally, latest_technicals, on='ticker', how='left', suffixes=('', '_price'))
    
    # Remove any duplicate tickers that might have been created during merging
    featured_df = featured_df.drop_duplicates(subset=['ticker'], keep='first')
    
    # 3. Engineer features from fundamental and estimate data
    
    # Earnings in next 3 weeks (21 days) - use next_earnings_date if available
    if 'next_earnings_date' in featured_df.columns:
        featured_df['earnings_in_3_weeks'] = (featured_df['next_earnings_date'] - pd.to_datetime(today)).dt.days.between(0, 21)
    else:
        featured_df['earnings_in_3_weeks'] = False
    
    # Enhanced earnings surprise
    featured_df['last_eps_surprise_pct'] = featured_df.get('earnings_surprise_enhanced', 0)
    
    # Profit Margin from last quarter - use available columns
    if 'earnings_m' in featured_df.columns and 'revenue_m' in featured_df.columns:
        featured_df['profit_margin'] = featured_df['earnings_m'] / featured_df['revenue_m']
    else:
        featured_df['profit_margin'] = 0

    # Check if last 2 quarters had positive surprises (a proxy for "doing well")
    # This is a more advanced feature, for now we use last quarter's surprise as a flag
    featured_df['positive_surprise_last_q'] = featured_df['last_eps_surprise_pct'] > 0

    # Continuous upward movement (using the trend slope we calculated)
    featured_df['is_upward_trending'] = featured_df['trend_slope_15d'] > 0
    # Trend persistence signal
    featured_df['trend_persistence_20d'] = featured_df.get('up_day_ratio_20d', 0) > 0.55
    
    # 4. Add missing features based on requirements
    # Days since last earnings if available
    if 'earnings_date' in featured_df.columns:
        try:
            featured_df['days_since_last_earnings'] = (pd.to_datetime(today) - pd.to_datetime(featured_df['earnings_date'])).dt.days
        except Exception:
            featured_df['days_since_last_earnings'] = 0
    else:
        featured_df['days_since_last_earnings'] = 0

    # Weighted confidence score as meta-feature (similar to selector)
    def _weighted_confidence(row):
        score = 0.0
        score += 20.0 if bool(row.get('broke_resistance', False)) else 0.0
        score += 15.0 if bool(row.get('post_earnings_dip_rally', False)) else 0.0
        score += 10.0 if bool(row.get('strong_momentum', False)) else 0.0
        score += 10.0 if bool(row.get('breakout_confirmed', False)) else 0.0
        score += 5.0 if bool(row.get('golden_cross', False)) else 0.0
        score += 15.0 if bool(row.get('earnings_in_3_weeks', False)) else 0.0
        score += 10.0 if bool(row.get('last_2q_positive_surprises', False)) else 0.0
        score += 10.0 if bool(row.get('bullish_momentum', False)) else 0.0
        score += 10.0 if bool(row.get('risk_adjusted_momentum', False)) else 0.0
        score += 5.0 if float(row.get('sharpe_ratio', 0)) > 0.5 else 0.0
        score += 5.0 if float(row.get('volatility_30d', 1)) < 0.3 else 0.0
        score += 5.0 if float(row.get('volume_ratio', 0)) > 1.2 else 0.0
        score += 5.0 if float(row.get('bb_position', 0)) > 0.7 else 0.0
        return min(score, 100.0)

    featured_df['weighted_confidence_score'] = featured_df.apply(_weighted_confidence, axis=1)
    
    # Feature 1: Earnings in next 3 weeks (already implemented above)
    
    # Feature 2: Last 2 quarters stock has done well (positive surprises)
    featured_df['last_2q_positive_surprises'] = featured_df['last_eps_surprise_pct'] > 0
    
    # Feature 3: Continuous upward movement in last 3 weeks (already implemented as is_upward_trending)
    
    # Feature 4: More complex upward trending that crosses historical resistance (already implemented as broke_resistance)
    
    # Feature 5: Post-earnings dip rally (already implemented as post_earnings_dip_rally)
    
    # Feature 6: Bullish momentum after good quarter results
    # This combines positive earnings surprise with upward price movement
    featured_df['bullish_momentum'] = (
        (featured_df['last_eps_surprise_pct'] > 0.05) &  # Good earnings surprise > 5%
        (featured_df['is_upward_trending']) &  # Upward trend
        (featured_df['broke_resistance'])  # Broke resistance level
    )
    
    # Add long-term growth rate if available
    if 'growth' in featured_df.columns:
        featured_df['long_term_growth_rate'] = featured_df['growth']
    else:
        featured_df['long_term_growth_rate'] = 0
    
    # Add next earnings date if available
    if 'next_earnings_date' in featured_df.columns:
        featured_df['next_earnings_date'] = featured_df['next_earnings_date']
    else:
        featured_df['next_earnings_date'] = pd.NaT
    
    # Enhanced momentum signals
    featured_df['strong_momentum'] = (
        (featured_df['rsi_14d'] > 50) &  # Above neutral RSI
        (featured_df['macd'] > featured_df['macd_signal']) &  # MACD above signal
        (featured_df['volume_ratio'] > 1.2)  # Above average volume
    )
    
    # Risk-adjusted momentum
    featured_df['risk_adjusted_momentum'] = (
        (featured_df['momentum_20d'] > 0) &  # Positive momentum
        (featured_df['volatility_30d'] < 0.3) &  # Low volatility
        (featured_df['sharpe_ratio'] > 0.5)  # Good risk-adjusted returns
    )

    # Momentum winner: persistent multi-week trend, supportive signals
    featured_df['momentum_winner'] = (
        (
            (featured_df.get('momentum_60d', 0) > 0.5) |
            (featured_df.get('momentum_30d', 0) > 0.2)
        ) &
        (featured_df['is_upward_trending']) &
        (featured_df['close'] > featured_df['ma_20']) &
        ((featured_df['macd'] > featured_df['macd_signal']) | (featured_df.get('golden_cross', False)))
    )
    
    # Breakout confirmation
    featured_df['breakout_confirmed'] = (
        (featured_df['broke_resistance']) &  # Broke resistance
        (featured_df['volume_ratio'] > 1.5) &  # High volume
        (featured_df['bb_position'] > 0.8)  # Near upper Bollinger Band
    )
    
    # Keep all columns and add missing ones with defaults
    all_possible_cols = [
        'ticker', 'close', 'volume', 'ma_20', 'ma_50', 'volatility_30d', 'rsi_14d',
        'trend_slope_15d', 'trend_slope_30d', 'broke_resistance', 'post_earnings_dip_rally', 'is_upward_trending', 
        'earnings_in_3_weeks', 'profit_margin', 'last_eps_surprise_pct', 'positive_surprise_last_q',
        'last_2q_positive_surprises', 'bullish_momentum', 'long_term_growth_rate', 'next_earnings_date',
        'macd', 'macd_signal', 'macd_histogram', 'bb_position', 'volume_ratio', 'pvt', 'obv',
        'drawdown_30d', 'var_95_30d', 'sharpe_ratio', 'trend_strength', 'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_30d', 'momentum_60d', 'up_day_ratio_20d',
        'golden_cross', 'death_cross', 'distance_from_support', 'distance_from_resistance', 'breakout_strength',
        'strong_momentum', 'risk_adjusted_momentum', 'breakout_confirmed', 'momentum_winner', 'trend_persistence_20d',
        'atr_14d', 'atr_pct', 'volume_zscore_20', 'volume_spike', 'gap_pct', 'gap_up_2pct', 'gap_down_2pct', 'relative_strength_20d',
        'vol_adj_momentum_20d', 'days_since_last_broke_resistance', 'return_1d', 'days_since_last_earnings', 'weighted_confidence_score'
    ]
    
    # Ensure all columns exist, fill missing with appropriate defaults
    for col in all_possible_cols:
        if col not in featured_df.columns:
            if col in ['earnings_in_3_weeks', 'positive_surprise_last_q', 'last_2q_positive_surprises', 
                      'bullish_momentum', 'strong_momentum', 'risk_adjusted_momentum', 'breakout_confirmed',
                      'golden_cross', 'death_cross']:
                featured_df[col] = False
            elif col == 'next_earnings_date':
                featured_df[col] = pd.NaT
            else:
                featured_df[col] = 0
    
    # Return all columns that exist in the DataFrame
    return featured_df.copy()