import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime, timedelta


def calculate_confidence_score(row):
    """
    Calculate a confidence score based on multiple factors.
    Higher score = higher confidence in the prediction.
    """
    score = 0
    
    # Technical indicators (40% weight)
    if row.get('broke_resistance', False):
        score += 20
    if row.get('post_earnings_dip_rally', False):
        score += 15
    if row.get('strong_momentum', False):
        score += 10
    if row.get('breakout_confirmed', False):
        score += 10
    if row.get('golden_cross', False):
        score += 5
    
    # Fundamental indicators (30% weight)
    if row.get('earnings_in_3_weeks', False):
        score += 15
    if row.get('last_2q_positive_surprises', False):
        score += 10
    if row.get('bullish_momentum', False):
        score += 10
    
    # Risk metrics (20% weight)
    if row.get('risk_adjusted_momentum', False):
        score += 10
    if row.get('sharpe_ratio', 0) > 0.5:
        score += 5
    if row.get('volatility_30d', 1) < 0.3:
        score += 5
    
    # Volume confirmation (10% weight)
    if row.get('volume_ratio', 0) > 1.2:
        score += 5
    if row.get('bb_position', 0) > 0.7:
        score += 5
    
    return min(score, 100)  # Cap at 100


def calculate_risk_score(row):
    """
    Calculate a risk score based on volatility and other risk metrics.
    Lower score = lower risk.
    """
    risk_score = 0
    
    # Volatility risk (40% weight)
    volatility = row.get('volatility_30d', 0.5)
    if volatility > 0.5:
        risk_score += 40
    elif volatility > 0.3:
        risk_score += 20
    elif volatility > 0.2:
        risk_score += 10
    
    # Drawdown risk (30% weight)
    drawdown = abs(row.get('drawdown_30d', 0))
    if drawdown > 0.2:
        risk_score += 30
    elif drawdown > 0.1:
        risk_score += 15
    elif drawdown > 0.05:
        risk_score += 5
    
    # Value at Risk (20% weight)
    var = abs(row.get('var_95_30d', 0))
    if var > 0.1:
        risk_score += 20
    elif var > 0.05:
        risk_score += 10
    elif var > 0.02:
        risk_score += 5
    
    # Market cap risk (10% weight) - smaller caps are riskier
    market_cap = row.get('market_cap', 10000000000)  # Default to 10B
    if market_cap < 1000000000:  # < 1B
        risk_score += 10
    elif market_cap < 5000000000:  # < 5B
        risk_score += 5
    
    return min(risk_score, 100)  # Cap at 100


def diversify_portfolio(stocks_df, max_per_sector=3, max_per_industry=2):
    """
    Diversify the portfolio by limiting exposure per sector/industry.
    """
    if 'sector' not in stocks_df.columns or 'industry' not in stocks_df.columns:
        return stocks_df
    
    diversified_stocks = []
    sector_counts = {}
    industry_counts = {}
    
    for _, stock in stocks_df.iterrows():
        sector = stock.get('sector', 'Unknown')
        industry = stock.get('industry', 'Unknown')
        
        # Check sector limit
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
            
        # Check industry limit
        if industry_counts.get(industry, 0) >= max_per_industry:
            continue
        
        # Add stock to diversified list
        diversified_stocks.append(stock)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        
        if len(diversified_stocks) >= 20:  # Stop at 20 stocks
            break
    
    return pd.DataFrame(diversified_stocks)


def get_top_stocks(n=20, min_confidence=30, max_risk=70, diversify=True):
    """
    Loads the latest features, makes predictions, and ranks stocks to find the top n.

    Args:
        n (int): The number of top stocks to return.
        min_confidence (int): Minimum confidence score (0-100).
        max_risk (int): Maximum risk score (0-100).
        diversify (bool): Whether to apply portfolio diversification.

    Returns:
        pd.DataFrame: A DataFrame of the top n ranked stocks, or an empty DataFrame if an error occurs.
    """
    # --- 1. Load Model and Features ---
    model_path = os.path.join('models', 'stock_predictor.joblib')
    feature_path = 'data/featured_stocks.csv'

    try:
        model = joblib.load(model_path)
        features_df = pd.read_csv(feature_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure 'run_pipeline.py' has been run successfully.")
        return pd.DataFrame()

    # Check for NaN values before cleaning
    print(f"Data shape before cleaning: {features_df.shape}")
    print(f"NaN counts per column:")
    for col in features_df.columns:
        nan_count = features_df[col].isna().sum()
        if nan_count > 0:
            print(f"  {col}: {nan_count} NaN values")
    
    # Fill NaN values instead of dropping rows
    features_df.fillna(0, inplace=True)
    
    print(f"Data shape after cleaning: {features_df.shape}")
    if features_df.empty:
        print("No data available for prediction after cleaning.")
        return pd.DataFrame()

    # --- 2. Make Predictions ---
    # Prepare the feature set for prediction (must match the columns used in training)
    training_cols = model.get_booster().feature_names
    print(f"Model expects these features: {training_cols}")
    print(f"Available columns: {list(features_df.columns)}")
    
    # Check which training columns are available
    available_cols = [col for col in training_cols if col in features_df.columns]
    missing_cols = [col for col in training_cols if col not in features_df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        print("Filling missing columns with 0...")
        for col in missing_cols:
            features_df[col] = 0
    
    X_predict = features_df[training_cols]
    
    predictions = model.predict(X_predict)
    features_df['predicted_change'] = predictions
    
    # --- 3. Calculate Predicted Return % ---
    # Use 'close' column instead of 'price'
    if 'close' in features_df.columns:
        price_col = 'close'
    elif 'price' in features_df.columns:
        price_col = 'price'
    else:
        print("Error: No price column found")
        return pd.DataFrame()
    
    features_df['predicted_return_pct'] = (features_df['predicted_change'] / features_df[price_col]) * 100
    
    # --- 4. Calculate Confidence and Risk Scores ---
    features_df['confidence_score'] = features_df.apply(calculate_confidence_score, axis=1)
    features_df['risk_score'] = features_df.apply(calculate_risk_score, axis=1)
    
    # --- 5. Apply Filters ---
    # Filter by confidence and risk
    filtered_stocks = features_df[
        (features_df['confidence_score'] >= min_confidence) &
        (features_df['risk_score'] <= max_risk) &
        (features_df['predicted_change'] > 0)  # Only positive predictions
    ].copy()
    
    print(f"Stocks after confidence/risk filtering: {len(filtered_stocks)}")
    
    if filtered_stocks.empty:
        print("No stocks passed the confidence/risk filters. Relaxing constraints...")
        # Relax constraints
        filtered_stocks = features_df[
            (features_df['predicted_change'] > 0)  # Only positive predictions
        ].copy()
        print(f"Stocks with positive predictions: {len(filtered_stocks)}")
    
    if filtered_stocks.empty:
        print("No stocks with positive predictions. Showing all stocks...")
        filtered_stocks = features_df.copy()
    
    # --- 6. Calculate Composite Score ---
    # Combine predicted return, confidence, and risk into a single score
    filtered_stocks['composite_score'] = (
        filtered_stocks['predicted_return_pct'] * 0.5 +  # 50% weight to predicted return
        filtered_stocks['confidence_score'] * 0.3 +      # 30% weight to confidence
        (100 - filtered_stocks['risk_score']) * 0.2      # 20% weight to risk (inverted)
    )
    
    # --- 7. Apply Diversification ---
    if diversify and len(filtered_stocks) > n:
        filtered_stocks = diversify_portfolio(filtered_stocks)
    
    # --- 8. Rank and Select Top N ---
    # Sort by composite score in descending order
    ranked_stocks = filtered_stocks.sort_values('composite_score', ascending=False)
    
    # Remove duplicate tickers (keep the first occurrence with highest score)
    ranked_stocks = ranked_stocks.drop_duplicates(subset=['ticker'], keep='first')
    
    # Select the top N stocks
    top_n_stocks = ranked_stocks.head(n)
    
    # Debug: Check for duplicates
    if len(top_n_stocks) != len(top_n_stocks['ticker'].unique()):
        print(f"Warning: Found {len(top_n_stocks) - len(top_n_stocks['ticker'].unique())} duplicate tickers in final selection")
        print(f"Duplicate tickers: {top_n_stocks[top_n_stocks['ticker'].duplicated()]['ticker'].tolist()}")
    
    # --- 9. Prepare Display Columns ---
    display_cols = [
        'ticker', 
        'close',
        'predicted_change', 
        'predicted_return_pct',
        'confidence_score',
        'risk_score',
        'composite_score',
        'broke_resistance', 
        'post_earnings_dip_rally',
        'strong_momentum',
        'breakout_confirmed',
        'earnings_in_3_weeks',
        'trend_slope_15d',
        'rsi_14d',
        'volume_ratio',
        'volatility_30d',
        'momentum_5d',
        'momentum_10d',
        'momentum_20d'
    ]
    
    # Add sector/industry if available
    if 'sector' in top_n_stocks.columns:
        display_cols.append('sector')
    if 'industry' in top_n_stocks.columns:
        display_cols.append('industry')
    
    # Ensure all display columns exist before trying to select them
    final_cols = [col for col in display_cols if col in top_n_stocks.columns]
    
    print(f"Found and ranked {len(top_n_stocks)} top stocks.")
    print(f"Average confidence score: {top_n_stocks['confidence_score'].mean():.1f}")
    print(f"Average risk score: {top_n_stocks['risk_score'].mean():.1f}")
    print(f"Average predicted return: {top_n_stocks['predicted_return_pct'].mean():.2f}%")
    
    return top_n_stocks[final_cols]


def get_stock_analysis(ticker):
    """
    Get detailed analysis for a specific stock.
    """
    feature_path = 'data/featured_stocks.csv'
    
    try:
        features_df = pd.read_csv(feature_path)
        stock_data = features_df[features_df['ticker'] == ticker].iloc[0]
        
        analysis = {
            'ticker': ticker,
            'current_price': stock_data.get('close', 0),
            'technical_signals': {
                'broke_resistance': stock_data.get('broke_resistance', False),
                'post_earnings_dip_rally': stock_data.get('post_earnings_dip_rally', False),
                'strong_momentum': stock_data.get('strong_momentum', False),
                'breakout_confirmed': stock_data.get('breakout_confirmed', False),
                'golden_cross': stock_data.get('golden_cross', False),
                'rsi_14d': stock_data.get('rsi_14d', 0),
                'trend_slope_15d': stock_data.get('trend_slope_15d', 0),
                'volume_ratio': stock_data.get('volume_ratio', 0)
            },
            'fundamental_signals': {
                'earnings_in_3_weeks': stock_data.get('earnings_in_3_weeks', False),
                'last_2q_positive_surprises': stock_data.get('last_2q_positive_surprises', False),
                'bullish_momentum': stock_data.get('bullish_momentum', False),
                'profit_margin': stock_data.get('profit_margin', 0),
                'long_term_growth_rate': stock_data.get('long_term_growth_rate', 0)
            },
            'risk_metrics': {
                'volatility_30d': stock_data.get('volatility_30d', 0),
                'drawdown_30d': stock_data.get('drawdown_30d', 0),
                'var_95_30d': stock_data.get('var_95_30d', 0),
                'sharpe_ratio': stock_data.get('sharpe_ratio', 0)
            }
        }
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing stock {ticker}: {e}")
        return None


# You can add this to test the function directly
if __name__ == '__main__':
    print("--- Testing Enhanced Stock Selector ---")
    top_20 = get_top_stocks(n=20, min_confidence=30, max_risk=70, diversify=True)
    if not top_20.empty:
        print("Top 20 Recommended Stocks:")
        print(top_20.to_string())
        
        # Test individual stock analysis
        if len(top_20) > 0:
            first_stock = top_20.iloc[0]['ticker']
            analysis = get_stock_analysis(first_stock)
            if analysis:
                print(f"\nDetailed Analysis for {first_stock}:")
                print(f"Technical Signals: {analysis['technical_signals']}")
                print(f"Fundamental Signals: {analysis['fundamental_signals']}")
                print(f"Risk Metrics: {analysis['risk_metrics']}")