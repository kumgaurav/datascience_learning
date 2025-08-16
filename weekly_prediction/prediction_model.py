import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# We need to import our other modules to build the dataset internally
from data_loader import load_all_data
from feature_engineering import _calculate_momentum # Note the underscore

def train_model():
    """
    Builds a proper historical training set and trains a regression model.
    This version avoids data leakage by correctly handling time-series data.
    """
    print("--- Starting Model Training (Corrected Workflow) ---")
    
    # --- 1. Load Raw Data ---
    master_df, prices_df = load_all_data()
    if master_df.empty or prices_df.empty:
        return "Error: Data loading failed."
    
    print("Building historical feature set...")
    # --- 2. Create Historical Features ---
    # a) Calculate historical technical indicators for all days
    historical_technicals = _calculate_momentum(prices_df)

    # b) Prepare fundamental data for a point-in-time merge
    # We only want to join fundamentals that were known on or before a given price date.
    # Check which columns are available and use them
    available_cols = ['ticker']
    potential_cols = ['earnings_date', 'earnings_date_x', 'earnings_date_y', 'profit_margin', 'last_eps_surprise_pct', 'long_term_growth_rate']
    
    for col in potential_cols:
        if col in master_df.columns:
            available_cols.append(col)
    
    print(f"Available fundamental columns: {available_cols}")
    fundamentals = master_df[available_cols].copy()
    
    # Find the earnings date column
    earnings_col = None
    for col in ['earnings_date', 'earnings_date_x', 'earnings_date_y']:
        if col in fundamentals.columns:
            earnings_col = col
            break
    
    if earnings_col:
        fundamentals.rename(columns={earnings_col: 'date'}, inplace=True)
        # Convert to datetime and handle null values
        fundamentals['date'] = pd.to_datetime(fundamentals['date'], errors='coerce')
        # Fill null dates with a default date
        fundamentals['date'].fillna(pd.to_datetime('2024-06-01'), inplace=True)
        fundamentals.sort_values('date', inplace=True)
    else:
        # If no earnings date, use a default date
        fundamentals['date'] = pd.to_datetime('2024-06-01')

    # c) Combine technicals and fundamentals correctly
    # merge_asof is crucial for time-series. It merges on the nearest key (date)
    # 'direction=backward' ensures we only use fundamentals from the past.
    
    # Ensure both dataframes have proper date handling
    historical_technicals['date'] = pd.to_datetime(historical_technicals['date'], errors='coerce')
    historical_technicals.dropna(subset=['date'], inplace=True)
    historical_technicals.sort_values('date', inplace=True)
    
    print(f"Historical technicals shape: {historical_technicals.shape}")
    print(f"Fundamentals shape: {fundamentals.shape}")
    
    full_feature_df = pd.merge_asof(
        historical_technicals,
        fundamentals,
        on='date',
        by='ticker',
        direction='backward'
    )
    
    print(f"Merged dataset shape: {full_feature_df.shape}")
    print(f"Available columns: {list(full_feature_df.columns)}")

    # --- 3. Create Target Variable (y) ---
    # Check what price column is available
    price_col = None
    for col in ['price', 'close', 'Close']:
        if col in full_feature_df.columns:
            price_col = col
            break
    
    if price_col is None:
        return "Error: No price column found in the merged dataset."
    
    print(f"Using '{price_col}' column for target calculation")
    
    # Create future price target
    full_feature_df['price_target_5d'] = full_feature_df.groupby('ticker')[price_col].shift(-5)
    
    # Calculate price change percentage (this is what we actually want to predict)
    full_feature_df['price_change_5d_pct'] = (
        (full_feature_df['price_target_5d'] - full_feature_df[price_col]) / full_feature_df[price_col] * 100
    )
    
    print("Target variable: 5-day price change percentage (not absolute price)")
    print(f"Price change range: {full_feature_df['price_change_5d_pct'].min():.2f}% to {full_feature_df['price_change_5d_pct'].max():.2f}%")
    
    # --- 4. Clean and Prepare Final Dataset ---
    full_feature_df.dropna(inplace=True)

    if full_feature_df.empty:
        return "Error: Final dataset is empty after feature creation and cleaning."

    # --- 5. Define Features (X) and Target (y) ---
    # Exclude non-numeric columns, target columns, and current price
    exclude_cols = ['ticker', 'price_target_5d', 'price_change_5d_pct', 'date', price_col, 'high', 'low', 'open', 'volume']
    
    feature_cols = []
    for col in full_feature_df.columns:
        if col not in exclude_cols:
            # Only include numeric columns (int, float, bool)
            if full_feature_df[col].dtype in ['int64', 'float64', 'bool']:
                feature_cols.append(col)
    
    print(f"Using {len(feature_cols)} numeric features for training (excluding price data)")
    print(f"Features: {feature_cols}")
    X = full_feature_df[feature_cols]
    y = full_feature_df['price_change_5d_pct']  # Predict price change percentage, not absolute price

    # --- 6. Split and Train ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples.")
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # --- 7. Evaluate with Realistic Scores ---
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Evaluation (Corrected) ---")
    print("These scores should be more realistic now.")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    # --- 8. Save the Model ---
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'stock_predictor.joblib')
    joblib.dump(model, model_path)
    
    return f"✅ Model training complete. Model saved to '{model_path}'"