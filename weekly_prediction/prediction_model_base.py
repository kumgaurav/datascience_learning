import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from data_loader import load_all_data
from feature_engineering import _calculate_momentum


class BaseModelTrainer:
    """
    Shared training logic for our stock predictors.
    """

    def __init__(self, features_path: str, model_path: str) -> None:
        self.features_path = features_path
        self.model_path = model_path

    def _build_training_frame(self) -> pd.DataFrame:
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"features file not found: {self.features_path}")

        # Load the point-in-time features (for schema reference) and full data to reconstruct target
        _ = pd.read_csv(self.features_path)

        master_df, prices_df = load_all_data()
        if master_df.empty or prices_df.empty:
            raise RuntimeError("Data loading failed.")

        historical_technicals = _calculate_momentum(prices_df)
        historical_technicals['date'] = pd.to_datetime(historical_technicals['date'], errors='coerce')
        historical_technicals.dropna(subset=['date'], inplace=True)
        historical_technicals.sort_values('date', inplace=True)

        # Choose available fundamentals to merge
        available_cols = ['ticker']
        for col in [
            'earnings_date', 'earnings_date_x', 'earnings_date_y',
            'profit_margin', 'last_eps_surprise_pct', 'long_term_growth_rate',
        ]:
            if col in master_df.columns:
                available_cols.append(col)
        fundamentals = master_df[available_cols].copy()

        # If we have a usable date column, do time-aware merge; otherwise, merge on ticker only
        earnings_col = None
        for col in ['earnings_date', 'earnings_date_x', 'earnings_date_y']:
            if col in fundamentals.columns:
                earnings_col = col
                break

        if earnings_col is not None:
            fundamentals = fundamentals.rename(columns={earnings_col: 'date'})
            fundamentals['date'] = pd.to_datetime(fundamentals['date'], errors='coerce')
            fundamentals = fundamentals.dropna(subset=['date']).sort_values('date')
            merged = pd.merge_asof(
                historical_technicals,
                fundamentals,
                on='date',
                by='ticker',
                direction='backward',
            )
        else:
            # No reliable date field in fundamentals; avoid merging to prevent leakage
            merged = historical_technicals

        return merged

    def train(self) -> str:
        full_feature_df = self._build_training_frame()

        price_col = None
        for col in ['price', 'close', 'Close']:
            if col in full_feature_df.columns:
                price_col = col
                break
        if price_col is None:
            raise RuntimeError("No price column found in the merged dataset.")

        # Target: 5-day percentage change
        full_feature_df['price_target_5d'] = full_feature_df.groupby('ticker')[price_col].shift(-5)
        full_feature_df['price_change_5d_pct'] = (
            (full_feature_df['price_target_5d'] - full_feature_df[price_col]) / full_feature_df[price_col] * 100
        )
        # Only require target and price to be present; do not drop rows due to unrelated NaNs
        full_feature_df = full_feature_df.dropna(subset=['price_change_5d_pct', price_col])
        if full_feature_df.empty:
            raise RuntimeError("Final dataset is empty after target creation.")

        exclude_cols = [
            'ticker', 'price_target_5d', 'price_change_5d_pct', 'date',
            price_col, 'high', 'low', 'open', 'volume',
        ]
        feature_cols = [
            c for c in full_feature_df.columns
            if c not in exclude_cols and full_feature_df[c].dtype in ['int64', 'float64', 'bool']
        ]

        # Build model matrix and handle missing/inf values gracefully
        X = full_feature_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = full_feature_df['price_change_5d_pct']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=600, learning_rate=0.05,
            max_depth=5, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        joblib.dump(model, self.model_path)

        return f"✅ Trained {os.path.basename(self.model_path)} | MAE: {mae:.2f}, R2: {r2:.2f}"


