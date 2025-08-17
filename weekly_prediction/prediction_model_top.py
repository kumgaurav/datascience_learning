from prediction_model_base import BaseModelTrainer
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Dict, Any


class TopModelTrainer(BaseModelTrainer):
    def __init__(self) -> None:
        super().__init__(
            features_path='data/featured_stocks_top.csv',
            model_path='models/stock_predictor_top.joblib',
        )

    def train(self) -> str:
        # Build base frame with technicals + fundamentals
        df = self._build_training_frame()

        # Identify price column
        price_col = None
        for col in ['price', 'close', 'Close']:
            if col in df.columns:
                price_col = col
                break
        if price_col is None:
            raise RuntimeError("No price column found in the merged dataset.")

        # Create 5-day target as percentage change
        df['price_target_5d'] = df.groupby('ticker')[price_col].shift(-5)
        df['price_change_5d_pct'] = (
            (df['price_target_5d'] - df[price_col]) / df[price_col] * 100
        )

        # Ensure momentum features 5d/10d/20d exist
        for p in [5, 10, 20]:
            col = f'momentum_{p}d'
            if col not in df.columns:
                df[col] = df.groupby('ticker')[price_col].pct_change(periods=p)

        # Keep only rows with valid target and price
        df = df.dropna(subset=['price_change_5d_pct', price_col, 'date'])
        if df.empty:
            raise RuntimeError("Final dataset is empty after target creation.")

        # Log raw target range
        raw_min = float(df['price_change_5d_pct'].min())
        raw_max = float(df['price_change_5d_pct'].max())
        print(f"Target (5d % change) raw range: {raw_min:.2f}% to {raw_max:.2f}%")

        # Clip extreme target outliers to stabilize training
        lower, upper = np.percentile(df['price_change_5d_pct'], [1, 99])
        df['price_change_5d_pct'] = df['price_change_5d_pct'].clip(lower, upper)
        clip_min = float(df['price_change_5d_pct'].min())
        clip_max = float(df['price_change_5d_pct'].max())
        print(f"Target after clipping to [1,99] pct: {clip_min:.2f}% to {clip_max:.2f}%")

        # Feature set: numeric columns excluding obvious leakage/raw price fields
        exclude_cols = {
            'ticker', 'price_target_5d', 'price_change_5d_pct', 'date',
            price_col, 'high', 'low', 'open', 'volume',
        }
        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'bool']
        ]

        # Explicitly include momentum features
        for col in ['momentum_5d', 'momentum_10d', 'momentum_20d']:
            if col not in feature_cols and col in df.columns:
                feature_cols.append(col)

        # Ensure key requirement features exist (fill if missing)
        required_flags = [
            'broke_resistance', 'post_earnings_dip_rally', 'earnings_in_3_weeks',
            'last_2q_positive_surprises', 'is_upward_trending',
            'strong_momentum', 'breakout_confirmed',
        ]
        for col in required_flags:
            if col not in df.columns:
                df[col] = False
            if col not in feature_cols:
                feature_cols.append(col)

        # Add engineered signal features to reflect UI summary requirements
        if 'rsi_14d' in df.columns:
            df['rsi_overbought'] = df['rsi_14d'] > 70
            df['rsi_oversold'] = df['rsi_14d'] < 30
            feature_cols += [c for c in ['rsi_overbought', 'rsi_oversold'] if c not in feature_cols]
        if 'volume_ratio' in df.columns:
            df['volume_above_avg'] = df['volume_ratio'] > 1.2
            feature_cols.append('volume_above_avg') if 'volume_above_avg' not in feature_cols else None
        if 'volatility_30d' in df.columns:
            df['low_volatility'] = df['volatility_30d'] < 0.3
            feature_cols.append('low_volatility') if 'low_volatility' not in feature_cols else None
        # Count positive technical signals
        pos_signals = [
            'broke_resistance', 'breakout_confirmed', 'strong_momentum', 'post_earnings_dip_rally'
        ]
        df['signal_pos_count'] = df[pos_signals].astype(int).sum(axis=1)
        feature_cols.append('signal_pos_count') if 'signal_pos_count' not in feature_cols else None

        # Include additional features from improvement doc
        for extra in [
            'atr_pct', 'volume_spike', 'volume_zscore_20', 'gap_up_2pct', 'gap_down_2pct',
            'relative_strength_20d', 'vol_adj_momentum_20d', 'days_since_last_broke_resistance',
            'return_1d', 'days_since_last_earnings', 'weighted_confidence_score'
        ]:
            if extra in df.columns and extra not in feature_cols:
                feature_cols.append(extra)

        # Time-based split: train on older dates, test on most recent 20%
        df = df.sort_values('date')
        cutoff = df['date'].quantile(0.8)
        train_df = df[df['date'] <= cutoff]
        test_df = df[df['date'] > cutoff]
        if train_df.empty or test_df.empty:
            # Fallback: last 60 calendar days as test
            fallback_cutoff = df['date'].max() - pd.Timedelta(days=60)
            train_df = df[df['date'] <= fallback_cutoff]
            test_df = df[df['date'] > fallback_cutoff]

        # Create a validation slice from the tail of train for early stopping and tuning
        val_cut = train_df['date'].quantile(0.9)
        inner_train_df = train_df[train_df['date'] <= val_cut]
        val_df = train_df[train_df['date'] > val_cut]
        if inner_train_df.empty or val_df.empty:
            # Fallback: last 30 days of train as validation
            fallback_val_cut = train_df['date'].max() - pd.Timedelta(days=30)
            inner_train_df = train_df[train_df['date'] <= fallback_val_cut]
            val_df = train_df[train_df['date'] > fallback_val_cut]

        X_train = inner_train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = inner_train_df['price_change_5d_pct']
        X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_val = val_df['price_change_5d_pct']
        X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_test = test_df['price_change_5d_pct']

        # Increase weight for rows matching confidence filter: broke_resistance OR post_earnings_dip_rally
        sample_weight = np.ones(len(inner_train_df), dtype=float)
        # Positive signals boost
        boost = (
            inner_train_df['broke_resistance'].astype(int) * 0.6 +
            inner_train_df['breakout_confirmed'].astype(int) * 0.6 +
            inner_train_df['strong_momentum'].astype(int) * 0.4 +
            inner_train_df['post_earnings_dip_rally'].astype(int) * 0.6 +
            inner_train_df.get('low_volatility', pd.Series(False, index=inner_train_df.index)).astype(int) * 0.2 +
            inner_train_df.get('volume_above_avg', pd.Series(False, index=inner_train_df.index)).astype(int) * 0.2
        )
        sample_weight += boost.values
        # If no positive signals, slightly downweight
        no_signal = (inner_train_df['signal_pos_count'] == 0)
        sample_weight[no_signal.values] *= 0.7
        # Clamp weights
        sample_weight = np.clip(sample_weight, 0.5, 3.0)

        # Small randomized hyperparameter search with early stopping
        rng = np.random.RandomState(42)
        param_space = []
        for _ in range(10):
            param_space.append({
                'max_depth': int(rng.choice([3, 4, 5, 6, 7])),
                'learning_rate': float(rng.uniform(0.015, 0.08)),
                'subsample': float(rng.uniform(0.6, 0.9)),
                'colsample_bytree': float(rng.uniform(0.6, 0.9)),
                'min_child_weight': int(rng.choice([1, 3, 5, 7])),
                'reg_lambda': float(rng.uniform(0.5, 2.0)),
                'reg_alpha': float(rng.uniform(0.0, 0.5)),
                'n_estimators': 1500,
            })

        def fit_eval(params: Dict[str, Any]) -> Dict[str, Any]:
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                eval_metric='mae',
                **params,
            )
            model.fit(
                X_train, y_train,
                sample_weight=sample_weight[: len(X_train)],
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            val_pred = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)
            return {'model': model, 'val_mae': val_mae, 'params': params}

        results = [fit_eval(p) for p in param_space]
        best = min(results, key=lambda r: r['val_mae'])
        model = best['model']

        # Evaluate on the unseen test period
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        dir_acc = float((np.sign(preds) == np.sign(y_test.values)).mean()) * 100.0

        # Persist model
        import os
        import joblib
        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        joblib.dump(model, self.model_path)

        return (
            f"✅ Trained {os.path.basename(self.model_path)} | "
            f"MAE: {mae:.2f}, R2: {r2:.2f}, DirectionalAcc: {dir_acc:.1f}%"
        )


def train_top_model() -> str:
    return TopModelTrainer().train()


