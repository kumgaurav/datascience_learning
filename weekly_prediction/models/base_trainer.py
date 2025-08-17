# models/base_trainer.py
import pandas as pd
import numpy as np

class BaseModelTrainer:
    def __init__(self, features_path: str):
        self.features_path = features_path
        self.df = pd.read_csv(features_path, parse_dates=['date'])
        self.prepared = False

    def prepare_data(self, price_col='close', horizon=5):
        df = self.df.copy()

        # Ensure consistent ordering for group-wise operations
        df = df.sort_values(['ticker', 'date'])

        # --- Core lagged returns (captures recent momentum over multiple horizons) ---
        for periods in [1, 3, 5, 10, 20]:
            df[f'return_{periods}d'] = (
                df.groupby('ticker')[price_col].pct_change(periods=periods)
            )

        # --- Rolling price stats (mean/std) and realized volatility over returns ---
        for window in [5, 20]:
            df[f'close_ma_{window}d'] = df.groupby('ticker')[price_col].transform(
                lambda s: s.rolling(window).mean()
            )
            df[f'close_std_{window}d'] = df.groupby('ticker')[price_col].transform(
                lambda s: s.rolling(window).std()
            )
            # Realized volatility from 1d returns (no annualization here; short-horizon)
            df[f'volatility_{window}d'] = df.groupby('ticker')['return_1d'].transform(
                lambda s: s.rolling(window).std()
            )

        # --- RSI(14) per ticker ---
        def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
            delta = series.diff(1)
            gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        df['rsi_14d'] = df.groupby('ticker')[price_col].transform(lambda s: _compute_rsi(s))

        # --- MACD (12,26,9) per ticker ---
        ema_fast = df.groupby('ticker')[price_col].transform(
            lambda s: s.ewm(span=12, adjust=False).mean()
        )
        ema_slow = df.groupby('ticker')[price_col].transform(
            lambda s: s.ewm(span=26, adjust=False).mean()
        )
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df.groupby('ticker')['macd'].transform(
            lambda s: s.ewm(span=9, adjust=False).mean()
        )

        # --- Bollinger Band position (20d) ---
        bb_ma20 = df.groupby('ticker')[price_col].transform(lambda s: s.rolling(20).mean())
        bb_std20 = df.groupby('ticker')[price_col].transform(lambda s: s.rolling(20).std())
        bb_upper = bb_ma20 + 2.0 * bb_std20
        bb_lower = bb_ma20 - 2.0 * bb_std20
        # Avoid division by zero
        band_width = (bb_upper - bb_lower).replace(0, np.nan)
        df['bb_position'] = (df[price_col] - bb_lower) / band_width

        # --- Volume trends ---
        df['volume_ma_20'] = df.groupby('ticker')['volume'].transform(lambda s: s.rolling(20).mean())
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']

        # --- Optional market-wide feature (SPY 1d return, if present) ---
        try:
            spy_mask = df['ticker'].astype(str).str.upper() == 'SPY'
            if spy_mask.any():
                spy_returns = (
                    df.loc[spy_mask, ['date', price_col]]
                      .sort_values('date')
                )
                spy_returns['spy_return_1d'] = spy_returns[price_col].pct_change()
                df = df.merge(
                    spy_returns[['date', 'spy_return_1d']],
                    on='date',
                    how='left'
                )
            else:
                # Fill with NaN if SPY not present; downstream will handle
                df['spy_return_1d'] = np.nan
        except Exception:
            df['spy_return_1d'] = np.nan

        # Create target: % change over horizon days
        df['target'] = (
            df.groupby('ticker')[price_col]
              .shift(-horizon)
              .sub(df[price_col])
              .div(df[price_col]) * 100
        )

        # Basic momentum features (kept for continuity with earlier versions)
        for p in [5, 10, 20]:
            df[f'momentum_{p}d'] = (
                df.groupby('ticker')[price_col].pct_change(periods=p)
            )

        # Ensure momentum aliases exist for compatibility with selector/trainer downstream
        # (Some paths might rely on these names explicitly)

        # Drop rows without target
        df = df.dropna(subset=['target'])

        # Train/test split (time-based)
        df = df.sort_values(['ticker', 'date'])
        cutoff = df['date'].quantile(0.8)
        train_df = df[df['date'] <= cutoff]
        test_df = df[df['date'] > cutoff]

        self.train_df, self.test_df = train_df, test_df
        self.prepared = True
        return train_df, test_df
