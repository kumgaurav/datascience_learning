# models/base_trainer.py
import os
import pandas as pd
import numpy as np

class BaseModelTrainer:
    def __init__(self, features_path: str, tracker=None, run_name: str = None, feature_config: dict | None = None):
        self.features_path = features_path
        # Validate CSV exists and is readable
        if not os.path.isfile(features_path):
            raise FileNotFoundError(f"Features CSV not found: {features_path}")
        try:
            self.df = pd.read_csv(features_path, parse_dates=['date'])
            try:
                print(f"[BASE INIT] Loaded features from {features_path} (rows={len(self.df)}, cols={len(self.df.columns)})")
            except Exception:
                pass
        except Exception as e:
            # Retry without parse_dates, then try to coerce date if present
            try:
                self.df = pd.read_csv(features_path)
                if 'date' in self.df.columns:
                    self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
                try:
                    print(f"[BASE INIT] Loaded features from {features_path} without parse_dates (rows={len(self.df)})")
                except Exception:
                    pass
            except Exception:
                raise ValueError(f"Failed to read features CSV: {features_path} → {e}")

        self.prepared = False
        # Optional experiment tracker (MLflow/W&B)
        self.tracker = tracker
        self.run_name = run_name
        # Feature configuration with sensible defaults; caller can override via dict
        self.feature_config = {
            'lag_returns_periods': [1, 3, 5, 10, 20],
            'rolling_windows': [5, 20],
            'rsi_period': 14,
            'macd_spans': (12, 26, 9),  # (fast, slow, signal)
            'bb_window': 20,
            'bb_k': 2.0,
            'volume_ma_window': 20,
            'momentum_periods': [5, 10, 20],
        }
        if feature_config:
            # Shallow merge; provided keys override defaults
            try:
                self.feature_config.update({k: v for k, v in feature_config.items() if k in self.feature_config})
            except Exception:
                pass
        try:
            print(f"[BASE INIT] Feature config: {self.feature_config}")
        except Exception:
            pass
        if self.tracker is None:
            try:
                # Lazy import to avoid hard dependency
                from utils.tracking import get_tracker
                tracking_mode = os.environ.get('TRACKING_MODE', 'none')
                exp_name = os.environ.get('EXPERIMENT_NAME')
                mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI')
                wandb_project = os.environ.get('WANDB_PROJECT')
                wandb_entity = os.environ.get('WANDB_ENTITY')
                self.tracker = get_tracker(
                    mode=tracking_mode,
                    experiment_name=exp_name,
                    mlflow_uri=mlflow_uri,
                    wandb_project=wandb_project,
                    wandb_entity=wandb_entity,
                )
            except Exception:
                self.tracker = None

    def prepare_data(self, price_col='close', horizon=5):
        df = self.df.copy()

        # Ensure consistent ordering for group-wise operations
        df = df.sort_values(['ticker', 'date'])
        try:
            print(f"[BASE PREP] Starting prepare_data(price_col={price_col}, horizon={horizon})")
        except Exception:
            pass

        # --- Core lagged returns (captures recent momentum over multiple horizons) ---
        for periods in self.feature_config.get('lag_returns_periods', [1, 3, 5, 10, 20]):
            df[f'return_{periods}d'] = (
                df.groupby('ticker')[price_col].pct_change(periods=periods)
            )

        # --- Rolling price stats (mean/std) and realized volatility over returns ---
        for window in self.feature_config.get('rolling_windows', [5, 20]):
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
            gain = delta.where(delta > 0, 0.0).rolling(window=period, min_periods=period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
            # Avoid divide-by-zero and force NaN when insufficient history
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = gain / loss
                rs = rs.replace([np.inf, -np.inf], np.nan)
                rsi = 100 - (100 / (1 + rs))
            return rsi
        rsi_period = int(self.feature_config.get('rsi_period', 14))
        df['rsi_14d'] = df.groupby('ticker')[price_col].transform(lambda s: _compute_rsi(s, period=rsi_period))

        # --- MACD (12,26,9) per ticker ---
        macd_fast, macd_slow, macd_signal = self.feature_config.get('macd_spans', (12, 26, 9))
        ema_fast = df.groupby('ticker')[price_col].transform(
            lambda s: s.ewm(span=macd_fast, adjust=False).mean()
        )
        ema_slow = df.groupby('ticker')[price_col].transform(
            lambda s: s.ewm(span=macd_slow, adjust=False).mean()
        )
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df.groupby('ticker')['macd'].transform(
            lambda s: s.ewm(span=macd_signal, adjust=False).mean()
        )

        # --- Bollinger Band position (20d) ---
        bb_window = int(self.feature_config.get('bb_window', 20))
        bb_k = float(self.feature_config.get('bb_k', 2.0))
        bb_ma20 = df.groupby('ticker')[price_col].transform(lambda s: s.rolling(bb_window).mean())
        bb_std20 = df.groupby('ticker')[price_col].transform(lambda s: s.rolling(bb_window).std())
        bb_upper = bb_ma20 + bb_k * bb_std20
        bb_lower = bb_ma20 - bb_k * bb_std20
        # Avoid division by zero
        band_width = (bb_upper - bb_lower).replace(0, np.nan)
        df['bb_position'] = (df[price_col] - bb_lower) / band_width

        # --- Volume trends ---
        vol_ma = int(self.feature_config.get('volume_ma_window', 20))
        df['volume_ma_20'] = df.groupby('ticker')['volume'].transform(lambda s: s.rolling(vol_ma).mean())
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
                print("[BASE DIAG] SPY not present in input; 'spy_return_1d' filled with NaN.")
        except Exception:
            df['spy_return_1d'] = np.nan

        # Create target: % change over horizon days
        # Ensure sorted within ticker prior to shift to avoid misalignment
        df = df.sort_values(['ticker', 'date'])
        df['target'] = (
            df.groupby('ticker')[price_col]
              .shift(-horizon)
              .sub(df[price_col])
              .div(df[price_col]) * 100
        )

        # Basic momentum features (kept for continuity with earlier versions)
        for p in self.feature_config.get('momentum_periods', [5, 10, 20]):
            df[f'momentum_{p}d'] = (
                df.groupby('ticker')[price_col].pct_change(periods=p)
            )

        # Ensure momentum aliases exist for compatibility with selector/trainer downstream
        # (Some paths might rely on these names explicitly)

        # Drop rows without target
        df = df.dropna(subset=['target'])

        # Train/test split (time-based) using date range to avoid imbalance bias
        df = df.sort_values(['ticker', 'date'])
        date_min = df['date'].min()
        date_max = df['date'].max()
        try:
            cutoff = date_min + (date_max - date_min) * 0.8
            try:
                print(f"[BASE PREP] Split cutoff at {cutoff} (min={date_min}, max={date_max})")
            except Exception:
                pass
        except Exception:
            cutoff = df['date'].quantile(0.8)
        train_df = df[df['date'] <= cutoff]
        test_df = df[df['date'] > cutoff]
        try:
            print(f"[BASE PREP] Train rows={len(train_df)}, Test rows={len(test_df)}")
        except Exception:
            pass

        self.train_df, self.test_df = train_df, test_df
        self.prepared = True
        return train_df, test_df
