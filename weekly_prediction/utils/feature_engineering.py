import os
import logging
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd


def _normalize_prices_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    lower_map = {c.lower(): c for c in out.columns}
    t_cands = ['ticker', 'symbol', 'security', 'code']
    d_cands = ['date', 'trade_date', 'timestamp', 'datetime', 'as_of_date']
    o_cands = ['open']
    h_cands = ['high']
    l_cands = ['low']
    c_cands = ['close', 'adj_close', 'close_price', 'last', 'closeprice', 'c']
    v_cands = ['volume', 'vol']

    def pick(cands):
        for k in cands:
            if k in lower_map:
                return lower_map[k]
        return None

    tcol = pick(t_cands)
    dcol = pick(d_cands)
    ocol = pick(o_cands)
    hcol = pick(h_cands)
    lcol = pick(l_cands)
    ccol = pick(c_cands)
    vcol = pick(v_cands)

    if tcol is None or dcol is None or ccol is None or vcol is None:
        raise ValueError(
            f"Missing required columns: ticker={tcol}, date={dcol}, close={ccol}, volume={vcol}"
        )

    out['ticker'] = out[tcol].astype(str).str.upper()
    try:
        out['date'] = pd.to_datetime(out[dcol], errors='coerce')
    except Exception:
        out['date'] = out[dcol]
    out['open'] = pd.to_numeric(out[ocol], errors='coerce') if ocol in out.columns else np.nan
    out['high'] = pd.to_numeric(out[hcol], errors='coerce') if hcol in out.columns else np.nan
    out['low'] = pd.to_numeric(out[lcol], errors='coerce') if lcol in out.columns else np.nan
    out['close'] = pd.to_numeric(out[ccol], errors='coerce')
    out['volume'] = pd.to_numeric(out[vcol], errors='coerce')
    return out


def _compute_features(
    prices: pd.DataFrame,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_k: float = 2.0,
    fill_strategy: dict | None = None,
) -> pd.DataFrame:
    """Compute technical features per (ticker,date) with vectorized pandas ops.

    Args:
        prices: Input OHLCV dataframe with columns `ticker`, `date`, `open`, `high`, `low`, `close`, `volume`.
        rsi_period: Lookback window for RSI.
        macd_fast: Fast EMA span for MACD.
        macd_slow: Slow EMA span for MACD.
        macd_signal: Signal EMA span for MACD.
        bb_window: Window for Bollinger calculations.
        bb_k: Standard deviation multiplier for Bollinger bands.
        fill_strategy: Optional mapping of column -> method (e.g., {'trend_slope_15d': 'ffill'}).
    """
    if prices.empty:
        return prices

    # Helpers
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index over `period` using mean of gains/losses."""
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0.0).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
            rs = rs.replace([np.inf, -np.inf], np.nan)
            return 100 - (100 / (1 + rs))

    def macd(series: pd.Series, fast=12, slow=26, signal=9):
        """MACD line, signal and histogram with EMA spans (fast, slow, signal)."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def boll(series: pd.Series, window=20, k=2.0):
        ma = series.rolling(window=window, min_periods=window).mean()
        std = series.rolling(window=window, min_periods=window).std()
        upper = ma + k * std
        lower = ma - k * std
        return upper, ma, lower

    df = prices.sort_values(['ticker', 'date']).copy()

    # Convenience groupers (flattened results via group_keys=False)
    g_close = df.groupby('ticker', group_keys=False)['close']
    g_vol = df.groupby('ticker', group_keys=False)['volume']
    g_high = df.groupby('ticker', group_keys=False)['high']
    g_low = df.groupby('ticker', group_keys=False)['low']

    # Moving averages and volatility
    df['ma_20'] = g_close.rolling(window=20, min_periods=20).mean().reset_index(level=0, drop=True)
    df['ma_50'] = g_close.rolling(window=50, min_periods=50).mean().reset_index(level=0, drop=True)
    df['return_1d'] = g_close.pct_change()
    df['volatility_30d'] = (
        df.groupby('ticker')['return_1d']
        .rolling(window=30, min_periods=30)
        .std()
        .reset_index(level=0, drop=True)
    )

    # RSI
    df['rsi_14d'] = g_close.apply(lambda s: rsi(s, rsi_period))

    # MACD
    macd_line = g_close.apply(lambda s: macd(s, fast=macd_fast, slow=macd_slow, signal=macd_signal)[0])
    macd_signal_s = g_close.apply(lambda s: macd(s, fast=macd_fast, slow=macd_slow, signal=macd_signal)[1])
    macd_hist = g_close.apply(lambda s: macd(s, fast=macd_fast, slow=macd_slow, signal=macd_signal)[2])
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal_s
    df['macd_histogram'] = macd_hist

    # Bollinger and position
    bb_up = g_close.rolling(window=bb_window, min_periods=bb_window).mean().reset_index(level=0, drop=True)
    bb_std = g_close.rolling(window=bb_window, min_periods=bb_window).std().reset_index(level=0, drop=True)
    df['bb_middle'] = bb_up
    df['bb_upper'] = bb_up + float(bb_k) * bb_std
    df['bb_lower'] = bb_up - float(bb_k) * bb_std
    with np.errstate(divide='ignore', invalid='ignore'):
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume indicators
    df['volume_ma_20'] = g_vol.rolling(window=20, min_periods=20).mean().reset_index(level=0, drop=True)
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    df['pvt'] = g_close.apply(lambda s: ((s - s.shift(1)) / s.shift(1))).mul(df['volume']).groupby(df['ticker']).cumsum()
    df['obv'] = g_close.apply(lambda s: np.sign(s.diff())).mul(df['volume']).groupby(df['ticker']).cumsum()

    # Risk metrics
    rolling_max_30 = g_close.rolling(window=30, min_periods=30).max().reset_index(level=0, drop=True)
    df['rolling_max'] = rolling_max_30
    with np.errstate(divide='ignore', invalid='ignore'):
        df['drawdown_30d'] = (df['close'] - df['rolling_max']) / df['rolling_max']
    df['var_95_30d'] = (
        df.groupby('ticker')['return_1d']
        .rolling(window=30, min_periods=30)
        .quantile(0.05)
        .reset_index(level=0, drop=True)
    )
    mean_30 = df.groupby('ticker')['return_1d'].rolling(30, min_periods=30).mean().reset_index(level=0, drop=True)
    std_30 = df.groupby('ticker')['return_1d'].rolling(30, min_periods=30).std().reset_index(level=0, drop=True)
    df['sharpe_ratio'] = (mean_30 - 0.02 / 252) / std_30

    # Advanced trend (direction over windows)
    def trend_dir_window(series: pd.Series, window: int) -> pd.Series:
        def f(x: pd.Series) -> float:
            if x.isna().any():
                return np.nan
            return 1.0 if x.iloc[-1] > x.iloc[0] else (-1.0 if x.iloc[-1] < x.iloc[0] else 0.0)
        return series.rolling(window=window).apply(f, raw=False)

    for period in [5, 10, 20, 50]:
        df[f'trend_{period}d'] = g_close.apply(lambda s, w=period: trend_dir_window(s, w))
    df['trend_strength'] = df['trend_20d'].abs() if 'trend_20d' in df.columns else df.get('trend_10d', 0).abs()

    # Momentum windows
    for p in [5, 10, 20, 30, 60]:
        df[f'momentum_{p}d'] = g_close.pct_change(periods=p)

    # Support/Resistance and distances
    df['support_20d'] = g_close.rolling(window=20, min_periods=20).min().reset_index(level=0, drop=True)
    df['resistance_20d'] = g_close.rolling(window=20, min_periods=20).max().reset_index(level=0, drop=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['distance_from_support'] = (df['close'] - df['support_20d']) / df['close']
        df['distance_from_resistance'] = (df['resistance_20d'] - df['close']) / df['close']
        df['breakout_strength'] = df['close'] / df.groupby('ticker')['resistance_20d'].shift(1)

    # Trend slopes with validation
    def slope_apply(series: pd.Series, window: int) -> pd.Series:
        x = np.arange(window)
        def calc_slope(y: pd.Series) -> float:
            if y.isna().sum() > 0 or len(y) != window:
                return np.nan
            return float(np.polyfit(x, y, 1)[0])
        return series.rolling(window=window).apply(calc_slope, raw=False)

    df['trend_slope_15d'] = g_close.apply(lambda s: slope_apply(s, 15))
    df['trend_slope_30d'] = g_close.apply(lambda s: slope_apply(s, 30))

    # Up-day ratio 20d
    def up_ratio(arr: np.ndarray) -> float:
        mask = ~np.isnan(arr)
        if not mask.any():
            return np.nan
        return float(np.mean(arr[mask] > 0))
    df['up_day_ratio_20d'] = (
        df.groupby('ticker')['return_1d']
        .rolling(window=20)
        .apply(up_ratio, raw=True)
        .reset_index(level=0, drop=True)
    )

    # Engineered features
    df['price_change_pct'] = g_close.pct_change(1)
    df['vol_change_pct'] = g_vol.pct_change(1)
    df['selloff_flag'] = ((df['vol_change_pct'] > 0.20) & (df['price_change_pct'] < 0)).astype(int)
    df['rsi_price_interaction'] = df['rsi_14d'] * df['price_change_pct']
    df['volratio_price_interaction'] = df['volume_ratio'] * df['price_change_pct']
    df['overbought_spike'] = ((df['rsi_14d'] > 75) & (df['volume_ratio'] > 1.2)).astype(int)

    # Broke resistance vs previous 50d high (shifted)
    prev50_unshift = df.groupby('ticker', group_keys=False)['close'].transform(lambda s: s.rolling(window=50, min_periods=1).max())
    df['prev50d_high'] = prev50_unshift.groupby(df['ticker']).shift(1)
    df['broke_resistance'] = df['close'] > df['prev50d_high']

    # Days since last broke_resistance (vectorized per-group)
    def days_since_true_bool(arr: np.ndarray) -> np.ndarray:
        n = arr.size
        idx = np.arange(n)
        last_true = np.where(arr, idx, -1)
        last_true_cum = np.maximum.accumulate(last_true)
        out = (idx - last_true_cum).astype(float)
        out[last_true_cum == -1] = np.nan
        return out
    df['days_since_last_broke_resistance'] = df.groupby('ticker', group_keys=False)['broke_resistance'].apply(
        lambda s: pd.Series(days_since_true_bool(s.values), index=s.index)
    )

    # ATR 14d and pct
    prev_close = g_close.shift(1)
    # Compute true range components per row
    prev_close_series = prev_close
    high_series = df['high']
    low_series = df['low']
    tr1 = (high_series - low_series)
    tr2 = (high_series - prev_close_series).abs()
    tr3 = (low_series - prev_close_series).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14d'] = (
        true_range.groupby(df['ticker']).rolling(window=14, min_periods=14).mean().reset_index(level=0, drop=True)
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        df['atr_pct'] = df['atr_14d'] / df['close']

    # Volume spike
    vol_mean_20 = g_vol.rolling(window=20, min_periods=20).mean().reset_index(level=0, drop=True)
    vol_std_20 = g_vol.rolling(window=20, min_periods=20).std().reset_index(level=0, drop=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['volume_zscore_20'] = (df['volume'] - vol_mean_20) / vol_std_20
    df['volume_spike'] = df['volume_zscore_20'] > 2.0

    # Gap analysis
    prev_close_flat = prev_close.reset_index(level=0, drop=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['gap_pct'] = (df['open'] - prev_close_flat) / prev_close_flat
    df['gap_up_2pct'] = df['gap_pct'] > 0.02
    df['gap_down_2pct'] = df['gap_pct'] < -0.02

    out_all = df

    # Cross-sectional relative strength vs median momentum on each date
    if not out_all.empty and 'momentum_20d' in out_all.columns:
        date_median = out_all.groupby('date')['momentum_20d'].median().rename('momentum_20d_median')
        out_all = out_all.merge(date_median, on='date', how='left')
        out_all['relative_strength_20d'] = out_all['momentum_20d'] - out_all['momentum_20d_median']
        out_all.drop(columns=['momentum_20d_median'], inplace=True)
    else:
        out_all['relative_strength_20d'] = 0

    # Volatility-adjusted momentum
    if 'momentum_20d' in out_all.columns and 'volatility_30d' in out_all.columns:
        out_all['vol_adj_momentum_20d'] = out_all['momentum_20d'] / (out_all['volatility_30d'] + 1e-6)
    else:
        out_all['vol_adj_momentum_20d'] = 0

    # Replace infs and apply optional fill strategies before generic fills
    out_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    base_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
    bool_cols = ['broke_resistance', 'gap_up_2pct', 'gap_down_2pct', 'volume_spike']
    if isinstance(fill_strategy, dict) and not out_all.empty:
        for col, method in fill_strategy.items():
            if col in out_all.columns and method == 'ffill':
                try:
                    out_all[col] = out_all.sort_values('date').groupby('ticker')[col].ffill()
                except Exception:
                    pass
    for col in out_all.columns:
        if col in base_cols:
            continue
        if col in bool_cols:
            out_all[col] = out_all[col].fillna(False).astype(bool)
        else:
            if pd.api.types.is_numeric_dtype(out_all[col]):
                out_all[col] = out_all[col].fillna(0.0)

    # Column ordering similar to the rich feature set
    preferred_cols = [
        'ticker', 'date', 'open', 'high', 'low', 'close', 'volume',
        'ma_20', 'ma_50', 'volatility_30d', 'rsi_14d',
        'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_position',
        'volume_ma_20', 'volume_ratio', 'pvt', 'obv',
        'drawdown_30d', 'var_95_30d', 'sharpe_ratio', 'trend_strength',
        'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_30d', 'momentum_60d',
        'support_20d', 'resistance_20d', 'distance_from_support', 'distance_from_resistance', 'breakout_strength',
        'trend_slope_15d', 'trend_slope_30d', 'up_day_ratio_20d',
        'price_change_pct', 'vol_change_pct', 'rsi_price_interaction', 'volratio_price_interaction',
        'selloff_flag', 'overbought_spike', 'prev50d_high', 'broke_resistance', 'days_since_last_broke_resistance',
        'volume_zscore_20', 'volume_spike', 'gap_pct', 'gap_up_2pct', 'gap_down_2pct',
        'relative_strength_20d', 'vol_adj_momentum_20d'
    ]
    existing = [c for c in preferred_cols if c in out_all.columns]
    return out_all[existing]


def build_features_for_inputs(clean_path: str, unclean_path: str, out_dir: str = 'data/features') -> Tuple[Optional[str], Optional[str]]:
    os.makedirs(out_dir, exist_ok=True)
    out_clean = out_unclean = None

    # Parameterization from env (populated by config)
    rsi_p = int(os.getenv('FE_RSI_PERIOD', '14'))
    macd_fast = int(os.getenv('FE_MACD_FAST', '12'))
    macd_slow = int(os.getenv('FE_MACD_SLOW', '26'))
    macd_sig = int(os.getenv('FE_MACD_SIGNAL', '9'))
    bb_win = int(os.getenv('FE_BB_WINDOW', '20'))
    bb_k = float(os.getenv('FE_BB_K', '2.0'))
    fill_slopes = os.getenv('FE_FILL_SLOPES_FFILL', 'false') in ('1','true','True')
    fill_strategy = {'trend_slope_15d': 'ffill', 'trend_slope_30d': 'ffill'} if fill_slopes else None

    if os.path.exists(clean_path):
        dfc = pd.read_csv(clean_path)
        dfc = _normalize_prices_columns(dfc)
        feats_c = _compute_features(
            dfc,
            rsi_period=rsi_p,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_sig,
            bb_window=bb_win,
            bb_k=bb_k,
            fill_strategy=fill_strategy,
        )
        out_clean = os.path.join(out_dir, 'stock_features_clean.csv')
        feats_c.to_csv(out_clean, index=False)
        logging.getLogger(__name__).info(f"[FE] Wrote clean features -> {out_clean} (rows={len(feats_c)})")
    else:
        logging.getLogger(__name__).warning(f"[FE] Clean input not found: {clean_path}")

    if os.path.exists(unclean_path):
        dfu = pd.read_csv(unclean_path)
        dfu = _normalize_prices_columns(dfu)
        feats_u = _compute_features(
            dfu,
            rsi_period=rsi_p,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_sig,
            bb_window=bb_win,
            bb_k=bb_k,
            fill_strategy=fill_strategy,
        )
        out_unclean = os.path.join(out_dir, 'stock_features_unclean.csv')
        feats_u.to_csv(out_unclean, index=False)
        logging.getLogger(__name__).info(f"[FE] Wrote unclean features -> {out_unclean} (rows={len(feats_u)})")
    else:
        logging.getLogger(__name__).warning(f"[FE] Unclean input not found: {unclean_path}")

    return out_clean, out_unclean


def main():
    import argparse
    from utils.config import load_config, apply_env_from_config
    from utils.logging_utils import configure_logging
    p = argparse.ArgumentParser(description='Generate features for clean and unclean price files')
    p.add_argument('--clean', default='data/input/stock_prices_with_clean_data.csv', help='Path to clean prices CSV')
    p.add_argument('--unclean', default='data/input/stock_prices_with_unclean_data.csv', help='Path to unclean prices CSV')
    p.add_argument('--outdir', default='data/features', help='Output directory for feature CSVs')
    args = p.parse_args()
    # Optional config + logging
    cfg_path = os.getenv('APP_CONFIG', 'config.yaml')
    cfg = load_config(cfg_path)
    apply_env_from_config(cfg)
    configure_logging()
    build_features_for_inputs(args.clean, args.unclean, args.outdir)


if __name__ == '__main__':
    main()


