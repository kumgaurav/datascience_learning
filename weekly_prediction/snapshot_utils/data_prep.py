"""Utility routines to prepare model-ready datasets.

This module builds two families of artifacts from base OHLCV prices:

- XGBoost tabular (per (ticker,date) snapshot):
  Adds lag returns, rolling stats, RSI/MACD, Bollinger, volume trends,
  market context (SPY 1–5d, 20d vol), excess return, 60d beta,
  volatility regime flag, and momentum aliases.

- LSTM sliding windows: 3D arrays of shape (num_windows, lookback, num_features)
  with per-window z-score normalization and forward-return labels.

All functions avoid forward-looking leakage by using only information up to each
row's date. Where SPY context is missing, per-date cross-sectional medians are
used as proxies to minimize NaNs.
"""
import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def _compute_technical_features(
    df: pd.DataFrame,
    price_col: str = "close",
    *,
    rsi_period: int = 14,
    bb_window: int = 20,
    bb_k: float = 2.0,
    volume_ma_window: int = 20,
    impute_bb_position_zero: bool = False,
) -> pd.DataFrame:
    """Compute technical, volume, risk and market-context features.

    Returns a copy of df with additional columns, preserving input rows. Newly
    added columns include (non-exhaustive):
      - return_{1,2,3,4,5,10,20}d, momentum_{5,10,20}d
      - close_ma_{5,20}d, close_std_{5,20}d, volatility_{5,20}d
      - rsi_14d, macd, macd_signal, bb_position
      - volume_ma_20, volume_ratio
      - spy_return_{1..5}d (with market medians as fallback), spy_vol_20d
      - market_return_{1..5}d, market_vol_20d, excess_return_1d, beta_60d
      - high_vol_regime (bool)
    """
    df = df.sort_values(["ticker", "date"]).copy()

    # Lag returns (include 2d/4d to support SPY proxy fill)
    for periods in [1, 2, 3, 4, 5, 10, 20]:
        df[f"return_{periods}d"] = df.groupby("ticker")[price_col].pct_change(periods=periods)

    # Momentum features (alias to multi-period returns, kept for model compatibility)
    for periods in [5, 10, 20]:
        df[f"momentum_{periods}d"] = df.groupby("ticker")[price_col].pct_change(periods=periods)

    # Rolling price stats and realized vol
    for window in [5, 20]:
        df[f"close_ma_{window}d"] = df.groupby("ticker")[price_col].transform(lambda s: s.rolling(window).mean())
        df[f"close_std_{window}d"] = df.groupby("ticker")[price_col].transform(lambda s: s.rolling(window).std())
        df[f"volatility_{window}d"] = df.groupby("ticker")["return_1d"].transform(lambda s: s.rolling(window).std())

    # RSI(14) with min_periods
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0.0).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
            rs = rs.replace([np.inf, -np.inf], np.nan)
            rsi = 100 - (100 / (1 + rs))
        return rsi

    df["rsi_14d"] = df.groupby("ticker")[price_col].transform(lambda s: _rsi(s, period=rsi_period))

    # MACD (12,26,9)
    ema_fast = df.groupby("ticker")[price_col].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema_slow = df.groupby("ticker")[price_col].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df.groupby("ticker")["macd"].transform(lambda s: s.ewm(span=9, adjust=False).mean())

    # Bollinger position (configurable window/k)
    bb_ma20 = df.groupby("ticker")[price_col].transform(lambda s: s.rolling(bb_window, min_periods=bb_window).mean())
    bb_std20 = df.groupby("ticker")[price_col].transform(lambda s: s.rolling(bb_window, min_periods=bb_window).std())
    bb_upper = bb_ma20 + float(bb_k) * bb_std20
    bb_lower = bb_ma20 - float(bb_k) * bb_std20
    band_width = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"] = (df[price_col] - bb_lower) / band_width
    if impute_bb_position_zero:
        df["bb_position"] = df["bb_position"].fillna(0.0)

    # Volume trends
    df["volume_ma_20"] = df.groupby("ticker")["volume"].transform(lambda s: s.rolling(volume_ma_window, min_periods=volume_ma_window).mean())
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

    # --- Engineered features per E1_08222025.md ---
    # Price/Volume changes
    df["price_change_pct"] = df.groupby("ticker")[price_col].pct_change(1)
    df["vol_change_pct"] = df.groupby("ticker")["volume"].pct_change(1)
    # Sell-off flag: sharp vol up and price down
    df["selloff_flag"] = ((df["vol_change_pct"] > 0.20) & (df["price_change_pct"] < 0)).astype(int)
    # Interactions
    df["rsi_price_interaction"] = df["rsi_14d"] * df["price_change_pct"]
    df["volratio_price_interaction"] = df["volume_ratio"] * df["price_change_pct"]
    # Overbought spike flag
    df["overbought_spike"] = ((df["rsi_14d"] > 75) & (df["volume_ratio"] > 1.2)).astype(int)

    # Optional market feature (SPY 1d return and extended context)
    try:
        spy_mask = df["ticker"].astype(str).str.upper() == "SPY"
        if spy_mask.any():
            spy = df.loc[spy_mask, ["date", price_col]].sort_values("date").copy()
            spy["spy_return_1d"] = spy[price_col].pct_change()
            # Add multi-horizon SPY returns (1–5d)
            for p in [2, 3, 4, 5]:
                spy[f"spy_return_{p}d"] = spy[price_col].pct_change(periods=p)
            # Rolling 20d volatility of SPY
            spy["spy_vol_20d"] = spy["spy_return_1d"].rolling(20).std()
            df = df.merge(spy[["date", "spy_return_1d", "spy_return_2d", "spy_return_3d", "spy_return_4d", "spy_return_5d", "spy_vol_20d"]], on="date", how="left")
        else:
            df["spy_return_1d"] = np.nan
            df["spy_return_2d"] = np.nan
            df["spy_return_3d"] = np.nan
            df["spy_return_4d"] = np.nan
            df["spy_return_5d"] = np.nan
            df["spy_vol_20d"] = np.nan
    except Exception:
        df["spy_return_1d"] = np.nan
        df["spy_return_2d"] = np.nan
        df["spy_return_3d"] = np.nan
        df["spy_return_4d"] = np.nan
        df["spy_return_5d"] = np.nan
        df["spy_vol_20d"] = np.nan

    # Fallback: if SPY is missing or not aligned on a given date, use cross-sectional median returns as market proxy
    try:
        for p in [1, 2, 3, 4, 5]:
            colr = f"return_{p}d"
            if colr in df.columns:
                market_proxy = df.groupby("date")[colr].median().rename(f"market_return_{p}d")
                df = df.merge(market_proxy, on="date", how="left")
                spy_col = f"spy_return_{p}d" if p > 1 else "spy_return_1d"
                if spy_col in df.columns:
                    df[spy_col] = df[spy_col].fillna(df[f"market_return_{p}d"])
    except Exception:
        pass

    # Derive market 20d volatility proxy and fill spy_vol_20d when missing
    try:
        if "market_return_1d" in df.columns:
            market_daily = (
                df[["date", "market_return_1d"]]
                  .drop_duplicates(subset=["date"])  # one per date
                  .sort_values("date")
            )
            market_daily["market_vol_20d"] = market_daily["market_return_1d"].rolling(20).std()
            # Backfill initial NaNs with expanding std, then zeros if still NaN
            market_daily["market_vol_20d"] = (
                market_daily["market_vol_20d"]
                    .fillna(market_daily["market_return_1d"].expanding().std())
                    .fillna(0.0)
            )
            df = df.merge(market_daily[["date", "market_vol_20d"]], on="date", how="left")
            if "spy_vol_20d" in df.columns:
                df["spy_vol_20d"] = df["spy_vol_20d"].fillna(df["market_vol_20d"]).fillna(0.0)
            else:
                df["spy_vol_20d"] = df["market_vol_20d"].fillna(0.0)
    except Exception:
        pass

    # Global fallbacks: ensure SPY/market features are present and finite
    try:
        for p in [1, 2, 3, 4, 5]:
            spy_col = f"spy_return_{p}d" if p > 1 else "spy_return_1d"
            mkt_col = f"market_return_{p}d"
            if spy_col not in df.columns:
                df[spy_col] = 0.0
            else:
                df[spy_col] = pd.to_numeric(df[spy_col], errors='coerce').fillna(0.0)
            if mkt_col not in df.columns:
                df[mkt_col] = 0.0
            else:
                df[mkt_col] = pd.to_numeric(df[mkt_col], errors='coerce').fillna(0.0)
        if 'spy_vol_20d' not in df.columns:
            df['spy_vol_20d'] = 0.0
        else:
            df['spy_vol_20d'] = pd.to_numeric(df['spy_vol_20d'], errors='coerce').fillna(0.0)
        if 'market_vol_20d' not in df.columns:
            df['market_vol_20d'] = 0.0
        else:
            df['market_vol_20d'] = pd.to_numeric(df['market_vol_20d'], errors='coerce').fillna(0.0)
    except Exception:
        pass

    # Excess return vs market proxy
    try:
        if "market_return_1d" in df.columns:
            df["excess_return_1d"] = df["return_1d"] - df["market_return_1d"]
        else:
            df["excess_return_1d"] = df["return_1d"]
    except Exception:
        df["excess_return_1d"] = df.get("return_1d", 0)

    # Rolling 60d beta to SPY using returns (avoid DataFrameGroupBy.apply warning)
    try:
        if "spy_return_1d" in df.columns:
            cov_60 = df.groupby("ticker", group_keys=False)["return_1d"].apply(
                lambda s: s.rolling(60).cov(df.loc[s.index, "spy_return_1d"])  # aligned other series
            )
            var_60 = df.groupby("ticker")["spy_return_1d"].transform(lambda s: s.rolling(60).var())
            df["beta_60d"] = cov_60 / var_60
        else:
            df["beta_60d"] = np.nan
    except Exception:
        df["beta_60d"] = np.nan

    # Volatility regime flag based on SPY 20d vol percentile
    try:
        if "spy_vol_20d" in df.columns and df["spy_vol_20d"].notna().any():
            # Compute threshold from available spy_vol_20d values
            thr = float(df["spy_vol_20d"].dropna().quantile(0.6))
            df["high_vol_regime"] = (df["spy_vol_20d"] > thr).fillna(False)
        else:
            df["high_vol_regime"] = False
    except Exception:
        df["high_vol_regime"] = False

    return df


def build_xgb_dataset(
    prices: pd.DataFrame,
    horizon: int = 5,
    price_col: str = "close",
    *,
    feature_config: dict | None = None,
    selected_features: list[str] | None = None,
) -> pd.DataFrame:
    df = prices.copy()
    # Ensure correct dtypes
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    # Basic return for technicals
    df["return_1d"] = df.groupby("ticker")[price_col].pct_change()
    cfg = {
        "rsi_period": 14,
        "bb_window": 20,
        "bb_k": 2.0,
        "volume_ma_window": 20,
        "impute_bb_position_zero": False,
    }
    if feature_config:
        try:
            cfg.update({k: v for k, v in feature_config.items() if k in cfg})
        except Exception:
            pass

    df = _compute_technical_features(
        df,
        price_col=price_col,
        rsi_period=int(cfg["rsi_period"]),
        bb_window=int(cfg["bb_window"]),
        bb_k=float(cfg["bb_k"]),
        volume_ma_window=int(cfg["volume_ma_window"]),
        impute_bb_position_zero=bool(cfg["impute_bb_position_zero"]),
    )

    # Forward return label
    df["target"] = (
        df.groupby("ticker")[price_col].shift(-horizon).sub(df[price_col]).div(df[price_col])
    )

    # Optionally reduce dimensionality to a subset of features
    if selected_features:
        keep = set(["ticker", "date", price_col, "target"]) | set(selected_features)
        df = df[[c for c in df.columns if c in keep]]

    # Drop rows without target
    df = df.dropna(subset=["target"])  # keep NaNs in features; models/trainer will handle
    return df


def build_xgb_features(
    prices: pd.DataFrame,
    horizon: int = 5,
    price_col: str = "close",
    *,
    feature_config: dict | None = None,
    selected_features: list[str] | None = None,
) -> pd.DataFrame:
    """Build XGB-style features for inference without dropping rows missing target.

    Keeps the most recent rows (which typically have NaN targets due to forward shift)
    so we can rank on the latest snapshot with the same feature set used in training.
    """
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    df["return_1d"] = df.groupby("ticker")[price_col].pct_change()
    cfg = {
        "rsi_period": 14,
        "bb_window": 20,
        "bb_k": 2.0,
        "volume_ma_window": 20,
        "impute_bb_position_zero": False,
    }
    if feature_config:
        try:
            cfg.update({k: v for k, v in feature_config.items() if k in cfg})
        except Exception:
            pass

    df = _compute_technical_features(
        df,
        price_col=price_col,
        rsi_period=int(cfg["rsi_period"]),
        bb_window=int(cfg["bb_window"]),
        bb_k=float(cfg["bb_k"]),
        volume_ma_window=int(cfg["volume_ma_window"]),
        impute_bb_position_zero=bool(cfg["impute_bb_position_zero"]),
    )
    df["target"] = (
        df.groupby("ticker")[price_col].shift(-horizon).sub(df[price_col]).div(df[price_col])
    )
    # Optionally reduce dimensionality to a subset of features
    if selected_features:
        keep = set(["ticker", "date", price_col, "target"]) | set(selected_features)
        df = df[[c for c in df.columns if c in keep]]

    return df


def build_lstm_sequences(
    prices: pd.DataFrame,
    lookback: int = 30,
    horizon: int = 5,
    price_col: str = "close",
    *,
    feature_config: dict | None = None,
    selected_features: list[str] | None = None,
    skip_if_nan_ratio_gt: float = 0.25,
    skip_if_low_variance_ratio_gt: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, pd.Timestamp]]]:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    # Engineer features (similar to XGB)
    df["return_1d"] = df.groupby("ticker")[price_col].pct_change()
    cfg = {
        "rsi_period": 14,
        "bb_window": 20,
        "bb_k": 2.0,
        "volume_ma_window": 20,
        "impute_bb_position_zero": False,
    }
    if feature_config:
        try:
            cfg.update({k: v for k, v in feature_config.items() if k in cfg})
        except Exception:
            pass
    df = _compute_technical_features(
        df,
        price_col=price_col,
        rsi_period=int(cfg["rsi_period"]),
        bb_window=int(cfg["bb_window"]),
        bb_k=float(cfg["bb_k"]),
        volume_ma_window=int(cfg["volume_ma_window"]),
        impute_bb_position_zero=bool(cfg["impute_bb_position_zero"]),
    )
    df["target"] = (
        df.groupby("ticker")[price_col].shift(-horizon).sub(df[price_col]).div(df[price_col])
    )

    # Select features: include OHLCV + engineered numeric
    non_feature = {"ticker", "date", "target"}
    feature_cols = [c for c in df.select_dtypes(include=["number", "bool"]).columns if c not in non_feature]
    if selected_features:
        # Keep intersection only
        feature_cols = [c for c in feature_cols if c in set(selected_features)]

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    index_info: List[Tuple[str, pd.Timestamp]] = []  # (ticker, window_end_date)
    eps = 1e-8
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        feat_vals = grp[feature_cols].to_numpy(dtype=float, copy=False)
        targets = grp["target"].to_numpy(dtype=float, copy=False)
        dates = grp["date"].to_numpy()
        n = len(grp)
        if n < lookback + horizon + 1:
            continue
        for start in range(0, n - lookback - horizon + 1):
            end = start + lookback
            window = feat_vals[start:end]
            # Per-window z-score (nan/zero-variance safe)
            # Skip windows with no finite values at all
            if not np.isfinite(window).any():
                continue
            # Skip windows with too many NaNs across features
            nan_ratio = np.isnan(window).mean()
            if nan_ratio > float(skip_if_nan_ratio_gt):
                continue
            with np.errstate(invalid='ignore', divide='ignore'):
                import warnings as _warnings
                with _warnings.catch_warnings():
                    _warnings.simplefilter('ignore', category=RuntimeWarning)
                    mean = np.nanmean(window, axis=0)
                    std = np.nanstd(window, axis=0, ddof=0)
                # Optionally skip windows where too many features are near-constant (low variance)
                low_var_mask = (std < eps) | ~np.isfinite(std)
                if low_var_mask.mean() > float(skip_if_low_variance_ratio_gt):
                    continue
                std_safe = np.where(low_var_mask, 1.0, std)
                window_centered = window - mean
                window_norm = np.divide(window_centered, std_safe, out=np.zeros_like(window_centered), where=std_safe != 0)
                # Replace any residual NaNs/Infs from all-NaN columns with zeros
                window_norm = np.nan_to_num(window_norm, nan=0.0, posinf=0.0, neginf=0.0)
            X_list.append(window_norm)
            y_list.append(targets[end])
            index_info.append((str(ticker), pd.Timestamp(dates[end])))

    if not X_list:
        return np.empty((0, lookback, len(feature_cols))), np.empty((0,)), []

    X = np.asarray(X_list)
    y = np.asarray(y_list)
    return X, y, index_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare XGB tabular dataset and LSTM sequences from stock prices")
    parser.add_argument("--prices", default=os.path.join("data", "stock_prices.csv"), help="Path to prices CSV")
    parser.add_argument("--xgb_out", default=os.path.join("data", "xgb_dataset.csv"), help="Output CSV for XGB dataset")
    parser.add_argument("--lstm_x_out", default=os.path.join("data", "lstm_X.npy"), help="Output .npy for LSTM X")
    parser.add_argument("--lstm_y_out", default=os.path.join("data", "lstm_y.npy"), help="Output .npy for LSTM y")
    parser.add_argument("--lstm_index_out", default=os.path.join("data", "lstm_index.csv"), help="Output CSV mapping sequences to (ticker,date)")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon in trading days")
    parser.add_argument("--lookback", type=int, default=30, help="Sequence length for LSTM")

    args = parser.parse_args()

    # Load prices
    prices = pd.read_csv(args.prices, parse_dates=["date"])
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Prices file missing required columns: {', '.join(sorted(missing))}")

    # XGB dataset
    xgb_df = build_xgb_dataset(prices, horizon=args.horizon)
    xgb_df.to_csv(args.xgb_out, index=False)
    print(f"[PREP] Wrote XGB dataset to: {args.xgb_out} (rows={len(xgb_df)})")

    # LSTM sequences
    X, y, index_info = build_lstm_sequences(prices, lookback=args.lookback, horizon=args.horizon)
    np.save(args.lstm_x_out, X)
    np.save(args.lstm_y_out, y)
    print(f"[PREP] Wrote LSTM arrays to: {args.lstm_x_out}, {args.lstm_y_out} (X.shape={X.shape}, y.shape={y.shape})")

    if index_info:
        idx_df = pd.DataFrame(index_info, columns=["ticker", "window_end_date"])
        idx_df.to_csv(args.lstm_index_out, index=False)
        print(f"[PREP] Wrote LSTM index to: {args.lstm_index_out}")


if __name__ == "__main__":
    main()


