import numpy as np
import pandas as pd

from utils.feature_engineering import _normalize_prices_columns, _compute_features


def test_normalize_empty_df_returns_empty():
    df = pd.DataFrame()
    out = _normalize_prices_columns(df)
    assert out.empty


def test_normalize_missing_required_columns_raises():
    df = pd.DataFrame({
        'date': ['2025-01-01'],
        'close': [100.0],
        'volume': [1000],
    })
    try:
        _ = _normalize_prices_columns(df)
        assert False, "Expected ValueError for missing ticker column"
    except ValueError as e:
        assert 'Missing required columns' in str(e)


def test_normalize_maps_columns_and_parses_types():
    df = pd.DataFrame({
        'Symbol': ['aapl', 'msft'],
        'Trade_Date': ['2025-01-01', '2025-01-02'],
        'Open': [100, 102],
        'High': [105, 104],
        'Low': [99, 101],
        'ClosePrice': [104.5, 103.5],
        'Vol': [1_000_000, 800_000],
    })
    out = _normalize_prices_columns(df)
    assert set(['ticker','date','open','high','low','close','volume']).issubset(set(out.columns))
    assert out['ticker'].tolist() == ['AAPL', 'MSFT']
    assert pd.api.types.is_datetime64_any_dtype(out['date'])


def test_compute_features_empty_returns_empty():
    df = pd.DataFrame()
    out = _compute_features(df)
    assert out.empty


def test_compute_features_basic_no_nans_in_features():
    # Build minimal valid price history for one ticker
    n = 30
    dates = pd.date_range('2025-01-01', periods=n, freq='D')
    price = np.linspace(100, 120, n)
    volume = np.linspace(1_000_000, 1_500_000, n)
    df = pd.DataFrame({
        'ticker': ['TEST'] * n,
        'date': dates,
        'open': price * 0.99,
        'high': price * 1.01,
        'low': price * 0.98,
        'close': price,
        'volume': volume,
    })

    out = _compute_features(df)
    # Ensure key feature columns exist
    expected_cols = [
        'ticker','date','open','high','low','close','volume',
        'ma_20','ma_50','rsi_14d','macd','bb_upper','volume_ratio',
        'trend_slope_15d','up_day_ratio_20d','price_change_pct','prev50d_high'
    ]
    for c in expected_cols:
        assert c in out.columns

    # Numeric feature columns (excluding identity columns) should not contain NaN or inf after fill
    base_cols = {'ticker','date','open','high','low','close','volume'}
    num_cols = [c for c in out.columns if c not in base_cols and pd.api.types.is_numeric_dtype(out[c])]
    for c in num_cols:
        s = pd.to_numeric(out[c], errors='coerce')
        assert np.isfinite(s).all(), f"Column {c} contains non-finite values"


