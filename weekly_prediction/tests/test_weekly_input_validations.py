import pandas as pd
import numpy as np
import pytest

from utils.train_lstm_weekly import _build_latest_sequences


def test_duplicate_ticker_date_raises():
    df = pd.DataFrame({
        'ticker': ['AAA', 'AAA'],
        'date': ['2024-01-01', '2024-01-01'],
        'close': [100.0, 100.0],
        'volume': [1_000_000, 1_000_000],
        'feat1': [1.0, 1.0],
    })
    with pytest.raises(ValueError):
        _build_latest_sequences(df, ['feat1'], lookback=1)


def test_zero_variance_window_raises():
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    df = pd.DataFrame({
        'ticker': ['AAA'] * len(dates),
        'date': dates,
        'close': [100.0] * len(dates),
        'volume': [1_000_000] * len(dates),
        'feat1': [5.0] * len(dates),  # constant feature → zero variance
    })
    with pytest.raises(ValueError):
        _build_latest_sequences(df, ['feat1'], lookback=5)


