import os
import numpy as np
import pandas as pd

from models.lstm.lstm_trainer import LSTMTrainer


def _make_synthetic_prices(num_days: int = 10, tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ['AAA', 'BBB']
    dates = pd.date_range('2024-01-01', periods=num_days, freq='D')
    rows = []
    for t in tickers:
        base = np.linspace(100.0, 110.0, num_days)
        noise = np.random.normal(scale=0.1, size=num_days)
        close = base + noise
        volume = np.random.randint(1_000_000, 1_500_000, size=num_days).astype(float)
        for d, c, v in zip(dates, close, volume):
            rows.append({'ticker': t, 'date': d, 'close': float(c), 'volume': float(v)})
    df = pd.DataFrame(rows)
    return df


def test_lstm_trains_with_penalty_weights_multi_ticker(tmp_path):
    df = _make_synthetic_prices(num_days=12, tickers=['AAA', 'BBB', 'CCC'])
    csv_path = tmp_path / 'features.csv'
    df.to_csv(csv_path, index=False)

    # Small model configuration for speed; enable penalty weights path
    model_config = {
        'lookback': 5,
        'horizon': 1,
        'epochs': 1,
        'batch_size': 8,
        'use_vectorized_windows': True,
        'use_sequence_weights': True,
        'use_penalty_weights': True,
        'penalty_weight_mode': 'multiply',
        'tensorboard': False,
    }

    trainer = LSTMTrainer(features_path=str(csv_path), model_config=model_config)
    model, preds = trainer.train(lookback=5, horizon=1)

    assert model is not None
    # preds can be empty if test split is empty; ensure call succeeds and returns an array
    assert hasattr(preds, 'shape')


