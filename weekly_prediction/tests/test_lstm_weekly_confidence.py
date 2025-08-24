import json
import numpy as np
import pandas as pd

from utils.train_lstm_weekly import train_and_export


class DummyModel:
    def predict(self, X, verbose=0):
        # Use the last timestep of the first feature to ensure variability across tickers
        X = np.asarray(X)
        vals = X[:, -1, 0].astype(float)
        return vals.reshape(-1, 1)


def test_weekly_confidence_not_all_zero_with_csz_generation(tmp_path, monkeypatch):
    # Create synthetic features for two tickers over shared dates
    dates = pd.date_range('2024-01-01', periods=8, freq='D')
    rows = []
    rng = np.random.default_rng(42)
    # Three tickers with distinct patterns to avoid symmetric cross-sectional z-scores
    for tkr, base, slope, rsi_start, rsi_end, vr_start, vr_end, noise_scale in [
        ('AAA', 100.0, +2.0, 35, 65, 0.7, 1.3, 0.05),
        ('BBB', 105.0, -1.5, 65, 35, 1.3, 0.7, 0.02),
        ('CCC', 98.0, 0.0, 50, 55, 1.0, 1.1, 0.10),
    ]:
        close = base + slope * np.linspace(0, 1.0, len(dates)) + rng.normal(0, noise_scale, len(dates))
        volume = np.linspace(1_000_000, 1_200_000, len(dates))
        rsi = np.linspace(rsi_start, rsi_end, len(dates)) + rng.normal(0, 0.5, len(dates))
        vol_ratio = np.linspace(vr_start, vr_end, len(dates)) + rng.normal(0, 0.03, len(dates))
        pc = np.gradient(close) / np.maximum(close[:-1].mean() if len(close) > 1 else 100.0, 1.0)
        pc = np.pad(pc, (0, max(0, len(dates) - len(pc))), mode='edge')
        for d, c, v, r, vr, p in zip(dates, close, volume, rsi, vol_ratio, pc):
            rows.append({
                'ticker': tkr,
                'date': d,
                'close': float(c),
                'volume': float(v),
                'rsi_14d': float(r),
                'volume_ratio': float(vr),
                'price_change_pct': float(p),
            })
    df = pd.DataFrame(rows)
    feats_p = tmp_path / 'feats.csv'
    df.to_csv(feats_p, index=False)

    # Meta expecting *_csz features (which are missing in df initially)
    meta = {
        'feature_cols': ['close_csz', 'rsi_14d_csz'],
        'lookback': 5,
    }
    meta_p = tmp_path / 'meta.json'
    meta_p.write_text(json.dumps(meta))

    # Dummy model path just to pass existence checks
    model_p = tmp_path / 'model.keras'
    model_p.write_text('dummy')

    # Monkeypatch loader to return a dummy model with variable predictions
    monkeypatch.setattr('tensorflow.keras.models.load_model', lambda _: DummyModel(), raising=False)

    out_p = tmp_path / 'out.csv'
    train_and_export(
        features_path=str(feats_p),
        out_path=str(out_p),
        lookback=5,
        skip_train=True,
        model_path=str(model_p),
        meta_path=str(meta_p),
    )

    out_df = pd.read_csv(out_p)
    assert 'confidence_score' in out_df.columns
    # Ensure not all zeros and at least one positive
    cs = np.asarray(out_df['confidence_score'], dtype=float)
    assert np.nanmax(cs) > 0.0
    assert not np.allclose(cs, 0.0)


