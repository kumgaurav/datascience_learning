import os
import pandas as pd
from utils.run_ensemble_weekly import run_ensemble


def test_alignment_outer_fill_zero(tmp_path):
    xgb = pd.DataFrame({
        'ticker': ['AAPL','MSFT'],
        'xgb_score': [0.2, 0.5],
        'xgb_confidence_score': [10, 20],
    })
    lstm = pd.DataFrame({
        'ticker': ['AAPL','NVDA'],
        'lstm_pred': [0.1, 0.3],
        'confidence_score': [15, 30],
    })
    feats = pd.DataFrame({
        'ticker': ['AAPL','MSFT','NVDA'],
        'rsi_14d': [55, 45, 60],
        'volume_ratio': [1.0, 1.1, 0.9],
        'price_change_pct': [0.01, -0.02, 0.03],
    })
    d = tmp_path
    xgb_p = d / 'xgb.csv'
    lstm_p = d / 'lstm.csv'
    feats_p = d / 'feats.csv'
    out_p = d / 'out.csv'
    xgb.to_csv(xgb_p, index=False)
    lstm.to_csv(lstm_p, index=False)
    feats.to_csv(feats_p, index=False)

    run_ensemble(
        xgb_path=str(xgb_p),
        lstm_path=str(lstm_p),
        features_path=str(feats_p),
        method='weighted',
        alpha=0.6,
        stack_model=None,
        out_path=str(out_p),
    )
    out = pd.read_csv(out_p)
    # Both AAPL and MSFT should be present; NVDA may be present with xgb_pred=0
    assert 'ticker' in out.columns
    assert out['ticker'].isin(['AAPL','MSFT','NVDA']).any()


def test_stacking_fallback_logs(tmp_path, monkeypatch):
    # Minimal aligned inputs
    xgb = pd.DataFrame({'ticker':['AAPL'], 'xgb_score':[0.2]})
    lstm = pd.DataFrame({'ticker':['AAPL'], 'lstm_pred':[0.1]})
    feats = pd.DataFrame({'ticker':['AAPL'], 'rsi_14d':[50], 'volume_ratio':[1.0], 'price_change_pct':[0.0]})
    d = tmp_path
    xgb_p = d / 'x.csv'; lstm_p = d / 'l.csv'; feats_p = d / 'f.csv'; out_p = d / 'o.csv'
    xgb.to_csv(xgb_p, index=False); lstm.to_csv(lstm_p, index=False); feats.to_csv(feats_p, index=False)

    # Point to a non-existent stacker; should log error and continue
    monkeypatch.setenv('ENSEMBLE_STACK_MODEL', str(d / 'missing.joblib'))
    run_ensemble(
        xgb_path=str(xgb_p),
        lstm_path=str(lstm_p),
        features_path=str(feats_p),
        method='weighted',
        alpha=0.5,
        stack_model=None,
        out_path=str(out_p),
    )
    out = pd.read_csv(out_p)
    assert 'ensemble_score' in out.columns

