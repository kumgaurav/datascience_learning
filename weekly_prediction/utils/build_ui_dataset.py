import os
import argparse
import pandas as pd


def _apply_config_env():
    try:
        from utils.config import load_config, apply_env_from_config
        cfg_path = os.getenv('APP_CONFIG', 'config.yaml')
        cfg = load_config(cfg_path)
        apply_env_from_config(cfg)
    except Exception:
        pass


def _read_csv(path: str) -> pd.DataFrame:
    if not path or not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_ui_dataset(features_path: str, ensemble_path: str, out_path: str) -> str:
    feats = _read_csv(features_path)
    ens = _read_csv(ensemble_path)

    # Normalize ticker
    for df in (feats, ens):
        if 'ticker' in df.columns:
            df['ticker'] = df['ticker'].astype(str).str.upper()

    # Normalize ensemble columns for consistent consumption
    ens_cols_map = {
        'xgb_pred': 'xgb_predicted_return_pct',
        'lstm_pred': 'lstm_predicted_return_pct',
        'confidence_score_xgb': 'xgb_confidence_score',
        'confidence_score_lstm': 'lstm_confidence_score',
    }
    ens = ens.rename(columns=ens_cols_map)

    # Pick a subset of features relevant for UI
    keep_feats = [
        'ticker','date','close','open','high','low','volume',
        'support_20d','resistance_20d','rsi_14d',
        'momentum_5d','momentum_10d','momentum_20d','momentum_30d','momentum_60d',
        'broke_resistance','breakout_confirmed','post_earnings_dip_rally',
    ]
    feats_keep = [c for c in keep_feats if c in feats.columns]
    feats_small = feats[feats_keep].copy() if feats_keep else feats.copy()

    # Merge on ticker; take latest feature row per ticker by date
    if 'date' in feats_small.columns:
        feats_small['date'] = pd.to_datetime(feats_small['date'], errors='coerce')
        feats_latest = feats_small.sort_values('date').groupby('ticker', as_index=False).tail(1)
    else:
        feats_latest = feats_small.drop_duplicates(subset=['ticker']) if 'ticker' in feats_small.columns else feats_small

    ui = feats_latest.copy()
    if not ens.empty and 'ticker' in ens.columns:
        # Only one row per ticker in ensemble; if more, dedup by keeping best score row
        if 'ensemble_score' in ens.columns:
            ens = ens.sort_values('ensemble_score', ascending=False).drop_duplicates(subset=['ticker'])
        else:
            ens = ens.drop_duplicates(subset=['ticker'])
        ui = ui.merge(ens, on='ticker', how='left', suffixes=('', '_ens'))

    # Derive unified prediction/confidence columns for display
    def _pick(row, candidates):
        for c in candidates:
            if c in row.index and pd.notna(row[c]):
                return row[c]
        return pd.NA

    if not ui.empty:
        ui['predicted_return_pct'] = ui.apply(lambda r: _pick(r, [
            'ensemble_score', 'xgb_predicted_return_pct', 'lstm_predicted_return_pct', 'xgb_pred', 'lstm_pred'
        ]), axis=1)
        ui['confidence_score'] = ui.apply(lambda r: _pick(r, [
            'confidence_score', 'xgb_confidence_score', 'lstm_confidence_score', 'confidence_score_xgb', 'confidence_score_lstm'
        ]), axis=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ui.to_csv(out_path, index=False)
    return out_path


def main():
    _apply_config_env()
    p = argparse.ArgumentParser(description='Build unified UI dataset from features and ensemble output')
    p.add_argument('--features', default=os.getenv('PATH_FEATURES_CLEAN', 'data/features/stock_features_clean.csv'))
    p.add_argument('--ensemble', default=os.getenv('PATH_ENSEMBLE_OUT', 'data/ensemble/ensemble_weekly_output.csv'))
    p.add_argument('--out', default='data/features/ui_unified_features.csv')
    args = p.parse_args()
    out = build_ui_dataset(args.features, args.ensemble, args.out)
    print(f"Wrote unified UI dataset -> {out}")


if __name__ == '__main__':
    main()


