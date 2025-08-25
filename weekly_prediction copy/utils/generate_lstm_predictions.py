import os
import argparse
import json
import numpy as np
import pandas as pd


def load_lstm(model_path: str, meta_path: str):
    try:
        from tensorflow import keras
    except Exception as e:
        raise RuntimeError(f"TensorFlow/Keras not available: {e}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model not found at {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"LSTM meta not found at {meta_path}")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    model = keras.models.load_model(model_path)
    feature_cols = meta.get('feature_cols')
    lookback = int(meta.get('lookback', 30))
    return model, feature_cols, lookback


def build_sequences_for_latest(features_df: pd.DataFrame, feature_cols: list[str], lookback: int) -> dict[str, np.ndarray]:
    seq_map: dict[str, np.ndarray] = {}
    if 'ticker' not in features_df.columns or 'date' not in features_df.columns:
        raise ValueError("Features CSV must contain 'ticker' and 'date' columns")
    features_df = features_df.copy()
    features_df['ticker'] = features_df['ticker'].astype(str).str.upper()
    # Ensure proper date sorting
    try:
        features_df['date'] = pd.to_datetime(features_df['date'])
    except Exception:
        pass
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

    skipped = []
    eps = 1e-8
    for ticker, grp in features_df.groupby('ticker'):
        g = grp.sort_values('date')
        if len(g) < lookback:
            skipped.append((str(ticker), len(g)))
            continue
        window = g[feature_cols].tail(lookback).to_numpy(dtype=float)
        # Per-window z-score normalization (match training logic)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.nanmean(window, axis=0)
            std = np.nanstd(window, axis=0, ddof=0)
            low_var_mask = (std < eps) | ~np.isfinite(std)
            std_safe = np.where(low_var_mask, 1.0, std)
            centered = window - mean
            normed = np.divide(centered, std_safe, out=np.zeros_like(centered), where=std_safe != 0)
            normed = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)
        seq_map[ticker] = normed.reshape((1, normed.shape[0], normed.shape[1]))
    if skipped:
        print(f"[LSTM WARN] Skipped {len(skipped)} tickers lacking lookback history: "
              f"{', '.join([f'{t}({n})' for t, n in skipped][:20])}{'...' if len(skipped) > 20 else ''}")
    return seq_map


def main():
    parser = argparse.ArgumentParser(description="Generate LSTM weekly return predictions and save to data/top")
    parser.add_argument('--features', type=str, default=os.getenv('ENSEMBLE_LSTM_FEATURES_CSV', 'data/top/xgb_features_latest.csv'),
                        help='CSV with historical features including date/ticker and LSTM feature columns')
    parser.add_argument('--model', type=str, default='models/lib/keras/lstm_model.keras', help='Path to saved LSTM model')
    parser.add_argument('--meta', type=str, default='models/top/json/lstm_model_meta.json', help='Path to LSTM metadata JSON')
    parser.add_argument('--output', type=str, default='data/top/lstm_weekly_predictions.csv', help='Output CSV path')
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    model, feature_cols, lookback = load_lstm(args.model, args.meta)

    seq_map = build_sequences_for_latest(df, feature_cols, lookback)
    if not seq_map:
        raise RuntimeError("No valid sequences built. Ensure features have enough history per ticker and correct columns.")

    # Batch prediction for efficiency
    tickers = list(seq_map.keys())
    X_batch = np.concatenate([seq_map[t] for t in tickers], axis=0) if tickers else np.empty((0,))
    preds = {}
    if len(tickers) == 0:
        raise RuntimeError("No sequences to predict.")
    try:
        y_pred = model.predict(X_batch, verbose=0).ravel()
        for t, y in zip(tickers, y_pred):
            try:
                preds[t] = float(y)
            except Exception:
                preds[t] = np.nan
    except Exception as e:
        print(f"[LSTM ERROR] Batch prediction failed: {e}. Attempting per-ticker fallback...")
        for tkr, X_seq in seq_map.items():
            try:
                y = float(model.predict(X_seq, verbose=0).ravel()[0])
                preds[tkr] = y
            except Exception as _e:
                print(f"[LSTM WARN] Prediction failed for {tkr}: {_e}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df = pd.DataFrame({'ticker': list(preds.keys()), 'lstm_predicted_return_pct': list(preds.values())})
    out_df.to_csv(args.output, index=False)
    print(f"[LSTM STEP] Wrote predictions to {args.output} (rows={len(out_df)})")


if __name__ == '__main__':
    main()


