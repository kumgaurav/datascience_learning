import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb


def w_penalty(rsi, vr, pc, alpha=8.0, gamma=0.35, bonus=0.10, wmin=0.5, wmax=1.10):
    # pc in decimal (e.g., -0.03 = -3%)
    vr_excess = max(0.0, float(vr) - 1.0)
    drop_mag = max(0.0, -float(pc))
    w1 = float(np.exp(-alpha * vr_excess * drop_mag))
    over = max(0.0, (float(rsi) - 70.0) / 30.0)
    w2 = float(1.0 - gamma * min(1.0, over))
    w3 = 1.0 + bonus * ((45.0 - float(rsi)) / 45.0) if (float(pc) > 0 and float(rsi) < 45.0 and float(vr) <= 1.2) else 1.0
    return float(np.clip(w1 * w2 * w3, wmin, wmax))


def load_features_df():
    candidates = [
        'data/top/features_raw_full.csv',
        'data/top/xgb_features_input.csv',
        'data/top/featured_stocks_top.csv',
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=['date'])
            df['ticker'] = df['ticker'].astype(str).str.upper()
            return df
    raise FileNotFoundError("Features file not found. Checked: " + ", ".join(candidates))


def load_prices_df():
    p = 'data/stock_prices.csv'
    if not os.path.exists(p):
        raise FileNotFoundError("data/stock_prices.csv not found")
    df = pd.read_csv(p, parse_dates=['date'])
    df['ticker'] = df['ticker'].astype(str).str.upper()
    return df


def add_forward_target(prices: pd.DataFrame, horizon_days: int = 5) -> pd.DataFrame:
    prices = prices.sort_values(['ticker', 'date']).copy()
    prices['fwd_5d'] = prices.groupby('ticker')['close'].shift(-horizon_days)
    prices['target_5d_pct'] = (prices['fwd_5d'] - prices['close']) / prices['close'] * 100.0
    return prices[['ticker', 'date', 'target_5d_pct']]


def attach_xgb_pred(df: pd.DataFrame) -> pd.DataFrame:
    model_path = 'models/stock_predictor_top.joblib'
    if not os.path.exists(model_path):
        df['xgb_pred'] = 0.0
        return df
    m = joblib.load(model_path)
    try:
        fcols = m.get_booster().feature_names
    except Exception:
        drop = {'ticker', 'date', 'target', 'target_5d_pct'}
        fcols = [c for c in df.select_dtypes(include=['number', 'bool']).columns if c not in drop]
    for c in fcols:
        if c not in df.columns:
            df[c] = 0.0
    X = df[fcols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df['xgb_pred'] = m.predict(X)
    return df


def try_attach_lstm_pred(df: pd.DataFrame) -> pd.DataFrame:
    # Allow skipping LSTM to avoid TF/Metal issues on macOS by default
    if os.getenv('STACK_ADD_LSTM', '0') != '1':
        df['lstm_pred'] = np.nan
        return df
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow import keras  # type: ignore
        import json
        meta_path = 'models/top/json/lstm_model_meta.json'
        model_path = 'models/lib/keras/lstm_model.keras'
        if not (os.path.exists(model_path) and os.path.exists(meta_path)):
            df['lstm_pred'] = np.nan
            return df
        # Force CPU-only and limit threads to avoid segfaults on macOS/Metal
        try:
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        feat_cols = meta.get('feature_cols', [])
        lookback = int(meta.get('lookback', 30))
        model = keras.models.load_model(model_path)

        df = df.sort_values(['ticker', 'date']).copy()
        lstm_rows = []
        for tkr, grp in df.groupby('ticker'):
            g = grp.copy()
            for c in feat_cols:
                if c not in g.columns:
                    g[c] = 0.0
            gX = g[feat_cols].to_numpy(dtype=float, copy=False)
            pred_vals = np.full(len(g), np.nan, dtype=float)
            for end in range(lookback, len(gX) + 1):
                window = gX[end - lookback:end]
                X_seq = window.reshape((1, window.shape[0], window.shape[1]))
                try:
                    pred_vals[end - 1] = float(model.predict(X_seq, verbose=0).ravel()[0])
                except Exception:
                    pred_vals[end - 1] = np.nan
            g['lstm_pred'] = pred_vals
            lstm_rows.append(g[['ticker', 'date', 'lstm_pred']])
        lstm_df = pd.concat(lstm_rows, ignore_index=True)
        df = df.merge(lstm_df, on=['ticker', 'date'], how='left')
        return df
    except Exception:
        df['lstm_pred'] = np.nan
        return df


def main():
    feat = load_features_df()
    prices = load_prices_df()
    target = add_forward_target(prices, horizon_days=int(os.getenv('STACK_HORIZON_DAYS', '5')))
    df = feat.merge(target, on=['ticker', 'date'], how='inner').dropna(subset=['target_5d_pct'])

    # Attach model-derived signals
    df = attach_xgb_pred(df)
    df = try_attach_lstm_pred(df)

    # Penalty weight (continuous) – convert price_change_pct to decimal if needed
    rsi = pd.to_numeric(df.get('rsi_14d', 50.0), errors='coerce').fillna(50.0)
    vr = pd.to_numeric(df.get('volume_ratio', 1.0), errors='coerce').fillna(1.0)
    pc_raw = pd.to_numeric(df.get('price_change_pct', 0.0), errors='coerce').fillna(0.0)
    pc_dec = np.where(pc_raw.abs().median() > 1.0, pc_raw / 100.0, pc_raw)
    df['w_pen'] = [
        w_penalty(
            a,
            b,
            c,
            alpha=float(os.getenv('PEN_ALPHA', '8.0')),
            gamma=float(os.getenv('RSI_GAMMA', '0.35')),
            bonus=float(os.getenv('RSI_BONUS', '0.10')),
            wmin=float(os.getenv('PEN_W_MIN', '0.50')),
            wmax=float(os.getenv('PEN_W_MAX', '1.10')),
        )
        for a, b, c in zip(rsi, vr, pc_dec)
    ]

    # Ensure engineered features exist
    for col, default in [
        ('overbought_spike', 0),
        ('vol_change_pct', 0.0),
        ('rsi_price_interaction', rsi * pc_dec),
        ('volratio_price_interaction', vr * pc_dec),
        ('selloff_flag', 0),
    ]:
        if col not in df.columns:
            df[col] = default

    meta_cols = [
        'xgb_pred',
        'lstm_pred',
        'w_pen',
        'rsi_14d',
        'overbought_spike',
        'volume_ratio',
        'price_change_pct',
        'vol_change_pct',
        'rsi_price_interaction',
        'volratio_price_interaction',
        'selloff_flag',
    ]
    for c in ['confidence_score', 'confidence_score_xgb', 'confidence_score_lstm']:
        if c in df.columns and c not in meta_cols:
            meta_cols.append(c)

    for c in meta_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Train window
    recent_days = int(os.getenv('STACK_RECENT_DAYS', '180'))
    cutoff = df['date'].max() - pd.Timedelta(days=recent_days)
    train_df = df[df['date'] >= cutoff].copy()
    X = train_df[meta_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = pd.to_numeric(train_df['target_5d_pct'], errors='coerce').fillna(0.0)

    stack = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=int(os.getenv('STACK_N_ESTIMATORS', '800')),
        max_depth=int(os.getenv('STACK_MAX_DEPTH', '4')),
        learning_rate=float(os.getenv('STACK_LR', '0.05')),
        subsample=float(os.getenv('STACK_SUBSAMPLE', '0.8')),
        colsample_bytree=float(os.getenv('STACK_COLSAMPLE', '0.8')),
        random_state=42,
    )
    stack.fit(X, y)

    out_dir = os.getenv('STACK_OUT_DIR', 'models/stack')
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'stacker.joblib')
    meta_info = {
        'meta_cols': meta_cols,
        'horizon_days': int(os.getenv('STACK_HORIZON_DAYS', '5')),
        'recent_days': recent_days,
    }
    joblib.dump({'model': stack, 'meta': meta_info}, model_path)
    print(f"Saved {model_path} with features: {meta_cols}")
    # Convenience: also copy into models directory root for discovery
    try:
        root_path = os.path.join('models', 'stacker.joblib')
        joblib.dump({'model': stack, 'meta': meta_info}, root_path)
        print(f"Saved {root_path} for pipeline convenience.")
    except Exception:
        pass


if __name__ == "__main__":
    main()


