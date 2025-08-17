from typing import List
# stock_selector_v2.py
import pandas as pd
import numpy as np
import os
import joblib
from stock_selector import calculate_confidence_score, calculate_risk_score

class StockSelector:
    def __init__(self, model, features_df, horizon=5, feature_cols=None, lstm_lookback=20):
        """
        model: trained model (XGB, LSTM, or ensemble)
        features_df: full dataframe with features (per ticker/date)
        horizon: prediction horizon in days
        feature_cols: explicit feature list to use for inference
        lstm_lookback: number of timesteps to construct per-ticker sequence for LSTM
        """
        self.model = model
        self.df = features_df.copy()
        self.horizon = horizon
        self.feature_cols = feature_cols
        self.lstm_lookback = lstm_lookback

    def _prepare_latest(self):
        """Take the most recent feature row per ticker"""
        latest = self.df.sort_values("date").groupby("ticker").tail(1)
        return latest

    def _predict_tabular(self, latest: pd.DataFrame, feature_cols: List[str]):
        """Predict returns for tabular models (e.g., XGB)."""
        X = latest[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        preds = self.model.predict(X)
        return preds

    def _is_lstm_model(self) -> bool:
        """Heuristic to detect a Keras LSTM model without hard TF dependency here."""
        try:
            layers = getattr(self.model, "layers", None)
            if layers is None:
                return False
            for layer in layers:
                if getattr(layer.__class__, "__name__", "").upper() == "LSTM":
                    return True
        except Exception:
            return False
        return False

    def rank_stocks(self, top_n=20, explain=False):
        """
        Rank stocks by confidence score which combines predicted return and positive signal count.
        If the model is an LSTM, builds per-ticker sequences of length `lstm_lookback` from the full df.
        """
        latest = self._prepare_latest()

        # Diagnostics: show columns present vs expected
        print(f"[SELECTOR DIAG] Latest shape: {latest.shape}")
        print(f"[SELECTOR DIAG] Latest columns: {list(latest.columns)}")

        # Determine feature columns for inference
        if self.feature_cols is not None:
            feature_cols = [c for c in self.feature_cols if c in latest.columns]
        else:
            exclude = ["ticker", "date", "close", "open", "high", "low", "volume", "target"]
            feature_cols = [c for c in latest.columns if c not in exclude]

        print(f"[SELECTOR DIAG] Using feature columns: {feature_cols}")

        # Predict using appropriate path
        if self._is_lstm_model():
            tickers, X_seq = [], []
            for ticker, g in self.df.groupby("ticker"):
                g = g.sort_values("date").tail(self.lstm_lookback)
                if len(g) < self.lstm_lookback:
                    continue
                use_cols = [c for c in feature_cols if c in g.columns]
                missing = [c for c in feature_cols if c not in g.columns]
                if missing:
                    print(f"[SELECTOR DIAG] Missing LSTM cols for {ticker}: {missing}")
                if not use_cols:
                    continue
                X_seq.append(g[use_cols].values)
                tickers.append(ticker)
            if not X_seq:
                return latest.head(0)
            X_arr = np.array(X_seq)
            preds = self.model.predict(X_arr).ravel()
            pred_df = pd.DataFrame({"ticker": tickers, "pred_return_pct": preds})
            latest = latest.merge(pred_df, on="ticker", how="inner")
        else:
            preds = self._predict_tabular(latest, feature_cols)
            latest["pred_return_pct"] = preds

        # Apply bullish confidence filter
        bullish_filter = (
            (latest.get("broke_resistance", False).astype(bool)) |
            (latest.get("post_earnings_dip_rally", False).astype(bool))
        )
        filtered = latest[bullish_filter].copy()

        if filtered.empty:
            print("⚠️ No stocks passed bullish filter, falling back to top N by prediction.")
            filtered = latest

        # Compute signal count and confidence score for ranking
        pos_signals = [
            "broke_resistance", "breakout_confirmed", "strong_momentum", "post_earnings_dip_rally"
        ]
        for col in pos_signals:
            if col not in filtered.columns:
                filtered.loc[:, col] = False
        filtered.loc[:, "signal_pos_count"] = filtered[pos_signals].astype(int).sum(axis=1)

        # Add momentum-based confidence from 5d/10d/20d trends (returns preferred, else momentum aliases)
        momentum_sources = {
            "m5": ["return_5d", "momentum_5d"],
            "m10": ["return_10d", "momentum_10d"],
            "m20": ["return_20d", "momentum_20d"],
        }
        def _pick_first_present(row, candidates):
            for c in candidates:
                if c in row.index and pd.notna(row.get(c)):
                    return row.get(c)
            return 0
        m5 = filtered.apply(lambda r: _pick_first_present(r, momentum_sources["m5"]), axis=1)
        m10 = filtered.apply(lambda r: _pick_first_present(r, momentum_sources["m10"]), axis=1)
        m20 = filtered.apply(lambda r: _pick_first_present(r, momentum_sources["m20"]), axis=1)
        # Count positive momentum windows
        filtered.loc[:, "momentum_pos_count"] = (
            (m5 > 0).astype(int) + (m10 > 0).astype(int) + (m20 > 0).astype(int)
        )
        # Combine into confidence: predicted return scaled by total positive signals
        filtered.loc[:, "confidence_score"] = filtered["pred_return_pct"] * (
            1 + filtered["signal_pos_count"] + filtered["momentum_pos_count"]
        )

        ranked = filtered.sort_values("confidence_score", ascending=False).head(top_n)

        # Optional SHAP explainability for XGB-like models
        if explain:
            try:
                booster = getattr(self.model, "get_booster", None)
                if booster is not None:
                    import shap
                    Xr = ranked[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                    explainer = shap.TreeExplainer(self.model)
                    shap_values = explainer.shap_values(Xr)
                    abs_vals = np.abs(shap_values)
                    top_idx = abs_vals.argmax(axis=1)
                    ranked["top_feature"] = [feature_cols[i] for i in top_idx]
            except Exception:
                pass

        return ranked[["ticker", "pred_return_pct", "confidence_score"] + feature_cols]


# --- Compatibility helpers for UI (function-style API) ---
def get_top_stocks(n: int = 20, min_confidence: int = 30, max_risk: int = 70, diversify: bool = True, verbose: bool = False, bullish_only: bool = True) -> pd.DataFrame:
    """
    Wrapper to produce top-N stocks using the enhanced selector.
    Reads model from models/stock_predictor_top.joblib and features from data/featured_stocks_top.csv.
    Adds predicted_change, risk_score, composite_score for UI compatibility.
    """
    model_path = os.path.join('models', 'stock_predictor_top.joblib')
    # Use XGB inference features for prediction alignment; use featured CSV for display/extras
    pred_path = os.getenv('XGB_FEATURES_CSV', 'data/xgb_features_latest.csv')
    disp_path = os.getenv('FEATURED_STOCKS_CSV', 'data/featured_stocks_top.csv')
    try:
        model = joblib.load(model_path)
        # Prediction DataFrame (for model inference)
        if os.path.exists(pred_path):
            pred_df = pd.read_csv(pred_path)
        else:
            pred_df = pd.read_csv(disp_path)
        # Display DataFrame (for UI fields like close/signals)
        try:
            disp_df = pd.read_csv(disp_path)
        except Exception:
            disp_df = pred_df.copy()
    except Exception as e:
        print(f"[SELECTOR_V2] Failed to load model or features: {e}")
        return pd.DataFrame()

    # Determine model feature columns
    try:
        feature_cols = model.get_booster().feature_names
    except Exception:
        # Fallback: infer numeric/bool excluding non-features
        exclude = {"ticker", "date", "target", "open", "high", "low", "close", "volume"}
        feature_cols = [c for c in pred_df.select_dtypes(include=['number', 'bool']).columns if c not in exclude]

    # Ensure all model feature columns exist; add missing as zeros for alignment
    missing = [c for c in feature_cols if c not in pred_df.columns]
    if missing:
        for col in missing:
            pred_df[col] = 0.0

    selector = StockSelector(model, pred_df, feature_cols=feature_cols)
    ranked = selector.rank_stocks(top_n=n)
    if ranked.empty:
        return ranked

    # Ensure consistent naming
    ranked = ranked.rename(columns={"pred_return_pct": "predicted_return_pct"})

    # Merge UI/display fields from featured CSV (latest per ticker)
    disp_latest = disp_df.sort_values('date').groupby('ticker').tail(1)
    ui_cols = [
        'ticker', 'date', 'close', 'open', 'high', 'low', 'volume',
        'broke_resistance', 'post_earnings_dip_rally', 'strong_momentum', 'breakout_confirmed',
        'golden_cross', 'trend_slope_15d', 'trend_slope_30d', 'rsi_14d', 'volume_ratio', 'volatility_30d',
        'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_30d', 'momentum_60d', 'up_day_ratio_20d',
        'momentum_winner', 'trend_persistence_20d', 'earnings_in_3_weeks', 'last_2q_positive_surprises',
        'profit_margin', 'long_term_growth_rate', 'sector', 'industry'
    ]
    merge_cols = [c for c in ui_cols if c in disp_latest.columns]
    ranked = ranked.merge(disp_latest[merge_cols], on='ticker', how='left')
    # Price-based convenience
    if 'close' in ranked.columns:
        ranked['predicted_change'] = (ranked['predicted_return_pct'] / 100.0) * ranked['close']
    else:
        ranked['predicted_change'] = 0.0

    # Add risk and composite scores
    try:
        ranked['risk_score'] = ranked.apply(calculate_risk_score, axis=1)
    except Exception:
        ranked['risk_score'] = 50.0
    try:
        # Keep the confidence score computed inside ranker if present; otherwise compute via helper
        if 'confidence_score' not in ranked.columns:
            ranked['confidence_score'] = ranked.apply(calculate_confidence_score, axis=1)
    except Exception:
        ranked['confidence_score'] = 50.0

    ranked['composite_score'] = (
        ranked['predicted_return_pct'] * 0.5 +
        ranked['confidence_score'] * 0.3 +
        (100 - ranked['risk_score']) * 0.2
    )

    # Apply UI thresholds; keep ability to backfill to N
    filt = (
        (ranked['confidence_score'] >= min_confidence) &
        (ranked['risk_score'] <= max_risk)
    )
    filtered = ranked[filt].copy()
    filtered = filtered.sort_values('composite_score', ascending=False).drop_duplicates('ticker')
    if len(filtered) < n:
        remaining = ranked[~ranked['ticker'].isin(filtered['ticker'])].sort_values('predicted_return_pct', ascending=False)
        need = n - len(filtered)
        filtered = pd.concat([filtered, remaining.head(need)], ignore_index=True)
    return filtered.head(n)


def get_stock_analysis(ticker: str):
    """Compatibility helper to fetch per-ticker feature row for analysis panels."""
    feature_path = os.getenv('FEATURED_STOCKS_CSV', 'data/featured_stocks_top.csv')
    try:
        features_df = pd.read_csv(feature_path)
        row = features_df.sort_values('date').groupby('ticker').tail(1)
        row = row[row['ticker'] == ticker]
        if row.empty:
            return None
        return row.iloc[0]
    except Exception as e:
        print(f"[SELECTOR_V2] get_stock_analysis failed for {ticker}: {e}")
        return None
