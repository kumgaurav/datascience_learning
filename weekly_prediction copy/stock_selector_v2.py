from typing import List, Optional, Dict, Any
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
        try:
            print(f"[SELECTOR INIT] model={type(model).__name__}, rows={len(self.df)}, cols={len(self.df.columns)}, horizon={horizon}")
        except Exception:
            pass

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
        try:
            print(f"[SELECTOR DIAG] Latest columns sample: {list(latest.columns)[:8]} ...")
        except Exception:
            pass

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

        # Include all stocks for ranking (filter only at UI later)
        filtered = latest.copy()

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
        # Build realized momentum rank scores across the full universe and persist to data/top
        latest = latest.copy()
        latest.loc[:, 'ticker'] = latest['ticker'].astype(str).str.upper()
        filtered.loc[:, 'ticker'] = filtered['ticker'].astype(str).str.upper()

        def _pick_first_present_col(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
            vals = pd.Series([None] * len(df), index=df.index, dtype='float64')
            for c in candidates:
                if c in df.columns:
                    s = pd.to_numeric(df[c], errors='coerce')
                    vals = vals.where(~vals.isna(), s)
            vals = vals.fillna(0.0)
            return vals

        # Candidate columns for realized returns
        s5_raw = _pick_first_present_col(latest, [
            'price_change_5d_pct', 'return_5d', 'momentum_5d'
        ])
        s15_raw = _pick_first_present_col(latest, [
            'price_change_15d_pct', 'return_15d', 'momentum_15d'
        ])
        s30_raw = _pick_first_present_col(latest, [
            'price_change_30d_pct', 'return_30d', 'momentum_30d'
        ])

        N = max(len(latest), 1)
        rank_5 = s5_raw.rank(method='min', ascending=False)
        rank_15 = s15_raw.rank(method='min', ascending=False)
        rank_30 = s30_raw.rank(method='min', ascending=False)
        score_5 = 1.0 - (rank_5 - 1.0) / float(N)
        score_15 = 1.0 - (rank_15 - 1.0) / float(N)
        score_30 = 1.0 - (rank_30 - 1.0) / float(N)

        recent_return_score_series = 0.4 * score_5 + 0.3 * score_15 + 0.3 * score_30

        # Persist scores for transparency and reuse
        try:
            out_dir = os.path.join('data', 'top')
            os.makedirs(out_dir, exist_ok=True)
            out_df = pd.DataFrame({
                'ticker': latest['ticker'],
                'rank_score_5d': score_5.astype(float).round(6),
                'rank_score_15d': score_15.astype(float).round(6),
                'rank_score_30d': score_30.astype(float).round(6),
                'recent_return_score': recent_return_score_series.astype(float).round(6),
                'value_5d': s5_raw.astype(float).round(6),
                'value_15d': s15_raw.astype(float).round(6),
                'value_30d': s30_raw.astype(float).round(6),
            })
            out_df = out_df.drop_duplicates('ticker')
            out_df.to_csv(os.path.join(out_dir, 'realized_rank_scores.csv'), index=False)
        except Exception:
            pass

        # Attach to filtered for scoring
        scores_map = dict(zip(latest['ticker'], recent_return_score_series))
        filtered.loc[:, 'recent_return_score'] = filtered['ticker'].map(lambda t: scores_map.get(t, 0.0))

        # Confidence score
        filtered.loc[:, "confidence_score"] = (
            filtered["pred_return_pct"]
            * (1 + 0.5 * filtered["signal_pos_count"] + 0.5 * filtered["momentum_pos_count"])
            * (1 + filtered['recent_return_score'])
        )

        # Apply dynamic selloff penalty to confidence (deprioritize names with sharp sell-off)
        try:
            if 'selloff_flag' not in filtered.columns:
                filtered.loc[:, 'selloff_flag'] = 0
            import os as _os
            vr_min = float(_os.getenv('SELLOFF_VRATIO_MIN', '1.2'))
            vr_ref = float(_os.getenv('SELLOFF_VRATIO_REF', '2.5'))
            drop_ref = float(_os.getenv('SELLOFF_REF_DROP_PCT', '0.05'))
            f_min = float(_os.getenv('SELLOFF_FACTOR_MIN', '0.60'))
            f_max = float(_os.getenv('SELLOFF_FACTOR_MAX', '0.75'))
            vr = pd.to_numeric(filtered.get('volume_ratio', 0.0), errors='coerce').fillna(0.0)
            pc = pd.to_numeric(filtered.get('price_change_pct', filtered.get('return_1d', 0.0)), errors='coerce').fillna(0.0)
            so = filtered['selloff_flag'].astype(float).fillna(0.0) >= 1.0
            def _factor_row(v, p, s):
                if bool(s) and (v > vr_min) and (p < 0.0):
                    sv = max(0.0, min(1.0, (v - vr_min) / max(1e-6, (vr_ref - vr_min))))
                    sd = max(0.0, min(1.0, (-p) / max(1e-6, drop_ref)))
                    sev = 0.5 * sv + 0.5 * sd
                    return float(f_max - (f_max - f_min) * sev)
                return 1.0
            factors = pd.Series([_factor_row(float(v), float(p), bool(s)) for v, p, s in zip(vr, pc, so)], index=filtered.index, dtype='float64')
            filtered.loc[:, 'confidence_score'] = pd.to_numeric(filtered['confidence_score'], errors='coerce') * factors
            # Also scale model predicted return to propagate penalty into ranked outputs
            if 'pred_return_pct' in filtered.columns:
                filtered.loc[:, 'pred_return_pct'] = pd.to_numeric(filtered['pred_return_pct'], errors='coerce') * factors
            if _os.getenv('DEBUG_SELLOFF', '0') == '1':
                try:
                    dbg_cols = ['ticker','confidence_score','pred_return_pct','selloff_flag','volume_ratio','price_change_pct']
                    dbg_df = filtered[[c for c in dbg_cols if c in filtered.columns]].copy()
                    dbg_df = dbg_df.sort_values('confidence_score', ascending=False).head(50)
                    import os
                    os.makedirs('data/top', exist_ok=True)
                    dbg_df.to_csv('data/top/debug_selector_penalties.csv', index=False)
                    print('[DEBUG] Wrote selector penalty debug to data/top/debug_selector_penalties.csv')
                except Exception:
                    pass
        except Exception:
            # Best-effort; do not fail ranking if penalty application errors
            pass

        # Compute risk for UI and finalize ranking
        try:
            filtered.loc[:, "risk_score"] = filtered.apply(calculate_risk_score, axis=1)
        except Exception:
            filtered.loc[:, "risk_score"] = 50.0

        ranked = filtered.sort_values("confidence_score", ascending=False)
        if top_n is not None:
            ranked = ranked.head(top_n)

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

        # Minimal UI-ready columns plus signals and computed fields
        extra_cols = [
            "signal_pos_count", "momentum_pos_count", "risk_score"
        ]
        keep_cols = [c for c in extra_cols if c in ranked.columns]
        base_cols = ["ticker", "pred_return_pct", "confidence_score"] + keep_cols
        # Avoid duplicate columns when model feature list already contains computed fields
        feature_cols_extra = [c for c in feature_cols if c not in base_cols]
        final_cols = base_cols + feature_cols_extra
        # Safety: enforce uniqueness order-preserving
        seen = set()
        final_cols_unique = []
        for c in final_cols:
            if c not in seen:
                final_cols_unique.append(c)
                seen.add(c)
        return ranked[final_cols_unique]


# --- Compatibility helpers for UI (function-style API) ---
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
_DF_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_cached_model(path: str):
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = None
    entry = _MODEL_CACHE.get(path)
    if entry and entry.get("mtime") == mtime:
        return entry["obj"]
    obj = joblib.load(path)
    _MODEL_CACHE[path] = {"mtime": mtime, "obj": obj}
    return obj


def _get_cached_csv(path: str) -> pd.DataFrame:
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = None
    entry = _DF_CACHE.get(path)
    if entry and entry.get("mtime") == mtime:
        return entry["obj"].copy()
    df = pd.read_csv(path)
    _DF_CACHE[path] = {"mtime": mtime, "obj": df}
    return df.copy()


def get_top_stocks(
    n: int = 25,
    min_confidence: int = 30,
    max_risk: int = 70,
    diversify: bool = True,
    verbose: bool = False,
    bullish_only: bool = True,
    model: Optional[Any] = None,
    pred_df: Optional[pd.DataFrame] = None,
    disp_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Wrapper to produce top-N stocks using the enhanced selector.
    Reads model from models/stock_predictor_top.joblib and features from data/featured_stocks_top.csv.
    Adds predicted_change, risk_score, composite_score for UI compatibility.
    """
    model_path = os.path.join('models', 'stock_predictor_top.joblib')
    pred_path = os.getenv('XGB_FEATURES_CSV', 'data/top/xgb_features_latest.csv')
    disp_path = os.getenv('FEATURED_STOCKS_CSV', 'data/featured_stocks_top.csv')
    # Load with caching only if not provided by caller
    try:
        if model is None:
            model = _get_cached_model(model_path)
        if pred_df is None:
            pred_df = _get_cached_csv(pred_path) if os.path.exists(pred_path) else _get_cached_csv(disp_path)
        if disp_df is None:
            try:
                disp_df = _get_cached_csv(disp_path)
            except Exception:
                disp_df = pred_df.copy()
    except Exception as e:
        print(f"[SELECTOR_V2] Failed to load model or features: {e}")
        return pd.DataFrame()

    # Determine model feature columns
    if feature_cols is None:
        try:
            feature_cols = model.get_booster().feature_names
        except Exception:
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

    # Ensure consistent naming (note: for ranker this is a score, not true pct)
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
    # Persist XGB-ranked output for two-step workflow
    try:
        out_dir = os.path.join('data', 'top')
        os.makedirs(out_dir, exist_ok=True)
        # Add xgb_pred alias for convenience
        if 'xgb_pred' not in ranked.columns and 'predicted_return_pct' in ranked.columns:
            try:
                ranked['xgb_pred'] = pd.to_numeric(ranked['predicted_return_pct'], errors='coerce')
            except Exception:
                ranked['xgb_pred'] = ranked['predicted_return_pct']
        ranked[['ticker', 'predicted_return_pct', 'xgb_pred', 'confidence_score', 'risk_score', 'composite_score']].to_csv(
            os.path.join(out_dir, 'xgb_ranked.csv'), index=False
        )
    except Exception:
        pass

    # --- Optional LSTM ensemble for weekly return regression ---
    # Enable only if explicitly requested to support a clean two-step pipeline
    if os.getenv('ENABLE_LSTM_ENSEMBLE', '').lower() != 'true':
        return ranked
    # Try to load an LSTM Keras model and its meta; if available, predict next-week returns
    try:
        import json as _json  # local import to avoid global dep
        import numpy as _np
        from tensorflow import keras as _keras  # type: ignore

        # New locations per request
        lstm_model_path = os.path.join('models', 'lib', 'keras', 'lstm_model.keras')
        lstm_meta_path = os.path.join('models', 'top', 'json', 'lstm_model_meta.json')
        lstm_model = None
        feature_cols_lstm = None
        lookback_lstm = None
        if os.path.exists(lstm_model_path) and os.path.exists(lstm_meta_path):
            try:
                with open(lstm_meta_path, 'r') as f:
                    meta = _json.load(f)
                feature_cols_lstm = meta.get('feature_cols')
                lookback_lstm = int(meta.get('lookback', 30))
                lstm_model = _keras.models.load_model(lstm_model_path)
            except Exception:
                lstm_model = None
        if lstm_model is not None and feature_cols_lstm:
            # Build per-ticker sequences for the ranked tickers using pred_df
            preds_map = {}
            # Prefer a historical features CSV for sequence building if provided
            hist_path = os.getenv('ENSEMBLE_LSTM_FEATURES_CSV', '').strip()
            seq_df = pred_df
            try:
                if hist_path and os.path.exists(hist_path):
                    seq_df = pd.read_csv(hist_path)
            except Exception:
                seq_df = pred_df
            for tkr in ranked['ticker'].astype(str).str.upper().tolist():
                g = seq_df[seq_df['ticker'].astype(str).str.upper() == tkr].copy()
                if g.empty:
                    continue
                g = g.sort_values('date')
                # Ensure required LSTM feature columns exist
                use_cols = []
                for c in feature_cols_lstm:
                    if c not in g.columns:
                        g[c] = 0.0
                    use_cols.append(c)
                if len(g) < int(lookback_lstm or 30):
                    continue
                window = g[use_cols].tail(int(lookback_lstm or 30)).to_numpy(dtype=float)
                X_seq = window.reshape((1, window.shape[0], window.shape[1]))
                try:
                    y_pred = float(lstm_model.predict(X_seq, verbose=0).ravel()[0])
                    preds_map[tkr] = y_pred
                except Exception:
                    continue
            # Attach predictions
            if preds_map:
                ranked['lstm_predicted_return_pct'] = ranked['ticker'].astype(str).str.upper().map(lambda t: preds_map.get(t, _np.nan))
                # Persist to data/top
                try:
                    out_dir = os.path.join('data', 'top')
                    os.makedirs(out_dir, exist_ok=True)
                    pd.DataFrame({
                        'ticker': list(preds_map.keys()),
                        'lstm_predicted_return_pct': list(preds_map.values()),
                    }).to_csv(os.path.join(out_dir, 'lstm_weekly_predictions.csv'), index=False)
                except Exception:
                    pass
                # Normalize scores and build ensemble
                def _minmax(s: pd.Series) -> pd.Series:
                    try:
                        s = pd.to_numeric(s, errors='coerce')
                        mn, mx = s.min(), s.max()
                        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                            return pd.Series(0.5, index=s.index)
                        return (s - mn) / (mx - mn)
                    except Exception:
                        return pd.Series(0.5, index=s.index)
                ranker_norm = _minmax(ranked.get('confidence_score', pd.Series(0.0, index=ranked.index)))
                lstm_norm = _minmax(ranked.get('lstm_predicted_return_pct', pd.Series(0.0, index=ranked.index)))
                alpha = float(os.getenv('ENSEMBLE_ALPHA', '0.6'))  # weight on ranker
                ranked['ensemble_score'] = alpha * ranker_norm + (1 - alpha) * lstm_norm
                # Save ensemble scores
                try:
                    out_dir = os.path.join('data', 'top')
                    os.makedirs(out_dir, exist_ok=True)
                    ranked[['ticker', 'confidence_score', 'lstm_predicted_return_pct', 'ensemble_score']].to_csv(
                        os.path.join(out_dir, 'final_ensemble_scores.csv'), index=False
                    )
                except Exception:
                    pass
                # Re-rank by ensemble
                ranked = ranked.sort_values('ensemble_score', ascending=False)
    except Exception:
        # If TF not installed or model missing, silently skip ensemble
        pass

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


def get_stock_analysis(ticker: str) -> Optional[Dict[str, Any]]:
    """Compatibility helper to fetch per-ticker feature row for analysis panels."""
    feature_path = os.getenv('FEATURED_STOCKS_CSV', 'data/featured_stocks_top.csv')
    try:
        features_df = pd.read_csv(feature_path)
        row = features_df.sort_values('date').groupby('ticker').tail(1)
        row = row[row['ticker'] == ticker]
        if row.empty:
            return None
        return row.iloc[0].to_dict()
    except Exception as e:
        print(f"[SELECTOR_V2] get_stock_analysis failed for {ticker}: {e}")
        return None
