import os
import argparse
import pandas as pd
from typing import Dict, Any, Optional


class SymbolComparator:
    """
    Compare two symbols using available artifacts and explain who wins and why.

    Inputs (loaded if present):
      - data/xgboost/xgboost_weekly_output.csv (new XGB weekly output)
      - data/lstm/lstm_weekly_output.csv (new LSTM weekly output)
      - data/ensemble/ensemble_weekly_output.csv (new Ensemble weekly output)
      - data/features/stock_features_clean.csv (new latest features snapshot)
      - data/top/xgb_weekly_output.csv (legacy features + XGB outputs)
      - data/top/xgb_ranked_output.csv (XGB ranker outputs; fallback)
      - data/top/lstm_weekly_predictions_output.csv (LSTM predictions)
      - data/top/ensemble_scores_output.csv (Ensemble scores)
      - data/top/realized_rank_scores.csv (Recent momentum rank scores)
      - data/featured_stocks_top.csv (Signals/indicators for explanations)
    """

    def __init__(self,
                 xgb_csv: str = 'data/top/xgb_ranked_output.csv',
                 lstm_csv: str = 'data/top/lstm_weekly_predictions_output.csv',
                 ens_csv: str = 'data/top/ensemble_scores_output.csv',
                 recent_csv: str = 'data/top/realized_rank_scores.csv',
                 features_csv: str = 'data/top/featured_stocks_top.csv',
                 raw_input_csv: str = 'data/stock_prices_input.csv',
                 filtered_csv: str = 'data/stock_prices.csv',
                 data_creation_report: str = 'data/top/data_creation_report.csv'):
        self.frames: Dict[str, Optional[pd.DataFrame]] = {}
        for key, path in [
            # New pipeline artifacts
            ('xgb_new', 'data/xgboost/xgboost_weekly_output.csv'),
            ('lstm_new', 'data/lstm/lstm_weekly_output.csv'),
            ('ens_new', 'data/ensemble/ensemble_weekly_output.csv'),
            ('feat_new', 'data/features/stock_features_clean.csv'),
            # Legacy/compat artifacts
            ('xgbw', 'data/top/xgb_weekly_output.csv'),
            ('xgb', xgb_csv),
            ('lstm', lstm_csv),
            ('ens', ens_csv),
            ('recent', recent_csv),
            ('feat', features_csv),
        ] + [
            # Raw/filtered prices for diagnostics
            ('raw', raw_input_csv),
            ('filtered', filtered_csv),
            ('raw_new', 'data/input/stock_prices_with_clean_data.csv'),
            ('raw_unclean', 'data/input/stock_prices_with_unclean_data.csv'),
            ('data_report', data_creation_report),
            ('feat_raw', 'data/top/featured_stocks_raw.csv'),
        ]:
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    if 'ticker' in df.columns:
                        df['ticker'] = df['ticker'].astype(str).str.upper()
                    self.frames[key] = df
                else:
                    self.frames[key] = None
            except Exception:
                self.frames[key] = None

    def _collect(self, symbol: str) -> Dict[str, Any]:
        t = symbol.upper().strip()
        out: Dict[str, Any] = {'ticker': t}
        # Prefer NEW weekly XGB output
        xgbn = self.frames.get('xgb_new')
        if xgbn is not None:
            rown = xgbn[xgbn['ticker'] == t]
            if not rown.empty:
                # Support adjusted and raw columns
                for col_old, col_new in [
                    ('xgb_score_adj', 'xgb_predicted_return_pct'),
                    ('xgb_confidence_score_adj', 'xgb_confidence_score'),
                ]:
                    if col_old in rown.columns and col_new not in out:
                        out[col_new] = rown.iloc[0].get(col_old)
                # Also retain raw for display
                for col in ['xgb_score', 'xgb_confidence_score', 'w_pen']:
                    if col in rown.columns and col not in out:
                        out[col] = rown.iloc[0].get(col)

        # Legacy weekly merged output for XGB metrics if available
        xgbw = self.frames.get('xgbw')
        if xgbw is not None:
            roww = xgbw[xgbw['ticker'] == t]
            if not roww.empty:
                for col in ['xgb_predicted_return_pct', 'xgb_confidence_score', 'predicted_return_pct', 'confidence_score', 'risk_score', 'composite_score']:
                    if col in roww.columns:
                        out[col] = roww.iloc[0].get(col)
        # Fallback to ranked-only CSV
        if ('xgb_confidence_score' not in out and 'confidence_score' not in out) or ('xgb_predicted_return_pct' not in out and 'predicted_return_pct' not in out):
            xgb = self.frames.get('xgb')
            if xgb is not None:
                row = xgb[xgb['ticker'] == t]
                if not row.empty:
                    for col in ['xgb_predicted_return_pct', 'xgb_confidence_score', 'predicted_return_pct', 'confidence_score', 'risk_score', 'composite_score']:
                        if col in row.columns and col not in out:
                            out[col] = row.iloc[0].get(col)
        # Prefer NEW LSTM weekly output
        lstm_new = self.frames.get('lstm_new')
        if lstm_new is not None:
            rowln = lstm_new[lstm_new['ticker'] == t]
            if not rowln.empty:
                for col_old, col_new in [
                    ('lstm_pred_adj','lstm_predicted_return_pct'),
                    ('confidence_score_adj','confidence_score_lstm'),
                ]:
                    if col_old in rowln.columns and col_new not in out:
                        out[col_new] = rowln.iloc[0].get(col_old)
                for col in ['lstm_pred','confidence_score','w_pen']:
                    if col in rowln.columns and col not in out:
                        out[col] = rowln.iloc[0].get(col)

        lstm = self.frames.get('lstm')
        if lstm is not None:
            row = lstm[lstm['ticker'] == t]
            if not row.empty and 'lstm_predicted_return_pct' in row.columns:
                out['lstm_predicted_return_pct'] = row.iloc[0]['lstm_predicted_return_pct']
        # Prefer NEW ensemble weekly
        ens_new = self.frames.get('ens_new')
        ens = ens_new if ens_new is not None else self.frames.get('ens')
        if ens is not None:
            row = ens[ens['ticker'] == t]
            if not row.empty:
                if 'ensemble_score' in row.columns:
                    out['ensemble_score'] = row.iloc[0]['ensemble_score']
                if 'w_pen' in row.columns and 'w_pen' not in out:
                    out['w_pen'] = row.iloc[0]['w_pen']
                # If only LSTM exists in ensemble file, surface it too
                if 'lstm_predicted_return_pct' in row.columns and 'lstm_predicted_return_pct' not in out:
                    out['lstm_predicted_return_pct'] = row.iloc[0]['lstm_predicted_return_pct']
        recent = self.frames.get('recent')
        if recent is not None:
            row = recent[recent['ticker'] == t]
            if not row.empty and 'recent_return_score' in row.columns:
                out['recent_return_score'] = row.iloc[0]['recent_return_score']
                for rcol in ['rank_score_5d', 'rank_score_15d', 'rank_score_30d']:
                    if rcol in row.columns:
                        out[rcol] = row.iloc[0][rcol]
        feat_new = self.frames.get('feat_new')
        feat = feat_new if feat_new is not None else self.frames.get('feat')
        if feat is not None:
            row = feat[feat['ticker'] == t].copy()
            if not row.empty:
                # Prefer latest date snapshot for this ticker to avoid early NaNs
                try:
                    if 'date' in row.columns:
                        row['date'] = pd.to_datetime(row['date'], errors='coerce')
                        row = row.sort_values('date').tail(1)
                except Exception:
                    pass
                r = row.iloc[0]
                for scol in ['broke_resistance', 'breakout_confirmed', 'strong_momentum', 'post_earnings_dip_rally', 'pre_earning_rally',
                             'golden_cross', 'earnings_in_3_weeks', 'last_2q_positive_surprises']:
                    if scol in row.columns:
                        out[scol] = bool(r.get(scol))
                for ncol in ['momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_30d', 'momentum_60d',
                             'rsi_14d', 'trend_slope_15d', 'trend_slope_30d', 'volume_ratio', 'volatility_30d']:
                    if ncol in row.columns:
                        out[ncol] = r.get(ncol)
                # New engineered features (if present)
                for ecol in ['price_change_pct','vol_change_pct','selloff_flag','rsi_price_interaction','volratio_price_interaction','overbought_spike']:
                    if ecol in row.columns:
                        out[ecol] = r.get(ecol)

        # Fallback: compute missing core features (RSI/VolRatio/momentums) from raw prices
        try:
            import numpy as _np
            import pandas as _pd
            def _is_nan(v):
                try:
                    return v is None or _np.isnan(float(v))
                except Exception:
                    return False
            need_any = any(
                k not in out or _is_nan(out.get(k)) for k in [
                    'rsi_14d','volume_ratio','momentum_5d','momentum_10d','momentum_20d',
                    'price_change_pct','vol_change_pct','rsi_price_interaction','volratio_price_interaction','overbought_spike'
                ]
            )
            if need_any:
                raw_df = self.frames.get('raw_new') if self.frames.get('raw_new') is not None else self.frames.get('raw')
                if raw_df is not None and {'ticker','date','close','volume'}.issubset(raw_df.columns):
                    g = raw_df[raw_df['ticker'].astype(str).str.upper() == t].copy()
                    if not g.empty:
                        g['date'] = _pd.to_datetime(g['date'], errors='coerce')
                        g = g.sort_values('date')
                        # Compute
                        g['price_change_pct'] = _pd.to_numeric(g['close'], errors='coerce').pct_change(1)
                        g['vol_change_pct'] = _pd.to_numeric(g['volume'], errors='coerce').pct_change(1)
                        # RSI(14) simple version
                        delta = g['close'].diff(1)
                        gain = delta.where(delta > 0, 0.0).rolling(window=14, min_periods=14).mean()
                        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=14).mean()
                        with _np.errstate(divide='ignore', invalid='ignore'):
                            rs = gain / loss
                            rs = rs.replace([_np.inf, -_np.inf], _np.nan)
                            g['rsi_14d_calc'] = 100 - (100 / (1 + rs))
                        g['volume_ma_20'] = _pd.to_numeric(g['volume'], errors='coerce').rolling(20, min_periods=20).mean()
                        g['volume_ratio_calc'] = _pd.to_numeric(g['volume'], errors='coerce') / g['volume_ma_20']
                        g['momentum_5d_calc'] = _pd.to_numeric(g['close'], errors='coerce').pct_change(5)
                        g['momentum_10d_calc'] = _pd.to_numeric(g['close'], errors='coerce').pct_change(10)
                        g['momentum_20d_calc'] = _pd.to_numeric(g['close'], errors='coerce').pct_change(20)
                        last = g.tail(1).iloc[0]
                        def _set_if_missing(k, val):
                            if k not in out or _is_nan(out.get(k)):
                                out[k] = float(val) if val is not None and _np.isfinite(val) else out.get(k)
                        _set_if_missing('rsi_14d', last.get('rsi_14d_calc'))
                        _set_if_missing('volume_ratio', last.get('volume_ratio_calc'))
                        _set_if_missing('momentum_5d', last.get('momentum_5d_calc'))
                        _set_if_missing('momentum_10d', last.get('momentum_10d_calc'))
                        _set_if_missing('momentum_20d', last.get('momentum_20d_calc'))
                        _set_if_missing('price_change_pct', last.get('price_change_pct'))
                        _set_if_missing('vol_change_pct', last.get('vol_change_pct'))
                        # Derived interactions/flags
                        try:
                            rsi_v = float(out.get('rsi_14d')) if not _is_nan(out.get('rsi_14d')) else float(last.get('rsi_14d_calc'))
                        except Exception:
                            rsi_v = _np.nan
                        try:
                            pc_v = float(out.get('price_change_pct')) if not _is_nan(out.get('price_change_pct')) else float(last.get('price_change_pct'))
                        except Exception:
                            pc_v = _np.nan
                        try:
                            vr_v = float(out.get('volume_ratio')) if not _is_nan(out.get('volume_ratio')) else float(last.get('volume_ratio_calc'))
                        except Exception:
                            vr_v = _np.nan
                        if 'rsi_price_interaction' not in out or _is_nan(out.get('rsi_price_interaction')):
                            if _np.isfinite(rsi_v) and _np.isfinite(pc_v):
                                out['rsi_price_interaction'] = rsi_v * pc_v
                        if 'volratio_price_interaction' not in out or _is_nan(out.get('volratio_price_interaction')):
                            if _np.isfinite(vr_v) and _np.isfinite(pc_v):
                                out['volratio_price_interaction'] = vr_v * pc_v
                        if 'overbought_spike' not in out or _is_nan(out.get('overbought_spike')):
                            if _np.isfinite(rsi_v) and _np.isfinite(vr_v):
                                out['overbought_spike'] = int((rsi_v > 75.0) and (vr_v > 1.2))
        except Exception:
            pass

        # Backfill confidence from ensemble if present there but missing in XGB
        if ('xgb_confidence_score' not in out and 'confidence_score' not in out):
            ens = self.frames.get('ens')
            if ens is not None:
                rowe = ens[ens['ticker'] == t]
                if not rowe.empty and 'confidence_score' in rowe.columns:
                    out['confidence_score'] = rowe.iloc[0].get('confidence_score')

        # Fallback for volatility_30d: try raw features, else compute from raw prices (last 30d)
        try:
            import numpy as _np
            import pandas as _pd
            vol = out.get('volatility_30d')
            if vol is None or (isinstance(vol, float) and _np.isnan(vol)):
                feat_raw = self.frames.get('feat_raw')
                if feat_raw is not None:
                    rowr = feat_raw[feat_raw.get('ticker','').astype(str).str.upper() == t]
                    if not rowr.empty and 'volatility_30d' in rowr.columns:
                        v = rowr.iloc[0].get('volatility_30d')
                        if v is not None and not _np.isnan(float(v)):
                            out['volatility_30d'] = float(v)
                if 'volatility_30d' not in out or _np.isnan(out['volatility_30d']):
                    raw_df = self.frames.get('raw')
                    if raw_df is not None and {'ticker','date','close'}.issubset(raw_df.columns):
                        g = raw_df[raw_df['ticker'].astype(str).str.upper() == t].copy()
                        if not g.empty:
                            g['date'] = _pd.to_datetime(g['date'], errors='coerce')
                            g = g.sort_values('date').tail(40)
                            g['ret'] = g['close'].pct_change()
                            vol30 = _pd.to_numeric(g['ret'], errors='coerce').tail(30).std()
                            if vol30 is not None and not _np.isnan(vol30):
                                out['volatility_30d'] = float(vol30)
        except Exception:
            pass
        return out

    @staticmethod
    def _score_priority(a: Dict[str, Any]) -> float:
        # Prefer positive signals: check metrics in priority order and pick the FIRST positive.
        keys = ['ensemble_score', 'xgb_confidence_score', 'confidence_score', 'xgb_predicted_return_pct', 'predicted_return_pct', 'lstm_predicted_return_pct']
        vals = []
        for key in keys:
            v = a.get(key)
            try:
                if v is None:
                    continue
                fv = float(v)
                vals.append(fv)
            except Exception:
                continue
        # If any positive candidate exists, return the first positive encountered following priority
        for fv in vals:
            if fv > 0:
                return fv
        # Fallback: return first available (could be zero/negative)
        if vals:
            return vals[0]
        return float('-inf')

    def compare(self, symbol_a: str, symbol_b: str) -> Dict[str, Any]:
        a = self._collect(symbol_a)
        b = self._collect(symbol_b)
        a_score = self._score_priority(a)
        b_score = self._score_priority(b)
        # Apply a selloff penalty consistently for comparison
        try:
            import os as _os
            import pandas as _pd
            vr_thr = float(_os.getenv('SELLOFF_STRONG_VRATIO', '2.0'))
            dp_thr = float(_os.getenv('SELLOFF_STRONG_DROP_PCT', '-0.03'))
            strong_fac = float(_os.getenv('SELLOFF_FACTOR_STRONG', '0.60'))
            mild_fac = float(_os.getenv('SELLOFF_FACTOR_MILD', '0.90'))
            def _factor(d: Dict[str, Any]) -> float:
                try:
                    so = int(d.get('selloff_flag', 0)) == 1
                    vr = float(d.get('volume_ratio', 0.0) or 0.0)
                    pc = float(d.get('price_change_pct', d.get('return_1d', 0.0)) or 0.0)
                except Exception:
                    so, vr, pc = False, 0.0, 0.0
                if so:
                    return strong_fac if (vr > vr_thr and pc < dp_thr) else mild_fac
                return 1.0
            a_score *= _factor(a)
            b_score *= _factor(b)
        except Exception:
            pass
        winner = symbol_a.upper() if a_score >= b_score else symbol_b.upper()
        loser = symbol_b.upper() if winner == symbol_a.upper() else symbol_a.upper()

        reasons = []
        # Explicit per-model winners using model artifacts
        def _first_positive_or_first(d: Dict[str, Any], keys: list[str]) -> float:
            vals = []
            for k in keys:
                v = d.get(k)
                try:
                    if v is None:
                        continue
                    fv = float(v)
                    vals.append(fv)
                except Exception:
                    continue
            for fv in vals:
                if fv > 0:
                    return fv
            return vals[0] if vals else float('-inf')

        # XGB-only winner from weekly output fields (prefer confidence, then predicted pct)
        ax = _first_positive_or_first(a, ['xgb_confidence_score', 'confidence_score', 'xgb_predicted_return_pct', 'predicted_return_pct'])
        bx = _first_positive_or_first(b, ['xgb_confidence_score', 'confidence_score', 'xgb_predicted_return_pct', 'predicted_return_pct'])
        if ax != float('-inf') or bx != float('-inf'):
            xgb_only_winner = symbol_a.upper() if ax >= bx else symbol_b.upper()
            reasons.append(f"XGB Weekly Winner: {xgb_only_winner} (A={('n/a' if ax==float('-inf') else f'{ax:.3f}')}, B={('n/a' if bx==float('-inf') else f'{bx:.3f}')})")

        # LSTM-only winner from lstm output
        al = _first_positive_or_first(a, ['lstm_predicted_return_pct'])
        bl = _first_positive_or_first(b, ['lstm_predicted_return_pct'])
        if al != float('-inf') or bl != float('-inf'):
            lstm_only_winner = symbol_a.upper() if al >= bl else symbol_b.upper()
            reasons.append(f"LSTM Weekly Winner: {lstm_only_winner} (A={('n/a' if al==float('-inf') else f'{al:.3f}')}, B={('n/a' if bl==float('-inf') else f'{bl:.3f}')})")
        def fmt(name: str, d: Dict[str, Any]) -> Optional[str]:
            v = d.get(name)
            try:
                if v is None:
                    return None
                return f"{float(v):.3f}"
            except Exception:
                return str(v)

        # Core score comparison
        for key, label in [
            ('ensemble_score', 'Ensemble'),
            ('xgb_confidence_score', 'XGB Confidence'),
            ('confidence_score', 'Confidence'),
            ('xgb_predicted_return_pct', 'XGB Predicted Return %'),
            ('predicted_return_pct', 'Predicted Return %'),
            ('lstm_predicted_return_pct', 'LSTM Predicted Return %'),
        ]:
            av = fmt(key, a)
            bv = fmt(key, b)
            if av is not None or bv is not None:
                reasons.append(f"{label}: {symbol_a.upper()}={av or 'n/a'} vs {symbol_b.upper()}={bv or 'n/a'}")

        # Momentum/recency
        rv_a = fmt('recent_return_score', a)
        rv_b = fmt('recent_return_score', b)
        if rv_a is not None or rv_b is not None:
            reasons.append(f"Recent Return Score: {symbol_a.upper()}={rv_a or 'n/a'} vs {symbol_b.upper()}={rv_b or 'n/a'}")

        # Signals (use the three weighted ones per spec)
        def pos_signals(d: Dict[str, Any]) -> int:
            return sum(int(bool(d.get(k))) for k in ['broke_resistance','breakout_confirmed','post_earnings_dip_rally','pre_earning_rally'])
        reasons.append(
            f"Positive Signals (weighted set): {symbol_a.upper()}={pos_signals(a)} vs {symbol_b.upper()}={pos_signals(b)} "
            f"(broke_resistance/breakout_confirmed/post_earnings_dip_rally/pre_earning_rally)"
        )
        # Detailed per-signal breakdown
        sig_labels = [
            ('broke_resistance', 'Broke Resistance'),
            ('breakout_confirmed', 'Breakout Confirmed'),
            ('post_earnings_dip_rally', 'Post-Earnings Dip Rally'),
            ('pre_earning_rally', 'Pre-Earnings Rally'),
        ]
        for key, label in sig_labels:
            av = 'Yes' if bool(a.get(key)) else 'No'
            bv = 'Yes' if bool(b.get(key)) else 'No'
            reasons.append(f"{label}: {symbol_a.upper()}={av} vs {symbol_b.upper()}={bv}")

        # Derived training-time weight per spec: (1 + 0.5*signals) * (1 + 0.5*recent_return_score) clamped [0.5, 3.0]
        def derived_weight(d: Dict[str, Any]) -> Optional[float]:
            try:
                br = 1.0 if bool(d.get('broke_resistance')) else 0.0
                bc = 1.0 if bool(d.get('breakout_confirmed')) else 0.0
                pedr = 1.0 if bool(d.get('post_earnings_dip_rally')) else 0.0
                rrs = float(d.get('recent_return_score')) if d.get('recent_return_score') is not None else 0.0
                signal_weight = 1.0 + 0.5 * (br + bc + pedr)
                w = signal_weight * (1.0 + 0.5 * rrs)
                return max(0.5, min(3.0, w))
            except Exception:
                return None
        wa = derived_weight(a)
        wb = derived_weight(b)
        if wa is not None or wb is not None:
            reasons.append(
                f"Derived Training Weight: {symbol_a.upper()}={(f'{wa:.3f}' if wa is not None else 'n/a')} "
                f"vs {symbol_b.upper()}={(f'{wb:.3f}' if wb is not None else 'n/a')}"
            )

        # Risk if available
        ra = fmt('risk_score', a)
        rb = fmt('risk_score', b)
        if ra is not None or rb is not None:
            reasons.append(f"Risk Score (lower better): {symbol_a.upper()}={ra or 'n/a'} vs {symbol_b.upper()}={rb or 'n/a'}")

        # Other notable indicators
        for k, label in [('rsi_14d','RSI 14d'), ('volume_ratio','Volume Ratio'), ('volatility_30d','Volatility 30d'),
                         ('momentum_5d','Momentum 5d'), ('momentum_10d','Momentum 10d'), ('momentum_20d','Momentum 20d'),
                         ('price_change_pct','Price Change % (1d)'), ('vol_change_pct','Volume Change % (1d)'),
                         ('rsi_price_interaction','RSI×PriceChange'), ('volratio_price_interaction','VolRatio×PriceChange')]:
            av = fmt(k, a)
            bv = fmt(k, b)
            if av is not None or bv is not None:
                reasons.append(f"{label}: {symbol_a.upper()}={av or 'n/a'} vs {symbol_b.upper()}={bv or 'n/a'}")

        # New feature flags interpretation
        def _flag_str(name: str, d: Dict[str, Any]) -> str:
            v = d.get(name)
            try:
                return 'Yes' if int(v) == 1 else 'No'
            except Exception:
                return 'No' if not v else 'Yes'
        if ('selloff_flag' in a or 'selloff_flag' in b) or ('overbought_spike' in a or 'overbought_spike' in b):
            asf = _flag_str('selloff_flag', a)
            bsf = _flag_str('selloff_flag', b)
            aos = _flag_str('overbought_spike', a)
            bos = _flag_str('overbought_spike', b)
            reasons.append(f"Sell-Off Flag: {symbol_a.upper()}={asf} vs {symbol_b.upper()}={bsf}")
            reasons.append(f"Overbought Spike: {symbol_a.upper()}={aos} vs {symbol_b.upper()}={bos}")

        # Simple recommendation cue from new features
        def rec_from_new_feats(d: Dict[str, Any]) -> str:
            cues = []
            try:
                if float(d.get('price_change_pct', 0)) > 0 and float(d.get('rsi_14d', 50)) < 40:
                    cues.append('oversold_bounce')
            except Exception:
                pass
            try:
                if float(d.get('volume_ratio', 1)) > 1.2 and float(d.get('rsi_14d', 50)) > 75:
                    cues.append('overbought_exhaustion')
            except Exception:
                pass
            try:
                if int(d.get('selloff_flag', 0)) == 1:
                    cues.append('selloff_penalty')
            except Exception:
                pass
            return ','.join(cues) if cues else 'neutral'
        reasons.append(
            f"New feature cues: {symbol_a.upper()}={rec_from_new_feats(a)} vs {symbol_b.upper()}={rec_from_new_feats(b)}"
        )

        # Data availability diagnostics (prefer new raw paths)
        raw_new = self.frames.get('raw_new')
        raw = raw_new if raw_new is not None else self.frames.get('raw')
        filt = self.frames.get('filtered')
        rep = self.frames.get('data_report')
        def _presence_line(sym: str) -> str:
            if raw is None and filt is None:
                return f"Data presence for {sym}: input=n/a, filtered=n/a"
            in_raw = None if raw is None else (not raw[raw.get('ticker','').astype(str).str.upper() == sym].empty)
            in_filt = None if filt is None else (not filt[filt.get('ticker','').astype(str).str.upper() == sym].empty)
            return f"Data presence for {sym}: input={'yes' if in_raw else 'no' if in_raw is not None else 'n/a'}, filtered={'yes' if in_filt else 'no' if in_filt is not None else 'n/a'}"
        reasons.insert(0, _presence_line(symbol_b.upper()))
        reasons.insert(0, _presence_line(symbol_a.upper()))
        if rep is not None:
            row = rep[rep.get('ticker','').astype(str).str.upper() == symbol_b.upper()]
            if not row.empty:
                r = row.iloc[0]
                # Append diagnostics
                for k,label in [('rows_2y','rows_2y'),('avg_volume','avg_volume'),('close','latest_close'),('pass_liquidity','pass_liquidity'),('pass_length','pass_length')]:
                    if k in row.columns:
                        reasons.append(f"{symbol_b.upper()} {label}: {r.get(k)}")
                # Elimination reason
                try:
                    if (in_raw is True) and (in_filt is False):
                        pl = bool(r.get('pass_liquidity')) if 'pass_liquidity' in row.columns else None
                        ple = bool(r.get('pass_length')) if 'pass_length' in row.columns else None
                        latest_close = r.get('close') if 'close' in row.columns else None
                        avg_vol = r.get('avg_volume') if 'avg_volume' in row.columns else None
                        if pl is False:
                            reasons.insert(1, f"Elimination reason: failed liquidity filter (latest_close={latest_close}, avg_volume={avg_vol})")
                        elif ple is False:
                            reasons.insert(1, f"Elimination reason: insufficient history for length filter (rows_2y={r.get('rows_2y')})")
                        else:
                            reasons.insert(1, "Elimination reason: filtered by intermediate cleaning (e.g., outlier removal)")
                except Exception:
                    pass
        # Fallback diagnostics if data_creation_report.csv missing
        elif self.frames.get('raw') is not None:
            try:
                raw_df = self.frames['raw'].copy()
                raw_df['date'] = pd.to_datetime(raw_df['date'], errors='coerce')
                max_date = raw_df['date'].max()
                if pd.notna(max_date):
                    cutoff = max_date - pd.DateOffset(years=2)
                    raw_df = raw_df[raw_df['date'] >= cutoff]
                # Compute metrics for B and AAPL
                bdf = raw_df[raw_df['ticker'] == symbol_b.upper()]
                aapl_df = raw_df[raw_df['ticker'] == 'AAPL']
                rows_b = len(bdf)
                rows_aapl = len(aapl_df)
                latest_close = float(bdf.sort_values('date')['close'].tail(1).iloc[0]) if rows_b > 0 else None
                avg_vol = float(bdf['volume'].mean()) if rows_b > 0 else None
                min_required = max(65, int(0.8 * rows_aapl)) if rows_aapl > 0 else 65
                pass_liquidity = (latest_close is not None and latest_close >= 3.0) and (avg_vol is not None and avg_vol >= 300000)
                pass_length = rows_b >= min_required
                reasons.append(f"{symbol_b.upper()} latest_close: {latest_close if latest_close is not None else 'n/a'}")
                reasons.append(f"{symbol_b.upper()} avg_volume: {avg_vol if avg_vol is not None else 'n/a'}")
                reasons.append(f"{symbol_b.upper()} rows_2y: {rows_b}")
                reasons.append(f"{symbol_b.upper()} pass_liquidity: {pass_liquidity}")
                reasons.append(f"{symbol_b.upper()} pass_length: {pass_length} (min_required={min_required})")
                if (in_raw is True) and (in_filt is False):
                    if not pass_liquidity:
                        reasons.insert(1, f"Elimination reason: failed liquidity filter (latest_close={latest_close}, avg_volume={avg_vol})")
                    elif not pass_length:
                        reasons.insert(1, f"Elimination reason: insufficient history for length filter (rows_2y={rows_b}, min_required={min_required})")
                    else:
                        reasons.insert(1, "Elimination reason: filtered by intermediate cleaning (e.g., outlier removal)")
            except Exception:
                pass
        # Add broke_resistance explanation (close vs previous 50d high)
        try:
            def _br_line(tkr: str) -> str:
                rdf = self.frames.get('raw')
                if rdf is None or not {'ticker','date','close'}.issubset(rdf.columns):
                    return f"{tkr}: n/a"
                g = rdf[rdf['ticker'].astype(str).str.upper()==tkr].copy()
                if g.empty:
                    return f"{tkr}: n/a"
                g['date'] = pd.to_datetime(g['date'], errors='coerce')
                g = g.sort_values('date')
                if len(g) < 2:
                    return f"{tkr}: n/a"
                prev50 = g.iloc[:-1].tail(50)
                if prev50.empty:
                    return f"{tkr}: n/a"
                res = float(prev50['close'].max())
                close_now = float(g.iloc[-1]['close'])
                br = close_now > res
                return f"{tkr}: close={close_now:.2f} vs prev50d_high={res:.2f} → {'Yes' if br else 'No'}"
            reasons.append("Broke Resistance calc (close vs prev 50d high):")
            reasons.append("- " + _br_line(symbol_a.upper()))
            reasons.append("- " + _br_line(symbol_b.upper()))
        except Exception:
            pass

        # Penalty weight summary if available
        try:
            def _pen_str(d: Dict[str, Any]) -> str:
                v = d.get('w_pen')
                try:
                    return f"{float(v):.3f}" if v is not None else 'n/a'
                except Exception:
                    return 'n/a'
            pa = _pen_str(a)
            pb = _pen_str(b)
            if pa != 'n/a' or pb != 'n/a':
                reasons.append(f"Penalty Weight (w_pen): {symbol_a.upper()}={pa} vs {symbol_b.upper()}={pb}")
        except Exception:
            pass

        return {
            'winner': winner,
            'loser': loser,
            'a': a,
            'b': b,
            'reasons': reasons,
        }


def main():
    p = argparse.ArgumentParser(description='Compare two symbols using model outputs and features')
    p.add_argument('--a', required=True, help='First symbol (e.g., AAPL)')
    p.add_argument('--b', required=True, help='Second symbol (e.g., MSFT)')
    p.add_argument('--out', default='', help='Optional path to write a markdown report')
    args = p.parse_args()

    cmp = SymbolComparator()
    res = cmp.compare(args.a, args.b)
    winner = res['winner']

    print(f"Winner: {winner}")
    print("\nComparison details:")
    for line in res['reasons']:
        print(f"- {line}")

    if args.out:
        try:
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            with open(args.out, 'w') as f:
                f.write(f"# {args.a.upper()} vs {args.b.upper()}\n\n")
                f.write(f"Winner: **{winner}**\n\n")
                f.write("## Details\n")
                for line in res['reasons']:
                    f.write(f"- {line}\n")
            print(f"\nWrote report to {args.out}")
        except Exception as e:
            print(f"Failed to write report: {e}")


if __name__ == '__main__':
    main()


