# models/xgb_trainer.py
import os
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from .base_trainer import BaseModelTrainer


def _precision_at_k_by_date(df, preds, k=20):
    # df must contain columns: 'date', 'target'
    out = []
    df_local = df.copy()
    df_local["_pred"] = preds
    for dt, g in df_local.groupby("date"):
        g = g.sort_values("_pred", ascending=False)
        kk = min(k, len(g))
        if kk == 0:
            continue
        top = g.head(kk)
        prec = float((top["target"] > 0).mean())
        out.append(prec)
    return float(np.mean(out)) if out else float("nan")


def _ndcg_at_k_by_date(df, preds, k=20, gain: str = "linear"):
    """Compute average NDCG@k across dates using true returns as relevance.
    Assumes df has columns 'date' and 'target' (realized forward return).
    """
    def dcg(rel):
        rel = np.asarray(rel)[:k]
        if rel.size == 0:
            return 0.0
        # Gain scheme
        discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
        if gain == "exp2":
            gains = (2.0 ** rel - 1.0)
        else:
            gains = rel
        return float(np.sum(gains * discounts))

    out = []
    if 'target' not in df.columns:
        return float("nan")
    df_local = df.copy()
    df_local["_pred"] = preds
    for dt, g in df_local.groupby("date"):
        if "target" not in g.columns:
            continue
        g_sorted = g.sort_values("_pred", ascending=False)
        rel_sorted = g_sorted["target"].values
        ideal_sorted = np.sort(g["target"].values)[::-1]
        dcg_val = dcg(rel_sorted)
        idcg_val = dcg(ideal_sorted)
        if idcg_val > 0:
            out.append(dcg_val / idcg_val)
    return float(np.mean(out)) if out else float("nan")


class XGBTrainer(BaseModelTrainer):
    def _engineer_ranking_features(self, inner_train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, exclude_cols):
        # Add per-ticker z-scores (selected cols) and simple interactions
        cols_for_z = ['volume', 'volatility_20d', 'close_std_20d', 'volume_ratio']
        for col in cols_for_z:
            if col in inner_train_df.columns:
                inner_train_df[f'{col}_z'] = inner_train_df.groupby('ticker')[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
                if len(val_df):
                    val_df[f'{col}_z'] = val_df.groupby('ticker')[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
                test_df[f'{col}_z'] = test_df.groupby('ticker')[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
        if 'momentum_20d' in inner_train_df.columns and 'volume_ratio' in inner_train_df.columns:
            inner_train_df['mom20_x_volratio'] = inner_train_df['momentum_20d'] * inner_train_df.get('volume_ratio', 0)
            if len(val_df):
                val_df['mom20_x_volratio'] = val_df.get('momentum_20d', 0) * val_df.get('volume_ratio', 0)
            test_df['mom20_x_volratio'] = test_df.get('momentum_20d', 0) * test_df.get('volume_ratio', 0)

        # Rebuild feature list to include engineered cols
        numeric_cols_all = inner_train_df.select_dtypes(include=['number', 'bool']).columns.tolist()
        feature_cols = [c for c in numeric_cols_all if c not in exclude_cols]
        return inner_train_df, val_df, test_df, feature_cols

    def _clip_features_by_quantile(self, df_train: pd.DataFrame, df_list: list[pd.DataFrame], feature_cols: list[str], q_low: float = 1.0, q_high: float = 99.0):
        # Compute clipping bounds on train, apply to provided dfs
        bounds = {}
        for c in feature_cols:
            try:
                arr = df_train[c].to_numpy(dtype=float)
                mask = np.isfinite(arr)
                if not mask.any():
                    continue
                lo, hi = np.nanpercentile(arr[mask], [q_low, q_high])
                if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                    continue
                bounds[c] = (lo, hi)
            except Exception:
                continue
        for df in df_list:
            for c, (lo, hi) in bounds.items():
                try:
                    df[c] = np.clip(df[c].to_numpy(dtype=float), lo, hi)
                except Exception:
                    pass
        return df_list

    def _hyperparam_search(self, X_train, y_train, train_group, X_val, y_val, val_group, rng, val_df_local: pd.DataFrame, train_df_local: pd.DataFrame, n_trials: int = 0):
        # If Optuna available and trials requested, run it; else random search
        use_optuna = False
        n_trials = int(n_trials or int(os.environ.get('XGB_TUNER_TRIALS', '0')))
        try:
            import optuna  # type: ignore
            use_optuna = n_trials > 0
        except Exception:
            use_optuna = False

        def fit_eval(params):
            model = xgb.XGBRanker(
                objective='rank:pairwise',
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                **params,
            )
            if len(X_val) > 0 and len(val_group) > 0:
                try:
                    es_cb = [xgb.callback.EarlyStopping(rounds=100, save_best=True)]
                    model.fit(
                        X_train, y_train, group=train_group,
                        eval_set=[(X_val, y_val)], eval_group=[val_group],
                        verbose=False, callbacks=es_cb,
                    )
                except TypeError:
                    try:
                        model.fit(
                            X_train, y_train, group=train_group,
                            eval_set=[(X_val, y_val)], eval_group=[val_group],
                            verbose=False, early_stopping_rounds=100,
                        )
                    except TypeError:
                        model.fit(
                            X_train, y_train, group=train_group,
                            verbose=False,
                        )
            else:
                model.fit(X_train, y_train, group=train_group, verbose=False)

            if len(X_val) > 0:
                val_pred = model.predict(X_val)
                p20 = _precision_at_k_by_date(val_df_local, val_pred, k=20)
            else:
                val_pred = model.predict(X_train)
                p20 = _precision_at_k_by_date(train_df_local, val_pred, k=20)
            # Convert NaN to -inf for robust selection
            score = float(p20) if np.isfinite(p20) else float('-inf')
            return {'model': model, 'p20': p20, 'score': score, 'params': params}

        results = []
        if use_optuna:
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 7),
                    'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.08, log=False),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 6),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 8.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'n_estimators': 3000,
                }
                res = fit_eval(params)
                # maximize P@20 → minimize negative
                return -res['score']

            study = optuna.create_study(direction='minimize', study_name='xgb_ranker')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_params = study.best_trial.user_attrs.get('params') if hasattr(study.best_trial, 'user_attrs') else None
            if best_params is None:
                best_params = study.best_params
            # Fit/eval with best
            results.append(fit_eval(best_params))
        else:
            # Random search
            param_space = []
            for _ in range(12):
                param_space.append({
                    'max_depth': int(rng.choice([3, 4, 5, 6, 7])),
                    'learning_rate': float(rng.uniform(0.02, 0.06)),
                    'subsample': float(rng.uniform(0.6, 0.95)),
                    'colsample_bytree': float(rng.uniform(0.6, 0.95)),
                    'min_child_weight': int(rng.choice([1, 2, 3, 4, 5])),
                    'reg_lambda': float(rng.uniform(0.5, 8.0)),
                    'reg_alpha': float(rng.uniform(0.0, 1.0)),
                    'n_estimators': 3000,
                })
            results = [fit_eval(p) for p in param_space]

        best = max(results, key=lambda r: r['score'])
        return best

    def train(self):
        try:
            print(f"[XGB INIT] Starting training with features file: {self.features_path}")
        except Exception:
            pass
        """Train an XGBoost ranking model (pairwise) for next-week selection.

        Key design choices:
          - Rank-normalized target per date (`target_rank`) to remove scale bias.
          - Time-aware validation split and group-by-date ranking with XGBRanker.
          - Per-ticker z-scores and simple interactions to stabilize split dynamics.
          - Model selection by Precision@20; reports NDCG@20 and simple weekly backtest.
          - Metrics persisted under `metrics/` with timestamped filenames.
        """
        # Diagnostics: required base columns
        required_base = ['ticker', 'date', 'close']
        have_cols = set(getattr(self, 'df', {}).columns if hasattr(self, 'df') else [])
        missing_base = [c for c in required_base if c not in have_cols]
        print(f"[XGB DIAG] Required base columns: {required_base}")
        print(f"[XGB DIAG] Missing base columns: {missing_base}")

        train_df, test_df = self.prepare_data()
        # Ensure datetime dtype
        try:
            train_df['date'] = pd.to_datetime(train_df['date'])
            test_df['date'] = pd.to_datetime(test_df['date'])
        except Exception:
            pass
        # Rank-normalize forward returns per date for ranking objective
        try:
            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df['target_rank'] = train_df.groupby('date')['target'].rank(pct=True)
            test_df['target_rank'] = test_df.groupby('date')['target'].rank(pct=True)
        except Exception:
            train_df['target_rank'] = train_df['target']
            test_df['target_rank'] = test_df['target']

        exclude = ['ticker', 'date', 'target', 'target_rank', 'close', 'open', 'high', 'low', 'volume']
        numeric_cols = train_df.select_dtypes(include=['number', 'bool']).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in exclude]
        print(f"[XGB DIAG] Candidate feature columns (numeric/bool, excluding leakage): {len(feature_cols)}")
        if len(feature_cols) < 1:
            print("[XGB DIAG] ERROR: No usable feature columns found.")
        self.feature_cols = feature_cols

        # Target clipping for stability
        try:
            lower, upper = np.percentile(train_df['target'], [1, 99])
            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df['target'] = np.clip(train_df['target'], lower, upper)
            test_df['target'] = np.clip(test_df['target'], lower, upper)
            print(f"[XGB] Target clip bounds: [{lower:.2f}, {upper:.2f}]")
        except Exception:
            pass

        # Time-based validation split
        try:
            cutoff = train_df['date'].quantile(0.9)
            inner_train_df = train_df[train_df['date'] <= cutoff]
            val_df = train_df[train_df['date'] > cutoff]
            if inner_train_df.empty or val_df.empty:
                fallback_cut = train_df['date'].max() - (train_df['date'].max() - train_df['date'].min()) * 0.1
                inner_train_df = train_df[train_df['date'] <= fallback_cut]
                val_df = train_df[train_df['date'] > fallback_cut]
        except Exception:
            inner_train_df = train_df
            val_df = train_df.iloc[0:0]

        # Sort by date for grouping
        inner_train_df = inner_train_df.sort_values(['date'])
        val_df = val_df.sort_values(['date'])
        test_df = test_df.sort_values(['date'])

        inner_train_df, val_df, test_df, feature_cols = self._engineer_ranking_features(inner_train_df, val_df, test_df, exclude)
        self.feature_cols = feature_cols

        # Build matrices using rank target
        # Clip features by train quantiles for stability
        _ = self._clip_features_by_quantile(inner_train_df, [inner_train_df, val_df, test_df], feature_cols, 1.0, 99.0)
        try:
            print(f"[XGB PREP] Feature set size: {len(feature_cols)}; Train rows: {len(inner_train_df)}; Val rows: {len(val_df)}; Test rows: {len(test_df)}")
        except Exception:
            pass
        X_train = inner_train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = inner_train_df['target_rank']
        X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_val = val_df['target_rank'] if len(val_df) else val_df.get('target_rank', val_df)
        X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_test = test_df['target_rank']

        # Group sizes by date (for ranking); filter small groups (<20) in validation
        K = 20
        train_group = inner_train_df.groupby('date').size().tolist()
        if len(val_df):
            val_sizes = val_df.groupby('date').size()
            valid_dates = val_sizes[val_sizes >= K].index
            if len(valid_dates) < len(val_sizes):
                mask_val = val_df['date'].isin(valid_dates)
                X_val = X_val[mask_val]
                y_val = y_val[mask_val]
                val_group = val_df[mask_val].groupby('date').size().tolist()
                val_df = val_df[mask_val]
            else:
                val_group = val_sizes.tolist()
        else:
            val_group = []

        # Sample weights (requested scheme):
        # signal_weight = 1.0 + 0.5 per positive of {broke_resistance, breakout_confirmed, post_earnings_dip_rally}
        # weight = signal_weight * (1 + 0.5 * recent_return_score)
        # where recent_return_score ∈ [0,1] from normalized realized returns (5d/15d/30d)
        # and final weight is clamped to [0.5, 3.0]

        def _pick_first_present_col(df_local: pd.DataFrame, candidates: list[str]) -> pd.Series:
            vals = pd.Series([None] * len(df_local), index=df_local.index, dtype='float64')
            for c in candidates:
                if c in df_local.columns:
                    s = pd.to_numeric(df_local[c], errors='coerce')
                    vals = vals.where(~vals.isna(), s)
            return vals.fillna(0.0)

        def _norm_rank_within_date(df_local: pd.DataFrame, series: pd.Series) -> pd.Series:
            try:
                # Higher is better → descending rank; map to [0,1]
                r = series.groupby(df_local['date']).rank(method='min', ascending=False)
                sz = df_local.groupby('date')['date'].transform('size').astype(float)
                score = 1.0 - (r - 1.0) / sz.clip(lower=1.0)
                return score.fillna(0.0)
            except Exception:
                return pd.Series(0.0, index=df_local.index)

        # Build recent return score on inner_train_df
        s5 = _pick_first_present_col(inner_train_df, ['price_change_5d_pct', 'return_5d', 'momentum_5d'])
        s15 = _pick_first_present_col(inner_train_df, ['price_change_15d_pct', 'return_15d', 'momentum_15d'])
        s30 = _pick_first_present_col(inner_train_df, ['price_change_30d_pct', 'return_30d', 'momentum_30d'])
        score5 = _norm_rank_within_date(inner_train_df, s5)
        score15 = _norm_rank_within_date(inner_train_df, s15)
        score30 = _norm_rank_within_date(inner_train_df, s30)
        recent_return_score = 0.4 * score5 + 0.3 * score15 + 0.3 * score30

        # Signal component (include pre-earnings rally)
        br = inner_train_df.get('broke_resistance', False)
        bc = inner_train_df.get('breakout_confirmed', False)
        pedr = inner_train_df.get('post_earnings_dip_rally', False)
        # pre_earning_rally: prefer explicit flag; fallback to earnings_in_3_weeks
        per = inner_train_df.get('pre_earning_rally', inner_train_df.get('earnings_in_3_weeks', False))
        br = br.astype(int) if hasattr(br, 'astype') else int(bool(br))
        bc = bc.astype(int) if hasattr(bc, 'astype') else int(bool(bc))
        pedr = pedr.astype(int) if hasattr(pedr, 'astype') else int(bool(pedr))
        per = per.astype(int) if hasattr(per, 'astype') else int(bool(per))
        signal_weight = 1.0 + 0.5 * (br.values if hasattr(br, 'values') else br) \
                               + 0.5 * (bc.values if hasattr(bc, 'values') else bc) \
                               + 0.5 * (pedr.values if hasattr(pedr, 'values') else pedr) \
                               + 0.5 * (per.values if hasattr(per, 'values') else per)

        sw = signal_weight * (1.0 + 0.5 * recent_return_score.to_numpy())
        sw = np.clip(sw, 0.5, 3.0)

        # Parameter search optimizing Precision@20 on validation (Optuna optional)
        rng = np.random.RandomState(42)
        # Optional tracking start
        if getattr(self, 'tracker', None):
            try:
                self.tracker.start_run(run_name=self.run_name or 'xgb_trainer', params={
                    'model': 'xgboost_ranker',
                    'tuner_trials': int(os.environ.get('XGB_TUNER_TRIALS', '0')),
                    'features': len(feature_cols),
                })
            except Exception:
                pass

        best = self._hyperparam_search(
            X_train, y_train, train_group,
            X_val, y_val, val_group,
            rng,
            val_df, inner_train_df,
            n_trials=int(os.environ.get('XGB_TUNER_TRIALS', '0'))
        )
        model = best['model']
        try:
            print(f"[XGB TUNE] Best P@20: {best.get('p20')}, params: {best.get('params')}")
        except Exception:
            pass
        try:
            model._validation_p20 = float(best.get('p20', float('nan')))
        except Exception:
            pass

        # Evaluate on test period
        preds = model.predict(X_test)
        try:
            print(f"[XGB EVAL] Predicted on test set with shape {X_test.shape}")
        except Exception:
            pass
        # Report ranking metrics (Precision@20, NDCG@20)
        p20_test = _precision_at_k_by_date(test_df, preds, k=K)
        ndcg20_test = _ndcg_at_k_by_date(test_df, preds, k=K, gain=os.environ.get('NDCG_GAIN', 'linear'))
        # For reference, also MAE/R2/DirAcc on scores vs returns (not main objective)
        try:
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            dir_acc = (np.sign(preds) == np.sign(y_test)).mean()
        except Exception:
            mae = r2 = dir_acc = float('nan')

        print(f"XGBoost Ranker → P@{K} {p20_test:.3f}, NDCG@{K} {ndcg20_test:.3f}, MAE {mae:.2f}, R² {r2:.2f}, DirAcc {dir_acc*100:.1f}%")
        # Persist validation/test MAE on model for downstream ensemble weighting
        try:
            model._validation_mae = float(mae)
        except Exception:
            pass
        # Track metrics
        if getattr(self, 'tracker', None):
            try:
                self.tracker.log_metrics({
                    f'p_at_{K}': float(p20_test),
                    f'ndcg_at_{K}': float(ndcg20_test),
                    'mae_proxy_vs_rank': float(mae) if 'mae' in locals() else float('nan'),
                    'r2_proxy_vs_rank': float(r2) if 'r2' in locals() else float('nan'),
                })
            except Exception:
                pass

        # Simple top-20 backtest on test set using realized forward returns (unranked target)
        try:
            df_scores = test_df.copy()
            df_scores = df_scores.assign(_pred=preds)
            # Use clipped original percent target as proxy for realized
            realized = df_scores['target'] if 'target' in df_scores.columns else df_scores['target_rank']
            rets = []
            for dt, g in df_scores.groupby('date'):
                top = g.sort_values('_pred', ascending=False).head(K)
                if len(top) > 0:
                    # apply simple trading cost of 20 bps to the daily basket
                    rets.append(float(realized.loc[top.index].mean()) - 0.002)
            if rets:
                avg_ret = float(np.mean(rets))
                print(f"[BACKTEST] Avg top-{K} realized return: {avg_ret:.3f} (pct points)")
        except Exception:
            pass

        # Evaluate on most recent two 1-week windows (by unique dates)
        try:
            uniq = pd.unique(test_df['date'])
            uniq = pd.Series(uniq).sort_values().to_numpy()
            if uniq.size >= 5:
                last5 = uniq[-5:]
                prev5 = uniq[-10:-5] if uniq.size >= 10 else None

                def _eval_win(sel_dates):
                    mask = np.isin(test_df['date'].to_numpy(), sel_dates)
                    if not mask.any():
                        return None
                    dfw = test_df[mask]
                    predw = preds[mask]
                    p20w = _precision_at_k_by_date(dfw, predw, k=K)
                    # realized avg top-K
                    dfw_s = dfw.copy().assign(_pred=predw)
                    top_mean = float(
                        dfw_s.sort_values(['date', '_pred'], ascending=[True, False])
                             .groupby('date')
                             .head(K)
                             .groupby('date')['target']
                             .mean()
                             .mean()
                    ) if 'target' in dfw_s.columns else float('nan')
                    return p20w, top_mean

                res_last = _eval_win(last5)
                if res_last is not None:
                    print(f"[EVAL] Last 1w → P@{K} {res_last[0]:.3f}, avg top-{K} return {res_last[1]:.3f}")
                if prev5 is not None:
                    res_prev = _eval_win(prev5)
                    if res_prev is not None:
                        print(f"[EVAL] Prev 1w → P@{K} {res_prev[0]:.3f}, avg top-{K} return {res_prev[1]:.3f}")
        except Exception:
            pass

        # Persist key metrics to CSV for monitoring
        try:
            metrics = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'p_at_k': float(p20_test),
                'ndcg_at_k': float(ndcg20_test),
                'mae': float(mae) if 'mae' in locals() else np.nan,
                'r2': float(r2) if 'r2' in locals() else np.nan,
                'dir_acc': float(dir_acc) if 'dir_acc' in locals() else np.nan,
                'k': int(K),
                'rows_test': int(len(test_df)),
            }
            out_dir = os.path.join('metrics')
            os.makedirs(out_dir, exist_ok=True)
            ts_name = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            out_path = os.path.join(out_dir, f'metrics_ranker_{ts_name}.csv')
            pd.DataFrame([metrics]).to_csv(out_path, index=False)
            print(f"[METRICS] Logged metrics to {out_path}")
            # Also log as artifact if tracker exists
            if getattr(self, 'tracker', None):
                try:
                    self.tracker.log_artifact(out_path, artifact_path='metrics')
                except Exception:
                    pass
            # Save model as artifact on disk
            try:
                os.makedirs('models', exist_ok=True)
                model_out = os.path.join('models', 'stock_predictor_top.joblib')
                import joblib
                joblib.dump(model, model_out)
                print(f"[XGB SAVE] Saved model to {model_out}")
            except Exception:
                pass
        except Exception as e:
            print(f"[METRICS] Failed to log metrics: {e}")

        # End tracking
        if getattr(self, 'tracker', None):
            try:
                self.tracker.end_run()
            except Exception:
                pass

        return model, preds
