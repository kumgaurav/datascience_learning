# models/xgb_trainer.py
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


class XGBTrainer(BaseModelTrainer):
    def train(self):
        # Diagnostics: required base columns
        required_base = ['ticker', 'date', 'close']
        have_cols = set(getattr(self, 'df', {}).columns if hasattr(self, 'df') else [])
        missing_base = [c for c in required_base if c not in have_cols]
        print(f"[XGB DIAG] Required base columns: {required_base}")
        print(f"[XGB DIAG] Missing base columns: {missing_base}")

        train_df, test_df = self.prepare_data()
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

        # Add per-ticker z-scores (selected cols) and simple interactions
        for col in ['volume', 'volatility_20d', 'close_std_20d', 'volume_ratio']:
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
        feature_cols = [c for c in numeric_cols_all if c not in exclude]
        self.feature_cols = feature_cols

        # Build matrices using rank target
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

        # Sample weights emphasizing bullish signals
        sw = np.ones(len(inner_train_df), dtype=float)
        for col, weight in [
            ('broke_resistance', 0.6),
            ('breakout_confirmed', 0.6),
            ('strong_momentum', 0.4),
            ('post_earnings_dip_rally', 0.6),
            ('low_volatility', 0.2),
            ('volume_above_avg', 0.2),
        ]:
            if col in inner_train_df.columns:
                sw += inner_train_df[col].astype(int).values * weight
        if 'signal_pos_count' in inner_train_df.columns:
            no_signal_mask = (inner_train_df['signal_pos_count'] == 0).values
            sw[no_signal_mask] *= 0.7
        sw = np.clip(sw, 0.5, 3.0)

        # Parameter search optimizing Precision@20 on validation
        rng = np.random.RandomState(42)
        param_space = []
        for _ in range(8):
            param_space.append({
                'max_depth': int(rng.choice([3, 4, 5])),
                'learning_rate': float(rng.uniform(0.02, 0.06)),
                'subsample': float(rng.uniform(0.6, 0.9)),
                'colsample_bytree': float(rng.uniform(0.6, 0.9)),
                'min_child_weight': int(rng.choice([1, 3, 5])),
                'reg_lambda': float(rng.uniform(0.5, 3.0)),
                'reg_alpha': float(rng.uniform(0.0, 0.5)),
                'n_estimators': 1500,
            })

        def fit_eval(params):
            model = xgb.XGBRanker(
                objective='rank:pairwise',
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                **params,
            )
            if len(X_val) > 0 and len(val_group) > 0:
                # Try early stopping variants
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

            # Validation Precision@20
            if len(X_val) > 0:
                val_pred = model.predict(X_val)
                p20 = _precision_at_k_by_date(val_df, val_pred, k=20)
            else:
                val_pred = model.predict(X_train)
                p20 = _precision_at_k_by_date(inner_train_df, val_pred, k=20)
            return {'model': model, 'p20': p20, 'params': params}

        results = [fit_eval(p) for p in param_space]
        # pick best by precision@20 (higher is better)
        best = max(results, key=lambda r: (r['p20'] if r['p20'] == r['p20'] else -1))
        model = best['model']
        try:
            model._validation_p20 = float(best.get('p20', float('nan')))
        except Exception:
            pass

        # Evaluate on test period
        preds = model.predict(X_test)
        # Report ranking metrics (Precision@20)
        p20_test = _precision_at_k_by_date(test_df, preds, k=20)
        # For reference, also MAE/R2/DirAcc on scores vs returns (not main objective)
        try:
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            dir_acc = (np.sign(preds) == np.sign(y_test)).mean()
        except Exception:
            mae = r2 = dir_acc = float('nan')

        print(f"XGBoost Ranker → Precision@20 {p20_test:.3f}, MAE {mae:.2f}, R² {r2:.2f}, DirAcc {dir_acc*100:.1f}%")

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
                    rets.append(float(realized.loc[top.index].mean()))
            if rets:
                avg_ret = float(np.mean(rets))
                print(f"[BACKTEST] Avg top-{K} realized return: {avg_ret:.3f} (pct points)")
        except Exception:
            pass

        return model, preds
