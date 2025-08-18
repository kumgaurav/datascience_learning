# models/lstm_trainer.py
import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_absolute_error, r2_score
from .base_trainer import BaseModelTrainer

class LSTMTrainer(BaseModelTrainer):
    """Sequence model trainer for short-horizon returns.

    Builds sliding windows per ticker with per-window z-score normalization and
    trains a compact LSTM with dropout and early stopping. Also performs a
    fallback time-based split on window end dates if the standard test split is
    empty, to ensure at least a minimal evaluation.
    """
    def __init__(self, features_path: str, tracker=None, run_name: str = None, feature_config: dict | None = None, model_config: dict | None = None):
        super().__init__(features_path, tracker=tracker, run_name=run_name, feature_config=feature_config)
        self.model_config = {
            'lookback': 60,
            'horizon': 5,
            'lstm_units': 64,
            'dense_units': 32,
            'dropout': 0.2,
            'stacked_lstm_layers': 1,
            'epochs': 40,
            'batch_size': 32,
            'validation_split': 0.1,
            'early_stopping_patience': 5,
            'reduce_lr_patience': 3,
            'use_vectorized_windows': True,
            'save_model_path': 'models/lstm_model.keras',
            'cv_folds': 0,
        }
        if model_config:
            try:
                self.model_config.update(model_config)
            except Exception:
                pass
    def _make_sequences_with_dates(self, df, feature_cols, lookback=60, horizon=5):
        """
        Same as _make_sequences but also returns the window end dates for splitting.
        """
        X, y, end_dates = [], [], []
        eps = 1e-8
        for ticker, grp in df.groupby('ticker'):
            grp = grp.sort_values('date')
            feat_vals = grp[feature_cols].to_numpy(dtype=float, copy=False)
            targets = grp['target'].to_numpy(dtype=float, copy=False)
            dates = grp['date'].to_numpy()
            num_rows = len(grp)
            if num_rows < lookback + horizon + 1:
                continue
            for start in range(0, num_rows - lookback - horizon + 1):
                end = start + lookback
                window = feat_vals[start:end]
                mean = window.mean(axis=0)
                std = window.std(axis=0)
                std_safe = np.where(std < eps, 1.0, std)
                window_norm = (window - mean) / std_safe
                X.append(window_norm)
                y.append(targets[end])
                end_dates.append(dates[end])
        return np.asarray(X), np.asarray(y), np.asarray(end_dates)
    def _make_sequences(self, df, feature_cols, lookback=20, horizon=5):
        """
        Build per-ticker sliding windows of length `lookback` with label as the
        forward return at the window end (precomputed in df['target']).

        Normalization: per-window z-score (mean/std across the window) to avoid
        leakage across time/tickers. Small epsilon guards against zero-std.
        """
        X, y = [], []
        eps = 1e-8
        for ticker, grp in df.groupby('ticker'):
            grp = grp.sort_values('date')
            feat_vals = grp[feature_cols].to_numpy(dtype=float, copy=False)
            targets = grp['target'].to_numpy(dtype=float, copy=False)
            num_rows = len(grp)
            if num_rows < lookback + horizon + 1:
                continue
            for start in range(0, num_rows - lookback - horizon + 1):
                end = start + lookback
                window = feat_vals[start:end]
                # Per-window z-score normalization
                mean = window.mean(axis=0)
                std = window.std(axis=0)
                std_safe = np.where(std < eps, 1.0, std)
                window_norm = (window - mean) / std_safe
                X.append(window_norm)
                y.append(targets[end])
        return np.asarray(X), np.asarray(y)

    def _make_sequences_vectorized(self, df, feature_cols, lookback=20, horizon=5):
        X_out, y_out, end_dates = [], [], []
        eps = 1e-8
        for _, grp in df.groupby('ticker'):
            grp = grp.sort_values('date')
            feat_vals = grp[feature_cols].to_numpy(dtype=float, copy=False)
            targets = grp['target'].to_numpy(dtype=float, copy=False)
            dates = grp['date'].to_numpy()
            n = len(grp)
            max_start = n - lookback - horizon + 1
            if max_start <= 0:
                continue
            try:
                shape = (n - lookback + 1, lookback, feat_vals.shape[1])
                strides = (feat_vals.strides[0], feat_vals.strides[0], feat_vals.strides[1])
                windows = np.lib.stride_tricks.as_strided(feat_vals, shape=shape, strides=strides)
                windows = windows[:max_start]
                means = windows.mean(axis=1)
                stds = windows.std(axis=1)
                stds_safe = np.where(stds < eps, 1.0, stds)
                windows_norm = (windows - means[:, None, :]) / stds_safe[:, None, :]
                X_out.append(windows_norm)
                y_out.append(targets[lookback:lookback + max_start])
                end_dates.append(dates[lookback:lookback + max_start])
            except Exception:
                X_grp, y_grp = self._make_sequences(grp, feature_cols, lookback, horizon)
                if X_grp.size > 0:
                    X_out.append(X_grp)
                    y_out.append(y_grp)
                    end_dates.append(grp['date'].to_numpy()[lookback:lookback + len(y_grp)])
        if not X_out:
            return np.empty((0, lookback, len(feature_cols))), np.empty((0,)), np.empty((0,))
        X_cat = np.concatenate(X_out, axis=0)
        y_cat = np.concatenate(y_out, axis=0)
        ends = np.concatenate(end_dates, axis=0)
        return X_cat, y_cat, ends

    def train(self, lookback=None, horizon=None, cv_folds: int = 0):
        # Diagnostics: required base columns
        required_base = ['ticker', 'date', 'close']
        have_cols = set(getattr(self, 'df', {}).columns if hasattr(self, 'df') else [])
        missing_base = [c for c in required_base if c not in have_cols]
        print(f"[LSTM DIAG] Required base columns: {required_base}")
        print(f"[LSTM DIAG] Missing base columns: {missing_base}")
        try:
            print(f"[LSTM INIT] Starting training with features file: {self.features_path}")
        except Exception:
            pass

        lookback = int(self.model_config.get('lookback') if lookback is None else lookback)
        horizon = int(self.model_config.get('horizon') if horizon is None else horizon)
        train_df, test_df = self.prepare_data(horizon=horizon)
        print(f"[LSTM] Prepared data with train rows: {len(train_df)}, test rows: {len(test_df)}")
        try:
            print(f"[LSTM PREP] Using lookback={lookback}, horizon={horizon}")
        except Exception:
            pass

        # For sequence models, include OHLCV and engineered numeric features.
        # Exclude only non-features and the label to avoid leakage.
        exclude = {'ticker', 'date', 'target'}
        numeric_cols = train_df.select_dtypes(include=['number', 'bool']).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in exclude]
        print(f"[LSTM DIAG] Candidate feature columns (incl. OHLCV + engineered): {len(feature_cols)}")
        if len(feature_cols) < 1:
            print("[LSTM DIAG] ERROR: No usable feature columns found.")
        self.feature_cols = feature_cols

        # Cross-sectional z-score per date (create _csz columns and prefer them)
        try:
            def _csz(df_local, cols):
                out = df_local.copy()
                for c in cols:
                    if c in out.columns:
                        out[f"{c}_csz"] = out.groupby('date')[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
                return out
            cs_cols = feature_cols
            train_df = _csz(train_df, cs_cols)
            test_df = _csz(test_df, cs_cols)
            feature_cols = [f"{c}_csz" if f"{c}_csz" in train_df.columns else c for c in feature_cols]
        except Exception:
            pass

        # Build per-ticker sliding windows
        if bool(self.model_config.get('use_vectorized_windows', True)):
            train_seq, train_y, _ = self._make_sequences_vectorized(train_df, feature_cols, lookback, horizon)
            test_seq, test_y, _ = self._make_sequences_vectorized(test_df, feature_cols, lookback, horizon)
        else:
            train_seq, train_y = self._make_sequences(train_df, feature_cols, lookback, horizon)
            test_seq, test_y = self._make_sequences(test_df, feature_cols, lookback, horizon)
        print(f"[LSTM] Sequence shapes → train: {train_seq.shape}, test: {test_seq.shape}")

        # Replace NaN/Inf in sequences and drop NaN targets to avoid training/eval crashes
        if train_seq.size == 0:
            print("[LSTM] WARNING: No train sequences; skipping training.")
            return models.Sequential(), np.array([])
        if test_seq.size == 0:
            # Fallback: build sequences on train_df and split by most recent end-date windows
            try:
                all_seq, all_y, all_ends = self._make_sequences_with_dates(train_df, feature_cols, lookback, horizon)
                if all_seq.size > 0:
                    # Determine cutoff on end dates (last 20%)
                    # Sort unique dates
                    unique_ends = np.unique(all_ends)
                    if unique_ends.size > 5:
                        cutoff_idx = int(max(1, np.floor(0.8 * unique_ends.size)))
                        cutoff_date = unique_ends[cutoff_idx]
                        test_mask = all_ends >= cutoff_date
                        train_mask = ~test_mask
                        if test_mask.any() and train_mask.any():
                            train_seq, train_y = all_seq[train_mask], all_y[train_mask]
                            test_seq, test_y = all_seq[test_mask], all_y[test_mask]
                            print(f"[LSTM] Fallback split on sequence end-dates → train: {train_seq.shape}, test: {test_seq.shape}")
                        else:
                            print("[LSTM] WARNING: Fallback split produced empty test; proceeding without evaluation.")
                    else:
                        print("[LSTM] WARNING: Not enough unique end-dates for fallback split; proceeding without evaluation.")
                else:
                    print("[LSTM] WARNING: No sequences from fallback builder; proceeding without evaluation.")
            except Exception as e:
                print(f"[LSTM] WARNING: Fallback sequence split failed: {e}; proceeding without evaluation.")

        train_seq = np.nan_to_num(train_seq, nan=0.0, posinf=0.0, neginf=0.0)
        test_seq = np.nan_to_num(test_seq, nan=0.0, posinf=0.0, neginf=0.0)

        # Align by removing rows where targets are NaN/Inf
        train_mask = np.isfinite(train_y)
        test_mask = np.isfinite(test_y)
        if not train_mask.all():
            print(f"[LSTM] Dropping {np.size(train_y) - np.count_nonzero(train_mask)} train targets that are NaN/Inf")
        if not test_mask.all():
            print(f"[LSTM] Dropping {np.size(test_y) - np.count_nonzero(test_mask)} test targets that are NaN/Inf")
        train_seq, train_y = train_seq[train_mask], train_y[train_mask]
        test_seq, test_y = test_seq[test_mask], test_y[test_mask]
        print(f"[LSTM] After cleaning → train: {train_seq.shape}, test: {test_seq.shape}")
        try:
            print(f"[LSTM PREP] Sequence dims → lookback={lookback}, features={len(feature_cols)}")
        except Exception:
            pass

        # Force CPU-only and limit TensorFlow threads to avoid macOS segfaults
        try:
            try:
                tf.config.set_visible_devices([], 'GPU')
            except Exception:
                pass
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass

        model = models.Sequential()
        model.add(layers.Input(shape=(lookback, len(feature_cols))))
        lstm_units = int(self.model_config.get('lstm_units', 64))
        dense_units = int(self.model_config.get('dense_units', 32))
        dropout = float(self.model_config.get('dropout', 0.2))
        stacked = int(self.model_config.get('stacked_lstm_layers', 1))
        if stacked > 1:
            for _ in range(stacked - 1):
                model.add(layers.LSTM(lstm_units, return_sequences=True))
                if dropout > 0:
                    model.add(layers.Dropout(dropout))
            model.add(layers.LSTM(lstm_units, return_sequences=False))
        else:
            model.add(layers.LSTM(lstm_units, return_sequences=False))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
        if dense_units > 0:
            model.add(layers.Dense(dense_units, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mae')

        # Persist training-time metadata on the model for downstream inference/ensembling
        try:
            model.feature_cols = list(feature_cols)
            model.lookback = int(lookback)
        except Exception:
            pass

        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, total_epochs: int):
                super().__init__()
                self.total_epochs = total_epochs

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                loss_val = logs.get('loss')
                msg_loss = f"{loss_val:.4f}" if loss_val is not None else "n/a"
                print(f"[LSTM] Epoch {epoch + 1}/{self.total_epochs} - loss: {msg_loss}")

        num_epochs = int(self.model_config.get('epochs', 40))
        print(f"[LSTM] Starting training for {num_epochs} epochs (lookback={lookback}, features={len(feature_cols)})...")
        # Add EarlyStopping and ReduceLROnPlateau with configurable patience
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(self.model_config.get('early_stopping_patience', 5)), restore_best_weights=True)
        rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=int(self.model_config.get('reduce_lr_patience', 3)), min_lr=1e-5)
        # Start tracking
        if getattr(self, 'tracker', None):
            try:
                self.tracker.start_run(run_name=self.run_name or 'lstm_trainer', params={
                    'model': 'lstm',
                    'epochs': int(num_epochs),
                    'lookback': lookback,
                    'features': len(feature_cols),
                })
            except Exception:
                pass

        model.fit(
            train_seq, train_y,
            epochs=int(num_epochs),
            batch_size=int(self.model_config.get('batch_size', 32)),
            verbose=0,
            validation_split=(
                float(self.model_config.get('validation_split', 0.1))
                if (len(train_seq) > 50 and len(train_seq) * float(self.model_config.get('validation_split', 0.1)) >= 1.0)
                else 0.0
            ),
            callbacks=[ProgressCallback(num_epochs), es, rlrop]
        )
        print("[LSTM] Training complete.")
        try:
            print(f"[LSTM TRAIN] Completed epochs={num_epochs}, batch_size={int(self.model_config.get('batch_size', 32))}")
        except Exception:
            pass

        if len(test_seq) == 0:
            print("[LSTM] WARNING: No test sequences available after cleaning; skipping evaluation.")
            return model, np.array([])

        preds = model.predict(test_seq).ravel()
        # Filter out any NaN/Inf predictions for fair eval
        pred_mask = np.isfinite(preds)
        if not pred_mask.all():
            print(f"[LSTM] Dropping {np.size(preds) - np.count_nonzero(pred_mask)} NaN/Inf predictions before metrics")
        eval_mask = pred_mask & np.isfinite(test_y)
        preds_eval = preds[eval_mask]
        y_eval = test_y[eval_mask]
        if len(y_eval) == 0:
            print("[LSTM] WARNING: No valid pairs for evaluation after filtering; skipping metrics.")
            # End tracking before returning
            if getattr(self, 'tracker', None):
                try:
                    self.tracker.end_run()
                except Exception:
                    pass
            try:
                os.makedirs('models', exist_ok=True)
                save_path = self.model_config.get('save_model_path', 'models/lstm_model.keras')
                model.save(save_path)
                meta = {
                    'feature_cols': list(feature_cols),
                    'lookback': int(lookback),
                    'horizon': int(horizon),
                    'epochs': int(num_epochs),
                }
                with open('models/lstm_model_meta.json', 'w') as f:
                    json.dump(meta, f)
                print(f"[LSTM SAVE] Saved model to {save_path} and metadata to models/lstm_model_meta.json")
            except Exception:
                pass
            return model, preds

        mae = mean_absolute_error(y_eval, preds_eval)
        r2 = r2_score(y_eval, preds_eval)
        dir_acc = (np.sign(preds_eval) == np.sign(y_eval)).mean()

        print(f"LSTM → MAE {mae:.2f}, R² {r2:.2f}, DirAcc {dir_acc*100:.1f}%")
        # Persist validation/test MAE on model for downstream ensemble weighting
        try:
            model._validation_mae = float(mae)
        except Exception:
            pass
        # Optional simple time-series CV evaluation (evaluation only)
        try:
            folds = int(cv_folds or self.model_config.get('cv_folds', 0))
            if folds > 1:
                combo = pd.concat([train_df, test_df], ignore_index=True)
                all_seq, all_y, all_ends = self._make_sequences_with_dates(combo, feature_cols, lookback, horizon)
                if all_seq.size > 0:
                    uniq = np.unique(all_ends)
                    for i in range(1, folds + 1):
                        cutoff = uniq[int(i / (folds + 1) * (len(uniq) - 1))]
                        test_mask = all_ends > cutoff
                        if not np.any(test_mask):
                            continue
                        preds_cv = model.predict(all_seq[test_mask]).ravel()
                        y_cv = all_y[test_mask]
                        mask_finite = np.isfinite(preds_cv) & np.isfinite(y_cv)
                        if mask_finite.any():
                            mae_cv = mean_absolute_error(y_cv[mask_finite], preds_cv[mask_finite])
                            print(f"[LSTM CV] Fold {i}/{folds} MAE: {mae_cv:.3f}")
        except Exception:
            pass
        # Persist model and metadata
        try:
            os.makedirs('models', exist_ok=True)
            save_path = self.model_config.get('save_model_path', 'models/lstm_model.keras')
            model.save(save_path)
            meta = {
                'feature_cols': list(feature_cols),
                'lookback': int(lookback),
                'horizon': int(horizon),
                'epochs': int(num_epochs),
                'lstm_units': int(lstm_units),
                'dense_units': int(dense_units),
                'dropout': float(dropout),
                'stacked_lstm_layers': int(stacked),
            }
            with open('models/lstm_model_meta.json', 'w') as f:
                json.dump(meta, f)
            print(f"[LSTM SAVE] Saved model to {save_path} and metadata to models/lstm_model_meta.json")
        except Exception:
            pass

        # Track metrics
        if getattr(self, 'tracker', None):
            try:
                self.tracker.log_metrics({
                    'mae': float(mae),
                    'r2': float(r2),
                    'directional_accuracy': float(dir_acc),
                })
            except Exception:
                pass
            finally:
                try:
                    self.tracker.end_run()
                except Exception:
                    pass
        return model, preds
