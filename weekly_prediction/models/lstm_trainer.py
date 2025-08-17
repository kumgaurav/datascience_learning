# models/lstm_trainer.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_absolute_error, r2_score
from .base_trainer import BaseModelTrainer

class LSTMTrainer(BaseModelTrainer):
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

    def train(self, lookback=20, horizon=5):
        # Diagnostics: required base columns
        required_base = ['ticker', 'date', 'close']
        have_cols = set(getattr(self, 'df', {}).columns if hasattr(self, 'df') else [])
        missing_base = [c for c in required_base if c not in have_cols]
        print(f"[LSTM DIAG] Required base columns: {required_base}")
        print(f"[LSTM DIAG] Missing base columns: {missing_base}")

        train_df, test_df = self.prepare_data(horizon=horizon)
        print(f"[LSTM] Prepared data with train rows: {len(train_df)}, test rows: {len(test_df)}")

        # For sequence models, include OHLCV and engineered numeric features.
        # Exclude only non-features and the label to avoid leakage.
        exclude = {'ticker', 'date', 'target'}
        numeric_cols = train_df.select_dtypes(include=['number', 'bool']).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in exclude]
        print(f"[LSTM DIAG] Candidate feature columns (incl. OHLCV + engineered): {len(feature_cols)}")
        if len(feature_cols) < 1:
            print("[LSTM DIAG] ERROR: No usable feature columns found.")
        self.feature_cols = feature_cols

        # Build per-ticker sliding windows
        train_seq, train_y = self._make_sequences(train_df, feature_cols, lookback, horizon)
        test_seq, test_y = self._make_sequences(test_df, feature_cols, lookback, horizon)
        print(f"[LSTM] Sequence shapes → train: {train_seq.shape}, test: {test_seq.shape}")

        # Replace NaN/Inf in sequences and drop NaN targets to avoid training/eval crashes
        if train_seq.size == 0 or test_seq.size == 0:
            print("[LSTM] WARNING: Empty sequence arrays; skipping training/evaluation.")
            return models.Sequential(), np.array([])

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

        model = models.Sequential([
            layers.Input(shape=(lookback, len(feature_cols))),
            layers.LSTM(64, return_sequences=False),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
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

        num_epochs = 20
        print(f"[LSTM] Starting training for {num_epochs} epochs (lookback={lookback}, features={len(feature_cols)})...")
        model.fit(train_seq, train_y, epochs=num_epochs, batch_size=32, verbose=0, callbacks=[ProgressCallback(num_epochs)])
        print("[LSTM] Training complete.")

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
            return model, preds

        mae = mean_absolute_error(y_eval, preds_eval)
        r2 = r2_score(y_eval, preds_eval)
        dir_acc = (np.sign(preds_eval) == np.sign(y_eval)).mean()

        print(f"LSTM → MAE {mae:.2f}, R² {r2:.2f}, DirAcc {dir_acc*100:.1f}%")
        return model, preds
