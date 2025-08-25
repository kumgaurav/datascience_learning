# models/ensemble_trainer.py
import numpy as np
from typing import List, Optional
import numpy as np
import pandas as pd

class EnsembleTrainer:
    """Flexible ensemble combiner for arbitrary model objects with .predict().

    Supports:
      - weighted average (default)
      - rank-based blending (scale-agnostic)
      - voting (for classification-style signals)
      - probability averaging (clip to [0,1])

    Notes:
      - Voting interprets predictions > voting_threshold as a positive (bullish) vote.
      - Rank method handles equal scores by returning 0.5 for all-equal cases, and
        otherwise uses average ranks (descending) normalized to [0,1].
    """

    def __init__(
        self,
        models: List[object],
        weights: Optional[List[float]] = None,
        method: str = "weighted",  # "weighted" | "rank" | "voting" | "prob"
        voting_threshold: float = 0.0,
    ):
        self.models = models
        self.weights = weights
        self.method = str(method).lower()
        self.voting_threshold = float(voting_threshold)

    def predict(self, Xs: List[np.ndarray]):
        collected: List[np.ndarray] = []
        weights: List[float] = []
        lengths: List[int] = []
        for idx, (m, X) in enumerate(zip(self.models, Xs)):
            try:
                # Prefer probabilities when in 'prob' mode and model supports predict_proba
                if self.method == 'prob' and hasattr(m, 'predict_proba'):
                    pp = m.predict_proba(X)
                    arr = np.asarray(pp)
                    # Take positive-class prob if 2-D
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        pred = arr[:, 1]
                    else:
                        pred = arr.ravel()
                else:
                    pred = m.predict(X)
                if pred is None:
                    print(f"[ENSEMBLE WARN] Model {idx} returned None; skipped.")
                    continue
                arr = np.asarray(pred)
                # Accept shapes: (n,), (n,1). If other, try ravel; if multi-d, take first column
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                else:
                    arr = arr.ravel()
                if arr.size == 0:
                    print(f"[ENSEMBLE WARN] Model {idx} produced empty predictions; skipped.")
                    continue
                if not np.isfinite(arr).any():
                    print(f"[ENSEMBLE WARN] Model {idx} predictions are non-finite; skipped.")
                    continue
                collected.append(arr.astype(float))
                lengths.append(arr.size)
                if self.weights is not None and idx < len(self.weights):
                    w = float(self.weights[idx])
                    if not np.isfinite(w) or w <= 0:
                        w = 0.0
                else:
                    w = 1.0
                weights.append(w)
            except Exception as e:
                # Skip failing model
                print(f"[ENSEMBLE WARN] Model {idx} predict() failed: {e}")
                continue

        if not collected:
            raise RuntimeError("No valid model predictions for ensemble")

        # Align lengths by truncating to the minimum length across valid models
        min_len = min(lengths)
        if any(l != min_len for l in lengths):
            print(f"[ENSEMBLE WARN] Mismatched prediction lengths {lengths}; truncating to min length {min_len}.")
        preds = np.vstack([p[:min_len] for p in collected])
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.size != preds.shape[0]:
            print("[ENSEMBLE WARN] Weights length mismatch; using equal weights.")
            weights_arr = np.ones(preds.shape[0], dtype=float)
        if weights_arr.sum() <= 0 or not np.isfinite(weights_arr).any():
            print("[ENSEMBLE WARN] Invalid weights; using equal weights.")
            weights_arr = np.ones_like(weights_arr)
        weights_arr = weights_arr / weights_arr.sum()

        method = self.method
        if method == "rank":
            # Rank-normalize each model prediction to [0,1]
            def _rank_norm(s: np.ndarray) -> np.ndarray:
                s = np.asarray(s, dtype=float)
                if s.size <= 1 or not np.isfinite(s).any():
                    return np.zeros_like(s, dtype=float)
                # All equal → neutral 0.5
                if np.nanstd(s) == 0.0:
                    return np.full_like(s, 0.5, dtype=float)
                # Average ranks (descending): higher value -> higher score
                ser = pd.Series(s)
                r = ser.rank(method='average', ascending=False)
                # Normalize to [0,1]
                mn, mx = float(r.min()), float(r.max())
                if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
                    return np.full_like(s, 0.5, dtype=float)
                return (r - mn) / (mx - mn)
            rn = np.vstack([_rank_norm(p) for p in preds])
            return np.average(rn, axis=0, weights=weights_arr)
        elif method == "voting":
            # Binary votes based on threshold; return vote proportion in [0,1]
            votes = (preds > self.voting_threshold).astype(float)
            return np.average(votes, axis=0, weights=weights_arr)
        elif method == "prob":
            # Probability averaging; clip to [0,1]
            avg = np.average(preds, axis=0, weights=weights_arr)
            return np.clip(avg, 0.0, 1.0)
        else:
            # Weighted average on raw predictions
            return np.average(preds, axis=0, weights=weights_arr)
