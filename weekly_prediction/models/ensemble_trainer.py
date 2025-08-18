# models/ensemble_trainer.py
import numpy as np
from typing import List, Optional

class EnsembleTrainer:
    def __init__(self, models: List[object], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights

    def predict(self, Xs: List[np.ndarray]):
        collected = []
        weights = []
        for idx, (m, X) in enumerate(zip(self.models, Xs)):
            try:
                pred = m.predict(X)
                if pred is None:
                    continue
                pred = np.asarray(pred).ravel()
                if not np.isfinite(pred).all():
                    continue
                collected.append(pred)
                if self.weights is not None and idx < len(self.weights):
                    w = float(self.weights[idx])
                    if not np.isfinite(w) or w <= 0:
                        w = 0.0
                else:
                    w = 1.0
                weights.append(w)
            except Exception:
                # Skip failing model
                continue

        if not collected:
            raise RuntimeError("No valid model predictions for ensemble")

        preds = np.vstack(collected)
        weights = np.asarray(weights, dtype=float)
        if weights.size != preds.shape[0]:
            # Fallback to equal weights
            weights = np.ones(preds.shape[0], dtype=float)
        if weights.sum() <= 0:
            # If all provided weights are zero/invalid, fallback to equal average
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        return np.average(preds, axis=0, weights=weights)
