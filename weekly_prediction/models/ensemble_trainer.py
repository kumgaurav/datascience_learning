# models/ensemble_trainer.py
import numpy as np

class EnsembleTrainer:
    def __init__(self, models):
        self.models = models

    def predict(self, Xs):
        preds = [m.predict(X) for m, X in zip(self.models, Xs)]
        return np.mean(preds, axis=0)
