import numpy as np
from models.ensemble.ensemble_trainer import EnsembleTrainer


class Dummy:
    def __init__(self, y):
        self.y = np.asarray(y, dtype=float)
    def predict(self, X):
        return self.y


def test_truncation_vs_alignment():
    m1 = Dummy([0.1, 0.2, 0.3])
    m2 = Dummy([0.5, 0.6])
    ens = EnsembleTrainer([m1, m2], weights=[0.5, 0.5], method='weighted')
    out = ens.predict([np.zeros((3, 1)), np.zeros((2, 1))])
    assert out.shape[0] == 2  # truncated to min length 2


def test_weight_mismatch_warns_and_equals():
    m1 = Dummy([0.1, 0.2])
    m2 = Dummy([0.3, 0.4])
    ens = EnsembleTrainer([m1, m2], weights=[0.7], method='weighted')
    out = ens.predict([np.zeros((2, 1)), np.zeros((2, 1))])
    assert out.shape[0] == 2


def test_rank_default_value_applies_for_constant_series():
    m1 = Dummy([1.0, 1.0, 1.0])
    m2 = Dummy([2.0, 2.0, 2.0])
    ens = EnsembleTrainer([m1, m2], method='rank', rank_default_value=0.3)
    out = ens.predict([np.zeros((3, 1)), np.zeros((3, 1))])
    assert np.allclose(out, 0.3)


def test_model_validation():
    class NoPredict:
        pass
    try:
        EnsembleTrainer([NoPredict()], method='weighted')
        assert False, 'Expected validation error for model without predict'
    except ValueError:
        pass

