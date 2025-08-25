from prediction_model_base import BaseModelTrainer


class MomentumModelTrainer(BaseModelTrainer):
    def __init__(self) -> None:
        super().__init__(
            features_path='data/momentum/featured_stocks_momentum.csv',
            model_path='models/stock_predictor_momentum.joblib',
        )


def train_momentum_model() -> str:
    return MomentumModelTrainer().train()


