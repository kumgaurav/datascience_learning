import os
import logging
from typing import Any, Dict, Optional


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON. Returns {} if file missing or parsing fails.

    Supports simple nested structures like:
    logging: { level: INFO, file: app.log }
    penalty: { alpha: 8.0, rsi_gamma: 0.35, rsi_bonus: 0.10, w_min: 0.5, w_max: 1.1 }
    ensemble: { top_n: 25, method: weighted, alpha: 0.6, stack_model: null }
    lstm: { model_path: ..., meta_path: ..., feature_cols: ..., pad_short_seqs: false }
    """
    if not path or not isinstance(path, str) or not os.path.isfile(path):
        return {}
    # Try YAML first
    try:
        import yaml  # type: ignore
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load YAML config from {path}: {e}")
    # Try JSON
    try:
        import json
        with open(path, 'r') as f:
            return json.load(f) or {}
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load JSON config from {path}: {e}")
        return {}


def _cfg_get(cfg: Dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def apply_env_from_config(cfg: Dict[str, Any], skip_existing: bool = True) -> None:
    """Map selected config values to environment variables consumed by modules."""
    # Logging
    lvl = _cfg_get(cfg, ['logging','level'], _cfg_get(cfg, ['log_level']))
    if lvl is not None and (not skip_existing or 'LOG_LEVEL' not in os.environ):
        os.environ['LOG_LEVEL'] = str(lvl)
    lf = _cfg_get(cfg, ['logging','file'], _cfg_get(cfg, ['log_file']))
    if lf and (not skip_existing or 'LOG_FILE' not in os.environ):
        os.environ['LOG_FILE'] = str(lf)

    # Penalty
    pen = _cfg_get(cfg, ['penalty'], {}) or {}
    def _num_or_none(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            if isinstance(v, bool):
                return float(v)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str) and v.strip() != '':
                return float(v)
        except Exception:
            return None
        return None
    mapping = {
        'PEN_ALPHA': _num_or_none(pen.get('alpha', _cfg_get(cfg, ['pen_alpha']))),
        'RSI_GAMMA': _num_or_none(pen.get('rsi_gamma', _cfg_get(cfg, ['rsi_gamma']))),
        'RSI_BONUS': _num_or_none(pen.get('rsi_bonus', _cfg_get(cfg, ['rsi_bonus']))),
        'PEN_W_MIN': _num_or_none(pen.get('w_min', _cfg_get(cfg, ['pen_w_min']))),
        'PEN_W_MAX': _num_or_none(pen.get('w_max', _cfg_get(cfg, ['pen_w_max']))),
    }
    for env_key, val in mapping.items():
        if val is not None and (not skip_existing or env_key not in os.environ):
            os.environ[env_key] = str(val)

    # Ensemble
    ens = _cfg_get(cfg, ['ensemble'], {}) or {}
    ens_map = {
        'ENSEMBLE_TOP_N': ens.get('top_n', _cfg_get(cfg, ['ensemble_top_n'])),
        'ENSEMBLE_METHOD': ens.get('method', _cfg_get(cfg, ['ensemble_method'])),
        'ENSEMBLE_ALPHA': ens.get('alpha', _cfg_get(cfg, ['ensemble_alpha'])),
        'ENSEMBLE_STACK_MODEL': ens.get('stack_model', _cfg_get(cfg, ['ensemble_stack_model'])),
    }
    for env_key, val in ens_map.items():
        if val is not None and (not skip_existing or env_key not in os.environ):
            os.environ[env_key] = str(val)

    # LSTM
    lstm = _cfg_get(cfg, ['lstm'], {}) or {}
    lstm_map = {
        'LSTM_MODEL_PATH': lstm.get('model_path', _cfg_get(cfg, ['lstm_model_path'])),
        'LSTM_META_PATH': lstm.get('meta_path', _cfg_get(cfg, ['lstm_meta_path'])),
        'LSTM_FEATURE_COLS': lstm.get('feature_cols', _cfg_get(cfg, ['lstm_feature_cols'])),
        'LSTM_PAD_SHORT_SEQS': lstm.get('pad_short_seqs', _cfg_get(cfg, ['lstm_pad_short_seqs'])),
        # Training hyperparameters
        'LSTM_LOOKBACK': lstm.get('lookback', _cfg_get(cfg, ['lstm_lookback'])),
        'LSTM_HORIZON': lstm.get('horizon', _cfg_get(cfg, ['lstm_horizon'])),
        'LSTM_EPOCHS': lstm.get('epochs', _cfg_get(cfg, ['lstm_epochs'])),
        'LSTM_BATCH_SIZE': lstm.get('batch_size', _cfg_get(cfg, ['lstm_batch_size'])),
        'LSTM_VALIDATION_SPLIT': lstm.get('validation_split', _cfg_get(cfg, ['lstm_validation_split'])),
        'LSTM_EARLY_STOPPING_PATIENCE': lstm.get('early_stopping_patience', _cfg_get(cfg, ['lstm_early_stopping_patience'])),
        'LSTM_REDUCE_LR_PATIENCE': lstm.get('reduce_lr_patience', _cfg_get(cfg, ['lstm_reduce_lr_patience'])),
        'LSTM_USE_SEQUENCE_WEIGHTS': lstm.get('use_sequence_weights', _cfg_get(cfg, ['lstm_use_sequence_weights'])),
        'LSTM_USE_PENALTY_WEIGHTS': lstm.get('use_penalty_weights', _cfg_get(cfg, ['lstm_use_penalty_weights'])),
        'LSTM_PENALTY_WEIGHT_MODE': lstm.get('penalty_weight_mode', _cfg_get(cfg, ['lstm_penalty_weight_mode'])),
        'LSTM_USE_CURRICULUM': lstm.get('use_curriculum', _cfg_get(cfg, ['lstm_use_curriculum'])),
        'LSTM_CURRICULUM_WARMUP_EPOCHS': lstm.get('curriculum_warmup_epochs', _cfg_get(cfg, ['lstm_curriculum_warmup_epochs'])),
        'LSTM_BIDIRECTIONAL': lstm.get('bidirectional', _cfg_get(cfg, ['lstm_bidirectional'])),
    }
    for env_key, val in lstm_map.items():
        if val is not None and (not skip_existing or env_key not in os.environ):
            os.environ[env_key] = str(val)

    # Momentum model
    momentum = _cfg_get(cfg, ['momentum'], {}) or {}
    momentum_map = {
        'MOMENTUM_MODEL_PATH': momentum.get('model_path', _cfg_get(cfg, ['momentum_model_path'])),
        'MOMENTUM_FEATURES_PATH': momentum.get('features_path', _cfg_get(cfg, ['momentum_features_path'])),
    }
    for env_key, val in momentum_map.items():
        if val is not None and (not skip_existing or env_key not in os.environ):
            os.environ[env_key] = str(val)

    # Feature engineering
    feats = _cfg_get(cfg, ['features'], {}) or {}
    feats_map = {
        'FE_RSI_PERIOD': feats.get('rsi_period', _cfg_get(cfg, ['fe_rsi_period'])),
        'FE_MACD_FAST': feats.get('macd_fast', _cfg_get(cfg, ['fe_macd_fast'])),
        'FE_MACD_SLOW': feats.get('macd_slow', _cfg_get(cfg, ['fe_macd_slow'])),
        'FE_MACD_SIGNAL': feats.get('macd_signal', _cfg_get(cfg, ['fe_macd_signal'])),
        'FE_BB_WINDOW': feats.get('bb_window', _cfg_get(cfg, ['fe_bb_window'])),
        'FE_BB_K': feats.get('bb_k', _cfg_get(cfg, ['fe_bb_k'])),
        'FE_FILL_SLOPES_FFILL': feats.get('fill_slopes_ffill', _cfg_get(cfg, ['fe_fill_slopes_ffill'])),
    }
    for env_key, val in feats_map.items():
        if val is not None and (not skip_existing or env_key not in os.environ):
            os.environ[env_key] = str(val)

    # Paths (inputs/outputs)
    paths = _cfg_get(cfg, ['paths'], {}) or {}
    paths_map = {
        'PATH_CLEAN_PRICES': paths.get('clean_prices', _cfg_get(cfg, ['clean_prices'])),
        'PATH_UNCLEAN_PRICES': paths.get('unclean_prices', _cfg_get(cfg, ['unclean_prices'])),
        'PATH_FEATURES_CLEAN': paths.get('features_clean', _cfg_get(cfg, ['features_clean'])),
        'PATH_XGB_OUT': paths.get('xgb_out', _cfg_get(cfg, ['xgb_out'])),
        'PATH_LSTM_OUT': paths.get('lstm_out', _cfg_get(cfg, ['lstm_out'])),
        'PATH_ENSEMBLE_OUT': paths.get('ensemble_out', _cfg_get(cfg, ['ensemble_out'])),
        'PATH_MOMENTUM_OUT': paths.get('momentum_out', _cfg_get(cfg, ['momentum_out'])),
        # UI-specific CSVs
        'EARNINGS_HISTORY_CSV': paths.get('earnings_history', _cfg_get(cfg, ['earnings_history'])),
    }
    for env_key, val in paths_map.items():
        if val is not None and (not skip_existing or env_key not in os.environ):
            os.environ[env_key] = str(val)

    # XGBoost
    xgb = _cfg_get(cfg, ['xgboost'], {}) or {}
    xgb_map = {
        'XGB_K': xgb.get('k', _cfg_get(cfg, ['xgb_k'])),
        'XGB_TUNER_TRIALS': xgb.get('tuner_trials', _cfg_get(cfg, ['xgb_tuner_trials'])),
        'XGB_TUNER_METRIC': xgb.get('tuner_metric', _cfg_get(cfg, ['xgb_tuner_metric'])),
        'XGB_OBJECTIVE': xgb.get('objective', _cfg_get(cfg, ['xgb_objective'])),
        'NDCG_GAIN': xgb.get('ndcg_gain', _cfg_get(cfg, ['ndcg_gain'])),
        'XGB_IMPORTANCE_TYPE': xgb.get('importance_type', _cfg_get(cfg, ['xgb_importance_type'])),
        'XGB_N_ESTIMATORS': xgb.get('n_estimators', _cfg_get(cfg, ['xgb_n_estimators'])),
        'XGB_RANDOM_STATE': xgb.get('random_state', _cfg_get(cfg, ['xgb_random_state'])),
        'XGB_N_JOBS': xgb.get('n_jobs', _cfg_get(cfg, ['xgb_n_jobs'])),
        'SELLOFF_TRAIN_WEIGHT_FACTOR': xgb.get('selloff_train_weight_factor', _cfg_get(cfg, ['selloff_train_weight_factor'])),
        'OVERSOLD_BOUNCE_WEIGHT_FACTOR': xgb.get('oversold_bounce_weight_factor', _cfg_get(cfg, ['oversold_bounce_weight_factor'])),
    }
    for env_key, val in xgb_map.items():
        if val is not None and (not skip_existing or env_key not in os.environ):
            os.environ[env_key] = str(val)


