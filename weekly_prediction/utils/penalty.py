# utils/penalty.py
import numpy as np
import os
import logging

_DEFAULTS_LOGGED = False


def compute_penalty_weight(
    rsi: float,
    vol_ratio: float,
    price_chg_pct: float,
    config: dict = None
) -> float:
    """Compute penalty weight based on RSI, volume ratio, and price change.

    Args:
        rsi (float): Relative Strength Index (e.g., rsi_14d).
        vol_ratio (float): Volume ratio (e.g., volume/volume_ma_20).
        price_chg_pct (float): Percentage price change (e.g., return_1d).
        config (dict, optional): Configuration parameters. Defaults to environment variables.

    Returns:
        float: Penalty weight clipped to [W_MIN, W_MAX].
    """
    global _DEFAULTS_LOGGED
    if config is None:
        # Read from env with defaults; log once if defaults are used
        env_specs = {
            'PEN_ALPHA': '8.0',
            'RSI_GAMMA': '0.35',
            'RSI_BONUS': '0.10',
            'PEN_W_MIN': '0.50',
            'PEN_W_MAX': '1.10',
        }
        used_defaults = [k for k, d in env_specs.items() if os.getenv(k) is None]
        config = {
            'PEN_ALPHA': float(os.getenv('PEN_ALPHA', env_specs['PEN_ALPHA'])),
            'RSI_GAMMA': float(os.getenv('RSI_GAMMA', env_specs['RSI_GAMMA'])),
            'RSI_BONUS': float(os.getenv('RSI_BONUS', env_specs['RSI_BONUS'])),
            'W_MIN': float(os.getenv('PEN_W_MIN', env_specs['PEN_W_MIN'])),
            'W_MAX': float(os.getenv('PEN_W_MAX', env_specs['PEN_W_MAX'])),
        }
        if used_defaults and not _DEFAULTS_LOGGED:
            logging.getLogger(__name__).info(
                f"[PENALTY] Using default env values for: {', '.join(used_defaults)}"
            )
            _DEFAULTS_LOGGED = True
    
    # Clamp inputs to safe ranges
    rsi_val = max(0.0, min(100.0, float(rsi)))
    vol_ratio_val = max(0.0, float(vol_ratio))
    price_chg_val = float(price_chg_pct)

    vr_excess = max(0.0, vol_ratio_val - 1.0)
    drop_mag = max(0.0, -price_chg_val)
    w1 = float(np.exp(-config['PEN_ALPHA'] * vr_excess * drop_mag))
    over = max(0.0, (rsi_val - 70.0) / 30.0)
    w2 = float(1.0 - config['RSI_GAMMA'] * min(1.0, over))
    w3 = 1.0
    if (price_chg_val > 0.0) and (rsi_val < 45.0) and (vol_ratio_val <= 1.2):
        w3 = float(1.0 + config['RSI_BONUS'] * ((45.0 - rsi_val) / 45.0))
    
    return float(np.clip(w1 * w2 * w3, config['W_MIN'], config['W_MAX']))