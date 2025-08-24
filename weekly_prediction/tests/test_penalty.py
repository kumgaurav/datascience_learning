import math
from utils.penalty import compute_penalty_weight


def test_penalty_weight_typical_values():
    w = compute_penalty_weight(rsi=60.0, vol_ratio=1.1, price_chg_pct=-0.02)
    assert 0.5 <= w <= 1.1


def test_penalty_weight_bonus_when_low_rsi_positive_return_low_vol():
    w = compute_penalty_weight(rsi=30.0, vol_ratio=1.0, price_chg_pct=0.01)
    assert w > 1.0


def test_penalty_weight_clipped_bounds():
    w_low = compute_penalty_weight(rsi=100.0, vol_ratio=3.0, price_chg_pct=-0.2)
    w_high = compute_penalty_weight(rsi=0.0, vol_ratio=0.5, price_chg_pct=0.2)
    assert 0.5 <= w_low <= 1.1
    assert 0.5 <= w_high <= 1.1


def test_penalty_input_clamps():
    # Negative RSI should clamp to 0, vol_ratio clamps to >=0
    w_neg = compute_penalty_weight(rsi=-10.0, vol_ratio=-5.0, price_chg_pct=0.1)
    assert 0.5 <= w_neg <= 1.1


