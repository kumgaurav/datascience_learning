import os
from utils.config import load_config, apply_env_from_config


def test_load_yaml_and_apply_env(tmp_path, monkeypatch):
    cfg_content = """
logging:
  level: DEBUG
  file: logs/test.log
penalty:
  alpha: 7.5
  rsi_gamma: "0.40"
  rsi_bonus: 0.1
  w_min: 0.55
  w_max: 1.05
ensemble:
  method: rank
  alpha: 0.7
  top_n: 30
lstm:
  pad_short_seqs: true
"""
    p = tmp_path / 'cfg.yaml'
    p.write_text(cfg_content)
    # Clear related envs to ensure they get set
    for k in ['LOG_LEVEL','LOG_FILE','PEN_ALPHA','RSI_GAMMA','RSI_BONUS','PEN_W_MIN','PEN_W_MAX','ENSEMBLE_METHOD','ENSEMBLE_ALPHA','ENSEMBLE_TOP_N','LSTM_PAD_SHORT_SEQS']:
        monkeypatch.delenv(k, raising=False)

    cfg = load_config(str(p))
    apply_env_from_config(cfg)

    assert os.environ.get('LOG_LEVEL') == 'DEBUG'
    assert os.environ.get('LOG_FILE').endswith('logs/test.log')
    assert os.environ.get('PEN_ALPHA') == '7.5'
    assert os.environ.get('RSI_GAMMA') == '0.4'
    assert os.environ.get('RSI_BONUS') == '0.1'
    assert os.environ.get('PEN_W_MIN') == '0.55'
    assert os.environ.get('PEN_W_MAX') == '1.05'
    assert os.environ.get('ENSEMBLE_METHOD') == 'rank'
    assert os.environ.get('ENSEMBLE_ALPHA') == '0.7'
    assert os.environ.get('ENSEMBLE_TOP_N') == '30'
    assert os.environ.get('LSTM_PAD_SHORT_SEQS') in ('True','true','1')


def test_apply_env_skip_existing(monkeypatch, tmp_path):
    monkeypatch.setenv('PEN_ALPHA', '9.9')
    p = tmp_path / 'cfg.yaml'
    p.write_text('penalty: { alpha: 7.5 }')
    cfg = load_config(str(p))
    apply_env_from_config(cfg, skip_existing=True)
    # should not overwrite
    assert os.environ.get('PEN_ALPHA') == '9.9'


def test_apply_env_overwrite_when_false(monkeypatch, tmp_path):
    monkeypatch.setenv('PEN_ALPHA', '9.9')
    p = tmp_path / 'cfg2.yaml'
    p.write_text('penalty: { alpha: 7.5 }')
    cfg = load_config(str(p))
    apply_env_from_config(cfg, skip_existing=False)
    # should overwrite
    assert os.environ.get('PEN_ALPHA') == '7.5'


