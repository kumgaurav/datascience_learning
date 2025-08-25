import os
import pandas as pd


def _apply_config_env():
    # Try to apply env from config to resolve paths
    try:
        from utils.config import load_config, apply_env_from_config
        cfg_path = os.getenv('APP_CONFIG', 'config.yaml')
        cfg = load_config(cfg_path)
        apply_env_from_config(cfg)
    except Exception:
        pass


def test_ui_required_files_exist_and_report_missing():
    _apply_config_env()
    # Logical name -> resolved path via env or default
    candidates = {
        'features_clean': os.getenv('PATH_FEATURES_CLEAN', 'data/features/stock_features_clean.csv'),
        'ensemble_out': os.getenv('PATH_ENSEMBLE_OUT', 'data/ensemble/ensemble_weekly_output.csv'),
        'xgb_out': os.getenv('PATH_XGB_OUT', 'data/xgboost/xgboost_weekly_output.csv'),
        'lstm_out': os.getenv('PATH_LSTM_OUT', 'data/lstm/lstm_weekly_output.csv'),
        'earnings_history': os.getenv('EARNINGS_HISTORY_CSV', 'data/input/earnings_history.csv'),
        'prices_clean': 'data/input/stock_prices_with_clean_data.csv',
    }
    missing = [(name, path) for name, path in candidates.items() if not os.path.isfile(path)]
    assert not missing, (
        'Missing required UI data files:\n' +
        '\n'.join([f"- {name}: expected at {path}" for name, path in missing]) +
        "\nSet APP_CONFIG=config.yaml and/or adjust paths.* in config.yaml (paths section) or export envs accordingly."
    )


def test_minimal_columns_in_present_files():
    _apply_config_env()
    # Only validate files that exist; this test is supplemental
    validations = []
    # Prices clean: require ticker,date,close
    pc = 'data/input/stock_prices_with_clean_data.csv'
    if os.path.isfile(pc):
        df = pd.read_csv(pc, nrows=5)
        need = {'ticker', 'date', 'close'}
        validations.append((pc, need.issubset(set(df.columns)), need - set(df.columns)))
    # Earnings: canonical or aliases resolvable
    eh = os.getenv('EARNINGS_HISTORY_CSV', 'data/input/earnings_history.csv')
    if os.path.isfile(eh):
        df = pd.read_csv(eh, nrows=5)
        lower = {c.lower(): c for c in df.columns}
        required = ['ticker','earnings_date','reported_eps','estimate_eps','surprise_percentage']
        aliases = {
            'symbol': 'ticker', 'date': 'earnings_date',
            'actualeps': 'reported_eps', 'eps': 'reported_eps',
            'epsestimate': 'estimate_eps', 'estimate': 'estimate_eps',
            'surprise_percent': 'surprise_percentage', 'surprise_pct': 'surprise_percentage',
        }
        ok = True
        missing = []
        for c in required:
            if c in lower:
                continue
            found = any(a in lower for a, tgt in aliases.items() if tgt == c)
            if not found:
                ok = False
                missing.append(c)
        validations.append((eh, ok, missing))
    # Weekly outputs: require ticker column
    for p in [
        os.getenv('PATH_ENSEMBLE_OUT', 'data/ensemble/ensemble_weekly_output.csv'),
        os.getenv('PATH_XGB_OUT', 'data/xgboost/xgboost_weekly_output.csv'),
        os.getenv('PATH_LSTM_OUT', 'data/lstm/lstm_weekly_output.csv'),
    ]:
        if os.path.isfile(p):
            df = pd.read_csv(p, nrows=5)
            validations.append((p, 'ticker' in df.columns, ['ticker'] if 'ticker' not in df.columns else []))

    # Report any schema issues
    bad = [(p, miss) for p, ok, miss in validations if not ok]
    assert not bad, 'Files missing required columns:\n' + '\n'.join([f"- {p}: missing {miss}" for p, miss in bad])


