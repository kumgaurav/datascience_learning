import os
import pandas as pd


def test_price_levels_available_from_features():
    feats_path = os.getenv('PATH_FEATURES_CLEAN', 'data/features/stock_features_clean.csv')
    if not os.path.isfile(feats_path):
        # Skip if features file not present in CI/dev
        return
    df = pd.read_csv(feats_path)
    # Required columns for UI price levels
    required = ['ticker', 'date', 'close']
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required columns in features: {missing}"
    # Support/resistance may be absent; verify we can compute from last 20 closes per ticker
    any_ticker = df['ticker'].astype(str).str.upper().unique()[:1]
    for t in any_ticker:
        td = df[df['ticker'].astype(str).str.upper() == t].copy()
        td['date'] = pd.to_datetime(td['date'], errors='coerce')
        td = td.sort_values('date').tail(20)
        assert len(td) > 0, "No recent closes available to compute support/resistance"
        sup = pd.to_numeric(td['close'], errors='coerce').min()
        res = pd.to_numeric(td['close'], errors='coerce').max()
        assert pd.notna(sup) and pd.notna(res), "Computed support/resistance are NaN"
        assert res >= sup, "Resistance should be >= support"


def test_earnings_history_has_expected_columns():
    path = os.getenv('EARNINGS_HISTORY_CSV', 'data/input/earnings_history.csv')
    if not os.path.isfile(path):
        return
    df = pd.read_csv(path)
    # Accept common aliases but ensure canonical columns resolvable
    lower = {c.lower(): c for c in df.columns}
    # canonical set
    canonical = ['ticker','earnings_date','reported_eps','estimate_eps','surprise_percentage']
    aliases = {
        'symbol': 'ticker',
        'date': 'earnings_date',
        'actualeps': 'reported_eps',
        'eps': 'reported_eps',
        'epsestimate': 'estimate_eps',
        'estimate': 'estimate_eps',
        'surprise_percent': 'surprise_percentage',
        'surprise_pct': 'surprise_percentage',
    }
    resolved = []
    for c in canonical:
        if c in lower:
            resolved.append(c)
        else:
            # try any alias
            got = None
            for a, tgt in aliases.items():
                if tgt == c and a in lower:
                    got = a
                    break
            assert got is not None, f"Missing column for {c} (allow aliases: {[k for k,v in aliases.items() if v==c]})"
            resolved.append(got)


