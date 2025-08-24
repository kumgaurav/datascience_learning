import os
import sys
from typing import Dict, Optional

import pandas as pd


def _get_engine():
    """Create a SQLAlchemy engine.

    Resolution order:
      1) Env DATABASE_URL
      2) conf/database_url.txt (single URL line)
      3) conf/db_config.json (driver, user, password, host, port, database)
      4) conf/database.yml (rails-like) if PyYAML available; use ['default'] or ['production']
      5) Component envs: DB_DRIVER, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
    """
    url = os.getenv("DATABASE_URL", "").strip()
    conf_dir = os.path.join(os.getcwd(), "conf")

    def _compose_url(cfg: dict) -> str:
        drv = cfg.get("driver", "postgresql+psycopg2")
        user = cfg.get("user", "")
        pwd = cfg.get("password", "")
        host = cfg.get("host", "localhost")
        port = cfg.get("port", "")
        db = cfg.get("database", "")
        auth = f"{user}:{pwd}@" if user or pwd else ""
        port_part = f":{port}" if port else ""
        return f"{drv}://{auth}{host}{port_part}/{db}"

    if not url:
        # Try conf/database_url.txt
        try:
            path_txt = os.path.join(conf_dir, "database_url.txt")
            if os.path.exists(path_txt):
                with open(path_txt, "r") as f:
                    url = f.read().strip()
                    if url:
                        print(f"[DB] Using DATABASE_URL from {path_txt}")
        except Exception:
            pass

    if not url:
        # Try conf/db_config.json
        try:
            import json  # noqa: F401
            path_json = os.path.join(conf_dir, "db_config.json")
            if os.path.exists(path_json):
                with open(path_json, "r") as f:
                    cfg = json.load(f)
                url = _compose_url(cfg)
                print(f"[DB] Using DB config from {path_json}")
        except Exception as e:
            print(f"[DB] Failed to read db_config.json: {e}")

    if not url:
        # Try conf/database.yml
        try:
            path_yml = os.path.join(conf_dir, "database.yml")
            if os.path.exists(path_yml):
                try:
                    import yaml  # type: ignore
                except Exception:
                    yaml = None
                if yaml is not None:
                    with open(path_yml, "r") as f:
                        y = yaml.safe_load(f)
                    node = y.get("production") or y.get("default") or y
                    cfg = {
                        "driver": node.get("driver", "postgresql+psycopg2"),
                        "user": node.get("username") or node.get("user"),
                        "password": node.get("password"),
                        "host": node.get("host", "localhost"),
                        "port": node.get("port"),
                        "database": node.get("database") or node.get("dbname"),
                    }
                    url = _compose_url(cfg)
                    print(f"[DB] Using DB config from {path_yml}")
                else:
                    print("[DB] PyYAML not installed; cannot parse conf/database.yml")
        except Exception as e:
            print(f"[DB] Failed to read database.yml: {e}")

    if not url:
        # Try conf/config.ini ([mysql])
        try:
            import configparser
            path_ini = os.path.join(conf_dir, "config.ini")
            if os.path.exists(path_ini):
                cfgp = configparser.ConfigParser()
                cfgp.read(path_ini)
                if cfgp.has_section("mysql"):
                    section = cfgp["mysql"]
                    cfg = {
                        "driver": section.get("driver", fallback="mysql+pymysql" if section else "mysql+pymysql"),
                        "user": section.get("username", fallback=""),
                        "password": section.get("password", fallback=""),
                        "host": section.get("url", fallback="localhost"),
                        "port": section.get("port", fallback=""),
                        "database": section.get("database", fallback=""),
                    }
                    # Map Java driver to SQLAlchemy
                    if cfg.get("driver") == "com.mysql.cj.jdbc.Driver":
                        cfg["driver"] = "mysql+pymysql"
                    url = _compose_url(cfg)
                    print(f"[DB] Using DB config from {path_ini} [mysql]")
        except Exception as e:
            print(f"[DB] Failed to read config.ini: {e}")

    if not url:
        # Try component envs
        drv = os.getenv("DB_DRIVER")
        user = os.getenv("DB_USER")
        pwd = os.getenv("DB_PASSWORD")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        name = os.getenv("DB_NAME")
        if host and name:
            url = _compose_url({
                "driver": drv or "postgresql+psycopg2",
                "user": user or "",
                "password": pwd or "",
                "host": host,
                "port": port or "",
                "database": name,
            })
            print("[DB] Using DB connection from component env vars")

    if not url:
        print("[DB] No DATABASE_URL or conf-based DB config found.")
        return None

    try:
        from sqlalchemy import create_engine  # type: ignore
    except Exception:
        print("[DB] SQLAlchemy not installed. Please `pip install sqlalchemy psycopg2-binary`.")
        return None
    try:
        return create_engine(url)
    except Exception as e:
        print(f"[DB] Failed to create engine from URL: {e}")
        return None


def fetch_table(engine, table: str, date_col: Optional[str] = None,
                start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch a whole table (optionally date-filtered) into a DataFrame.

    - engine: SQLAlchemy engine
    - table: table or schema.table
    - date_col: if provided, applies BETWEEN filter with start_date/end_date
    - start_date/end_date: ISO strings (YYYY-MM-DD)
    """
    if engine is None:
        print("[DB] No engine available; returning empty frame.")
        return pd.DataFrame()

    def _run(dc: Optional[str]) -> pd.DataFrame:
        if dc and (start_date or end_date):
            conds = []
            if start_date:
                conds.append(f"{dc} >= '{start_date}'")
            if end_date:
                conds.append(f"{dc} <= '{end_date}'")
            where = " WHERE " + " AND ".join(conds)
        else:
            where = ""
        sql = f"SELECT * FROM {table}{where}"
        return pd.read_sql(sql, engine)

    # Attempt with provided date_col first
    try:
        return _run(date_col)
    except Exception as e:
        print(f"[DB] Query failed for {table} (date_col={date_col}): {e}")
        # Try to infer a date column and retry
        try:
            probe = pd.read_sql(f"SELECT * FROM {table} LIMIT 1", engine)
            candidates = [
                'report_date', 'date', 'earnings_date', 'fiscaldateending', 'fiscal_date', 'as_of_date', 'period'
            ]
            picked = None
            lower = {c.lower(): c for c in probe.columns}
            for cand in candidates:
                if cand in lower:
                    picked = lower[cand]
                    break
            if picked:
                print(f"[DB] Retrying {table} with detected date column: {picked}")
                try:
                    return _run(picked)
                except Exception as e2:
                    print(f"[DB] Retry failed for {table} with {picked}: {e2}")
            # Fallback: fetch without date filter
            print(f"[DB] Falling back to full fetch without date filter for {table}")
            return _run(None)
        except Exception as eprobe:
            print(f"[DB] Probe failed for {table}: {eprobe}")
            return pd.DataFrame()


def fetch_all(output_dir: str = "data/input", start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, str]:
    """Fetch required datasets and write CSVs. Returns map of logical name -> path.

    Controlled by env table names (defaults provided):
      STOCK_PRICES_TABLE (default: stock_prices)
      STOCK_EARNINGS_TABLE (default: stock_earnings)
      EARNINGS_EST_TABLE (default: earnings_estimates)
      Q_INCOME_TABLE (default: quarterly_income)
      Q_REVENUE_TABLE (default: quarterly_revenue)
      REVENUE_EST_TABLE (default: revenue_estimates)
      GROWTH_EST_TABLE (default: growth_estimates)
    """
    os.makedirs(output_dir, exist_ok=True)
    engine = _get_engine()

    # Default table names
    tables = {
        "stock_prices": os.getenv("STOCK_PRICES_TABLE", "stock_prices"),
        "stock_earnings": os.getenv("STOCK_EARNINGS_TABLE", "stock_earnings"),
        "earnings_estimates": os.getenv("EARNINGS_EST_TABLE", "earnings_estimates"),
        "quarterly_income": os.getenv("Q_INCOME_TABLE", "quarterly_income"),
        "quarterly_revenue": os.getenv("Q_REVENUE_TABLE", "quarterly_revenue"),
        "revenue_estimates": os.getenv("REVENUE_EST_TABLE", "revenue_estimates"),
        "growth_estimates": os.getenv("GROWTH_EST_TABLE", "growth_estimates"),
        # Direct earnings history export (preferred): set EARNINGS_HISTORY_TABLE to a view/table providing the desired schema
        "earnings_history": os.getenv("EARNINGS_HISTORY_TABLE", "earnings_history"),
    }

    # Override table names from conf/config.ini if present; collect date col overrides too
    try:
        import configparser
        path_ini = os.path.join(os.getcwd(), "conf", "config.ini")
        if os.path.exists(path_ini):
            cfgp = configparser.ConfigParser()
            cfgp.read(path_ini)
            if cfgp.has_section("mysql"):
                section = cfgp["mysql"]
                # Prices table
                t_prices = section.get("table", fallback=None)
                if t_prices:
                    tables["stock_prices"] = t_prices
                # Earnings
                t_earn = section.get("earning_table", fallback=None)
                if t_earn:
                    tables["stock_earnings"] = t_earn
                # Revenue/quarterly
                t_rev = section.get("revenue_table", fallback=None)
                if t_rev:
                    # Some schemas store both income and revenue in the same table
                    tables["quarterly_revenue"] = t_rev
                print(f"[DB] Using table overrides from conf/config.ini: {tables}")
                # Date column overrides
                ini_date_overrides = {}
                if section.get("prices_date_col", fallback=None):
                    ini_date_overrides["stock_prices"] = section.get("prices_date_col")
                if section.get("earnings_date_col", fallback=None):
                    ini_date_overrides["stock_earnings"] = section.get("earnings_date_col")
                if section.get("income_date_col", fallback=None):
                    ini_date_overrides["quarterly_income"] = section.get("income_date_col")
                if section.get("revenue_date_col", fallback=None):
                    ini_date_overrides["quarterly_revenue"] = section.get("revenue_date_col")
                if section.get("rev_est_date_col", fallback=None):
                    ini_date_overrides["revenue_estimates"] = section.get("rev_est_date_col")
            else:
                ini_date_overrides = {}
        else:
            ini_date_overrides = {}
    except Exception as e:
        print(f"[DB] Failed to read table overrides from config.ini: {e}")
        ini_date_overrides = {}

    date_col_map = {
        "stock_prices": "date",
        "stock_earnings": "earnings_date",
        "earnings_estimates": "earnings_date",
        "quarterly_income": "report_date",
        "quarterly_revenue": "report_date",
        "revenue_estimates": "next_earnings_date",
        "growth_estimates": None,
        "earnings_history": "earnings_date",
    }
    # Apply INI overrides
    if ini_date_overrides:
        date_col_map.update(ini_date_overrides)

    def _normalize_prices_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure standard columns ticker/date/close exist with best-effort mapping."""
        if df is None or df.empty:
            return df
        out = df.copy()
        lower_map = {c.lower(): c for c in out.columns}
        # Candidates by priority
        t_cands = ['ticker', 'symbol', 'security', 'code']
        d_cands = ['date', 'trade_date', 'timestamp', 'datetime', 'as_of_date']
        c_cands = ['close', 'adj_close', 'close_price', 'last', 'closeprice', 'c']
        def _pick(cands):
            for k in cands:
                if k in lower_map:
                    return lower_map[k]
            return None
        tcol = _pick(t_cands)
        dcol = _pick(d_cands)
        ccol = _pick(c_cands)
        # Create standard columns if possible
        if tcol is not None:
            out['ticker'] = out[tcol].astype(str).str.upper()
        if dcol is not None:
            try:
                out['date'] = pd.to_datetime(out[dcol], errors='coerce')
            except Exception:
                out['date'] = out[dcol]
        if ccol is not None:
            out['close'] = pd.to_numeric(out[ccol], errors='coerce')
        return out

    def _standardize_stock_price_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with lowercase, underscore-separated column names and standard keys.
        Targets: date, open, high, low, close, adj_close, volume, ticker
        - Ensures a 'ticker' column (renames 'symbol' to 'ticker' if needed)
        - Maps common synonyms (e.g., 'Adj Close' -> 'adj_close')
        """
        if df is None or df.empty:
            return df
        out = df.copy()
        rename = {}
        lower_cols = {c.lower(): c for c in out.columns}
        # Ensure canonical 'ticker' column
        if 'ticker' not in lower_cols and 'symbol' in lower_cols:
            rename[lower_cols['symbol']] = 'ticker'
        # OHLCV synonyms
        syn_map = {
            'date': ['date', 'trade_date', 'timestamp', 'datetime', 'as_of_date'],
            'open': ['open', 'o', 'open_price'],
            'high': ['high', 'h', 'high_price'],
            'low': ['low', 'l', 'low_price'],
            'close': ['close', 'c', 'close_price', 'last', 'closeprice'],
            'adj_close': ['adj close', 'adj_close', 'adjusted close', 'adjusted_close', 'closeadj'],
            'volume': ['volume', 'vol', 'shares'],
        }
        for target, candidates in syn_map.items():
            for cand in candidates:
                if cand in lower_cols:
                    rename[lower_cols[cand]] = target
                    break
        # Apply rename and lowercase the rest (spaces -> underscores)
        out = out.rename(columns=rename)
        out.columns = [c.strip().lower().replace(' ', '_') for c in out.columns]
        # Ensure ticker uppercase if present
        if 'ticker' in out.columns:
            try:
                out['ticker'] = out['ticker'].astype(str).str.upper()
            except Exception:
                pass
        # Drop exact duplicate-named columns, keep first occurrence
        out = out.loc[:, ~out.columns.duplicated()]
        # Reorder if possible
        preferred = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'ticker']
        present = [c for c in preferred if c in out.columns]
        others = [c for c in out.columns if c not in present]
        out = out[present + others]
        return out

    out_paths: Dict[str, str] = {}
    for key, tbl in tables.items():
        df = fetch_table(engine, tbl, date_col=date_col_map.get(key), start_date=start_date, end_date=end_date)
        # Normalize and write
        if key == "stock_prices":
            df = _normalize_prices_columns(df)
            df = _standardize_stock_price_column_names(df)
        elif key in ("stock_earnings","earnings_estimates","quarterly_income","quarterly_revenue","revenue_estimates","growth_estimates"):
            try:
                if "ticker" in df.columns:
                    df["ticker"] = df["ticker"].astype(str).str.upper()
            except Exception:
                pass
        elif key == "earnings_history":
            # For earnings_history, write CSV exactly as selected from DB; do not transform columns
            pass
        out_path = os.path.join(output_dir, f"{key}.csv")
        try:
            df.to_csv(out_path, index=False)
            print(f"[DB] Wrote {key} -> {out_path} (rows={len(df)})")
        except Exception as e:
            print(f"[DB] Failed to write {out_path}: {e}")
        out_paths[key] = out_path

    # Derive cleaned/uncleaned prices outputs:
    #  - Clean: exclude tickers whose most recent close < 3.0, and ensure uniform row count per ticker
    #  - Unclean: all rows for tickers excluded from the clean set
    try:
        prices_path = out_paths.get("stock_prices")
        if prices_path and os.path.exists(prices_path):
            sp = pd.read_csv(prices_path)
            # Normalize essential columns
            cols_lower = {c.lower(): c for c in sp.columns}
            tcol = cols_lower.get("ticker")
            dcol = cols_lower.get("date")
            ccol = cols_lower.get("close")
            if not (tcol and dcol and ccol):
                print("[DB] stock_prices_with_clean_data skipped: required columns (ticker,date,close) not found.")
            else:
                sp[tcol] = sp[tcol].astype(str).str.upper()
                try:
                    sp[dcol] = pd.to_datetime(sp[dcol], errors='coerce')
                except Exception:
                    pass
                # Sort and compute latest close per ticker
                sp = sp.sort_values([tcol, dcol])
                latest = sp.groupby(tcol).tail(1)
                keep = latest[latest[ccol] >= 3.0][tcol].unique().tolist()
                sp_keep = sp[sp[tcol].isin(keep)].copy()
                sp_drop = sp[~sp[tcol].isin(keep)].copy()
                # Uniform row count per ticker
                counts = sp_keep.groupby(tcol).size()
                if counts.empty:
                    sp_filt = sp_keep.head(0)
                else:
                    min_count = int(counts.min())
                    if min_count <= 0:
                        sp_filt = sp_keep.head(0)
                    else:
                        # Efficient per-group tail without apply (avoids FutureWarning)
                        sp_filt = (
                            sp_keep.sort_values([tcol, dcol])
                                   .groupby(tcol, group_keys=False)
                                   .tail(min_count)
                        )
                # Write clean and unclean outputs (standardized lowercase column names)
                out_clean = os.path.join(output_dir, "stock_prices_with_clean_data.csv")
                sp_filt_std = _standardize_stock_price_column_names(sp_filt)
                sp_filt_std.to_csv(out_clean, index=False)
                print(f"[DB] Wrote stock_prices_with_clean_data -> {out_clean} (tickers={sp_filt[tcol].nunique() if not sp_filt.empty else 0}, rows={len(sp_filt)})")
                out_paths["stock_prices_with_clean_data"] = out_clean

                out_unclean = os.path.join(output_dir, "stock_prices_with_unclean_data.csv")
                sp_drop_std = _standardize_stock_price_column_names(sp_drop)
                sp_drop_std.to_csv(out_unclean, index=False)
                print(f"[DB] Wrote stock_prices_with_unclean_data -> {out_unclean} (tickers={sp_drop[tcol].nunique() if not sp_drop.empty else 0}, rows={len(sp_drop)})")
                out_paths["stock_prices_with_unclean_data"] = out_unclean
        else:
            print("[DB] stock_prices not found; skipping cleaned/uncleaned generation.")
    except Exception as e:
        print(f"[DB] Failed to build stock_prices_with_clean/unclean_data: {e}")

    # If earnings_history table/view exists we already wrote it; otherwise, attempt to compose from parts as fallback
    try:
        earn_p = out_paths.get("stock_earnings")
        est_p = out_paths.get("earnings_estimates")
        qrev_p = out_paths.get("quarterly_revenue")
        rev_est_p = out_paths.get("revenue_estimates")
        # Skip composition if direct export was available
        if out_paths.get('earnings_history') and os.path.exists(out_paths['earnings_history']):
            return out_paths
        if earn_p and os.path.exists(earn_p):
            df_e = pd.read_csv(earn_p)
            # normalize column names for merge
            ecols = {c.lower(): c for c in df_e.columns}
            tcol_e = ecols.get('ticker') or ecols.get('symbol')
            dcol_e = ecols.get('earnings_date') or ecols.get('date')
            rep_eps_col = ecols.get('reported_eps') or ecols.get('actualeps') or ecols.get('eps')
            df_e = df_e.rename(columns={
                tcol_e: 'ticker',
                dcol_e: 'earnings_date',
                rep_eps_col: 'reported_eps'
            }) if tcol_e and dcol_e and rep_eps_col else df_e
            df_e['ticker'] = df_e.get('ticker', pd.Series([], dtype=str)).astype(str).str.upper()
        else:
            df_e = pd.DataFrame()

        if est_p and os.path.exists(est_p):
            df_est = pd.read_csv(est_p)
            estcols = {c.lower(): c for c in df_est.columns}
            tcol_s = estcols.get('ticker') or estcols.get('symbol')
            dcol_s = estcols.get('earnings_date') or estcols.get('date')
            est_eps_col = estcols.get('estimate_eps') or estcols.get('epsestimate') or estcols.get('estimate')
            df_est = df_est.rename(columns={
                tcol_s: 'ticker',
                dcol_s: 'earnings_date',
                est_eps_col: 'estimate_eps'
            }) if tcol_s and dcol_s and est_eps_col else df_est
            df_est['ticker'] = df_est.get('ticker', pd.Series([], dtype=str)).astype(str).str.upper()
        else:
            df_est = pd.DataFrame()

        # Merge on ticker + earnings_date when possible
        needed_cols = {'ticker','earnings_date','reported_eps','estimate_eps','surprise_percentage'}
        if not df_e.empty and needed_cols.issubset(set(df_e.columns)):
            base = df_e[list(needed_cols)].copy()
        elif not df_e.empty and not df_est.empty and 'ticker' in df_e.columns and 'earnings_date' in df_e.columns and 'ticker' in df_est.columns and 'earnings_date' in df_est.columns:
            try:
                df_e['earnings_date'] = pd.to_datetime(df_e['earnings_date'], errors='coerce')
                df_est['earnings_date'] = pd.to_datetime(df_est['earnings_date'], errors='coerce')
            except Exception:
                pass
            base = pd.merge(df_e, df_est, on=['ticker', 'earnings_date'], how='outer')
        else:
            # Fallback: just use whichever exists
            base = df_e if not df_e.empty else df_est

        # Attach revenue if available
        if qrev_p and os.path.exists(qrev_p):
            try:
                df_rev = pd.read_csv(qrev_p)
                rcols = {c.lower(): c for c in df_rev.columns}
                tcol_r = rcols.get('ticker') or rcols.get('symbol')
                dcol_r = rcols.get('report_date') or rcols.get('date')
                rev_col = rcols.get('revenue') or rcols.get('totalrevenue')
                df_rev = df_rev.rename(columns={
                    tcol_r: 'ticker',
                    dcol_r: 'earnings_date',
                    rev_col: 'reported_revenue'
                }) if tcol_r and dcol_r and rev_col else df_rev
                df_rev['ticker'] = df_rev.get('ticker', pd.Series([], dtype=str)).astype(str).str.upper()
                try:
                    df_rev['earnings_date'] = pd.to_datetime(df_rev['earnings_date'], errors='coerce')
                except Exception:
                    pass
                if not base.empty and 'ticker' in base.columns and 'earnings_date' in base.columns:
                    base = pd.merge(base, df_rev[['ticker','earnings_date','reported_revenue']], on=['ticker','earnings_date'], how='left')
            except Exception as e:
                print(f"[DB] Failed to attach revenue to earnings_history: {e}")

        # Attach revenue estimates if available
        if rev_est_p and os.path.exists(rev_est_p):
            try:
                df_rev_est = pd.read_csv(rev_est_p)
                rcols = {c.lower(): c for c in df_rev_est.columns}
                tcol = rcols.get('ticker') or rcols.get('symbol')
                dcol = rcols.get('earnings_date') or rcols.get('next_earnings_date') or rcols.get('date')
                est_col = rcols.get('estimate_revenue') or rcols.get('revenueestimate') or rcols.get('estimate')
                df_rev_est = df_rev_est.rename(columns={
                    tcol: 'ticker',
                    dcol: 'earnings_date',
                    est_col: 'estimate_revenue'
                }) if tcol and dcol and est_col else df_rev_est
                df_rev_est['ticker'] = df_rev_est.get('ticker', pd.Series([], dtype=str)).astype(str).str.upper()
                try:
                    df_rev_est['earnings_date'] = pd.to_datetime(df_rev_est['earnings_date'], errors='coerce')
                except Exception:
                    pass
                if not base.empty and 'ticker' in base.columns and 'earnings_date' in base.columns:
                    base = pd.merge(base, df_rev_est[['ticker','earnings_date','estimate_revenue']], on=['ticker','earnings_date'], how='left')
            except Exception as e:
                print(f"[DB] Failed to attach revenue estimates to earnings_history: {e}")

        if not base.empty:
            # Do not recompute EPS surprise if column already exists per schema
            if 'surprise_percentage' not in base.columns:
                try:
                    est = pd.to_numeric(base.get('estimate_eps'), errors='coerce')
                    act = pd.to_numeric(base.get('reported_eps'), errors='coerce')
                    base['surprise_percentage'] = ((act - est) / est.abs() * 100.0).where(est.notna() & (est != 0))
                except Exception:
                    pass
            # Keep all available columns post-merge, with standardized keys present
            out_hist = base.copy()
            if {'ticker','earnings_date'}.issubset(out_hist.columns):
                out_hist = out_hist.sort_values(['ticker','earnings_date'])
            # Write to both canonical and input paths
            try:
                os.makedirs('data', exist_ok=True)
                out_hist.to_csv('data/earnings_history.csv', index=False)
                print(f"[DB] Wrote data/earnings_history.csv (rows={len(out_hist)})")
            except Exception as e:
                print(f"[DB] Failed to write data/earnings_history.csv: {e}")
            try:
                out_hist.to_csv(os.path.join(output_dir, 'earnings_history.csv'), index=False)
                print(f"[DB] Wrote {os.path.join(output_dir, 'earnings_history.csv')} (rows={len(out_hist)})")
                out_paths['earnings_history'] = os.path.join(output_dir, 'earnings_history.csv')
            except Exception as e:
                print(f"[DB] Failed to write {os.path.join(output_dir, 'earnings_history.csv')}: {e}")
        else:
            print("[DB] earnings_history not built (no earnings/estimates data).")
    except Exception as e:
        print(f"[DB] Failed to build earnings_history: {e}")

    return out_paths


def main():
    import argparse
    p = argparse.ArgumentParser(description="Fetch data from DB and write CSVs to data/")
    p.add_argument("--out", default="data/input", help="Output directory for CSVs")
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    args = p.parse_args()
    fetch_all(args.out, args.start, args.end)


if __name__ == "__main__":
    main()


