from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional
import configparser

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL


# -----------------------------
# Utility helpers (keep class slim)
# -----------------------------

def load_db_config(config_path: Path) -> Dict[str, str]:
    """Load database configuration from an INI file.

    Expected section: [mysql]
    Required keys: url, username, password, database
    """
    parser = configparser.ConfigParser()
    if not config_path.exists():
        raise FileNotFoundError(f"Database config not found at: {config_path}")

    parser.read(config_path)
    if "mysql" not in parser:
        raise KeyError("Missing [mysql] section in config.ini")

    section = parser["mysql"]
    required_keys = ["url", "username", "password", "database"]
    missing = [k for k in required_keys if k not in section or section.get(k) == ""]
    if missing:
        raise KeyError(f"Missing required mysql config keys: {', '.join(missing)}")

    return {
        "host": section.get("url"),
        "username": section.get("username"),
        "password": section.get("password"),
        "database": section.get("database"),
        # Optional: allow custom port; default 3306
        "port": section.getint("port", fallback=3306),
    }


def build_sqlalchemy_engine(db_conf: Dict[str, str]) -> Engine:
    """Create a SQLAlchemy engine for MySQL using PyMySQL driver.

    Using URL.create ensures proper escaping of special characters in credentials.
    """
    url = URL.create(
        drivername="mysql+pymysql",
        username=db_conf["username"],
        password=db_conf["password"],
        host=db_conf["host"],
        port=db_conf.get("port", 3306),
        database=db_conf["database"],
    )
    return create_engine(url)


def sanitize_table_name(table_name: str) -> str:
    """Conservatively sanitize a MySQL table name to prevent SQL injection.

    Allows alphanumerics, underscores, and backticks. Wrap final name in backticks.
    """
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_`")
    if not table_name or any(ch not in allowed_chars for ch in table_name):
        # Fallback: replace invalid chars with underscore
        cleaned = "".join(ch if ch in allowed_chars else "_" for ch in table_name or "")
    else:
        cleaned = table_name

    # Ensure wrapped in backticks; also remove stray backticks inside
    cleaned = cleaned.replace("`", "")
    return f"`{cleaned}`"


def read_table(engine: Engine, table_name: str) -> pd.DataFrame:
    """Read the full table into a pandas DataFrame."""
    table_expr = sanitize_table_name(table_name)
    query = f"SELECT * FROM {table_expr}"
    return pd.read_sql(query, con=engine)


def ensure_directory(directory_path: Path) -> None:
    """Create the directory if it does not already exist."""
    directory_path.mkdir(parents=True, exist_ok=True)


def save_dataframe_as_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to CSV with UTF-8 encoding and no index."""
    df.to_csv(output_path, index=False)


# -----------------------------
# Per-table dataframe transformers
# -----------------------------

def transform_dataframe_for_table(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply table-specific transformations prior to saving CSV.

    - stocksinfp: lowercase columns, rename symbol->ticker, enforce column order
      [date, ticker, close, high, low, open, volume]
    """
    if table_name == "stocksinfp":
        # Lowercase column names
        df.columns = [str(c).lower() for c in df.columns]

        # Rename symbol -> ticker if present
        if "symbol" in df.columns:
            df = df.rename(columns={"symbol": "ticker"})

        required_order = ["date", "ticker", "close", "high", "low", "open", "volume"]
        missing_columns = [c for c in required_order if c not in df.columns]
        if missing_columns:
            raise ValueError(
                f"stocksinfp is missing required columns: {', '.join(missing_columns)}"
            )

        # Restrict to last 2 years of data relative to the latest available date in the dataset
        # Ensure date is datetime
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        max_date = df["date"].max()
        if pd.notna(max_date):
            cutoff_date = max_date - pd.DateOffset(years=1)
            df = df[df["date"] >= cutoff_date]
        # Optional: sort by date for consistency
        df = df.sort_values(["ticker", "date"]) 

        # Enforce column order (and drop extras to match exact order requested)
        df = df[required_order]

        # ----- Liquidity/penny filters (based on latest close and avg volume over window) -----
        try:
            # Ensure numeric dtypes
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            latest_per_ticker = df.sort_values("date").groupby("ticker").tail(1)[["ticker", "close"]]
            liquid_price = set(latest_per_ticker[latest_per_ticker["close"] >= 3.0]["ticker"])

            avg_vol = df.groupby("ticker")["volume"].mean().rename("avg_volume")
            liquid_volume = set(avg_vol[avg_vol >= 300_000].index)

            keep_liquid = liquid_price & liquid_volume if liquid_price and liquid_volume else liquid_price or liquid_volume
            if keep_liquid:
                before_tickers = df["ticker"].nunique()
                df = df[df["ticker"].isin(keep_liquid)]
                after_tickers = df["ticker"].nunique()
                dropped = before_tickers - after_tickers
                if dropped > 0:
                    print(f"[stocksinfp] Dropped {dropped} illiquid/penny tickers (<$3 or avg vol <300k). Kept {after_tickers}.")
        except Exception as exc:
            print(f"[stocksinfp] Liquidity filter skipped due to error: {exc}")

        # ----- Outlier removal using per-ticker log returns (drop extreme moves) -----
        try:
            df = df.sort_values(["ticker", "date"]).copy()
            # Use transform to preserve the original index alignment
            ratio = df.groupby("ticker")["close"].transform(lambda s: s / s.shift(1))
            df["log_ret"] = np.log(ratio.where(ratio > 0))
            # Compute per-ticker 1st/99th percentiles on valid returns
            bounds = df.dropna(subset=["log_ret"]).groupby("ticker")["log_ret"].quantile([0.01, 0.99]).unstack()
            bounds.columns = ["p01", "p99"]
            df = df.merge(bounds, left_on="ticker", right_index=True, how="left")
            before_rows = len(df)
            mask_ok = (df["log_ret"].isna()) | ((df["log_ret"] >= df["p01"]) & (df["log_ret"] <= df["p99"]))
            df = df[mask_ok].drop(columns=["log_ret", "p01", "p99"], errors="ignore")
            after_rows = len(df)
            removed_rows = before_rows - after_rows
            if removed_rows > 0:
                print(f"[stocksinfp] Removed {removed_rows} outlier rows by log-return filter (1st–99th pct per ticker).")
        except Exception as exc:
            # Be conservative: if anything fails, proceed without outlier removal
            if "log_ret" in df.columns:
                df = df.drop(columns=["log_ret"], errors="ignore")
            print(f"[stocksinfp] Outlier filter skipped due to error: {exc}")

        # Filter: keep only tickers that have the same number of rows as Apple (AAPL)
        try:
            aapl_rows = len(df[df["ticker"].astype(str).str.upper() == "AAPL"])
            if aapl_rows > 0:
                counts = df["ticker"].value_counts()
                keep_tickers = set(counts[counts == aapl_rows].index)
                # Always keep SPY if present so market proxy features can be computed
                if "SPY" in set(df["ticker"].astype(str).str.upper().unique()):
                    keep_tickers.add("SPY")
                before_rows = len(df)
                before_tickers = df["ticker"].nunique()
                df = df[df["ticker"].isin(keep_tickers)]
                after_rows = len(df)
                after_tickers = df["ticker"].nunique()
                removed_tickers = before_tickers - after_tickers
                removed_rows = before_rows - after_rows
                if removed_tickers > 0:
                    print(
                        f"[stocksinfp] Filtered tickers by AAPL row count ({aapl_rows}). "
                        f"Removed {removed_tickers} tickers and {removed_rows} rows; kept {after_tickers} tickers."
                    )
            else:
                print("[stocksinfp] AAPL not found; skipping equal-length ticker filter.")
        except Exception as exc:
            # Do not fail export due to filtering; just report
            print(f"[stocksinfp] Skipped equal-length filter due to error: {exc}")

    return df


# -----------------------------
# Main class (thin facade)
# -----------------------------

class StockDataCreator:
    """Exports database tables to CSV files under the project `data` folder.

    Usage:
        creator = StockDataCreator()
        creator.export_tables(["table_a", "table_b"], name_mapping={"table_b": "custom_name"})
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.config_path = config_path or project_root / "stocks" / "conf" / "config.ini"
        self.data_dir = data_dir or project_root / "data"

        db_conf = load_db_config(self.config_path)
        self.engine = build_sqlalchemy_engine(db_conf)

    def export_table(self, table_name: str, csv_name: Optional[str] = None) -> Path:
        """Export a single table to CSV. Returns the CSV output path."""
        ensure_directory(self.data_dir)
        final_csv_name = (csv_name or table_name).strip()
        if not final_csv_name:
            raise ValueError("CSV name resolved to an empty value")

        output_path = self.data_dir / f"{final_csv_name}.csv"
        df = read_table(self.engine, table_name)
        df = transform_dataframe_for_table(table_name, df)
        save_dataframe_as_csv(df, output_path)
        return output_path

    def export_tables(
        self,
        table_names: Iterable[str],
        name_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Path]:
        """Export multiple tables to CSVs.

        - If `name_mapping` provided, it maps table_name -> csv_base_name.
        - By default, CSV file name equals the table name.
        Returns a dict mapping table_name -> exported CSV Path.
        """
        mapping = name_mapping or {}
        results: Dict[str, Path] = {}
        for table_name in table_names:
            csv_name = mapping.get(table_name, table_name)
            results[table_name] = self.export_table(table_name, csv_name)
        return results


if __name__ == "__main__":
    # Optional CLI for ad-hoc usage
    import argparse

    parser = argparse.ArgumentParser(description="Export MySQL tables to CSV under ./data")
    parser.add_argument(
        "--tables",
        type=str,
        required=True,
        help="Comma-separated list of table names (e.g., table1,table2)",
    )
    parser.add_argument(
        "--map",
        type=str,
        required=False,
        default="",
        help="Optional mapping table:csvname pairs, comma-separated (e.g., table2:custom2,table3:custom3)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="",
        help="Optional path to config.ini; defaults to stocks/conf/config.ini",
    )

    args = parser.parse_args()

    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    mapping: Dict[str, str] = {}
    if args.map:
        for pair in args.map.split(","):
            if not pair.strip():
                continue
            if ":" not in pair:
                raise ValueError(f"Invalid mapping entry: {pair}. Expected format 'table:csvname'.")
            t, c = pair.split(":", 1)
            mapping[t.strip()] = c.strip()

    config_path = Path(args.config).resolve() if args.config else None
    creator = StockDataCreator(config_path=config_path)
    outputs = creator.export_tables(tables, name_mapping=mapping)
    for table, path in outputs.items():
        print(f"Exported {table} -> {path}")


