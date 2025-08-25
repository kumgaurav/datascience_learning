import pandas as pd
import numpy as np
import pyarrow as pa
import os
import sys
import tempfile
import logging
import traceback

logger = logging.getLogger("arrow_csv_debug")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger.info(f"pandas version: {pd.__version__}")
logger.info(f"pyarrow version: {pa.__version__}")


def get_production_solution():
    """
    Return production-ready converter that creates a PyArrow Table directly
    from a pandas DataFrame without dtype headaches.
    """
    def convert_to_arrow_table(df, debug=True):
        arrays = {}

        # Heuristics
        date_columns = [col for col in df.columns if "date" in col.lower()]
        numeric_columns = ['close', 'high', 'low', 'volume', 'price', 'amount']

        for col in df.columns:
            series = df[col]

            # Handle datetime/date columns → use timezone-free date32 to avoid pandas tz issues
            if col in date_columns:
                parsed = pd.to_datetime(series, errors="coerce")  # no UTC, naive
                if parsed.notna().any():
                    # Build date objects for pa.date32()
                    date_values = [x.date() if pd.notna(x) else None for x in parsed]
                    arrays[col] = pa.array(date_values, type=pa.date32())
                    if debug:
                        print("Arrow schema:")
                    continue
                arrays[col] = pa.array(series.fillna('').astype(str).tolist(), type=pa.string())

            # Handle numeric columns
            elif col in numeric_columns:
                numeric = pd.to_numeric(series, errors="coerce")
                if numeric.notna().any():
                    arrays[col] = pa.array(numeric.tolist(), type=pa.float64())
                    continue
                arrays[col] = pa.array(series.fillna('').astype(str).tolist(), type=pa.string())

            # Everything else → string
            else:
                arrays[col] = pa.array(series.fillna('').astype(str).tolist(), type=pa.string())

        arrow_table = pa.table(arrays)

        if debug:
            print("Arrow schema:")
            print(arrow_table.schema)

        return arrow_table

    return convert_to_arrow_table


def _safe_key(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def get_cached_csv_path(week_title: str, prefix: str = "earnings_full_") -> str:
    tmp_root = os.path.join(tempfile.gettempdir(), "stockapp_tmp")
    safe = _safe_key(f"{prefix}{week_title}")
    return os.path.join(tmp_root, f"{safe}.csv")


def print_df_diagnostics(df: pd.DataFrame, header: str = "DataFrame diagnostics") -> None:
    logger.info(header)
    try:
        lines = []
        for col in df.columns:
            series = df[col]
            non_null = series[series.notna()] if hasattr(series, "notna") else series
            sample_type = type(non_null.iloc[0]).__name__ if hasattr(non_null, "iloc") and len(non_null) > 0 else "None"
            lines.append(f"{col}: dtype={getattr(series, 'dtype', type(series))}, sample={sample_type}")
        logger.info("; ".join(lines))
    except Exception as e:
        logger.warning(f"Failed to print diagnostics: {e}")


def debug_arrow_on_cached_csv(week_title: str) -> None:
    """
    Read the same cached CSV written by the UI for the given week_title,
    log dtypes/samples, run the production Arrow converter, and isolate
    failing columns if any. Includes safe fallbacks.
    """
    from test import get_production_solution  # local import to avoid circulars

    path = get_cached_csv_path(week_title)
    if not os.path.exists(path):
        logger.error(f"Cached CSV not found: {path}")
        return

    logger.info(f"Reading cached CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    print_df_diagnostics(df, header="Cached CSV column types")

    converter = get_production_solution()

    # Try full conversion
    try:
        table = converter(df, debug=True)
        logger.info("Full conversion succeeded. Arrow schema:")
        logger.info(str(table.schema))
        # Try safe to_pandas options first
        try:
            pdf = table.to_pandas(split_blocks=True, ignore_metadata=True)
            logger.info("Round-trip to pandas with safe options succeeded.")
            print_df_diagnostics(pdf, header="Round-tripped pandas dtypes")
            return
        except Exception as e_safe:
            logger.warning(f"to_pandas with safe options failed: {e_safe}")
            # Try plain to_pandas next
            try:
                _ = table.to_pandas()
                logger.info("Round-trip to pandas (plain) succeeded.")
                return
            except Exception as e_plain:
                logger.warning(f"Full conversion failed: {e_plain}\n{traceback.format_exc()}")
    except Exception as e:
        logger.warning(f"Full Arrow table conversion failed: {e}\n{traceback.format_exc()}")

    # Column-by-column isolation (including to_pandas)
    logger.info("Isolating failing columns...")
    failures = []
    for col in df.columns:
        try:
            single = df[[col]].copy()
            print_df_diagnostics(single, header=f"Diagnosing column '{col}'")
            table = converter(single, debug=True)
            # Try safe to_pandas
            try:
                _ = table.to_pandas(split_blocks=True, ignore_metadata=True)
            except Exception:
                _ = table.to_pandas()
            logger.info(f"Column '{col}' converted OK")
        except Exception as ce:
            logger.error(f"Column '{col}' conversion failed: {ce}")
            failures.append(col)

    if failures:
        logger.error(f"Failing columns: {failures}")
    else:
        logger.info("All individual columns succeeded; failure is due to pandas/pyarrow block assembly across multiple columns. Using pylist fallback...")
        # Reconstruct via to_pylist fallback
        try:
            fallback_data = {}
            for name, col in zip(table.schema.names, table.columns):
                fallback_data[name] = col.to_pylist()
            pdf_fb = pd.DataFrame(fallback_data)
            print_df_diagnostics(pdf_fb, header="Fallback DataFrame dtypes (pylist)")
            logger.info("Fallback DataFrame constructed successfully.")
        except Exception as fb_err:
            logger.error(f"Fallback reconstruction failed: {fb_err}")


if __name__ == "__main__":
    # Usage:
    # python test.py "Next Week"
    if len(sys.argv) >= 2:
        week = sys.argv[1]
        debug_arrow_on_cached_csv(week)
    else:
        # Fallback to the existing demo
        from test import get_production_solution as _gps
        converter = _gps()

        df = pd.DataFrame({
            'symbol': ['AFRM', 'AMBA', None, 'BBW'],
            'close': [69.14, 85.23, None, 45.67],
            'date': ['2025-06-30', None, '2025-07-02', '2025-07-03']
        })

        arrow_table = converter(df)
        print("PRODUCTION SOLUTION: SUCCESS!")
        print(arrow_table)
