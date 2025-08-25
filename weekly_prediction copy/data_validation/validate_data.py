import argparse
import json
import os
import sys
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pathlib import Path

# Reuse existing pipeline loader without modifying it
# Ensure project root is on sys.path when running as a script
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_all_data


def _build_default_path(base_name: str, suffix: Optional[str]) -> str:
    if suffix:
        return os.path.join("data", f"{base_name}_{suffix}.csv")
    return os.path.join("data", f"{base_name}.csv")


def _read_snapshot_csv(base_name: str, suffix: Optional[str], parse_dates: Optional[List[str]] = None) -> Tuple[Optional[pd.DataFrame], List[str], Optional[str]]:
    """Read a CSV for a snapshot, searching dated files in data/ then olddata/.

    Returns: (df or None, issues list, resolved_path or None)
    """
    issues: List[str] = []
    candidate_paths: List[str] = []
    if suffix:
        candidate_paths = [
            os.path.join("data", f"{base_name}_{suffix}.csv"),
            os.path.join("olddata", f"{base_name}_{suffix}.csv"),
        ]
    else:
        candidate_paths = [os.path.join("data", f"{base_name}.csv")]

    resolved_path: Optional[str] = None
    for path in candidate_paths:
        if os.path.exists(path):
            resolved_path = path
            break

    if resolved_path is None:
        # Report both attempted locations for clarity
        if len(candidate_paths) == 1:
            issues.append(f"Missing file: {candidate_paths[0]}")
        else:
            issues.append(f"Missing file in both locations: {candidate_paths[0]} and {candidate_paths[1]}")
        return None, issues, None

    df, read_issues = _read_csv_safe(resolved_path, parse_dates=parse_dates)
    return df, read_issues, resolved_path


def _read_csv_safe(path: str, parse_dates: Optional[List[str]] = None) -> Tuple[Optional[pd.DataFrame], List[str]]:
    issues: List[str] = []
    if not os.path.exists(path):
        issues.append(f"Missing file: {path}")
        return None, issues

    try:
        df = pd.read_csv(path, parse_dates=parse_dates or [])
        # Normalize headers for validation (do not write back)
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df, issues
    except Exception as exc:
        issues.append(f"Failed to read {path}: {exc}")
        return None, issues


def _validate_required_columns(df: pd.DataFrame, required: List[str], label: str) -> List[str]:
    issues: List[str] = []
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"{label}: missing required columns: {', '.join(missing)}")
    return issues


def _validate_prices(df: pd.DataFrame, label: str) -> List[str]:
    issues: List[str] = []
    issues += _validate_required_columns(df, ["date", "ticker", "open", "high", "low", "close", "volume"], label)
    if issues:
        return issues

    # Dtypes and basic sanity
    try:
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            issues.append(f"{label}: 'date' column is not datetime; parsing may have failed")
    except KeyError:
        pass

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"{label}: '{col}' should be numeric")

    # Per-ticker chronological order and non-empty history
    if {"ticker", "date"}.issubset(df.columns):
        bad_order = []
        empty_tickers = []
        for ticker, grp in df.groupby("ticker"):
            if grp.empty:
                empty_tickers.append(ticker)
                continue
            if not grp["date"].is_monotonic_increasing:
                if not grp.sort_values("date").index.equals(grp.index):
                    bad_order.append(str(ticker))
        if empty_tickers:
            issues.append(f"{label}: found {len(empty_tickers)} tickers with no rows")
        if bad_order:
            issues.append(f"{label}: non-chronological rows for tickers: {', '.join(bad_order[:10])}{'...' if len(bad_order) > 10 else ''}")

    # Recent coverage check (last 21 rows per ticker at least)
    if "ticker" in df.columns:
        too_short = [t for t, g in df.groupby("ticker") if len(g) < 21]
        if too_short:
            issues.append(f"{label}: insufficient history (<21 rows) for {len(too_short)} tickers; examples: {', '.join(map(str, too_short[:10]))}{'...' if len(too_short) > 10 else ''}")

    # Values sanity
    for col in ["open", "high", "low", "close"]:
        if col in df.columns and (df[col] <= 0).any():
            issues.append(f"{label}: non-positive values found in '{col}'")
    if "volume" in df.columns and (df["volume"] < 0).any():
        issues.append(f"{label}: negative values found in 'volume'")

    # New: Consistency check relative to AAPL row count
    try:
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str)
            aapl_rows = int((df["ticker"].str.upper() == "AAPL").sum())
            if aapl_rows > 0:
                counts = df["ticker"].value_counts()
                mismatched = counts[counts != aapl_rows]
                if not mismatched.empty:
                    # Show a concise sample of offenders
                    examples = ", ".join([f"{t}:{int(n)}" for t, n in mismatched.head(15).items()])
                    issues.append(
                        f"{label}: {int(len(mismatched))} tickers have row counts != AAPL ({aapl_rows}). Examples: {examples}"
                    )
            else:
                issues.append(f"{label}: AAPL not present; cannot validate equal-length histories vs AAPL")
    except Exception as exc:
        issues.append(f"{label}: failed AAPL length consistency check: {exc}")

    return issues


def _validate_calendar_files(income: Optional[pd.DataFrame], revenue: Optional[pd.DataFrame], earnings_hist: Optional[pd.DataFrame], suffix_label: str) -> List[str]:
    issues: List[str] = []
    if income is not None:
        issues += _validate_required_columns(income, ["ticker", "report_date"], f"quarterly_income {suffix_label}")
        if "report_date" in income.columns and not pd.api.types.is_datetime64_any_dtype(income["report_date"]):
            issues.append(f"quarterly_income {suffix_label}: 'report_date' not parsed as datetime")
    if revenue is not None:
        issues += _validate_required_columns(revenue, ["ticker", "report_date"], f"quarterly_revenue {suffix_label}")
        if "report_date" in revenue.columns and not pd.api.types.is_datetime64_any_dtype(revenue["report_date"]):
            issues.append(f"quarterly_revenue {suffix_label}: 'report_date' not parsed as datetime")
    if earnings_hist is not None:
        issues += _validate_required_columns(earnings_hist, ["ticker", "earnings_date"], f"stock_earnings {suffix_label}")
        if "earnings_date" in earnings_hist.columns and not pd.api.types.is_datetime64_any_dtype(earnings_hist["earnings_date"]):
            issues.append(f"stock_earnings {suffix_label}: 'earnings_date' not parsed as datetime")

    # Joinability checks
    if income is not None and revenue is not None and {"ticker", "report_date"}.issubset(income.columns) and {"ticker", "report_date"}.issubset(revenue.columns):
        merged = pd.merge(income[["ticker", "report_date"]], revenue[["ticker", "report_date"]], on=["ticker", "report_date"], how="outer", indicator=True)
        only_income = (merged["_merge"] == "left_only").sum()
        only_revenue = (merged["_merge"] == "right_only").sum()
        if only_income or only_revenue:
            issues.append(f"income/revenue join mismatch {suffix_label}: {only_income} only-in-income, {only_revenue} only-in-revenue on (ticker, report_date)")

    return issues


def _validate_estimates(earn_est: Optional[pd.DataFrame], rev_est: Optional[pd.DataFrame], growth: Optional[pd.DataFrame], suffix_label: str) -> List[str]:
    issues: List[str] = []
    if earn_est is not None:
        issues += _validate_required_columns(earn_est, ["ticker", "earnings_date"], f"earnings_estimates {suffix_label}")
        if "earnings_date" in earn_est.columns and not pd.api.types.is_datetime64_any_dtype(earn_est["earnings_date"]):
            issues.append(f"earnings_estimates {suffix_label}: 'earnings_date' not parsed as datetime")
        # Try to ensure some EPS estimate presence (recognize common synonyms)
        eps_synonyms = {
            "estimate_eps", "estimated_eps", "eps_estimate",
            "average_estimate", "avg_estimate", "mean_estimate"
        }
        found_any = any(c in earn_est.columns for c in eps_synonyms)
        if not found_any:
            observed = ", ".join(list(earn_est.columns)[:25])
            issues.append(
                f"earnings_estimates {suffix_label}: no EPS estimate-like column found (checked: {', '.join(sorted(eps_synonyms))}). Found columns: {observed}"
            )
    if rev_est is not None:
        issues += _validate_required_columns(rev_est, ["ticker", "next_earnings_date"], f"revenue_estimates {suffix_label}")
        if "next_earnings_date" in rev_est.columns and not pd.api.types.is_datetime64_any_dtype(rev_est["next_earnings_date"]):
            issues.append(f"revenue_estimates {suffix_label}: 'next_earnings_date' not parsed as datetime")
    if growth is not None:
        # Accept common synonyms for growth signal
        growth_synonyms = {"growth", "stock_growth"}
        has_growth = any(c in growth.columns for c in growth_synonyms)
        if not has_growth:
            observed = ", ".join(list(growth.columns)[:25])
            issues.append(
                f"growth_estimates {suffix_label}: missing growth-like column (checked: {', '.join(sorted(growth_synonyms))}). Found columns: {observed}"
            )
    return issues


def _summarize_prices(df: Optional[pd.DataFrame]) -> Dict[str, object]:
    if df is None or df.empty:
        return {"rows": 0, "tickers": 0, "date_min": None, "date_max": None}
    return {
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else None,
        "date_min": str(pd.to_datetime(df["date"]).min().date()) if "date" in df.columns else None,
        "date_max": str(pd.to_datetime(df["date"]).max().date()) if "date" in df.columns else None,
    }


def validate_snapshot(suffix: Optional[str]) -> Dict[str, object]:
    label = f"[{suffix}]" if suffix else "[undated]"
    report: Dict[str, object] = {"label": label, "issues": [], "summaries": {}}

    # Read all files (matching data_loader logic)
    prices, issues_p, prices_path = _read_snapshot_csv("stock_prices", suffix, parse_dates=["date"])
    earnings, issues_e, earnings_path = _read_snapshot_csv("stock_earnings", suffix, parse_dates=["earnings_date"])
    earn_est, issues_ee, earn_est_path = _read_snapshot_csv("earnings_estimates", suffix, parse_dates=["earnings_date"])
    income, issues_i, income_path = _read_snapshot_csv("quarterly_income", suffix, parse_dates=["report_date"])
    revenue, issues_r, revenue_path = _read_snapshot_csv("quarterly_revenue", suffix, parse_dates=["report_date"])
    rev_est, issues_re, rev_est_path = _read_snapshot_csv("revenue_estimates", suffix, parse_dates=["next_earnings_date"])
    growth, issues_g, growth_path = _read_snapshot_csv("growth_estimates", suffix)

    report["issues"] += issues_p + issues_e + issues_ee + issues_i + issues_r + issues_re + issues_g
    report.setdefault("paths", {})[label] = {
        "stock_prices": prices_path,
        "stock_earnings": earnings_path,
        "earnings_estimates": earn_est_path,
        "quarterly_income": income_path,
        "quarterly_revenue": revenue_path,
        "revenue_estimates": rev_est_path,
        "growth_estimates": growth_path,
    }

    # Schema validations
    if prices is not None:
        report["issues"] += _validate_prices(prices, f"stock_prices {label}")
        report["summaries"]["prices"] = _summarize_prices(prices)

    report["issues"] += _validate_calendar_files(income, revenue, earnings, label)
    report["issues"] += _validate_estimates(earn_est, rev_est, growth, label)

    # Loader-level validation: try to build master_df/prices_df
    loader_date: Optional[date] = None
    if suffix:
        try:
            loader_date = datetime.strptime(suffix, "%Y-%m-%d").date()
        except ValueError:
            report["issues"].append(f"Invalid date suffix format: {suffix}; expected YYYY-MM-DD")

    try:
        master_df, prices_df = load_all_data(file_date=loader_date)
        if master_df is None or master_df.empty:
            report["issues"].append(f"Loader produced empty master_df {label}")
        if prices_df is None or prices_df.empty:
            report["issues"].append(f"Loader produced empty prices_df {label}")

        # Feature prerequisites on master_df
        if master_df is not None and not master_df.empty:
            # Must have ticker
            if "ticker" not in master_df.columns:
                report["issues"].append(f"master_df {label}: missing 'ticker'")
            # Fields referenced by feature engineering
            required_for_features = [
                "earnings_date",  # used for post-earnings rally window
            ]
            missing = [c for c in required_for_features if c not in master_df.columns]
            # Special handling: if duplicates exist due to merges, pandas may suffix _x/_y
            if "earnings_date" in missing and ("earnings_date_x" in master_df.columns or "earnings_date_y" in master_df.columns):
                report["issues"].append(
                    f"master_df {label}: 'earnings_date' appears only as suffixed columns (earnings_date_x/earnings_date_y) due to merges. Feature code expects 'earnings_date'."
                )
                # Do not also report as plain missing to avoid confusion
                missing = [c for c in missing if c != "earnings_date"]
            if missing:
                report["issues"].append(f"master_df {label}: missing columns needed by features: {', '.join(missing)}")

            # Optional but recommended
            recommended = ["growth", "earnings_m", "revenue_m", "next_earnings_date"]
            rec_missing = [c for c in recommended if c not in master_df.columns]
            # Accept synonyms for growth
            if "growth" in rec_missing and "stock_growth" in master_df.columns:
                rec_missing.remove("growth")
            if rec_missing:
                report["issues"].append(f"master_df {label}: recommended columns absent (features will default): {', '.join(rec_missing)}")

        # Prices DF sanity (post-loader)
        if prices_df is not None and not prices_df.empty:
            report["issues"] += _validate_prices(prices_df, f"loader.prices_df {label}")
            report["summaries"]["loader_prices"] = _summarize_prices(prices_df)
    except Exception as exc:
        report["issues"].append(f"Exception during load_all_data for {label}: {exc}")

    return report


def compare_snapshots(report_a: Dict[str, object], report_b: Dict[str, object], prices_a: Optional[pd.DataFrame], prices_b: Optional[pd.DataFrame]) -> Dict[str, object]:
    comparison: Dict[str, object] = {"label_a": report_a.get("label"), "label_b": report_b.get("label"), "issues": [], "differences": {}}

    # Compare tickers present in prices
    if prices_a is not None and prices_b is not None and not prices_a.empty and not prices_b.empty:
        tickers_a = set(prices_a["ticker"].unique()) if "ticker" in prices_a.columns else set()
        tickers_b = set(prices_b["ticker"].unique()) if "ticker" in prices_b.columns else set()
        only_a = sorted(list(tickers_a - tickers_b))
        only_b = sorted(list(tickers_b - tickers_a))
        comparison["differences"]["tickers_only_in_a"] = only_a[:50]
        comparison["differences"]["tickers_only_in_b"] = only_b[:50]
        if only_a or only_b:
            comparison["issues"].append(f"Ticker set mismatch between {report_a.get('label')} and {report_b.get('label')}")

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate data snapshots and detect issues without modifying pipeline code")
    parser.add_argument("--date", type=str, default=None, help="Date suffix YYYY-MM-DD to validate against undated base as well")
    parser.add_argument("--only", type=str, choices=["undated", "dated", "both"], default="both", help="Whether to validate undated, dated, or both and compare")
    parser.add_argument("--output", type=str, default="", help="Optional path to save JSON report")

    args = parser.parse_args()

    suffix = args.date
    results: Dict[str, object] = {"generated_at": datetime.utcnow().isoformat() + "Z"}

    # Validate undated
    report_undated = None
    if args.only in ("undated", "both"):
        report_undated = validate_snapshot(None)
        results["undated"] = report_undated

    # Validate dated
    report_dated = None
    if suffix and args.only in ("dated", "both"):
        report_dated = validate_snapshot(suffix)
        results["dated"] = report_dated

    # Add comparison if both
    if report_undated and report_dated:
        # Reload prices only for comparison summaries
        prices_a, _, _ = _read_snapshot_csv("stock_prices", None, parse_dates=["date"])
        prices_b, _, _ = _read_snapshot_csv("stock_prices", suffix, parse_dates=["date"])
        results["comparison"] = compare_snapshots(report_undated, report_dated, prices_a, prices_b)

    # Human-readable printout
    def _print_section(title: str) -> None:
        print("\n" + title)
        print("-" * len(title))

    if report_undated:
        _print_section("Undated snapshot")
        print(f"Summary (prices): {report_undated.get('summaries', {}).get('prices', {})}")
        for issue in report_undated.get("issues", []):
            print(f"[ISSUE] {issue}")
    if report_dated:
        _print_section(f"Dated snapshot {suffix}")
        print(f"Summary (prices): {report_dated.get('summaries', {}).get('prices', {})}")
        for issue in report_dated.get("issues", []):
            print(f"[ISSUE] {issue}")
    if results.get("comparison"):
        _print_section("Comparison (undated vs dated)")
        diff = results["comparison"]
        for k, v in diff.get("differences", {}).items():
            print(f"{k}: {v}")
        for issue in diff.get("issues", []):
            print(f"[ISSUE] {issue}")

    # Optional JSON output
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved detailed JSON report to: {args.output}")
        except Exception as exc:
            print(f"Failed to save JSON report: {exc}")


if __name__ == "__main__":
    main()


