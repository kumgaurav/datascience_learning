from data_loader import load_all_data
from feature_engineering import create_all_features
from datetime import date, datetime
import argparse
import os
import pandas as pd
import numpy as np


def _score_change(pct: float) -> int:
    """Map percent change to a discrete score per features_top_momentum.md example."""
    try:
        v = float(pct)
    except Exception:
        return 0
    if v > 10.0:
        return 3
    if v > 5.0:
        return 2
    if v > 0.0:
        return 1
    return 0


def _compute_window_scores(latest_slice: pd.DataFrame, horizons: list[int]) -> tuple[int, dict]:
    """Compute aggregated price+volume scores for the provided horizons using the latest slice for a ticker.

    latest_slice must be sorted by date ascending and contain at least max(horizons)+1 rows.
    Returns (total_score, breakdown_dict).
    """
    total = 0
    breakdown: dict[str, float] = {}
    if latest_slice.empty:
        return 0, breakdown
    # Ensure we have numeric close/volume
    s_close = pd.to_numeric(latest_slice['close'], errors='coerce')
    s_vol = pd.to_numeric(latest_slice['volume'], errors='coerce')
    n = len(latest_slice)
    for h in horizons:
        if n <= h:
            # Not enough history for this horizon
            continue
        start_price = s_close.iloc[-h-1]
        end_price = s_close.iloc[-1]
        start_vol = s_vol.iloc[-h-1]
        end_vol = s_vol.iloc[-1]
        if pd.notna(start_price) and start_price != 0:
            price_pct = (end_price - start_price) / start_price * 100.0
        else:
            price_pct = np.nan
        if pd.notna(start_vol) and start_vol != 0:
            vol_pct = (end_vol - start_vol) / start_vol * 100.0
        else:
            vol_pct = np.nan
        breakdown[f'price_change_{h}d_pct'] = float(price_pct) if np.isfinite(price_pct) else np.nan
        breakdown[f'volume_change_{h}d_pct'] = float(vol_pct) if np.isfinite(vol_pct) else np.nan
        total += _score_change(price_pct) + _score_change(vol_pct)
    return total, breakdown


def _export_momentum_top_lists_from_prices(prices_df: pd.DataFrame, output_dir: str = 'data/momentum', top_n: int = 25) -> None:
    """Create top performers lists for 5d/15d/30d momentum and write CSV files.

    Produces:
      - data/momentum/top5d_performers.csv
      - data/momentum/top15d_performers.csv
      - data/momentum/top30d_performers.csv
    """
    if prices_df is None or prices_df.empty:
        raise ValueError('prices_df is empty; cannot compute momentum lists')

    # Ensure proper dtypes
    df = prices_df.copy()
    if 'date' not in df.columns or 'ticker' not in df.columns or 'close' not in df.columns or 'volume' not in df.columns:
        raise ValueError("prices_df must contain columns: 'date','ticker','close','volume'")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'ticker', 'close', 'volume'])
    df = df.sort_values(['ticker', 'date'])

    os.makedirs(output_dir, exist_ok=True)

    # Define horizon sets for scoring per output window
    config = {
        5: [1, 3, 5],
        15: [1, 3, 5, 10, 15],
        30: [1, 3, 5, 10, 15, 20, 30],
    }

    latest_date = df['date'].max()

    results: dict[int, list[dict]] = {5: [], 15: [], 30: []}
    for ticker, g in df.groupby('ticker'):
        g_sorted = g.sort_values('date')
        # For realized returns we need start/end prices for each window
        for window_days, horizons in config.items():
            if len(g_sorted) <= window_days:
                continue
            # Compute scores using multiple horizons
            score, breakdown = _compute_window_scores(g_sorted, horizons)

            # Realized return for this window
            end_row = g_sorted.iloc[-1]
            end_price = float(end_row['close'])
            end_date = pd.to_datetime(end_row['date'])
            start_row = g_sorted.iloc[-window_days-1]
            start_price = float(start_row['close'])
            start_date = pd.to_datetime(start_row['date'])
            if start_price == 0 or not np.isfinite(start_price):
                realized_return_pct = np.nan
            else:
                realized_return_pct = (end_price - start_price) / start_price * 100.0

            row = {
                'ticker': str(ticker),
                'start_date': start_date.date(),
                'end_date': end_date.date(),
                'start_price': start_price,
                'end_price': end_price,
                'realized_return_pct': realized_return_pct,
                'momentum_score': int(score),
            }
            # Add a few diagnostic fields: primary window price/volume change
            try:
                from_idx = -window_days-1
                to_idx = -1
                p0 = float(g_sorted['close'].iloc[from_idx])
                p1 = float(g_sorted['close'].iloc[to_idx])
                v0 = float(g_sorted['volume'].iloc[from_idx])
                v1 = float(g_sorted['volume'].iloc[to_idx])
                row[f'price_change_{window_days}d_pct'] = (p1 - p0) / p0 * 100.0 if p0 else np.nan
                row[f'volume_change_{window_days}d_pct'] = (v1 - v0) / v0 * 100.0 if v0 else np.nan
            except Exception:
                row[f'price_change_{window_days}d_pct'] = np.nan
                row[f'volume_change_{window_days}d_pct'] = np.nan

            # Optionally include horizon breakdown for debugging
            for k, v in breakdown.items():
                # Only include the per-window that matches output window or keep all; keep all for transparency
                row[k] = v

            results[window_days].append(row)

    # Build DataFrames, rank, and save top-N
    for window_days, rows in results.items():
        if not rows:
            continue
        out_df = pd.DataFrame(rows)
        out_df = out_df.replace([np.inf, -np.inf], np.nan)
        out_df = out_df.sort_values(['momentum_score', f'price_change_{window_days}d_pct'], ascending=[False, False])
        out_top = out_df.head(int(top_n))
        out_path = os.path.join(output_dir, f'top{window_days}d_performers.csv')
        out_top.to_csv(out_path, index=False)
        print(f"Saved momentum top-{top_n} for {window_days}d window to: {out_path} (rows={len(out_top)})")

def main():
    """
    This script runs the entire data processing and model training pipeline.
    """
    print("Starting the data pipeline...")
    
    parser = argparse.ArgumentParser(description="Run data processing and model training pipeline")
    parser.add_argument("--date", type=str, default=None, help="Date in YYYY-MM-DD format. If omitted, use undated files.")
    args = parser.parse_args()

    TARGET_DATE = None
    if args.date:
        try:
            TARGET_DATE = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"Invalid --date value '{args.date}'. Expected format YYYY-MM-DD. Falling back to undated files.")
            TARGET_DATE = None
    FEATURE_FILE_PATH_TOP = 'data/featured_stocks_top.csv'
    FEATURE_FILE_PATH_MOM = 'data/momentum/featured_stocks_momentum.csv'

    # Step 1: Load the data
    master_df, prices_df = load_all_data(file_date=TARGET_DATE)
    if master_df.empty or prices_df.empty:
        print("Pipeline stopped due to data loading errors.")
        return

    # Step 2: Engineer all features
    print("Engineering features...")
    # Use the provided date for features; if not provided, infer from prices_df
    features_date = TARGET_DATE
    if features_date is None:
        try:
            latest_prices_date = prices_df['date'].max()
            if hasattr(latest_prices_date, 'date'):
                features_date = latest_prices_date.date()
            else:
                # In case the column wasn't parsed as datetime for any reason
                features_date = date.today()
        except Exception:
            features_date = date.today()

    featured_stocks_df = create_all_features(master_df, prices_df, today=features_date)
    # Ensure momentum dir exists
    os.makedirs(os.path.dirname(FEATURE_FILE_PATH_MOM), exist_ok=True)
    # Save two separate feature files (initially identical; can diverge later)
    featured_stocks_df.to_csv(FEATURE_FILE_PATH_TOP, index=False)
    featured_stocks_df.to_csv(FEATURE_FILE_PATH_MOM, index=False)
    print(f"Validated features have been saved to: {FEATURE_FILE_PATH_TOP} and {FEATURE_FILE_PATH_MOM}")

    # Step 2.5: Generate momentum top performer CSVs per features_top_momentum.md
    try:
        _export_momentum_top_lists_from_prices(prices_df, output_dir='data/momentum', top_n=25)
    except Exception as e:
        print(f"Failed to create momentum top lists: {e}")
    
    #
    # V V V NEW STEP ADDED V V V
    #
    # Step 3: Train separate models (optional)
    try:
        # Delay import so that missing ML deps do not break feature generation
        from prediction_model import train_top_model, train_momentum_model  # noqa: WPS433
        top_result = train_top_model()
        print(top_result)
        mom_result = train_momentum_model()
        print(mom_result)
    except ModuleNotFoundError as e:
        # Gracefully skip training if ML dependencies (e.g., xgboost) are missing
        print(f"Skipping model training: {e}")
    except Exception as e:
        print(f"Training step failed but features were generated successfully: {e}")
    #
    # ^ ^ ^ END OF NEW STEP ^ ^ ^
    #

if __name__ == "__main__":
    main()