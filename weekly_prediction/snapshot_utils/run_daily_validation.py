#!/usr/bin/env python3
"""
Run daily validation pipeline as a single utility:
  1) Evaluator (build eval_input, compute actuals, run XGB/LSTM predictions, write eval CSV)
  2) Feature drift report
  3) Attribution (feature importance + per-ticker contributions)

Outputs are written under snapshot/<date>/eval, snapshot/<date>/drift, snapshot/<date>/attr.

Usage:
  python scripts/run_daily_validation.py \
    --snapshot snapshot/2025-08-18/snapshot.csv \
    --prices data/stock_prices_input.csv \
    --date 2025-08-18

Notes:
 - XGB and LSTM predictors are attempted by default by the evaluator; set env EVAL_DISABLE_XGB=1 or
   EVAL_DISABLE_LSTM=1 to disable them.
"""

import argparse
import os
import subprocess
import sys
import glob
import logging
from datetime import date

# Ensure project root is importable when invoked from snapshot_utils
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.config import load_config, apply_env_from_config  # type: ignore
from utils.logging_utils import configure_logging  # type: ignore


def _find_latest_snapshot_file(base_dir: str = "snapshot") -> str:
    subdirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    if not subdirs:
        raise FileNotFoundError("No snapshot directories found")
    latest_dir = sorted(subdirs)[-1]
    fp = os.path.join(latest_dir, "snapshot.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Snapshot CSV not found in {latest_dir}")
    return fp


def _infer_eval_date_from_path(path: str) -> str:
    base = os.path.basename(os.path.normpath(os.path.dirname(path) if path.endswith('.csv') else path))
    try:
        # crude validation of YYYY-MM-DD
        y, m, d = base.split('-')
        int(y); int(m); int(d)
        return base
    except Exception:
        return date.today().isoformat()


def _run(cmd, title: str, env: dict | None = None) -> tuple[bool, str]:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        ok = res.returncode == 0
        return ok, res.stdout
    except Exception as e:
        return False, f"{title} failed: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run snapshot → evaluator → drift → attribution in one go")
    parser.add_argument("--config", default=os.getenv("APP_CONFIG", os.path.join(os.path.dirname(__file__), "snapshot_config.yaml")), help="Path to YAML/JSON config for paths and logging")
    parser.add_argument("--snapshot", default=None, help="Path to snapshot CSV or snapshot directory. If omitted, a snapshot can be created if --features and --predictions are provided")
    parser.add_argument("--prices", default=None, help="Path to prices CSV (e.g., data/stock_prices_input.csv)")
    parser.add_argument("--date", dest="eval_date", default=None, help="Evaluation date YYYY-MM-DD. Defaults to snapshot folder name or today")
    parser.add_argument("--refresh_prices", action="store_true", help="Refresh prices in evaluator before running")
    parser.add_argument("--db_config", default=os.getenv("DB_CONFIG_PATH", ""), help="Optional DB config for refresh")
    # Optional: snapshot creation inputs
    parser.add_argument("--features", default=None, help="Features CSV to create snapshot when --snapshot is not given")
    parser.add_argument("--predictions", default=None, help="Predictions CSV to create snapshot when --snapshot is not given")
    args = parser.parse_args()

    # Load config and configure logging
    cfg = load_config(args.config)
    apply_env_from_config(cfg)
    configure_logging(level=os.getenv('LOG_LEVEL'), log_file=os.getenv('LOG_FILE'))
    logger = logging.getLogger(__name__)
    logger.info("Using config file: %s", args.config)

    # Resolve or create snapshot path
    snap = args.snapshot
    if not snap:
        if args.features and args.predictions:
            # Create a snapshot first
            # Ensure we can import snapshot_generator when run as a script
            from pathlib import Path
            scripts_dir = Path(__file__).resolve().parent
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            import snapshot_generator  # type: ignore
            import pandas as pd
            feats = pd.read_csv(args.features)
            preds = pd.read_csv(args.predictions)
            meta = {
                "features_source": os.path.abspath(args.features),
                "predictions_source": os.path.abspath(args.predictions),
            }
            snap_path = snapshot_generator.create_snapshot(feats, preds, meta, snapshot_date=args.eval_date)
            snap = snap_path
        else:
            snap_root = os.getenv('SNAPSHOT_ROOT_DIR', 'snapshot')
            snap = _find_latest_snapshot_file(snap_root)
    if os.path.isdir(snap):
        snap = os.path.join(snap, "snapshot.csv")
    if not os.path.exists(snap):
        raise FileNotFoundError(f"Snapshot CSV not found: {snap}")

    # Resolve date
    eval_date = args.eval_date or _infer_eval_date_from_path(snap)

    # Build a shared environment for subprocesses and pass config down
    child_env = os.environ.copy()
    child_env["APP_CONFIG"] = args.config

    # Resolve prices path: CLI > YAML cfg.paths.prices_csv/clean_prices > env STOCK_PRICES_CSV
    prices_to_use = args.prices
    if not prices_to_use:
        try:
            paths = cfg.get('paths') if isinstance(cfg, dict) else {}
            if isinstance(paths, dict):
                prices_to_use = paths.get('prices_csv') or paths.get('clean_prices') or paths.get('unclean_prices')
        except Exception:
            prices_to_use = None
    if not prices_to_use:
        prices_to_use = os.getenv('STOCK_PRICES_CSV')
    if not prices_to_use or not os.path.exists(prices_to_use):
        raise FileNotFoundError(f"Prices CSV required but not found. Provide --prices or set paths.prices_csv in {args.config} or STOCK_PRICES_CSV env.")
    logger.info("Using prices CSV: %s", prices_to_use)

    # 1) Evaluator
    eval_cmd = [sys.executable, "snapshot_utils/daily_evaluator.py", "--snapshot", snap, "--date", eval_date, "--prices", prices_to_use]
    if args.refresh_prices:
        eval_cmd += ["--refresh_prices"]
        if args.db_config:
            eval_cmd += ["--db_config", args.db_config]
    ok1, out1 = _run(eval_cmd, "Evaluator", env=child_env)
    logger.info("\n=== Evaluator ===\n%s", (out1 or ""))

    # 2) Drift
    drift_cmd = [sys.executable, "snapshot_utils/feature_drift.py", "--snapshot", snap, "--date", eval_date]
    ok2, out2 = _run(drift_cmd, "Feature drift", env=child_env)
    logger.info("\n=== Drift ===\n%s", (out2 or ""))

    # 3) Attribution
    attr_cmd = [sys.executable, "snapshot_utils/attribution.py", "--snapshot", snap, "--date", eval_date, "--prices", prices_to_use]
    ok3, out3 = _run(attr_cmd, "Attribution", env=child_env)
    logger.info("\n=== Attribution ===\n%s", (out3 or ""))

    # 4) Annotate eval with avoid reasons (best-effort)
    ann_cmd = [sys.executable, "snapshot_utils/annotate_eval.py", "--snapshot", snap, "--date", eval_date]
    ok4, out4 = _run(ann_cmd, "Annotate eval", env=child_env)
    if out4:
        logger.info("\n=== Annotate ===\n%s", out4)

    if ok1 and ok2 and ok3:
        logger.info("Daily validation pipeline completed successfully.")
    else:
        logger.warning("Pipeline completed with errors. See log sections above.")


if __name__ == "__main__":
    main()


