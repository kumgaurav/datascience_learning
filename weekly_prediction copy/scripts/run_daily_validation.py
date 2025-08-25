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
from datetime import date


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


def _run(cmd, title: str) -> tuple[bool, str]:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ok = res.returncode == 0
        return ok, res.stdout
    except Exception as e:
        return False, f"{title} failed: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run snapshot → evaluator → drift → attribution in one go")
    parser.add_argument("--snapshot", default=None, help="Path to snapshot CSV or snapshot directory. If omitted, a snapshot can be created if --features and --predictions are provided")
    parser.add_argument("--prices", default=None, help="Path to prices CSV (e.g., data/stock_prices_input.csv)")
    parser.add_argument("--date", dest="eval_date", default=None, help="Evaluation date YYYY-MM-DD. Defaults to snapshot folder name or today")
    parser.add_argument("--refresh_prices", action="store_true", help="Refresh prices in evaluator before running")
    parser.add_argument("--db_config", default=os.getenv("DB_CONFIG_PATH", ""), help="Optional DB config for refresh")
    # Optional: snapshot creation inputs
    parser.add_argument("--features", default=None, help="Features CSV to create snapshot when --snapshot is not given")
    parser.add_argument("--predictions", default=None, help="Predictions CSV to create snapshot when --snapshot is not given")
    args = parser.parse_args()

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
            snap = _find_latest_snapshot_file()
    if os.path.isdir(snap):
        snap = os.path.join(snap, "snapshot.csv")
    if not os.path.exists(snap):
        raise FileNotFoundError(f"Snapshot CSV not found: {snap}")

    # Resolve date
    eval_date = args.eval_date or _infer_eval_date_from_path(snap)

    # 1) Evaluator
    eval_cmd = [sys.executable, "scripts/daily_evaluator.py", "--snapshot", snap, "--date", eval_date]
    if args.prices:
        eval_cmd += ["--prices", args.prices]
    if args.refresh_prices:
        eval_cmd += ["--refresh_prices"]
        if args.db_config:
            eval_cmd += ["--db_config", args.db_config]
    ok1, out1 = _run(eval_cmd, "Evaluator")
    print("\n=== Evaluator ===\n" + (out1 or ""))

    # 2) Drift
    drift_cmd = [sys.executable, "scripts/feature_drift.py", "--snapshot", snap, "--date", eval_date]
    ok2, out2 = _run(drift_cmd, "Feature drift")
    print("\n=== Drift ===\n" + (out2 or ""))

    # 3) Attribution
    attr_cmd = [sys.executable, "scripts/attribution.py", "--snapshot", snap, "--date", eval_date]
    ok3, out3 = _run(attr_cmd, "Attribution")
    print("\n=== Attribution ===\n" + (out3 or ""))

    # 4) Annotate eval with avoid reasons (best-effort)
    ann_cmd = [sys.executable, "scripts/annotate_eval.py", "--snapshot", snap, "--date", eval_date]
    ok4, out4 = _run(ann_cmd, "Annotate eval")
    if out4:
        print("\n=== Annotate ===\n" + out4)

    if ok1 and ok2 and ok3:
        print("\n✅ Daily validation pipeline completed successfully.")
    else:
        print("\n⚠️ Pipeline completed with errors. See sections above.")


if __name__ == "__main__":
    main()


