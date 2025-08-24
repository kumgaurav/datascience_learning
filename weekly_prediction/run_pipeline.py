import argparse
import time
import logging
from utils.logging_utils import configure_logging
from utils.config import load_config, apply_env_from_config
import os

from utils.db_fetch import fetch_all as db_fetch_all
from utils.feature_engineering import build_features_for_inputs
from utils.train_xgboost_weekly import train_and_export as run_xgb_weekly
from utils.train_lstm_weekly import train_and_export as run_lstm_weekly
from utils.run_ensemble_weekly import run_ensemble as run_ensemble_weekly


def main():
    parser = argparse.ArgumentParser(description="Weekly prediction pipeline with optional DB fetch, features, XGB, LSTM and Ensemble steps")
    parser.add_argument("--skip_fetch", action="store_true", help="Skip fetching data from the database")
    parser.add_argument("--fetch_start", default=None, help="Optional fetch start date (YYYY-MM-DD)")
    parser.add_argument("--fetch_end", default=None, help="Optional fetch end date (YYYY-MM-DD)")
    parser.add_argument("--skip_features", action="store_true", help="Skip feature engineering step")
    parser.add_argument("--clean_prices", default=os.getenv('PATH_CLEAN_PRICES', "data/input/stock_prices_with_clean_data.csv"), help="Path to clean prices CSV")
    parser.add_argument("--unclean_prices", default=os.getenv('PATH_UNCLEAN_PRICES', "data/input/stock_prices_with_unclean_data.csv"), help="Path to unclean prices CSV")
    parser.add_argument("--features_out", default=os.getenv('PATH_FEATURES_DIR', "data/features"), help="Directory to write engineered features")
    # XGBoost step
    parser.add_argument("--skip_xgb", action="store_true", help="Skip XGBoost weekly training/export")
    parser.add_argument("--xgb_features", default=None, help="Path to features CSV for XGB (defaults to features_out/stock_features_clean.csv)")
    parser.add_argument("--xgb_out", default=os.getenv('PATH_XGB_OUT', "data/xgboost/xgboost_weekly_output.csv"), help="Output CSV for XGB weekly scores")
    parser.add_argument("--xgb_horizon", type=int, default=5, help="Forward return horizon for XGB")
    # LSTM step
    parser.add_argument("--skip_lstm", action="store_true", help="Skip LSTM weekly training/export")
    parser.add_argument("--lstm_features", default=None, help="Path to features CSV for LSTM (defaults to features_out/stock_features_clean.csv)")
    parser.add_argument("--lstm_out", default=os.getenv('PATH_LSTM_OUT', "data/lstm/lstm_weekly_output.csv"), help="Output CSV for LSTM weekly predictions")
    parser.add_argument("--lstm_lookback", type=int, default=None, help="Sequence length (override model default)")
    parser.add_argument("--lstm_horizon", type=int, default=None, help="Forward return horizon (override model default)")
    parser.add_argument("--lstm_skip_train", action="store_true", help="Skip LSTM training and use existing saved model/meta")
    parser.add_argument("--lstm_model", default=None, help="Path to saved LSTM model (.keras) if skipping training")
    parser.add_argument("--lstm_meta", default=None, help="Path to LSTM meta JSON if skipping training")
    # Ensemble step
    parser.add_argument("--skip_ensemble", action="store_true", help="Skip ensemble blend step")
    parser.add_argument("--ensemble_method", choices=["weighted","rank","voting","prob"], default=os.getenv('ENSEMBLE_METHOD','weighted'), help="Blending method")
    parser.add_argument("--ensemble_alpha", type=float, default=float(os.getenv('ENSEMBLE_ALPHA','0.6')), help="Weight on XGB component (0..1)")
    parser.add_argument("--ensemble_stack_model", default=os.getenv('ENSEMBLE_STACK_MODEL'), help="Optional path to joblib stacker model")
    parser.add_argument("--ensemble_out", default=os.getenv('PATH_ENSEMBLE_OUT', "data/ensemble/ensemble_weekly_output.csv"), help="Output CSV for ensemble weekly scores")
    args, unknown = parser.parse_known_args()

    # Load config file if provided via env or default path
    cfg_path = os.getenv('APP_CONFIG', 'config.yaml')
    cfg = load_config(cfg_path)
    apply_env_from_config(cfg)
    configure_logging()
    log = logging.getLogger('pipeline')
    log.info("[PIPE] Starting pipeline...")
    t0 = time.perf_counter()

    # Step 0: Fetch datasets from DB (optional)
    if not args.skip_fetch:
        try:
            log.info("[PIPE] Fetching datasets from database...")
            db_fetch_all(output_dir='data/input', start_date=args.fetch_start, end_date=args.fetch_end)
        except Exception as e:
            log.warning(f"[PIPE DIAG] DB fetch failed (continuing with existing CSVs): {e}")

    # Step 1: Feature engineering (optional)
    if not args.skip_features:
        try:
            log.info("[PIPE] Building features for clean and unclean price files...")
            build_features_for_inputs(args.clean_prices, args.unclean_prices, args.features_out)
        except Exception as e:
            log.error(f"[PIPE DIAG] Feature engineering failed: {e}")

    # Resolve default feature paths for downstream steps
    features_clean_default = os.path.join(args.features_out, 'stock_features_clean.csv')
    xgb_features_path = args.xgb_features or features_clean_default
    lstm_features_path = args.lstm_features or features_clean_default

    # Step 2: XGBoost weekly (optional)
    if not args.skip_xgb:
        try:
            log.info(f"[PIPE] Running XGB weekly → features={xgb_features_path}")
            t_xgb = time.perf_counter()
            run_xgb_weekly(xgb_features_path, args.xgb_out, args.xgb_horizon)
            log.info(f"[PIPE TIME] XGB weekly: {time.perf_counter() - t_xgb:.2f}s")
        except Exception as e:
            log.error(f"[PIPE DIAG] XGB weekly failed: {e}")

    # Step 3: LSTM weekly (optional)
    if not args.skip_lstm:
        try:
            log.info(f"[PIPE] Running LSTM weekly → features={lstm_features_path}")
            t_lstm = time.perf_counter()
            run_lstm_weekly(
                features_path=lstm_features_path,
                out_path=args.lstm_out,
                lookback=args.lstm_lookback,
                horizon=args.lstm_horizon,
                skip_train=bool(args.lstm_skip_train),
                model_path=args.lstm_model,
                meta_path=args.lstm_meta,
            )
            log.info(f"[PIPE TIME] LSTM weekly: {time.perf_counter() - t_lstm:.2f}s")
        except Exception as e:
            log.error(f"[PIPE DIAG] LSTM weekly failed: {e}")

    # Step 4: Ensemble weekly (optional)
    if not args.skip_ensemble:
        try:
            log.info("[PIPE] Running Ensemble weekly blend...")
            t_ens = time.perf_counter()
            run_ensemble_weekly(
                xgb_path=args.xgb_out,
                lstm_path=args.lstm_out,
                features_path=xgb_features_path,
                method=str(args.ensemble_method).lower(),
                alpha=float(args.ensemble_alpha),
                stack_model=args.ensemble_stack_model,
                out_path=args.ensemble_out,
            )
            log.info(f"[PIPE TIME] Ensemble weekly: {time.perf_counter() - t_ens:.2f}s")
        except Exception as e:
            log.error(f"[PIPE DIAG] Ensemble weekly failed: {e}")

    log.info("[PIPE] Completed pipeline steps.")
    log.info(f"[PIPE TIME] Total elapsed: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()


