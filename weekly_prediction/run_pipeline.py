from data_loader import load_all_data
from feature_engineering import create_all_features
from datetime import date, datetime
import argparse

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
    FEATURE_FILE_PATH_MOM = 'data/featured_stocks_momentum.csv'

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
    # Save two separate feature files (initially identical; can diverge later)
    featured_stocks_df.to_csv(FEATURE_FILE_PATH_TOP, index=False)
    featured_stocks_df.to_csv(FEATURE_FILE_PATH_MOM, index=False)
    print(f"Validated features have been saved to: {FEATURE_FILE_PATH_TOP} and {FEATURE_FILE_PATH_MOM}")
    
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