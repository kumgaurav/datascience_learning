from data_loader import load_all_data
from feature_engineering import create_all_features
from prediction_model import train_model # <-- Import the new function
from datetime import date

def main():
    """
    This script runs the entire data processing and model training pipeline.
    """
    print("Starting the data pipeline...")
    
    TARGET_DATE = date(2024, 6, 1) # This should match the date in your filenames
    FEATURE_FILE_PATH = 'data/featured_stocks.csv'

    # Step 1: Load the data
    master_df, prices_df = load_all_data(file_date=TARGET_DATE)
    if master_df.empty or prices_df.empty:
        print("Pipeline stopped due to data loading errors.")
        return

    # Step 2: Engineer all features
    print("Engineering features...")
    featured_stocks_df = create_all_features(master_df, prices_df, today=TARGET_DATE)
    featured_stocks_df.to_csv(FEATURE_FILE_PATH, index=False)
    print(f"Validated features have been saved to: {FEATURE_FILE_PATH}")
    
    #
    # V V V NEW STEP ADDED V V V
    #
    # Step 3: Train the prediction model
    training_result = train_model()
    print(training_result)
    #
    # ^ ^ ^ END OF NEW STEP ^ ^ ^
    #

if __name__ == "__main__":
    main()