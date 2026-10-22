import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_loading import (
    load_dataset,
    inspect_dataset,
    display_shape,
    display_columns,
    display_data_types,
    display_missing_values,
)
from src.data_cleaning import (
    handle_missing_values,
    remove_duplicates,
    remove_invalid_records,
    standardize_column_names,
    save_clean_dataset,
)
from src.exploratory_data_analysis import run_complete_eda
from src.feature_engineering import run_feature_engineering
from src.visualization import run_complete_visualization
from src.preprocessing import run_preprocessing
from src.model_training import run_model_training
from src.model_evaluation import evaluate_all_models, save_evaluation_report
from src.prediction import load_model, load_preprocessor, run_predictions
from src.utils import print_section_header, print_subsection_header, get_timestamp


def print_banner():
    print("\n" + "=" * 70)
    print("=" * 70)
    print("  AI IMPACT ON JOBS & LAYOFF RISK ANALYSIS")
    print("  Using Machine Learning")
    print("  GTU Semester-5: Python for Data Science (BE05000231)")
    print("  Student: Sharma Saurabh (250863131007)")
    print("=" * 70)
    print("=" * 70)


def run_month1():
    print_section_header("MONTH-1: FOUNDATION PHASE")

    print_subsection_header("PHASE 1: DATA LOADING")
    df = load_dataset(config.RAW_DATASET_PATH)

    print_subsection_header("PHASE 2: DATASET INSPECTION")
    rows, cols = display_shape(df)
    columns = display_columns(df)
    dtypes = display_data_types(df)
    missing = display_missing_values(df)
    inspect_dataset(df)

    print_subsection_header("PHASE 3: DATA CLEANING")
    df = handle_missing_values(df, strategy="drop")
    df = remove_duplicates(df)
    df = remove_invalid_records(df)
    df = standardize_column_names(df)

    print_subsection_header("PHASE 4: SAVE CLEANED DATASET")
    save_clean_dataset(df, config.CLEANED_DATASET_PATH)

    print(f"\n[Month-1 Complete] Cleaned dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    return df


def run_month2(df):
    print_section_header("MONTH-2: ANALYSIS PHASE")

    print_subsection_header("PHASE 1: EXPLORATORY DATA ANALYSIS")
    eda_results = run_complete_eda(df)

    print_subsection_header("PHASE 2: FEATURE ENGINEERING")
    df_engineered = run_feature_engineering(df)

    print_subsection_header("PHASE 3: DATA VISUALIZATION")
    run_complete_visualization(df_engineered, output_dir=config.CHARTS_OUTPUT_DIR)

    print_subsection_header("PHASE 4: DATA PREPROCESSING")
    X_train, X_test, y_train, y_test, scaler, encoders = run_preprocessing(
        df_engineered, config
    )

    print(f"\n[Month-2 Complete] Training set: {X_train.shape}, Testing set: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler, encoders


def run_month3(X_train, X_test, y_train, y_test, scaler, encoders):
    print_section_header("MONTH-3: MODELING PHASE")

    print_subsection_header("PHASE 1: MODEL TRAINING")
    trained_models, comparison_df, best_model_name = run_model_training(
        X_train, X_test, y_train, y_test, config
    )

    print_subsection_header("PHASE 2: MODEL EVALUATION")
    all_results = evaluate_all_models(
        trained_models, X_test, y_test, output_dir=config.CHARTS_OUTPUT_DIR
    )

    save_evaluation_report(all_results, config.EVALUATION_REPORT_PATH)

    print_subsection_header("PHASE 3: MAKING PREDICTIONS")
    best_model = trained_models[best_model_name]["model"]
    predictions_df = run_predictions(
        best_model, scaler, encoders, X_test, y_test, config
    )

    print(f"\n[Month-3 Complete] Best model: {best_model_name}")


def print_completion_summary():
    print("\n" + "=" * 70)
    print("=" * 70)
    print("  PROJECT COMPLETED SUCCESSFULLY!")
    print("")
    print("  Phases Completed:")
    print("  [Month-1] Foundation Phase")
    print("    - Project structure created")
    print("    - Dataset loaded and inspected")
    print("    - Data cleaned and saved")
    print("")
    print("  [Month-2] Analysis Phase")
    print("    - Exploratory Data Analysis performed")
    print("    - Features engineered")
    print("    - Visualizations generated")
    print("    - Data preprocessed for ML")
    print("")
    print("  [Month-3] Modeling Phase")
    print("    - Multiple ML models trained")
    print("    - Models evaluated and compared")
    print("    - Best model saved")
    print("    - Predictions generated")
    print("")
    print("  Output Files:")
    print(f"    - Cleaned Data: {config.CLEANED_DATASET_PATH}")
    print(f"    - Charts: {config.CHARTS_OUTPUT_DIR}")
    print(f"    - Best Model: {config.BEST_MODEL_PATH}")
    print(f"    - Evaluation Report: {config.EVALUATION_REPORT_PATH}")
    print(f"    - Predictions: {config.PREDICTIONS_PATH}")
    print("=" * 70)
    print("=" * 70 + "\n")


def main():
    try:
        print_banner()
        print(f"\nPipeline started at: {get_timestamp()}")

        df = run_month1()

        X_train, X_test, y_train, y_test, scaler, encoders = run_month2(df)

        run_month3(X_train, X_test, y_train, y_test, scaler, encoders)

        print_completion_summary()

        print(f"Pipeline completed at: {get_timestamp()}")

    except KeyboardInterrupt:
        print("\n\n[INFO] Process interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
