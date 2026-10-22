import os
import pandas as pd
import sys


def check_file_exists(file_path):
    try:
        if os.path.exists(file_path):
            print(f"[OK] File found: {file_path}")
            return True
        else:
            print(f"[ERROR] File not found: {file_path}")
            return False
    except Exception as e:
        print(f"[ERROR] Error checking file existence: {e}")
        return False


def load_dataset(file_path):
    try:
        if not check_file_exists(file_path):
            raise FileNotFoundError(f"Dataset not found at: {file_path}")

        print(f"\nLoading dataset from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"[OK] Dataset loaded successfully!")
        return df

    except FileNotFoundError as e:
        print(f"[ERROR] FileNotFoundError: {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"[ERROR] The file is empty: {file_path}")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"[ERROR] ParserError: Unable to parse CSV file. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error while loading dataset: {e}")
        sys.exit(1)


def inspect_dataset(df):
    try:
        if df is None or df.empty:
            print("[WARNING] DataFrame is None or empty.")
            return

        print("\n" + "=" * 60)
        print("DATASET INSPECTION")
        print("=" * 60)

        print("\n--- First 5 Rows ---")
        print(df.head())

        print("\n--- Last 5 Rows ---")
        print(df.tail())

        print("\n--- Basic Statistics ---")
        print(df.describe())

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"[ERROR] Error inspecting dataset: {e}")


def display_shape(df):
    try:
        if df is None or df.empty:
            print("[WARNING] DataFrame is None or empty.")
            return (0, 0)

        rows, cols = df.shape
        print(f"\nDataset Shape: {rows} rows x {cols} columns")
        return (rows, cols)

    except Exception as e:
        print(f"[ERROR] Error getting dataset shape: {e}")
        return (0, 0)


def display_columns(df):
    try:
        if df is None or df.empty:
            print("[WARNING] DataFrame is None or empty.")
            return []

        columns = df.columns.tolist()
        print(f"\nColumn Names ({len(columns)} columns):")
        for i, col in enumerate(columns, 1):
            print(f"  {i}. {col}")
        return columns

    except Exception as e:
        print(f"[ERROR] Error displaying columns: {e}")
        return []


def display_data_types(df):
    try:
        if df is None or df.empty:
            print("[WARNING] DataFrame is None or empty.")
            return None

        dtypes = df.dtypes
        print("\nColumn Data Types:")
        print("-" * 40)
        for col, dtype in dtypes.items():
            print(f"  {col}: {dtype}")
        return dtypes

    except Exception as e:
        print(f"[ERROR] Error displaying data types: {e}")
        return None


def display_missing_values(df):
    try:
        if df is None or df.empty:
            print("[WARNING] DataFrame is None or empty.")
            return None

        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100

        print("\nMissing Values Analysis:")
        print("-" * 50)
        print(f"{'Column':<35} {'Missing':<10} {'Percent':<10}")
        print("-" * 50)

        has_missing = False
        for col, count in missing_count.items():
            if count > 0:
                has_missing = True
                percent = missing_percent[col]
                print(f"  {col:<33} {count:<10} {percent:.2f}%")

        if not has_missing:
            print("  No missing values found in any column!")

        print("-" * 50)
        total_missing = missing_count.sum()
        print(f"\nTotal Missing Values: {total_missing}")

        return missing_count

    except Exception as e:
        print(f"[ERROR] Error displaying missing values: {e}")
        return None
