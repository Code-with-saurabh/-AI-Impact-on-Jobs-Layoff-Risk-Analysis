import os
import pandas as pd
import re


def handle_missing_values(df, strategy="drop", threshold=0.5):
    try:
        if df is None or df.empty:
            print("[WARNING] DataFrame is None or empty.")
            return df

        print("\n" + "=" * 60)
        print("HANDLING MISSING VALUES")
        print("=" * 60)

        initial_rows = len(df)
        missing_before = df.isnull().sum().sum()
        print(f"\nMissing values before cleaning: {missing_before}")

        missing_percent = df.isnull().mean()
        cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()

        if cols_to_drop:
            print(f"\nDropping columns with >{threshold*100}% missing values:")
            for col in cols_to_drop:
                print(f"  - {col} ({missing_percent[col]*100:.2f}% missing)")
            df = df.drop(columns=cols_to_drop)

        if strategy == "drop":
            rows_before = len(df)
            df = df.dropna()
            rows_dropped = rows_before - len(df)
            print(f"\nStrategy: Drop rows with missing values")
            print(f"Rows dropped: {rows_dropped}")

        elif strategy == "fill_mean":
            print(f"\nStrategy: Fill numeric columns with mean")
            numeric_cols = df.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    print(f"  - Filled '{col}' with mean: {mean_val:.2f}")

        elif strategy == "fill_median":
            print(f"\nStrategy: Fill numeric columns with median")
            numeric_cols = df.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"  - Filled '{col}' with median: {median_val:.2f}")

        elif strategy == "fill_mode":
            print(f"\nStrategy: Fill all columns with mode")
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(mode_val)
                    print(f"  - Filled '{col}' with mode: {mode_val}")

        else:
            print(f"[WARNING] Unknown strategy '{strategy}'. Using 'drop'.")
            df = df.dropna()

        missing_after = df.isnull().sum().sum()
        print(f"\nMissing values after cleaning: {missing_after}")
        print(f"Rows: {initial_rows} -> {len(df)}")

        print("=" * 60)
        return df

    except Exception as e:
        print(f"[ERROR] Error handling missing values: {e}")
        return df


def remove_duplicates(df):
    try:
        if df is None or df.empty:
            print("[WARNING] DataFrame is None or empty.")
            return df

        print("\n" + "=" * 60)
        print("REMOVING DUPLICATES")
        print("=" * 60)

        initial_rows = len(df)
        duplicate_count = df.duplicated().sum()

        print(f"\nDuplicate rows found: {duplicate_count}")

        if duplicate_count > 0:
            df = df.drop_duplicates()
            removed = initial_rows - len(df)
            print(f"Duplicate rows removed: {removed}")
            print(f"Remaining rows: {len(df)}")
        else:
            print("No duplicates found. Dataset is clean.")

        print("=" * 60)
        return df

    except Exception as e:
        print(f"[ERROR] Error removing duplicates: {e}")
        return df


def remove_invalid_records(df):
    try:
        if df is None or df.empty:
            print("[WARNING] DataFrame is None or empty.")
            return df

        print("\n" + "=" * 60)
        print("REMOVING INVALID RECORDS")
        print("=" * 60)

        initial_rows = len(df)
        invalid_count = 0

        if "Age" in df.columns:
            invalid_age = (df["Age"] < 0) | (df["Age"] > 100)
            if invalid_age.any():
                count = invalid_age.sum()
                print(f"\nRemoving {count} rows with invalid Age values (negative or >100)")
                df = df[~invalid_age]
                invalid_count += count

        if "Years_of_Experience" in df.columns:
            invalid_exp = df["Years_of_Experience"] < 0
            if invalid_exp.any():
                count = invalid_exp.sum()
                print(f"Removing {count} rows with negative Years_of_Experience")
                df = df[~invalid_exp]
                invalid_count += count

        if "Routine_Task_Percentage" in df.columns:
            invalid_routine = (df["Routine_Task_Percentage"] < 0) | (df["Routine_Task_Percentage"] > 100)
            if invalid_routine.any():
                count = invalid_routine.sum()
                print(f"Removing {count} rows with invalid Routine_Task_Percentage")
                df = df[~invalid_routine]
                invalid_count += count

        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            empty_strings = df[col].astype(str).str.strip() == ""
            if empty_strings.any():
                count = empty_strings.sum()
                print(f"Removing {count} rows with empty '{col}' values")
                df = df[~empty_strings]
                invalid_count += count

        total_removed = initial_rows - len(df)
        print(f"\nTotal invalid records removed: {total_removed}")
        print(f"Remaining valid records: {len(df)}")

        print("=" * 60)
        return df

    except Exception as e:
        print(f"[ERROR] Error removing invalid records: {e}")
        return df


def standardize_column_names(df):
    try:
        if df is None or df.empty:
            print("[WARNING] DataFrame is None or empty.")
            return df

        print("\n" + "=" * 60)
        print("STANDARDIZING COLUMN NAMES")
        print("=" * 60)

        original_columns = df.columns.tolist()

        new_columns = []
        for col in original_columns:
            new_col = col.lower()
            new_col = new_col.replace(" ", "_")
            new_col = re.sub(r"[^a-z0-9_]", "", new_col)
            new_col = re.sub(r"_+", "_", new_col)
            new_col = new_col.strip("_")
            new_columns.append(new_col)

        print("\nColumn Name Changes:")
        print("-" * 50)
        for old, new in zip(original_columns, new_columns):
            if old != new:
                print(f"  '{old}' -> '{new}'")

        df.columns = new_columns
        print(f"\nTotal columns standardized: {len(new_columns)}")

        print("=" * 60)
        return df

    except Exception as e:
        print(f"[ERROR] Error standardizing column names: {e}")
        return df


def save_clean_dataset(df, output_path):
    try:
        if df is None or df.empty:
            print("[WARNING] DataFrame is None or empty. Nothing to save.")
            return False

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"\nCreated output directory: {output_dir}")

        df.to_csv(output_path, index=False)

        print("\n" + "=" * 60)
        print("CLEANED DATASET SAVED")
        print("=" * 60)
        print(f"  Path: {output_path}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  File Size: {os.path.getsize(output_path) / 1024:.2f} KB")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"[ERROR] Error saving cleaned dataset: {e}")
        return False
