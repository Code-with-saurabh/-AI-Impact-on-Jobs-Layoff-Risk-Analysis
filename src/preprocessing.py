"""
Preprocessing Module for AI Impact on Jobs & Layoff Risk Analysis.
This module provides functions to preprocess data for machine learning.
It includes encoding, scaling, and train-test splitting.
Author: Sharma Saurabh (250863131007)
Course: BE05000231 - Python for Data Science (GTU Semester-5)
Month: 2 - Analysis Phase
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


def encode_categorical_columns(df, target_col="layoff_risk"):
    """
    Encode categorical columns using Label Encoding.
    Parameters:
        df (pandas.DataFrame): The DataFrame to encode.
        target_col (str): Target column name.
    Returns:
        pandas.DataFrame: DataFrame with encoded columns.
        dict: Mapping of encoders for each column.
    """
    try:
        print("\n[Preprocessing] Encoding categorical columns...")
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        encoders = {}
        df_encoded = df.copy()
        for col in categorical_cols:
            if col == target_col:
                continue
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
            print(f"  Encoded '{col}': {len(le.classes_)} classes")
        # Encode target column
        if target_col in df_encoded.columns:
            le_target = LabelEncoder()
            df_encoded[target_col] = le_target.fit_transform(df_encoded[target_col])
            encoders[target_col] = le_target
            print(f"  Encoded target '{target_col}': {list(le_target.classes_)}")
        print(f"\nTotal columns encoded: {len(encoders)}")
        return df_encoded, encoders
    except Exception as e:
        print(f"[ERROR] Error encoding categorical columns: {e}")
        return df, {}


def scale_numerical_columns(df, exclude_cols=None):
    """
    Scale numerical columns using StandardScaler.
    Parameters:
        df (pandas.DataFrame): The DataFrame to scale.
        exclude_cols (list): Columns to exclude from scaling.
    Returns:
        pandas.DataFrame: DataFrame with scaled columns.
        StandardScaler: The fitted scaler object.
    """
    try:
        print("\n[Preprocessing] Scaling numerical columns...")
        if exclude_cols is None:
            exclude_cols = []
        numerical_cols = df.select_dtypes(include=["number"]).columns
        cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]
        if len(cols_to_scale) == 0:
            print("[WARNING] No numerical columns to scale.")
            return df, None
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        print(f"  Scaled {len(cols_to_scale)} numerical columns")
        print(f"  Columns: {', '.join(cols_to_scale)}")
        return df_scaled, scaler
    except Exception as e:
        print(f"[ERROR] Error scaling numerical columns: {e}")
        return df, None


def split_features_target(df, target_col="layoff_risk"):
    """
    Split DataFrame into features (X) and target (y).
    Parameters:
        df (pandas.DataFrame): The DataFrame to split.
        target_col (str): Target column name.
    Returns:
        pandas.DataFrame: Features DataFrame.
        pandas.Series: Target Series.
    """
    try:
        print("\n[Preprocessing] Splitting features and target...")
        if target_col not in df.columns:
            print(f"[ERROR] Target column '{target_col}' not found.")
            return None, None
        X = df.drop(columns=[target_col])
        y = df[target_col]
        print(f"  Features shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        print(f"  Target distribution:\n{y.value_counts()}")
        return X, y
    except Exception as e:
        print(f"[ERROR] Error splitting features and target: {e}")
        return None, None


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    Parameters:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target.
        test_size (float): Proportion of data for testing.
        random_state (int): Random seed for reproducibility.
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    try:
        print("\n[Preprocessing] Splitting train and test data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Testing set: {X_test.shape[0]} samples")
        print(f"  Train target distribution:\n{y_train.value_counts()}")
        print(f"  Test target distribution:\n{y_test.value_counts()}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"[ERROR] Error in train-test split: {e}")
        return None, None, None, None


def save_preprocessor(scaler, encoders, output_dir):
    """
    Save preprocessor objects to disk.
    Parameters:
        scaler (StandardScaler): The fitted scaler.
        encoders (dict): Dictionary of fitted encoders.
        output_dir (str): Output directory path.
    """
    try:
        print("\n[Preprocessing] Saving preprocessor objects...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if scaler is not None:
            scaler_path = os.path.join(output_dir, "scaler.pkl")
            joblib.dump(scaler, scaler_path)
            print(f"  Saved scaler: {scaler_path}")
        if encoders:
            encoder_path = os.path.join(output_dir, "encoder.pkl")
            joblib.dump(encoders, encoder_path)
            print(f"  Saved encoders: {encoder_path}")
    except Exception as e:
        print(f"[ERROR] Error saving preprocessor: {e}")


def run_preprocessing(df, config):
    """
    Run complete preprocessing pipeline.
    Parameters:
        df (pandas.DataFrame): The engineered DataFrame.
        config: Configuration module.
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, encoders)
    """
    try:
        print("\n" + "#" * 70)
        print("# DATA PREPROCESSING")
        print("#" * 70)
        # Step 1: Encode categorical columns
        df_encoded, encoders = encode_categorical_columns(df, config.TARGET_COLUMN)
        # Step 2: Split features and target
        X, y = split_features_target(df_encoded, config.TARGET_COLUMN)
        if X is None or y is None:
            print("[ERROR] Failed to split features and target.")
            return None, None, None, None, None, None
        # Step 3: Scale numerical columns
        X_scaled, scaler = scale_numerical_columns(X)
        # Step 4: Train-test split
        X_train, X_test, y_train, y_test = train_test_split_data(
            X_scaled, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        # Step 5: Save preprocessor objects
        save_preprocessor(scaler, encoders, config.MODELS_OUTPUT_DIR)
        # Step 6: Save processed data
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        train_data.to_csv(config.TRAIN_DATA_PATH, index=False)
        test_data.to_csv(config.TEST_DATA_PATH, index=False)
        print(f"\nSaved training data: {config.TRAIN_DATA_PATH}")
        print(f"Saved testing data: {config.TEST_DATA_PATH}")
        print("\n" + "#" * 70)
        print("# PREPROCESSING COMPLETED SUCCESSFULLY")
        print("#" * 70)
        return X_train, X_test, y_train, y_test, scaler, encoders
    except Exception as e:
        print(f"[ERROR] Error in preprocessing: {e}")
        return None, None, None, None, None, None
