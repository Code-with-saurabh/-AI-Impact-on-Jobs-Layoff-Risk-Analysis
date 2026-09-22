"""
Configuration Module for AI Impact on Jobs & Layoff Risk Analysis Project.
This module centralizes all project paths and configuration settings.
It ensures no hardcoded paths are used throughout the project.
Author: Sharma Saurabh (250863131007)
Course: BE05000231 - Python for Data Science (GTU Semester-5)
"""
import os

# =============================================================================
# PROJECT ROOT PATH
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# DATA FOLDER PATHS
# =============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# =============================================================================
# RAW DATASET PATH
# =============================================================================
RAW_DATASET_PATH = os.path.join(
    RAW_DATA_DIR, "ai-impact-jobs-layoff-risk-dataset.csv"
)

# =============================================================================
# OUTPUT FOLDER PATHS
# =============================================================================
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PROCESSED_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "processed_data")
CHARTS_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "charts")
MODELS_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "models")
REPORTS_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "reports")
PREDICTIONS_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "predictions")

# =============================================================================
# FILE PATHS
# =============================================================================
CLEANED_DATASET_PATH = os.path.join(PROCESSED_OUTPUT_DIR, "cleaned_dataset.csv")
ENGINEERED_DATASET_PATH = os.path.join(PROCESSED_OUTPUT_DIR, "engineered_dataset.csv")
PREPROCESSED_DATASET_PATH = os.path.join(PROCESSED_OUTPUT_DIR, "preprocessed_dataset.csv")
TRAIN_DATA_PATH = os.path.join(PROCESSED_OUTPUT_DIR, "train_data.csv")
TEST_DATA_PATH = os.path.join(PROCESSED_OUTPUT_DIR, "test_data.csv")
BEST_MODEL_PATH = os.path.join(MODELS_OUTPUT_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_OUTPUT_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_OUTPUT_DIR, "encoder.pkl")
EVALUATION_REPORT_PATH = os.path.join(REPORTS_OUTPUT_DIR, "evaluation_report.txt")
PREDICTIONS_PATH = os.path.join(PREDICTIONS_OUTPUT_DIR, "predictions.csv")

# =============================================================================
# DOCUMENTATION FOLDER PATH
# =============================================================================
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

# =============================================================================
# TESTS FOLDER PATH
# =============================================================================
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = "layoff_risk"
