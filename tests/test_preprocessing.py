"""
Test module for preprocessing functions.
Author: Sharma Saurabh (250863131007)
Course: BE05000231 - Python for Data Science (GTU Semester-5)
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    encode_categorical_columns,
    scale_numerical_columns,
    split_features_target,
    train_test_split_data,
    save_preprocessor,
)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age": np.random.randint(22, 60, n),
        "education_level": np.random.choice(["Bachelor", "Master", "PhD"], n),
        "years_of_experience": np.random.randint(0, 35, n),
        "industry": np.random.choice(["Tech", "Finance", "Health"], n),
        "routine_task_percentage": np.random.uniform(10, 95, n),
        "creativity_requirement": np.random.uniform(5, 95, n),
        "ai_adoption_level": np.random.choice(["Low", "Medium", "High"], n),
        "layoff_risk": np.random.choice(["Low", "Medium", "High"], n),
    })


class TestEncodeCategoricalColumns:
    """Test cases for encode_categorical_columns function."""

    def test_encode_categorical_columns(self, sample_df):
        """Test encoding categorical columns."""
        df_encoded, encoders = encode_categorical_columns(sample_df, "layoff_risk")
        assert isinstance(df_encoded, pd.DataFrame)
        assert isinstance(encoders, dict)
        assert "layoff_risk" in encoders

    def test_encoded_values_numeric(self, sample_df):
        """Test that encoded values are numeric."""
        df_encoded, _ = encode_categorical_columns(sample_df, "layoff_risk")
        categorical_cols = sample_df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col != "layoff_risk":
                assert df_encoded[col].dtype in ["int64", "int32"]


class TestScaleNumericalColumns:
    """Test cases for scale_numerical_columns function."""

    def test_scale_numerical_columns(self, sample_df):
        """Test scaling numerical columns."""
        df_encoded, _ = encode_categorical_columns(sample_df, "layoff_risk")
        df_scaled, scaler = scale_numerical_columns(df_encoded)
        assert isinstance(df_scaled, pd.DataFrame)
        assert scaler is not None

    def test_scaled_mean_near_zero(self, sample_df):
        """Test that scaled values have mean near zero."""
        df_encoded, _ = encode_categorical_columns(sample_df, "layoff_risk")
        df_scaled, _ = scale_numerical_columns(df_encoded)
        numerical_cols = df_encoded.select_dtypes(include=["number"]).columns
        for col in numerical_cols:
            assert abs(df_scaled[col].mean()) < 0.1


class TestSplitFeaturesTarget:
    """Test cases for split_features_target function."""

    def test_split_features_target(self, sample_df):
        """Test splitting features and target."""
        df_encoded, _ = encode_categorical_columns(sample_df, "layoff_risk")
        X, y = split_features_target(df_encoded, "layoff_risk")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "layoff_risk" not in X.columns

    def test_split_shapes(self, sample_df):
        """Test shapes after splitting."""
        df_encoded, _ = encode_categorical_columns(sample_df, "layoff_risk")
        X, y = split_features_target(df_encoded, "layoff_risk")
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == len(sample_df.columns) - 1


class TestTrainTestSplitData:
    """Test cases for train_test_split_data function."""

    def test_train_test_split(self, sample_df):
        """Test train-test split."""
        df_encoded, _ = encode_categorical_columns(sample_df, "layoff_risk")
        X, y = split_features_target(df_encoded, "layoff_risk")
        X_train, X_test, y_train, y_test = train_test_split_data(X, y)
        assert X_train.shape[0] > X_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]

    def test_split_ratio(self, sample_df):
        """Test split ratio."""
        df_encoded, _ = encode_categorical_columns(sample_df, "layoff_risk")
        X, y = split_features_target(df_encoded, "layoff_risk")
        X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2)
        assert abs(len(X_test) / len(X) - 0.2) < 0.05


class TestSavePreprocessor:
    """Test cases for save_preprocessor function."""

    def test_save_preprocessor(self, sample_df, tmp_path):
        """Test saving preprocessor objects."""
        df_encoded, encoders = encode_categorical_columns(sample_df, "layoff_risk")
        _, scaler = scale_numerical_columns(df_encoded)
        save_preprocessor(scaler, encoders, str(tmp_path))
        assert os.path.exists(tmp_path / "scaler.pkl")
        assert os.path.exists(tmp_path / "encoder.pkl")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
