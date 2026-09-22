"""
Test module for data cleaning functions.
Author: Sharma Saurabh (250863131007)
Course: BE05000231 - Python for Data Science (GTU Semester-5)
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loading import load_dataset
from src.data_cleaning import (
    handle_missing_values,
    remove_duplicates,
    remove_invalid_records,
    standardize_column_names,
    save_clean_dataset,
)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    data = {
        "Age": [25, 30, 35, 40, 105],  # 105 is invalid
        "Education_Level": ["Bachelor", "Master", "PhD", "Bachelor", "High School"],
        "Years_of_Experience": [5, 10, 15, 20, -1],  # -1 is invalid
        "Industry": ["Tech", "Finance", "Health", "Tech", "Retail"],
        "Job_Role": ["Engineer", "Analyst", "Manager", "Engineer", "Analyst"],
        "Company_Size": ["Small", "Medium", "Large", "Small", "Medium"],
        "Job_Level": ["Entry", "Mid", "Senior", "Mid", "Entry"],
        "Routine_Task_Percentage": [40.0, 50.0, 60.0, 70.0, 110.0],  # 110 is invalid
        "Creativity_Requirement": [70.0, 60.0, 50.0, 40.0, 30.0],
        "Human_Interaction_Level": [50.0, 60.0, 70.0, 80.0, 90.0],
        "AI_Adoption_Level": ["High", "Medium", "Low", "High", "Medium"],
        "Number_of_AI_Tools_Used": [8, 5, 3, 10, 2],
        "AI_Usage_Hours_Per_Week": [25.0, 15.0, 10.0, 30.0, 5.0],
        "Tasks_Automated_Percentage": [50.0, 40.0, 30.0, 60.0, 70.0],
        "AI_Training_Hours": [40, 30, 20, 50, 10],
        "Layoff_Risk": ["Low", "Medium", "High", "Low", "Medium"],
    }
    return pd.DataFrame(data)


class TestHandleMissingValues:
    """Test cases for handle_missing_values function."""

    def test_drop_strategy(self, sample_df):
        """Test drop strategy."""
        df = handle_missing_values(sample_df, strategy="drop")
        assert isinstance(df, pd.DataFrame)

    def test_fill_mean_strategy(self, sample_df):
        """Test fill_mean strategy."""
        df = handle_missing_values(sample_df, strategy="fill_mean")
        assert isinstance(df, pd.DataFrame)

    def test_fill_median_strategy(self, sample_df):
        """Test fill_median strategy."""
        df = handle_missing_values(sample_df, strategy="fill_median")
        assert isinstance(df, pd.DataFrame)

    def test_fill_mode_strategy(self, sample_df):
        """Test fill_mode strategy."""
        df = handle_missing_values(sample_df, strategy="fill_mode")
        assert isinstance(df, pd.DataFrame)


class TestRemoveDuplicates:
    """Test cases for remove_duplicates function."""

    def test_remove_duplicates(self, sample_df):
        """Test removing duplicates."""
        # Add duplicate row
        df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
        initial_rows = len(df)
        df = remove_duplicates(df)
        assert len(df) < initial_rows

    def test_no_duplicates(self, sample_df):
        """Test when no duplicates exist."""
        df = remove_duplicates(sample_df)
        assert len(df) == len(sample_df)


class TestRemoveInvalidRecords:
    """Test cases for remove_invalid_records function."""

    def test_remove_invalid_age(self, sample_df):
        """Test removing invalid age records."""
        df = remove_invalid_records(sample_df)
        # Age 105 should be removed
        assert df["Age"].max() <= 100

    def test_remove_negative_experience(self, sample_df):
        """Test removing negative experience records."""
        df = remove_invalid_records(sample_df)
        # Years_of_Experience -1 should be removed
        assert df["Years_of_Experience"].min() >= 0


class TestStandardizeColumnNames:
    """Test cases for standardize_column_names function."""

    def test_standardize_column_names(self, sample_df):
        """Test standardizing column names."""
        df = standardize_column_names(sample_df)
        for col in df.columns:
            assert col == col.lower()
            assert " " not in col


class TestSaveCleanDataset:
    """Test cases for save_clean_dataset function."""

    def test_save_clean_dataset(self, sample_df, tmp_path):
        """Test saving clean dataset."""
        output_path = str(tmp_path / "test_clean.csv")
        result = save_clean_dataset(sample_df, output_path)
        assert result is True
        assert os.path.exists(output_path)

    def test_save_empty_dataset(self, tmp_path):
        """Test saving empty dataset."""
        output_path = str(tmp_path / "test_empty.csv")
        result = save_clean_dataset(pd.DataFrame(), output_path)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
