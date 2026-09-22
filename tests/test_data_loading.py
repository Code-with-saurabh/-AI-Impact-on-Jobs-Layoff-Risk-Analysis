"""
Test module for data loading functions.
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
from src.data_loading import (
    check_file_exists,
    load_dataset,
    inspect_dataset,
    display_shape,
    display_columns,
    display_data_types,
    display_missing_values,
)


class TestCheckFileExists:
    """Test cases for check_file_exists function."""

    def test_existing_file(self):
        """Test with existing file."""
        result = check_file_exists(config.RAW_DATASET_PATH)
        assert result is True

    def test_nonexistent_file(self):
        """Test with non-existent file."""
        result = check_file_exists("nonexistent_file.csv")
        assert result is False


class TestLoadDataset:
    """Test cases for load_dataset function."""

    def test_load_dataset(self):
        """Test loading dataset."""
        df = load_dataset(config.RAW_DATASET_PATH)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0

    def test_dataset_shape(self):
        """Test dataset shape."""
        df = load_dataset(config.RAW_DATASET_PATH)
        assert df.shape[0] == 20000
        assert df.shape[1] == 16


class TestDisplayShape:
    """Test cases for display_shape function."""

    def test_display_shape(self):
        """Test display_shape with valid DataFrame."""
        df = load_dataset(config.RAW_DATASET_PATH)
        rows, cols = display_shape(df)
        assert rows == 20000
        assert cols == 16

    def test_display_shape_empty(self):
        """Test display_shape with empty DataFrame."""
        rows, cols = display_shape(pd.DataFrame())
        assert rows == 0
        assert cols == 0

    def test_display_shape_none(self):
        """Test display_shape with None."""
        rows, cols = display_shape(None)
        assert rows == 0
        assert cols == 0


class TestDisplayColumns:
    """Test cases for display_columns function."""

    def test_display_columns(self):
        """Test display_columns with valid DataFrame."""
        df = load_dataset(config.RAW_DATASET_PATH)
        columns = display_columns(df)
        assert isinstance(columns, list)
        assert len(columns) == 16
        assert "Age" in columns
        assert "Layoff_Risk" in columns

    def test_display_columns_empty(self):
        """Test display_columns with empty DataFrame."""
        columns = display_columns(pd.DataFrame())
        assert columns == []


class TestDisplayDataTypes:
    """Test cases for display_data_types function."""

    def test_display_data_types(self):
        """Test display_data_types with valid DataFrame."""
        df = load_dataset(config.RAW_DATASET_PATH)
        dtypes = display_data_types(df)
        assert dtypes is not None
        assert len(dtypes) == 16

    def test_display_data_types_empty(self):
        """Test display_data_types with empty DataFrame."""
        dtypes = display_data_types(pd.DataFrame())
        assert dtypes is None


class TestDisplayMissingValues:
    """Test cases for display_missing_values function."""

    def test_display_missing_values(self):
        """Test display_missing_values with valid DataFrame."""
        df = load_dataset(config.RAW_DATASET_PATH)
        missing = display_missing_values(df)
        assert missing is not None
        assert missing.sum() == 0  # No missing values in our dataset

    def test_display_missing_values_empty(self):
        """Test display_missing_values with empty DataFrame."""
        missing = display_missing_values(pd.DataFrame())
        assert missing is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
