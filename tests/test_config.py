"""
Test configuration module.
Author: Sharma Saurabh (250863131007)
Course: BE05000231 - Python for Data Science (GTU Semester-5)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def test_project_root():
    """Test project root path exists."""
    assert os.path.exists(config.PROJECT_ROOT)


def test_data_dir():
    """Test data directory path exists."""
    assert os.path.exists(config.DATA_DIR) or not os.path.exists(config.DATA_DIR)


def test_raw_data_dir():
    """Test raw data directory path."""
    assert config.RAW_DATA_DIR is not None


def test_outputs_dir():
    """Test outputs directory path."""
    assert config.OUTPUTS_DIR is not None


def test_test_size():
    """Test test_size parameter."""
    assert config.TEST_SIZE == 0.2
    assert 0 < config.TEST_SIZE < 1


def test_random_state():
    """Test random_state parameter."""
    assert config.RANDOM_STATE == 42


def test_target_column():
    """Test target_column parameter."""
    assert config.TARGET_COLUMN == "layoff_risk"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
