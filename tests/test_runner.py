"""
Main test runner for AI Impact on Jobs & Layoff Risk Analysis.
Author: Sharma Saurabh (250863131007)
Course: BE05000231 - Python for Data Science (GTU Semester-5)

Usage:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short
    python -m pytest tests/test_data_loading.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])
