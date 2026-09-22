"""
Test module for feature engineering functions.
Author: Sharma Saurabh (250863131007)
Course: BE05000231 - Python for Data Science (GTU Semester-5)
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import (
    create_experience_categories,
    create_age_groups,
    create_ai_intensity_score,
    create_routine_creativity_ratio,
    create_human_ai_balance,
    create_risk_factors,
    run_feature_engineering,
)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age": np.random.randint(22, 60, n),
        "years_of_experience": np.random.randint(0, 35, n),
        "routine_task_percentage": np.random.uniform(10, 95, n),
        "creativity_requirement": np.random.uniform(5, 95, n),
        "human_interaction_level": np.random.uniform(10, 90, n),
        "ai_adoption_level": np.random.choice(["Low", "Medium", "High"], n),
        "number_of_ai_tools_used": np.random.randint(0, 15, n),
        "ai_usage_hours_per_week": np.random.uniform(0, 40, n),
        "tasks_automated_percentage": np.random.uniform(5, 85, n),
    })


class TestCreateExperienceCategories:
    """Test cases for create_experience_categories function."""

    def test_create_experience_categories(self, sample_df):
        """Test creating experience categories."""
        df = create_experience_categories(sample_df)
        assert "experience_category" in df.columns
        assert df["experience_category"].nunique() == 4

    def test_experience_categories_values(self, sample_df):
        """Test experience category values."""
        df = create_experience_categories(sample_df)
        expected_values = ["Junior", "Mid-Level", "Senior", "Expert"]
        actual_values = df["experience_category"].unique().tolist()
        for val in expected_values:
            assert val in actual_values


class TestCreateAgeGroups:
    """Test cases for create_age_groups function."""

    def test_create_age_groups(self, sample_df):
        """Test creating age groups."""
        df = create_age_groups(sample_df)
        assert "age_group" in df.columns
        assert df["age_group"].nunique() == 5

    def test_age_groups_values(self, sample_df):
        """Test age group values."""
        df = create_age_groups(sample_df)
        expected_values = ["18-25", "26-35", "36-45", "46-55", "56+"]
        actual_values = df["age_group"].unique().tolist()
        for val in expected_values:
            assert val in actual_values


class TestCreateAiIntensityScore:
    """Test cases for create_ai_intensity_score function."""

    def test_create_ai_intensity_score(self, sample_df):
        """Test creating AI intensity score."""
        df = create_ai_intensity_score(sample_df)
        assert "ai_intensity_score" in df.columns
        assert df["ai_intensity_score"].min() >= 0
        assert df["ai_intensity_score"].max() <= 100


class TestCreateRoutineCreativityRatio:
    """Test cases for create_routine_creativity_ratio function."""

    def test_create_routine_creativity_ratio(self, sample_df):
        """Test creating routine-creativity ratio."""
        df = create_routine_creativity_ratio(sample_df)
        assert "routine_creativity_ratio" in df.columns
        assert df["routine_creativity_ratio"].min() >= 0


class TestCreateHumanAiBalance:
    """Test cases for create_human_ai_balance function."""

    def test_create_human_ai_balance(self, sample_df):
        """Test creating human-AI balance score."""
        df = create_human_ai_balance(sample_df)
        assert "human_ai_balance" in df.columns
        assert df["human_ai_balance"].min() >= 0
        assert df["human_ai_balance"].max() <= 100


class TestCreateRiskFactors:
    """Test cases for create_risk_factors function."""

    def test_create_risk_factors(self, sample_df):
        """Test creating risk factor flags."""
        df = create_risk_factors(sample_df)
        assert "high_automation_risk" in df.columns
        assert "low_creativity_risk" in df.columns
        assert "high_routine_risk" in df.columns
        assert "low_interaction_risk" in df.columns

    def test_risk_factors_binary(self, sample_df):
        """Test that risk factors are binary (0 or 1)."""
        df = create_risk_factors(sample_df)
        for col in ["high_automation_risk", "low_creativity_risk",
                     "high_routine_risk", "low_interaction_risk"]:
            assert df[col].isin([0, 1]).all()


class TestRunFeatureEngineering:
    """Test cases for run_feature_engineering function."""

    def test_run_feature_engineering(self, sample_df):
        """Test complete feature engineering pipeline."""
        df = run_feature_engineering(sample_df)
        assert len(df.columns) > len(sample_df.columns)
        assert "experience_category" in df.columns
        assert "age_group" in df.columns
        assert "ai_intensity_score" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
