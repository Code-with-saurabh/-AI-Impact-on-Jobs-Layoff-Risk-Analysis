"""
Test module for prediction functions.
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
from src.prediction import (
    load_model,
    load_preprocessor,
    predict_single,
    predict_batch,
)


@pytest.fixture
def trained_model():
    """Load trained model for testing."""
    return load_model(config.BEST_MODEL_PATH)


@pytest.fixture
def preprocessor():
    """Load preprocessor for testing."""
    return load_preprocessor(config.SCALER_PATH, config.ENCODER_PATH)


class TestLoadModel:
    """Test cases for load_model function."""

    def test_load_model(self):
        """Test loading model."""
        model = load_model(config.BEST_MODEL_PATH)
        assert model is not None

    def test_load_nonexistent_model(self):
        """Test loading non-existent model."""
        model = load_model("nonexistent_model.pkl")
        assert model is None


class TestLoadPreprocessor:
    """Test cases for load_preprocessor function."""

    def test_load_preprocessor(self):
        """Test loading preprocessor."""
        scaler, encoders = load_preprocessor(config.SCALER_PATH, config.ENCODER_PATH)
        assert scaler is not None
        assert isinstance(encoders, dict)

    def test_load_nonexistent_preprocessor(self):
        """Test loading non-existent preprocessor."""
        scaler, encoders = load_preprocessor("nonexistent.pkl", "nonexistent.pkl")
        assert scaler is None
        assert encoders == {}


class TestPredictSingle:
    """Test cases for predict_single function."""

    def test_predict_single(self, trained_model, preprocessor):
        """Test single prediction."""
        scaler, encoders = preprocessor
        feature_columns = [
            "age", "education_level", "years_of_experience", "industry",
            "job_role", "company_size", "job_level", "routine_task_percentage",
            "creativity_requirement", "human_interaction_level", "ai_adoption_level",
            "number_of_ai_tools_used", "ai_usage_hours_per_week",
            "tasks_automated_percentage", "ai_training_hours",
            "experience_category", "age_group", "ai_intensity_score",
            "routine_creativity_ratio", "human_ai_balance",
            "high_automation_risk", "low_creativity_risk",
            "high_routine_risk", "low_interaction_risk",
        ]
        input_data = {
            "age": 35,
            "education_level": 0,
            "years_of_experience": 10,
            "industry": 0,
            "job_role": 0,
            "company_size": 0,
            "job_level": 0,
            "routine_task_percentage": 40.0,
            "creativity_requirement": 70.0,
            "human_interaction_level": 50.0,
            "ai_adoption_level": 0,
            "number_of_ai_tools_used": 8,
            "ai_usage_hours_per_week": 25.0,
            "tasks_automated_percentage": 50.0,
            "ai_training_hours": 40,
            "experience_category": 0,
            "age_group": 0,
            "ai_intensity_score": 50.0,
            "routine_creativity_ratio": 0.5,
            "human_ai_balance": 60.0,
            "high_automation_risk": 0,
            "low_creativity_risk": 0,
            "high_routine_risk": 0,
            "low_interaction_risk": 0,
        }
        result = predict_single(trained_model, scaler, encoders, input_data, feature_columns)
        assert "prediction" in result
        assert "probabilities" in result


class TestPredictBatch:
    """Test cases for predict_batch function."""

    def test_predict_batch(self, trained_model, preprocessor):
        """Test batch predictions."""
        scaler, encoders = preprocessor
        feature_columns = [
            "age", "education_level", "years_of_experience", "industry",
            "job_role", "company_size", "job_level", "routine_task_percentage",
            "creativity_requirement", "human_interaction_level", "ai_adoption_level",
            "number_of_ai_tools_used", "ai_usage_hours_per_week",
            "tasks_automated_percentage", "ai_training_hours",
            "experience_category", "age_group", "ai_intensity_score",
            "routine_creativity_ratio", "human_ai_balance",
            "high_automation_risk", "low_creativity_risk",
            "high_routine_risk", "low_interaction_risk",
        ]
        df = pd.DataFrame({
            "age": [35, 40, 45],
            "education_level": [0, 1, 2],
            "years_of_experience": [10, 15, 20],
            "industry": [0, 1, 2],
            "job_role": [0, 1, 2],
            "company_size": [0, 1, 2],
            "job_level": [0, 1, 2],
            "routine_task_percentage": [40.0, 50.0, 60.0],
            "creativity_requirement": [70.0, 60.0, 50.0],
            "human_interaction_level": [50.0, 60.0, 70.0],
            "ai_adoption_level": [0, 1, 2],
            "number_of_ai_tools_used": [8, 5, 3],
            "ai_usage_hours_per_week": [25.0, 15.0, 10.0],
            "tasks_automated_percentage": [50.0, 40.0, 30.0],
            "ai_training_hours": [40, 30, 20],
            "experience_category": [0, 1, 2],
            "age_group": [0, 1, 2],
            "ai_intensity_score": [50.0, 40.0, 30.0],
            "routine_creativity_ratio": [0.5, 0.8, 1.2],
            "human_ai_balance": [60.0, 50.0, 40.0],
            "high_automation_risk": [0, 0, 1],
            "low_creativity_risk": [0, 0, 1],
            "high_routine_risk": [0, 1, 1],
            "low_interaction_risk": [0, 0, 1],
        })
        result_df = predict_batch(trained_model, scaler, encoders, df, feature_columns)
        assert isinstance(result_df, pd.DataFrame)
        assert "predicted_risk" in result_df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
