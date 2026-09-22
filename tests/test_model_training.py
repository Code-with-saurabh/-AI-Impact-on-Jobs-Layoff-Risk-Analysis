"""
Test module for model training functions.
Author: Sharma Saurabh (250863131007)
Course: BE05000231 - Python for Data Science (GTU Semester-5)
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_training import (
    initialize_models,
    train_model,
    train_all_models,
    compare_models,
)


@pytest.fixture
def sample_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.randn(n),
    })
    y = pd.Series(np.random.choice([0, 1, 2], n))
    return X, y


class TestInitializeModels:
    """Test cases for initialize_models function."""

    def test_initialize_models(self):
        """Test initializing models."""
        models = initialize_models()
        assert isinstance(models, dict)
        assert len(models) == 3

    def test_model_names(self):
        """Test model names."""
        models = initialize_models()
        expected_names = ["Logistic Regression", "Decision Tree", "Random Forest"]
        for name in expected_names:
            assert name in models

    def test_model_types(self):
        """Test model types."""
        models = initialize_models()
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(models["Logistic Regression"], LogisticRegression)
        assert isinstance(models["Decision Tree"], DecisionTreeClassifier)
        assert isinstance(models["Random Forest"], RandomForestClassifier)


class TestTrainModel:
    """Test cases for train_model function."""

    def test_train_model(self, sample_data):
        """Test training single model."""
        X, y = sample_data
        models = initialize_models()
        model = models["Logistic Regression"]
        trained_model, train_time = train_model(model, X, y, "Logistic Regression")
        assert trained_model is not None
        assert train_time > 0

    def test_trained_model_can_predict(self, sample_data):
        """Test that trained model can make predictions."""
        X, y = sample_data
        models = initialize_models()
        model = models["Logistic Regression"]
        trained_model, _ = train_model(model, X, y, "Logistic Regression")
        predictions = trained_model.predict(X)
        assert len(predictions) == len(X)


class TestTrainAllModels:
    """Test cases for train_all_models function."""

    def test_train_all_models(self, sample_data):
        """Test training all models."""
        X, y = sample_data
        models = initialize_models()
        trained_models = train_all_models(models, X, y)
        assert isinstance(trained_models, dict)
        assert len(trained_models) == 3

    def test_trained_models_metadata(self, sample_data):
        """Test trained models have metadata."""
        X, y = sample_data
        models = initialize_models()
        trained_models = train_all_models(models, X, y)
        for name, metadata in trained_models.items():
            assert "model" in metadata
            assert "training_time" in metadata
            assert "train_accuracy" in metadata
            assert metadata["train_accuracy"] >= 0
            assert metadata["train_accuracy"] <= 1


class TestCompareModels:
    """Test cases for compare_models function."""

    def test_compare_models(self, sample_data):
        """Test comparing models."""
        X, y = sample_data
        models = initialize_models()
        trained_models = train_all_models(models, X, y)
        comparison_df = compare_models(trained_models, X, y)
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
