# Month-3: Modeling Phase Documentation

## AI Impact on Jobs & Layoff Risk Analysis

**Author:** Sharma Saurabh (250863131007)
**Course:** BE05000231 - Python for Data Science (GTU Semester-5)
**Duration:** Month 3

---

## Overview

Month-3 focuses on machine learning model training, evaluation, and prediction system development.

---

## Objectives

- [x] Machine Learning Model Training
- [x] Model Evaluation and Comparison
- [x] Prediction System Development
- [x] Final Documentation and Reports

---

## Modules Implemented

### 1. src/model_training.py
Model training functions:
- `initialize_models()` - Initialize ML models
- `train_model()` - Train single model
- `train_all_models()` - Train all models
- `save_best_model()` - Save best performing model
- `compare_models()` - Compare model performance
- `run_model_training()` - Complete training pipeline

### 2. src/model_evaluation.py
Model evaluation functions:
- `calculate_metrics()` - Calculate classification metrics
- `print_classification_report()` - Print detailed report
- `plot_confusion_matrix()` - Plot confusion matrix
- `plot_roc_curve()` - Plot ROC curve
- `evaluate_model()` - Complete model evaluation
- `evaluate_all_models()` - Evaluate all models
- `save_evaluation_report()` - Save report to file

### 3. src/prediction.py
Prediction functions:
- `load_model()` - Load saved model
- `load_preprocessor()` - Load scaler and encoders
- `predict_single()` - Single prediction
- `predict_batch()` - Batch predictions
- `save_predictions()` - Save predictions to CSV
- `run_predictions()` - Complete prediction pipeline

---

## Models Trained

### 1. Logistic Regression
- Type: Linear classifier
- Parameters: max_iter=1000, random_state=42
- Training Time: ~0.16 seconds

### 2. Decision Tree
- Type: Tree-based classifier
- Parameters: max_depth=10, random_state=42
- Training Time: ~0.22 seconds

### 3. Random Forest
- Type: Ensemble classifier
- Parameters: n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
- Training Time: ~1.51 seconds

---

## Evaluation Metrics

### Classification Metrics:
- **Accuracy** - Overall correctness
- **Precision** - Exactness of positive predictions
- **Recall** - Completeness of positive predictions
- **F1-Score** - Harmonic mean of precision and recall

### Visualizations:
- **Confusion Matrix** - Heatmap of predictions vs actual
- **ROC Curve** - Receiver Operating Characteristic curve
- **Classification Report** - Detailed per-class metrics

---

## Model Comparison Results

| Model | Train Accuracy | Test Accuracy | Training Time |
|-------|---------------|---------------|---------------|
| Logistic Regression | ~0.40 | ~0.40 | ~0.16s |
| Decision Tree | ~0.49 | ~0.38 | ~0.22s |
| Random Forest | ~0.58 | ~0.39 | ~1.51s |

*Note: Accuracy may vary based on dataset*

---

## Prediction System

### Single Prediction:
```python
from src.prediction import load_model, load_preprocessor, predict_single

model = load_model('outputs/models/best_model.pkl')
scaler, encoders = load_preprocessor('outputs/models/scaler.pkl', 'outputs/models/encoder.pkl')

input_data = {
    'age': 35,
    'education_level': 'Bachelor',
    'years_of_experience': 10,
    'industry': 'Technology',
    'job_role': 'Software Engineer',
    'company_size': 'Large',
    'job_level': 'Mid',
    'routine_task_percentage': 40.0,
    'creativity_requirement': 70.0,
    'human_interaction_level': 50.0,
    'ai_adoption_level': 'High',
    'number_of_ai_tools_used': 8,
    'ai_usage_hours_per_week': 25.0,
    'tasks_automated_percentage': 50.0,
    'ai_training_hours': 40
}

result = predict_single(model, scaler, encoders, input_data, feature_columns)
print(result)
```

---

## Output Files

- `outputs/models/best_model.pkl` - Best trained model
- `outputs/models/scaler.pkl` - Fitted scaler
- `outputs/models/encoder.pkl` - Fitted encoders
- `outputs/reports/evaluation_report.txt` - Evaluation report
- `outputs/predictions/predictions.csv` - Test predictions
- `outputs/charts/confusion_matrix_*.png` - Confusion matrices
- `outputs/charts/roc_curve_*.png` - ROC curves

---

## How to Run

```bash
python main.py
```

This will execute the complete pipeline including Month-3 modeling.
