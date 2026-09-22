# Month-2: Analysis Phase Documentation

## AI Impact on Jobs & Layoff Risk Analysis

**Author:** Sharma Saurabh (250863131007)
**Course:** BE05000231 - Python for Data Science (GTU Semester-5)
**Duration:** Month 2

---

## Overview

Month-2 focuses on data analysis including EDA, feature engineering, visualization, and preprocessing for machine learning.

---

## Objectives

- [x] Exploratory Data Analysis (EDA)
- [x] Data Visualization (Histograms, Heatmaps, Pie Charts)
- [x] Feature Engineering and Encoding
- [x] Data Preprocessing and Scaling

---

## Modules Implemented

### 1. src/exploratory_data_analysis.py
EDA functions:
- `basic_statistics()` - Statistical summary
- `distribution_analysis()` - Distribution of numerical columns
- `correlation_analysis()` - Correlation matrix
- `target_analysis()` - Target variable distribution
- `categorical_analysis()` - Categorical column analysis
- `group_analysis()` - Group-wise analysis
- `run_complete_eda()` - Complete EDA pipeline

### 2. src/feature_engineering.py
Feature engineering functions:
- `create_experience_categories()` - Experience categories (Junior/Mid/Senior/Expert)
- `create_age_groups()` - Age groups (18-25/26-35/36-45/46-55/56+)
- `create_ai_intensity_score()` - AI intensity score
- `create_routine_creativity_ratio()` - Routine-creativity ratio
- `create_human_ai_balance()` - Human-AI balance score
- `create_risk_factors()` - Risk factor flags
- `run_feature_engineering()` - Complete feature engineering pipeline

### 3. src/visualization.py
Visualization functions:
- `plot_target_distribution()` - Target variable distribution
- `plot_numerical_distributions()` - Numerical column distributions
- `plot_categorical_distributions()` - Categorical column distributions
- `plot_correlation_heatmap()` - Correlation heatmap
- `plot_risk_by_category()` - Risk by category plots
- `plot_boxplots()` - Boxplots by risk level
- `plot_ai_adoption_analysis()` - AI adoption analysis
- `plot_feature_importance_preview()` - Feature importance preview

### 4. src/preprocessing.py
Preprocessing functions:
- `encode_categorical_columns()` - Label encoding
- `scale_numerical_columns()` - Standard scaling
- `split_features_target()` - Features/target split
- `train_test_split_data()` - Train/test split
- `save_preprocessor()` - Save preprocessor objects
- `run_preprocessing()` - Complete preprocessing pipeline

---

## Feature Engineering Details

### New Features Created (9 total):

1. **experience_category** - Junior/Mid-Level/Senior/Expert
2. **age_group** - 18-25/26-35/36-45/46-55/56+
3. **ai_intensity_score** - Weighted score (0-100) based on AI usage, tools, and automation
4. **routine_creativity_ratio** - Ratio of routine tasks to creativity requirement
5. **human_ai_balance** - Balance score between human interaction and AI usage
6. **high_automation_risk** - Flag: tasks_automated_percentage > 70%
7. **low_creativity_risk** - Flag: creativity_requirement < 30%
8. **high_routine_risk** - Flag: routine_task_percentage > 70%
9. **low_interaction_risk** - Flag: human_interaction_level < 30%

---

## Preprocessing Steps

### 1. Encoding
- Label encoding for all categorical columns
- Target column (layoff_risk) encoded: Low=1, Medium=2, High=0

### 2. Scaling
- StandardScaler applied to all numerical columns
- Mean=0, Std=1 normalization

### 3. Train-Test Split
- 80% training (16,000 samples)
- 20% testing (4,000 samples)
- Stratified split to maintain class distribution

---

## Generated Charts

1. target_distribution.png
2. numerical_distributions.png
3. categorical_distributions.png
4. correlation_heatmap.png
5. risk_by_industry.png
6. risk_by_education_level.png
7. risk_by_company_size.png
8. risk_by_ai_adoption_level.png
9. boxplots_by_risk.png
10. ai_adoption_analysis.png
11. feature_importance_preview.png

---

## Output Files

- `outputs/processed_data/cleaned_dataset.csv`
- `outputs/processed_data/train_data.csv`
- `outputs/processed_data/test_data.csv`
- `outputs/charts/*.png` (11 charts)
- `outputs/models/scaler.pkl`
- `outputs/models/encoder.pkl`

---

## How to Run

```bash
python main.py
```

This will execute the complete pipeline including Month-2 analysis.
