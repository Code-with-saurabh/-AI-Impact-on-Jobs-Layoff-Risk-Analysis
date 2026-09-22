# Month-1: Foundation Phase Documentation

## AI Impact on Jobs & Layoff Risk Analysis

**Author:** Sharma Saurabh (250863131007)
**Course:** BE05000231 - Python for Data Science (GTU Semester-5)
**Duration:** Month 1

---

## Overview

Month-1 focuses on establishing a solid project foundation including project structure, data loading, inspection, and cleaning.

---

## Objectives

- [x] Create professional project structure
- [x] Configure project settings and paths
- [x] Load and validate dataset
- [x] Inspect dataset structure and quality
- [x] Perform basic data cleaning
- [x] Save cleaned dataset for future analysis

---

## Project Structure Created

```
AI-Impact-Jobs-Layoff-Risk/
├── README.md
├── requirements.txt
├── main.py
├── config.py
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── data_loading.py
│   └── data_cleaning.py
├── outputs/
│   └── processed_data/
├── notebooks/
├── docs/
└── tests/
```

---

## Modules Implemented

### 1. config.py
Centralized configuration module containing:
- Project root path
- Data folder paths (raw, processed)
- Output folder paths (charts, models, reports, predictions)
- File paths for all outputs
- Model parameters (TEST_SIZE, RANDOM_STATE, TARGET_COLUMN)

### 2. src/data_loading.py
Functions for data loading and inspection:
- `check_file_exists()` - Validates file existence
- `load_dataset()` - Loads CSV into DataFrame
- `inspect_dataset()` - Displays dataset info
- `display_shape()` - Shows rows x columns
- `display_columns()` - Lists all columns
- `display_data_types()` - Shows column data types
- `display_missing_values()` - Analyzes missing data

### 3. src/data_cleaning.py
Functions for data cleaning:
- `handle_missing_values()` - Handles missing data with multiple strategies
- `remove_duplicates()` - Removes duplicate rows
- `remove_invalid_records()` - Removes invalid records
- `standardize_column_names()` - Standardizes column names
- `save_clean_dataset()` - Saves cleaned data

---

## Data Cleaning Process

### Step 1: Handle Missing Values
- Strategy: Drop rows with missing values
- Threshold: Drop columns with >50% missing

### Step 2: Remove Duplicates
- Identify and remove exact duplicate rows

### Step 3: Remove Invalid Records
- Age must be 0-100
- Years_of_Experience must be non-negative
- Routine_Task_Percentage must be 0-100
- No empty strings in categorical columns

### Step 4: Standardize Column Names
- Convert to lowercase
- Replace spaces with underscores
- Remove special characters

---

## Output Files

- `outputs/processed_data/cleaned_dataset.csv` - Cleaned dataset

---

## How to Run

```bash
python main.py
```

This will execute the complete Month-1 pipeline:
1. Load raw dataset
2. Display dataset information
3. Clean the dataset
4. Save cleaned dataset
