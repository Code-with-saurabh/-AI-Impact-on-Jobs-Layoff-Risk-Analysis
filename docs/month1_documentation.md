# Month-1 Documentation: Foundation Phase

## Project: AI Impact on Jobs & Layoff Risk Analysis using Machine Learning

**Student ID:** BE05000231
**Course:** Python for Data Science (GTU Semester-5)
**Phase:** Month-1 - Foundation Phase
**Date:** August 2026

---

## 1. Phase Objectives

The Month-1 Foundation Phase focuses on:

1. Establishing a professional project structure
2. Configuring project settings and paths
3. Loading and validating the dataset
4. Inspecting dataset structure and quality
5. Performing basic data cleaning
6. Saving the cleaned dataset for future analysis

---

## 2. Project Structure

### 2.1 Directory Layout

```
AI-Impact-Jobs-Layoff-Risk/
│
├── main.py                   # Application entry point
├── config.py                 # Path configurations
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── data/
│   ├── raw/                  # Original dataset
│   │   └── ai-impact-jobs-layoff-risk-dataset.csv
│   └── processed/            # Processed data (future use)
│
├── src/
│   ├── __init__.py           # Package marker
│   ├── data_loading.py       # Loading functions
│   └── data_cleaning.py      # Cleaning functions
│
├── outputs/
│   └── processed_data/       # Output files
│
├── docs/                     # Documentation
│   └── month1_documentation.md
│
├── notebooks/                # Jupyter notebooks (future)
└── tests/                    # Unit tests (future)
```

### 2.2 File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Main entry point that orchestrates the entire pipeline |
| `config.py` | Centralized path configuration (no business logic) |
| `requirements.txt` | Lists required Python packages |
| `src/data_loading.py` | Functions for loading and inspecting data |
| `src/data_cleaning.py` | Functions for cleaning and preprocessing data |

---

## 3. Module Descriptions

### 3.1 config.py

**Purpose:** Centralize all project paths to avoid hardcoded paths throughout the codebase.

**Configurations:**
- `PROJECT_ROOT`: Root directory of the project
- `DATA_DIR`: Main data folder path
- `RAW_DATA_DIR`: Path to raw data folder
- `PROCESSED_DATA_DIR`: Path to processed data folder
- `RAW_DATASET_PATH`: Full path to the raw CSV file
- `OUTPUTS_DIR`: Main outputs folder
- `CLEANED_DATASET_PATH`: Path where cleaned dataset is saved

### 3.2 src/data_loading.py

**Purpose:** Provide robust data loading and inspection capabilities.

**Functions:**

| Function | Description |
|----------|-------------|
| `check_file_exists(file_path)` | Validates file existence before loading |
| `load_dataset(file_path)` | Loads CSV into pandas DataFrame with error handling |
| `inspect_dataset(df)` | Displays first/last rows and basic statistics |
| `display_shape(df)` | Shows dataset dimensions (rows x columns) |
| `display_columns(df)` | Lists all column names |
| `display_data_types(df)` | Shows data type of each column |
| `display_missing_values(df)` | Analyzes missing values per column |

### 3.3 src/data_cleaning.py

**Purpose:** Clean and preprocess the dataset for analysis.

**Functions:**

| Function | Description |
|----------|-------------|
| `handle_missing_values(df, strategy)` | Handles missing values using specified strategy |
| `remove_duplicates(df)` | Removes exact duplicate rows |
| `remove_invalid_records(df)` | Filters out invalid/inconsistent records |
| `standardize_column_names(df)` | Normalizes column names to lowercase with underscores |
| `save_clean_dataset(df, output_path)` | Saves cleaned DataFrame to CSV file |

---

## 4. Data Pipeline

### 4.1 Pipeline Flow

```
Raw CSV File
    ↓
[Phase 1: Loading]
    ↓
DataFrame
    ↓
[Phase 2: Inspection]
    ↓
Dataset Information Displayed
    ↓
[Phase 3: Cleaning]
    ↓
Cleaned DataFrame
    ↓
[Phase 4: Save]
    ↓
Cleaned CSV File
```

### 4.2 Execution Steps

1. **Load Dataset**
   - Check if file exists
   - Read CSV using pandas
   - Return DataFrame

2. **Inspect Dataset**
   - Display shape (rows, columns)
   - List column names
   - Show data types
   - Analyze missing values
   - Show basic statistics

3. **Clean Dataset**
   - Handle missing values (drop rows)
   - Remove duplicate rows
   - Remove invalid records
   - Standardize column names

4. **Save Cleaned Dataset**
   - Create output directory if needed
   - Save to CSV format
   - Display file information

---

## 5. Dataset Overview

### 5.1 Source

- **File:** ai-impact-jobs-layoff-risk-dataset.csv
- **Rows:** 20,000
- **Columns:** 16

### 5.2 Column Summary

| Column | Type | Description |
|--------|------|-------------|
| Age | int | Employee age |
| Education_Level | object | Education level |
| Years_of_Experience | int | Years of experience |
| Industry | object | Industry sector |
| Job_Role | object | Job title |
| Company_Size | object | Company size category |
| Job_Level | object | Seniority level |
| Routine_Task_Percentage | float | Routine task percentage |
| Creativity_Requirement | float | Creativity requirement score |
| Human_Interaction_Level | float | Human interaction level |
| AI_Adoption_Level | object | AI adoption level |
| Number_of_AI_Tools_Used | int | Number of AI tools used |
| AI_Usage_Hours_Per_Week | float | Weekly AI usage hours |
| Tasks_Automated_Percentage | float | Automatable task percentage |
| AI_Training_Hours | float | AI training hours |
| Layoff_Risk | object | Layoff risk level (Target) |

---

## 6. Data Cleaning Details

### 6.1 Missing Values Strategy

- **Method:** Drop rows with any missing values
- **Rationale:** Ensures complete data for analysis
- **Impact:** Removes incomplete records

### 6.2 Duplicate Removal

- **Method:** Drop exact duplicate rows
- **Rationale:** Ensures data integrity
- **Impact:** Removes redundant records

### 6.3 Invalid Record Removal

**Rules Applied:**
- Age must be between 0 and 100
- Years of Experience must be non-negative
- Routine Task Percentage must be between 0 and 100
- No empty string values in categorical columns

### 6.4 Column Name Standardization

**Transformation Rules:**
- Convert to lowercase
- Replace spaces with underscores
- Remove special characters
- Remove multiple consecutive underscores
- Strip leading/trailing underscores

---

## 7. Error Handling

### 7.1 File Operations

- FileNotFoundError handling for missing files
- Permission error handling for access issues
- Empty file detection

### 7.2 Data Operations

- Empty DataFrame checks
- Type validation
- NaN handling

### 7.3 General

- Try-except blocks around all major operations
- Informative error messages
- Graceful failure with exit codes

---

## 8. Testing

### 8.1 Current Status

- Empty `tests/` folder created
- No unit tests implemented yet (planned for future phases)

### 8.2 Future Testing Plan

- Unit tests for data loading functions
- Unit tests for data cleaning functions
- Integration tests for the complete pipeline

---

## 9. Output Files

### 9.1 Generated Files

| File | Location | Description |
|------|----------|-------------|
| cleaned_dataset.csv | outputs/processed_data/ | Cleaned dataset ready for analysis |

### 9.2 File Format

- Format: CSV (Comma-Separated Values)
- Encoding: UTF-8
- Index: Not included (index=False)

---

## 10. Dependencies

### 10.1 Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=1.5.0 | Data manipulation and analysis |
| numpy | >=1.23.0 | Numerical operations |

### 10.2 Installation

```bash
pip install -r requirements.txt
```

---

## 11. Month-1 Completion Checklist

- [x] Project structure created
- [x] Configuration module implemented
- [x] Requirements file created
- [x] Data loading module implemented
- [x] Data cleaning module implemented
- [x] Main entry point created
- [x] README documentation written
- [x] Month-1 documentation written
- [x] Dataset copied to raw data folder
- [x] Cleaned dataset saved to outputs folder

---

## 12. Next Phase (Month-2 Preview)

### 12.1 Planned Features

- Exploratory Data Analysis (EDA)
- Data Visualization (Histograms, Heatmaps, Pie Charts)
- Feature Engineering
- Data Preprocessing and Encoding

### 12.2 New Modules

- `src/exploratory_data_analysis.py`
- `src/feature_engineering.py`
- `src/visualization.py`
- `src/preprocessing.py`

---

## 13. Conclusion

The Month-1 Foundation Phase has been successfully completed with:

- Professional project structure
- Robust data loading capabilities
- Comprehensive data cleaning pipeline
- Complete documentation

The project is now ready for the Month-2 Analysis Phase.

---

**Document Version:** 1.0
**Last Updated:** August 2026
