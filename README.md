# AI Impact on Jobs & Layoff Risk Analysis using Machine Learning

## Project Overview
This project analyzes the impact of Artificial Intelligence on employment and assesses layoff risk factors using Machine Learning techniques. It is developed as part of the GTU Semester-5 course **BE05000231 - Python for Data Science**.

**Duration:** 3 Months (Currently in Month-1 Foundation Phase)

---

## Problem Statement
The rapid advancement of AI technology is transforming the job market, creating both opportunities and challenges. Understanding which jobs are at risk of being automated and identifying factors that contribute to layoff risk is crucial for workforce planning, career development, and policy making.

This project aims to:
- Analyze job characteristics and AI adoption patterns
- Identify key factors contributing to layoff risk
- Build predictive models (in future phases) to assess job vulnerability to AI automation

---

## Objectives

### Month-1 (Foundation Phase) - Current
- [x] Create professional project structure
- [x] Configure project settings and paths
- [x] Load and validate dataset
- [x] Inspect dataset structure and quality
- [x] Perform basic data cleaning
- [x] Save cleaned dataset for future analysis

### Month-2 (Analysis Phase) - Upcoming
- [ ] Exploratory Data Analysis (EDA)
- [ ] Data Visualization (Histograms, Heatmaps, Pie Charts)
- [ ] Feature Engineering and Encoding
- [ ] Data Preprocessing and Scaling

### Month-3 (Modeling Phase) - Upcoming
- [ ] Machine Learning Model Training
- [ ] Model Evaluation and Comparison
- [ ] Prediction System Development
- [ ] Final Documentation and Reports

---

## Dataset Description
**File:** `ai-impact-jobs-layoff-risk-dataset.csv`

| Column | Description | Type |
|--------|-------------|------|
| Age | Age of the employee | Numeric |
| Education_Level | Highest education attained | Categorical |
| Years_of_Experience | Professional experience in years | Numeric |
| Industry | Industry of employment | Categorical |
| Job_Role | Specific job title/role | Categorical |
| Company_Size | Size of the company (Small/Medium/Large) | Categorical |
| Job_Level | Seniority level (Entry/Mid/Senior) | Categorical |
| Routine_Task_Percentage | Percentage of routine tasks in job | Numeric |
| Creativity_Requirement | Level of creativity needed | Numeric |
| Human_Interaction_Level | Level of human interaction required | Numeric |
| AI_Adoption_Level | Current AI adoption in the role | Categorical |
| Number_of_AI_Tools_Used | Count of AI tools being used | Numeric |
| AI_Usage_Hours_Per_Week | Weekly AI usage hours | Numeric |
| Tasks_Automated_Percentage | Percentage of tasks that can be automated | Numeric |
| AI_Training_Hours | Hours spent on AI training | Numeric |
| Layoff_Risk | Risk level of being laid off (Target) | Categorical |

**Dataset Size:** 20,000 rows x 16 columns

---

## Folder Structure
```
AI-Impact-Jobs-Layoff-Risk/
│   README.md                 # Project documentation
│   requirements.txt          # Python dependencies
│   main.py                   # Main entry point
│   config.py                 # Configuration settings
├── data/
│   ├── raw/                  # Original, unmodified data
│   │   └── ai-impact-jobs-layoff-risk-dataset.csv
│   └── processed/            # Cleaned and processed data
├── src/                      # Source code modules
│   ├── __init__.py           # Package initializer
│   ├── data_loading.py       # Data loading functions
│   ├── data_cleaning.py      # Data cleaning functions
├── outputs/
│   ├── processed_data/       # Saved cleaned datasets
├── notebooks/                # Jupyter notebooks (future use)
├── docs/                     # Project documentation
│   └── month1_documentation.md
└── tests/                    # Unit tests (future use)
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd AI-Impact-Jobs-Layoff-Risk
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify dataset location**
   Ensure the CSV file is in `data/raw/` folder:
   ```
   data/raw/ai-impact-jobs-layoff-risk-dataset.csv
   ```

4. **Run the project**
   ```bash
   python main.py
   ```

---

## Usage

### Running Month-1 Pipeline
```bash
python main.py
```

This will:
1. Load the raw dataset
2. Display dataset information (shape, columns, data types, missing values)
3. Clean the dataset (handle missing values, remove duplicates, standardize columns)
4. Save the cleaned dataset to `outputs/processed_data/cleaned_dataset.csv`

### Output
After running, you will find:
- `outputs/processed_data/cleaned_dataset.csv` - The cleaned dataset ready for analysis

---

## Code Quality
- **PEP-8 Compliance:** Follows Python coding standards
- **Modular Design:** Reusable functions and clear separation of concerns
- **Error Handling:** Comprehensive exception handling throughout
- **Documentation:** Professional docstrings and inline comments
- **Configuration:** No hardcoded paths, centralized settings

---

## Author
**Student Name:** Sharma Saurabh
**Student ID:** 250863131007
**Course:** BE05000231 - Python for Data Science (GTU Semester-5)
**Project Duration:** 3 Months

---

## License
This project is developed for educational purposes as part of GTU Semester-5 coursework.
