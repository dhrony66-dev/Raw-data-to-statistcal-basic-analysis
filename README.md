# Automated Research Data Analysis Pipeline

A Python-based automated statistical analysis pipeline designed to transform raw research data into structured, analysis-ready results. The system combines data cleaning, validation, exploratory data analysis, assumption checking, and decision-based statistical test selection into a single workflow.

## 🔄 Analysis Pipeline

```text
                    RAW DATASET
                         ↓
              ┌─────────────────────┐
              │ PHASE 1             │
              │ DATA CLEANING       │
              └─────────────────────┘
                         ↓
          Duplicate | Missing | Space
                         ↓
                     Outliers
                         ↓
                  CLEAN DATASET
                         ↓
              ┌─────────────────────┐
              │ PHASE 2             │
              │ DATA VALIDATION     │
              └─────────────────────┘
                         ↓
          Data Type | Missing | Duplicate
                         ↓
              Range | Constant | Infinite
                         ↓
                 VALIDATED DATA
                         ↓
              ┌─────────────────────┐
              │ PHASE 3             │
              │ AUTOMATED EDA       │
              └─────────────────────┘
                         ↓
       Descriptive Statistics | Distribution
                         ↓
          Normality | Correlation | VIF
                         ↓
              ┌─────────────────────┐
              │ PHASE 4             │
              │ STATISTICAL         │
              │ DECISION ENGINE     │
              └─────────────────────┘
                         ↓
                USER SETS TARGET
                         ↓
                 TARGET VARIABLE
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
         NUMERICAL              CATEGORICAL
              ↓                     ↓
        CONTINUOUS             BINARY / MULTI
              ↓                     ↓
       Predictor Type          Appropriate
              ↓                Statistical Test
      ┌───────┴────────┐
      ↓                ↓
 Numerical        Categorical
 Predictor        Predictor
      ↓                ↓
Pearson/          2 Groups / >2 Groups
Spearman               ↓
                  t-test / ANOVA
                  Mann-Whitney /
                  Kruskal-Wallis
                         ↓
              Statistical Results
                         ↓
                Automated Reports
```

## Phase 1: Data Cleaning

The pipeline automatically prepares raw data by:

* Removing duplicate observations
* Handling missing values
* Using mean imputation for numerical variables
* Using mode imputation for categorical variables
* Removing unnecessary spaces from text values
* Detecting and treating outliers using the IQR method

## Phase 2: Data Validation

The cleaned dataset is validated through:

* Dataset dimension checks
* Data type identification
* Missing value verification
* Duplicate verification
* Constant variable detection
* Infinite value detection
* Overall validation status

## Phase 3: Automated EDA

The system automatically performs exploratory analysis including:

* Descriptive statistics
* Frequency distributions
* Histograms
* Boxplots
* Q-Q plots
* Normality analysis
* Correlation analysis
* Multicollinearity assessment using VIF

## Phase 4: Decision-Based Statistical Analysis

The key feature of this project is that it does not blindly apply the same statistical test to every dataset.

The user defines the target variable, and the system considers:

* Target variable type
* Predictor variable type
* Number of groups
* Distributional characteristics
* Statistical assumptions
* Sample characteristics

Based on these conditions, appropriate statistical procedures can be selected, such as:

```text
Numerical × Numerical
        ↓
Pearson / Spearman

Continuous Target × 2 Groups
        ↓
Independent t-test / Mann-Whitney U

Continuous Target × >2 Groups
        ↓
ANOVA / Kruskal-Wallis

Categorical × Categorical
        ↓
Chi-square / Fisher's Exact Test

Multiple Numerical Predictors
        ↓
VIF Analysis
```

## Technology Stack

* Python
* Pandas
* NumPy
* SciPy
* Matplotlib
* Statsmodels

## Project Goal

The goal of this project is to build a flexible, reproducible, and research-oriented statistical analysis framework that can adapt to different datasets instead of relying on a fixed set of statistical tests.

The pipeline is designed to reduce repetitive analytical work while keeping statistical test selection connected to the characteristics of the data and the research objective.

