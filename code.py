
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "C:/Users/HP/Downloads/messy_dataset.csv"
)

OUTPUT_FOLDER = Path(
    "D:/Ronys Project/Output"
)

RESULTS_FOLDER = (
    OUTPUT_FOLDER / "Automated_Analysis_Results"
)

# ------------------------------------------------------------
# IMPORTANT:
# এখানে তোমার dataset-এর outcome/target variable লিখবে।
#
# Example:
# TARGET_VARIABLE = "bmi"
# TARGET_VARIABLE = "diabetes"
#
# যদি target variable না থাকে, None রাখবে।
# ------------------------------------------------------------

TARGET_VARIABLE = None


# ============================================================
# PHASE 1: AUTOMATED DATA CLEANING
# ============================================================

def automate_data_cleaning(df):

    df_clean = df.copy()

    original_rows = len(df_clean)
    original_columns = len(df_clean.columns)

    # --------------------------------------------------------
    # Duplicate removal
    # --------------------------------------------------------

    duplicates_removed = df_clean.duplicated().sum()

    df_clean = df_clean.drop_duplicates()

    # --------------------------------------------------------
    # Identify variable types
    # --------------------------------------------------------

    numerical_cols = df_clean.select_dtypes(
        include=[np.number]
    ).columns

    categorical_cols = df_clean.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    # --------------------------------------------------------
    # Remove extra spaces + lowercase
    # --------------------------------------------------------

    for col in categorical_cols:

        df_clean[col] = (
            df_clean[col]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    # --------------------------------------------------------
    # Numerical missing → Mean
    # --------------------------------------------------------

    numerical_missing_filled = 0

    for col in numerical_cols:

        missing_count = df_clean[col].isna().sum()

        if missing_count > 0:

            mean_value = df_clean[col].mean()

            df_clean[col] = df_clean[col].fillna(
                mean_value
            )

            numerical_missing_filled += missing_count

    # --------------------------------------------------------
    # Categorical missing → Mode
    # --------------------------------------------------------

    categorical_missing_filled = 0

    for col in categorical_cols:

        missing_count = df_clean[col].isna().sum()

        if missing_count > 0:

            mode_values = df_clean[col].mode()

            if not mode_values.empty:

                mode_value = mode_values.iloc[0]

                df_clean[col] = df_clean[col].fillna(
                    mode_value
                )

                categorical_missing_filled += missing_count

            else:

                df_clean[col] = df_clean[col].fillna(
                    "unknown"
                )

                categorical_missing_filled += missing_count

    # --------------------------------------------------------
    # Outlier detection + IQR capping
    # --------------------------------------------------------

    outliers_capped = 0

    # ID/code columns should not be treated as outliers
    id_columns = [
        col for col in numerical_cols
        if (
            "id" in col.lower()
            or "code" in col.lower()
        )
    ]

    outlier_columns = [
        col for col in numerical_cols
        if col not in id_columns
    ]

    for col in outlier_columns:

        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)

        IQR = Q3 - Q1

        if IQR == 0:
            continue

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (
            (df_clean[col] < lower_bound)
            |
            (df_clean[col] > upper_bound)
        )

        outliers_capped += outlier_mask.sum()

        df_clean[col] = np.clip(
            df_clean[col],
            lower_bound,
            upper_bound
        )

    # --------------------------------------------------------
    # Cleaning report
    # --------------------------------------------------------

    report = {
        "Original rows": original_rows,
        "Original columns": original_columns,
        "Duplicates removed": duplicates_removed,
        "Numerical missing values filled":
            numerical_missing_filled,
        "Categorical missing values filled":
            categorical_missing_filled,
        "Outliers capped": outliers_capped,
        "Final rows": len(df_clean),
        "Final columns": len(df_clean.columns),
        "Remaining missing values":
            df_clean.isna().sum().sum(),
        "Remaining duplicates":
            df_clean.duplicated().sum()
    }

    return df_clean, report


# ============================================================
# PHASE 2: DATA VALIDATION
# ============================================================

def validate_data(df):

    results = []

    numerical_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    categorical_cols = df.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    ).columns

    results.append(
        ("Rows", len(df))
    )

    results.append(
        ("Columns", len(df.columns))
    )

    results.append(
        ("Numerical columns", len(numerical_cols))
    )

    results.append(
        ("Categorical columns", len(categorical_cols))
    )

    results.append(
        ("Missing values", df.isna().sum().sum())
    )

    results.append(
        ("Duplicate rows", df.duplicated().sum())
    )

    constant_columns = [
        col for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    ]

    results.append(
        ("Constant columns", len(constant_columns))
    )

    numerical_data = df.select_dtypes(
        include=[np.number]
    )

    if not numerical_data.empty:

        infinite_values = np.isinf(
            numerical_data
        ).sum().sum()

    else:

        infinite_values = 0

    results.append(
        ("Infinite values", infinite_values)
    )

    if (
        df.isna().sum().sum() == 0
        and df.duplicated().sum() == 0
        and len(constant_columns) == 0
        and infinite_values == 0
    ):

        status = "PASSED"

    else:

        status = "WARNING"

    results.append(
        ("Validation status", status)
    )

    return pd.DataFrame(
        results,
        columns=["Metric", "Value"]
    )


# ============================================================
# PHASE 3: AUTOMATED EDA
# ============================================================

def automated_eda(df, results_folder):

    tables_folder = (
        results_folder / "tables"
    )

    figures_folder = (
        results_folder / "figures"
    )

    tables_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    figures_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    numerical_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    categorical_cols = df.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    ).columns

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    if len(numerical_cols) > 0:

        descriptive = (
            df[numerical_cols]
            .describe()
            .T
        )

        descriptive["median"] = (
            df[numerical_cols].median()
        )

        descriptive["skewness"] = (
            df[numerical_cols].skew()
        )

        descriptive["kurtosis"] = (
            df[numerical_cols].kurtosis()
        )

        descriptive.to_csv(
            tables_folder /
            "descriptive_statistics.csv"
        )

    # --------------------------------------------------------
    # Categorical frequencies
    # --------------------------------------------------------

    for col in categorical_cols:

        frequency = (
            df[col]
            .value_counts(dropna=False)
            .reset_index()
        )

        frequency.columns = [
            col,
            "Frequency"
        ]

        frequency["Percentage"] = (
            frequency["Frequency"]
            /
            frequency["Frequency"].sum()
            * 100
        )

        frequency.to_csv(
            tables_folder /
            f"{col}_frequency.csv",
            index=False
        )

    # --------------------------------------------------------
    # Histograms
    # --------------------------------------------------------

    for col in numerical_cols:

        plt.figure(figsize=(8, 5))

        plt.hist(
            df[col].dropna(),
            bins=10,
            edgecolor="black"
        )

        plt.title(
            f"Distribution of {col}"
        )

        plt.xlabel(col)
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            figures_folder /
            f"{col}_histogram.png",
            dpi=300
        )

        plt.close()

    # --------------------------------------------------------
    # Boxplots
    # --------------------------------------------------------

    for col in numerical_cols:

        plt.figure(figsize=(6, 5))

        plt.boxplot(
            df[col].dropna()
        )

        plt.title(
            f"Boxplot of {col}"
        )

        plt.ylabel(col)

        plt.tight_layout()

        plt.savefig(
            figures_folder /
            f"{col}_boxplot.png",
            dpi=300
        )

        plt.close()

    # --------------------------------------------------------
    # Bar charts
    # --------------------------------------------------------

    for col in categorical_cols:

        counts = df[col].value_counts()

        plt.figure(figsize=(8, 5))

        counts.plot(kind="bar")

        plt.title(
            f"Distribution of {col}"
        )

        plt.xlabel(col)
        plt.ylabel("Frequency")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            figures_folder /
            f"{col}_bar_chart.png",
            dpi=300
        )

        plt.close()

    # --------------------------------------------------------
    # Correlation matrix
    # --------------------------------------------------------

    if len(numerical_cols) >= 2:

        correlation = (
            df[numerical_cols]
            .corr()
        )

        correlation.to_csv(
            tables_folder /
            "correlation_matrix.csv"
        )


# ============================================================
# PHASE 4
# DECISION-BASED AUTOMATED STATISTICAL ANALYSIS
# ============================================================

def phase4_statistical_analysis(
    df,
    results_folder,
    target_variable=None
):

    tables_folder = (
        results_folder / "phase4_tables"
    )

    figures_folder = (
        results_folder / "phase4_figures"
    )

    tables_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    figures_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # STEP 1: DATASET PROFILING
    # ========================================================

    numerical_cols = list(
        df.select_dtypes(
            include=[np.number]
        ).columns
    )

    categorical_cols = list(
        df.select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        ).columns
    )

    # Remove ID-like variables

    analysis_numerical_cols = [
        col for col in numerical_cols
        if not (
            "id" in col.lower()
            or "code" in col.lower()
        )
    ]

    # ========================================================
    # TARGET VARIABLE CHECK
    # ========================================================

    if target_variable is not None:

        if target_variable not in df.columns:

            print(
                f"WARNING: Target variable "
                f"'{target_variable}' not found."
            )

            target_variable = None

    # ========================================================
    # STEP 2: VARIABLE PROFILING
    # ========================================================

    profiling_results = []

    for col in df.columns:

        dtype = str(df[col].dtype)

        unique = df[col].nunique(
            dropna=True
        )

        if pd.api.types.is_numeric_dtype(
            df[col]
        ):

            variable_type = "Numerical"

        else:

            variable_type = "Categorical"

        if unique == 2:

            subtype = "Binary"

        elif unique <= 10:

            subtype = "Low-cardinality"

        else:

            subtype = "Continuous / High-cardinality"

        profiling_results.append(
            {
                "Variable": col,
                "Data_type": dtype,
                "Variable_type": variable_type,
                "Unique_values": unique,
                "Subtype": subtype
            }
        )

    profiling_df = pd.DataFrame(
        profiling_results
    )

    profiling_df.to_csv(
        tables_folder /
        "variable_profiling.csv",
        index=False
    )

    # ========================================================
    # STEP 3: NORMALITY ANALYSIS
    # ========================================================

    normality_results = []

    for col in analysis_numerical_cols:

        values = df[col].dropna()

        n = len(values)

        skewness = values.skew()

        kurtosis = values.kurtosis()

        if 3 <= n <= 5000:

            statistic, p_value = (
                stats.shapiro(values)
            )

            if p_value > 0.05:

                test_conclusion = (
                    "No strong evidence against normality"
                )

            else:

                test_conclusion = (
                    "Evidence of non-normality"
                )

        else:

            statistic = np.nan
            p_value = np.nan

            test_conclusion = (
                "Shapiro-Wilk not applied"
            )

        # ----------------------------------------------------
        # Q-Q plot
        # ----------------------------------------------------

        if n >= 3:

            plt.figure(figsize=(7, 5))

            stats.probplot(
                values,
                dist="norm",
                plot=plt
            )

            plt.title(
                f"Q-Q Plot: {col}"
            )

            plt.tight_layout()

            plt.savefig(
                figures_folder /
                f"{col}_qq_plot.png",
                dpi=300
            )

            plt.close()

        normality_results.append(
            {
                "Variable": col,
                "N": n,
                "Shapiro_statistic": statistic,
                "p_value": p_value,
                "Skewness": skewness,
                "Kurtosis": kurtosis,
                "Conclusion": test_conclusion
            }
        )

    normality_df = pd.DataFrame(
        normality_results
    )

    normality_df.to_csv(
        tables_folder /
        "normality_analysis.csv",
        index=False
    )

    # ========================================================
    # STEP 4: CORRELATION TEST SELECTION
    # ========================================================

    correlation_results = []

    for i in range(
        len(analysis_numerical_cols)
    ):

        for j in range(
            i + 1,
            len(analysis_numerical_cols)
        ):

            var1 = analysis_numerical_cols[i]
            var2 = analysis_numerical_cols[j]

            temp = df[
                [var1, var2]
            ].dropna()

            if len(temp) < 3:
                continue

            # ----------------------------------------------
            # Find normality status
            # ----------------------------------------------

            normal1 = False
            normal2 = False

            row1 = normality_df[
                normality_df["Variable"] == var1
            ]

            row2 = normality_df[
                normality_df["Variable"] == var2
            ]

            if not row1.empty:

                p1 = row1["p_value"].iloc[0]

                if (
                    not pd.isna(p1)
                    and p1 > 0.05
                ):

                    normal1 = True

            if not row2.empty:

                p2 = row2["p_value"].iloc[0]

                if (
                    not pd.isna(p2)
                    and p2 > 0.05
                ):

                    normal2 = True

            # ----------------------------------------------
            # Decision
            # ----------------------------------------------

            if normal1 and normal2:

                r, p = stats.pearsonr(
                    temp[var1],
                    temp[var2]
                )

                method = "Pearson"

            else:

                r, p = stats.spearmanr(
                    temp[var1],
                    temp[var2]
                )

                method = "Spearman"

            correlation_results.append(
                {
                    "Variable_1": var1,
                    "Variable_2": var2,
                    "Method": method,
                    "Correlation": r,
                    "p_value": p,
                    "N": len(temp)
                }
            )

    correlation_df = pd.DataFrame(
        correlation_results
    )

    correlation_df.to_csv(
        tables_folder /
        "automatic_correlation_tests.csv",
        index=False
    )

    # ========================================================
    # STEP 5: AUTOMATIC GROUP COMPARISON
    # ========================================================

    group_test_results = []

    if target_variable is not None:

        target_is_numeric = (
            pd.api.types.is_numeric_dtype(
                df[target_variable]
            )
        )

        # ----------------------------------------------------
        # Continuous target
        # ----------------------------------------------------

        if target_is_numeric:

            for col in categorical_cols:

                if col == target_variable:
                    continue

                number_of_groups = (
                    df[col]
                    .dropna()
                    .nunique()
                )

                if number_of_groups < 2:
                    continue

                if number_of_groups == 2:

                    groups = []

                    for _, group in (
                        df.groupby(col)
                    ):

                        values = (
                            group[target_variable]
                            .dropna()
                        )

                        groups.append(values)

                    if len(groups) != 2:
                        continue

                    group1 = groups[0]
                    group2 = groups[1]

                    # ------------------------------------------------
                    # Check normality within groups
                    # ------------------------------------------------

                    normal_groups = True

                    for group in groups:

                        if len(group) >= 3:

                            if len(group) <= 5000:

                                _, p_normal = (
                                    stats.shapiro(
                                        group
                                    )
                                )

                                if p_normal <= 0.05:

                                    normal_groups = False

                        else:

                            normal_groups = False

                    # ------------------------------------------------
                    # Test selection
                    # ------------------------------------------------

                    if normal_groups:

                        statistic, p_value = (
                            stats.ttest_ind(
                                group1,
                                group2,
                                equal_var=False
                            )
                        )

                        selected_test = (
                            "Independent t-test"
                        )

                    else:

                        statistic, p_value = (
                            stats.mannwhitneyu(
                                group1,
                                group2,
                                alternative="two-sided"
                            )
                        )

                        selected_test = (
                            "Mann-Whitney U"
                        )

                    group_test_results.append(
                        {
                            "Target": target_variable,
                            "Grouping_variable": col,
                            "Groups": number_of_groups,
                            "Selected_test":
                                selected_test,
                            "Statistic": statistic,
                            "p_value": p_value
                        }
                    )

                # ----------------------------------------------------
                # More than 2 groups
                # ----------------------------------------------------

                elif number_of_groups > 2:

                    groups = []

                    for _, group in (
                        df.groupby(col)
                    ):

                        values = (
                            group[target_variable]
                            .dropna()
                        )

                        if len(values) > 0:

                            groups.append(values)

                    if len(groups) < 3:
                        continue

                    normal_groups = True

                    for group in groups:

                        if len(group) >= 3:

                            if len(group) <= 5000:

                                _, p_normal = (
                                    stats.shapiro(
                                        group
                                    )
                                )

                                if p_normal <= 0.05:

                                    normal_groups = False

                        else:

                            normal_groups = False

                    if normal_groups:

                        statistic, p_value = (
                            stats.f_oneway(
                                *groups
                            )
                        )

                        selected_test = (
                            "One-way ANOVA"
                        )

                    else:

                        statistic, p_value = (
                            stats.kruskal(
                                *groups
                            )
                        )

                        selected_test = (
                            "Kruskal-Wallis"
                        )

                    group_test_results.append(
                        {
                            "Target": target_variable,
                            "Grouping_variable": col,
                            "Groups": number_of_groups,
                            "Selected_test":
                                selected_test,
                            "Statistic": statistic,
                            "p_value": p_value
                        }
                    )

    group_tests_df = pd.DataFrame(
        group_test_results
    )

    group_tests_df.to_csv(
        tables_folder /
        "automatic_group_comparison_tests.csv",
        index=False
    )

    # ========================================================
    # STEP 6: CATEGORICAL × CATEGORICAL
    # CHI-SQUARE OR FISHER
    # ========================================================

    categorical_test_results = []

    for i in range(
        len(categorical_cols)
    ):

        for j in range(
            i + 1,
            len(categorical_cols)
        ):

            var1 = categorical_cols[i]
            var2 = categorical_cols[j]

            contingency_table = pd.crosstab(
                df[var1],
                df[var2]
            )

            if (
                contingency_table.shape[0] < 2
                or contingency_table.shape[1] < 2
            ):
                continue

            try:

                chi2, p, dof, expected = (
                    stats.chi2_contingency(
                        contingency_table
                    )
                )

                # ------------------------------------------------
                # Expected frequency rule
                # ------------------------------------------------

                if (
                    contingency_table.shape == (2, 2)
                    and (expected < 5).any()
                ):

                    fisher_odds, fisher_p = (
                        stats.fisher_exact(
                            contingency_table
                        )
                    )

                    selected_test = (
                        "Fisher's Exact Test"
                    )

                    statistic = fisher_odds
                    p_value = fisher_p

                else:

                    selected_test = (
                        "Chi-square Test"
                    )

                    statistic = chi2
                    p_value = p

                categorical_test_results.append(
                    {
                        "Variable_1": var1,
                        "Variable_2": var2,
                        "Selected_test":
                            selected_test,
                        "Statistic": statistic,
                        "p_value": p_value
                    }
                )

            except Exception:

                pass

    categorical_tests_df = pd.DataFrame(
        categorical_test_results
    )

    categorical_tests_df.to_csv(
        tables_folder /
        "categorical_association_tests.csv",
        index=False
    )

    # ========================================================
    # STEP 7: VIF
    # ONLY WHEN MULTIPLE NUMERICAL PREDICTORS EXIST
    # ========================================================

    vif_results = []

    # If target is numerical, remove it from predictors

    vif_predictors = [
        col for col in analysis_numerical_cols
        if col != target_variable
    ]

    if len(vif_predictors) >= 2:

        vif_data = df[
            vif_predictors
        ].copy()

        vif_data = vif_data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        vif_data = vif_data.dropna()

        if (
            len(vif_data) > 2
            and len(vif_predictors) >= 2
        ):

            X = add_constant(
                vif_data,
                has_constant="add"
            )

            for i, col in enumerate(
                X.columns
            ):

                if col == "const":
                    continue

                try:

                    vif_value = (
                        variance_inflation_factor(
                            X.values,
                            i
                        )
                    )

                    tolerance = (
                        1 / vif_value
                    )

                    if vif_value < 5:

                        interpretation = (
                            "Acceptable"
                        )

                    elif vif_value < 10:

                        interpretation = (
                            "Potential concern"
                        )

                    else:

                        interpretation = (
                            "Serious multicollinearity"
                        )

                    vif_results.append(
                        {
                            "Variable": col,
                            "VIF": vif_value,
                            "Tolerance": tolerance,
                            "Interpretation":
                                interpretation
                        }
                    )

                except Exception:

                    pass

    vif_df = pd.DataFrame(
        vif_results
    )

    vif_df.to_csv(
        tables_folder /
        "vif_analysis.csv",
        index=False
    )

    # ========================================================
    # STEP 8: DECISION SUMMARY
    # ========================================================

    decision_summary = []

    decision_summary.append(
        {
            "Analysis": "Normality",
            "Decision":
                "Shapiro-Wilk + Skewness + Q-Q Plot"
        }
    )

    decision_summary.append(
        {
            "Analysis": "Numerical vs Numerical",
            "Decision":
                "Pearson if approximately normal; "
                "otherwise Spearman"
        }
    )

    decision_summary.append(
        {
            "Analysis": "Continuous target vs 2 groups",
            "Decision":
                "t-test if approximately normal; "
                "otherwise Mann-Whitney U"
        }
    )

    decision_summary.append(
        {
            "Analysis": "Continuous target vs >2 groups",
            "Decision":
                "ANOVA if approximately normal; "
                "otherwise Kruskal-Wallis"
        }
    )

    decision_summary.append(
        {
            "Analysis": "Categorical vs Categorical",
            "Decision":
                "Chi-square; Fisher's Exact when "
                "2×2 expected frequencies are small"
        }
    )

    decision_summary.append(
        {
            "Analysis": "Multicollinearity",
            "Decision":
                "VIF calculated when multiple "
                "numerical predictors exist"
        }
    )

    decision_df = pd.DataFrame(
        decision_summary
    )

    decision_df.to_csv(
        tables_folder /
        "automatic_test_decision_logic.csv",
        index=False
    )

    # ========================================================
    # STEP 9: PHASE 4 REPORT
    # ========================================================

    report = f"""
============================================================
PHASE 4: DECISION-BASED STATISTICAL ANALYSIS
============================================================

Dataset size:
{len(df)} rows × {len(df.columns)} columns

Target variable:
{target_variable}

Numerical variables analyzed:
{len(analysis_numerical_cols)}

Categorical variables:
{len(categorical_cols)}


------------------------------------------------------------
NORMALITY ANALYSIS
------------------------------------------------------------

Shapiro-Wilk:
Applied when 3 ≤ n ≤ 5000

Skewness:
Calculated

Kurtosis:
Calculated

Q-Q plots:
Generated


------------------------------------------------------------
CORRELATION ANALYSIS
------------------------------------------------------------

The system selects:

Pearson
    ↓
when both variables are approximately normal

Spearman
    ↓
when normality is not supported


------------------------------------------------------------
GROUP COMPARISON
------------------------------------------------------------

2 groups:

Approximately normal
    ↓
Independent t-test

Non-normal
    ↓
Mann-Whitney U


More than 2 groups:

Approximately normal
    ↓
One-way ANOVA

Non-normal
    ↓
Kruskal-Wallis


------------------------------------------------------------
CATEGORICAL ASSOCIATION
------------------------------------------------------------

Chi-square test is used when appropriate.

For a 2×2 table with small expected frequencies:

Fisher's Exact Test


------------------------------------------------------------
MULTICOLLINEARITY
------------------------------------------------------------

VIF is calculated only when:

Multiple numerical predictors exist.

VIF < 5:
Generally acceptable

VIF 5–10:
Potential concern

VIF > 10:
Serious concern


------------------------------------------------------------
IMPORTANT
------------------------------------------------------------

This system provides data-driven statistical test
selection.

Final statistical test selection should also consider:

1. Research question
2. Study design
3. Variable measurement scale
4. Independence of observations
5. Model assumptions
6. Sample size


============================================================
PHASE 4 COMPLETED
============================================================
"""

    report_file = (
        results_folder /
        "phase4_statistical_analysis_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    return report


# ============================================================
# MAIN PIPELINE
# ============================================================

print(
    "\n=================================================="
)

print(
    "AUTOMATED RESEARCH DATA ANALYSIS PIPELINE"
)

print(
    "=================================================="
)


# ============================================================
# LOAD DATA
# ============================================================

raw_data = pd.read_csv(
    INPUT_FILE
)

print(
    f"\nRaw dataset loaded: {raw_data.shape}"
)


# ============================================================
# PHASE 1
# ============================================================

print(
    "\nRunning Phase 1: Data Cleaning..."
)

cleaned_data, cleaning_report = (
    automate_data_cleaning(
        raw_data
    )
)

print(
    "Phase 1 completed."
)


# ============================================================
# PHASE 2
# ============================================================

print(
    "\nRunning Phase 2: Data Validation..."
)

validation_report = validate_data(
    cleaned_data
)

print(
    "Phase 2 completed."
)


# ============================================================
# PHASE 3
# ============================================================

print(
    "\nRunning Phase 3: Automated EDA..."
)

automated_eda(
    cleaned_data,
    RESULTS_FOLDER
)

print(
    "Phase 3 completed."
)


# ============================================================
# PHASE 4
# ============================================================

print(
    "\nRunning Phase 4: Decision-Based "
    "Statistical Analysis..."
)

phase4_report = (
    phase4_statistical_analysis(
        cleaned_data,
        RESULTS_FOLDER,
        TARGET_VARIABLE
    )
)

print(
    "Phase 4 completed."
)


# ============================================================
# SAVE CLEANED DATA
# ============================================================

cleaned_file = (
    OUTPUT_FOLDER /
    "cleaned_dataset.csv"
)

try:

    cleaned_data.to_csv(
        cleaned_file,
        index=False
    )

except PermissionError:

    cleaned_file = (
        OUTPUT_FOLDER /
        "cleaned_dataset_new.csv"
    )

    cleaned_data.to_csv(
        cleaned_file,
        index=False
    )


# ============================================================
# SAVE CLEANING REPORT
# ============================================================

cleaning_report_file = (
    OUTPUT_FOLDER /
    "cleaning_report.txt"
)

try:

    with open(
        cleaning_report_file,
        "w",
        encoding="utf-8"
    ) as file:

        for key, value in cleaning_report.items():

            file.write(
                f"{key}: {value}\n"
            )

except PermissionError:

    cleaning_report_file = (
        OUTPUT_FOLDER /
        "cleaning_report_new.txt"
    )

    with open(
        cleaning_report_file,
        "w",
        encoding="utf-8"
    ) as file:

        for key, value in cleaning_report.items():

            file.write(
                f"{key}: {value}\n"
            )


# ============================================================
# SAVE VALIDATION REPORT
# ============================================================

validation_report.to_csv(
    RESULTS_FOLDER /
    "validation_report.csv",
    index=False
)


# ============================================================
# COMPLETE PIPELINE REPORT
# ============================================================

complete_report = f"""
============================================================
AUTOMATED RESEARCH DATA ANALYSIS PIPELINE
============================================================


PHASE 1: DATA CLEANING
------------------------------------------------------------

Original rows:
{cleaning_report["Original rows"]}

Original columns:
{cleaning_report["Original columns"]}

Duplicates removed:
{cleaning_report["Duplicates removed"]}

Numerical missing values filled:
{cleaning_report["Numerical missing values filled"]}

Categorical missing values filled:
{cleaning_report["Categorical missing values filled"]}

Outliers capped:
{cleaning_report["Outliers capped"]}

Final rows:
{cleaning_report["Final rows"]}

Final columns:
{cleaning_report["Final columns"]}

Remaining missing values:
{cleaning_report["Remaining missing values"]}

Remaining duplicates:
{cleaning_report["Remaining duplicates"]}


PHASE 2: DATA VALIDATION
------------------------------------------------------------

Validation completed.

See:
validation_report.csv


PHASE 3: AUTOMATED EDA
------------------------------------------------------------

Descriptive statistics:
Generated

Frequency tables:
Generated

Histograms:
Generated

Boxplots:
Generated

Correlation matrix:
Generated


PHASE 4: DECISION-BASED STATISTICAL ANALYSIS
------------------------------------------------------------

Variable profiling:
Generated

Normality analysis:
Generated

Q-Q plots:
Generated

Automatic correlation test:
Generated

Automatic group comparison:
Generated

Categorical association test:
Generated

VIF:
Generated when appropriate

Decision logic:
Generated


============================================================
PIPELINE COMPLETED SUCCESSFULLY
============================================================
"""


complete_report_file = (
    OUTPUT_FOLDER /
    "complete_analysis_report.txt"
)

try:

    with open(
        complete_report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            complete_report
        )

except PermissionError:

    complete_report_file = (
        OUTPUT_FOLDER /
        "complete_analysis_report_new.txt"
    )

    with open(
        complete_report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            complete_report
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n=================================================="
)

print(
    "PIPELINE COMPLETED SUCCESSFULLY!"
)

print(
    "=================================================="
)

print(
    "\nCleaned dataset:"
)

print(
    cleaned_file
)

print(
    "\nAll results:"
)

print(
    RESULTS_FOLDER
)

print(
    "\nComplete report:"
)

print(
    complete_report_file
)
