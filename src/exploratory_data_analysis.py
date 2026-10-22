import pandas as pd
import numpy as np


def basic_statistics(df):
    try:
        print("\n" + "=" * 70)
        print("BASIC STATISTICAL SUMMARY")
        print("=" * 70)

        print("\n--- Numerical Columns ---")
        numerical_stats = df.describe()
        print(numerical_stats)

        print("\n--- Categorical Columns ---")
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            print(f"\n{col}:")
            print(f"  Unique Values: {df[col].nunique()}")
            print(f"  Top Value: {df[col].mode()[0] if not df[col].mode().empty else 'N/A'}")
            print(f"  Value Counts:")
            for val, count in df[col].value_counts().head(5).items():
                print(f"    {val}: {count} ({count/len(df)*100:.1f}%)")

        print("=" * 70)
        return numerical_stats

    except Exception as e:
        print(f"[ERROR] Error in basic statistics: {e}")
        return None


def distribution_analysis(df):
    try:
        print("\n" + "=" * 70)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 70)

        numerical_cols = df.select_dtypes(include=["number"]).columns
        distributions = {}

        for col in numerical_cols:
            skewness = df[col].skew()
            kurtosis = df[col].kurtosis()
            distributions[col] = {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "std": df[col].std(),
                "skewness": skewness,
                "kurtosis": kurtosis,
                "min": df[col].min(),
                "max": df[col].max(),
            }

            print(f"\n{col}:")
            print(f"  Mean: {distributions[col]['mean']:.2f}")
            print(f"  Median: {distributions[col]['median']:.2f}")
            print(f"  Std: {distributions[col]['std']:.2f}")
            print(f"  Skewness: {skewness:.2f}")
            print(f"  Kurtosis: {kurtosis:.2f}")

        print("=" * 70)
        return distributions

    except Exception as e:
        print(f"[ERROR] Error in distribution analysis: {e}")
        return {}


def correlation_analysis(df):
    try:
        print("\n" + "=" * 70)
        print("CORRELATION ANALYSIS")
        print("=" * 70)

        numerical_cols = df.select_dtypes(include=["number"]).columns
        corr_matrix = df[numerical_cols].corr()

        print("\nCorrelation Matrix:")
        print(corr_matrix.round(2))

        print("\n--- Highly Correlated Pairs (|r| > 0.5) ---")
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    high_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        if high_corr:
            for col1, col2, corr in high_corr:
                print(f"  {col1} <-> {col2}: {corr:.2f}")
        else:
            print("  No highly correlated pairs found.")

        print("=" * 70)
        return corr_matrix

    except Exception as e:
        print(f"[ERROR] Error in correlation analysis: {e}")
        return None


def target_analysis(df, target_col="layoff_risk"):
    try:
        print("\n" + "=" * 70)
        print(f"TARGET VARIABLE ANALYSIS: {target_col}")
        print("=" * 70)

        if target_col not in df.columns:
            print(f"[ERROR] Target column '{target_col}' not found.")
            return None

        target_counts = df[target_col].value_counts()
        target_percent = df[target_col].value_counts(normalize=True) * 100

        print(f"\nTarget Distribution:")
        print("-" * 40)
        for val, count in target_counts.items():
            percent = target_percent[val]
            print(f"  {val}: {count} ({percent:.1f}%)")

        print(f"\nTotal Samples: {len(df)}")
        print(f"Classes: {df[target_col].nunique()}")

        print("=" * 70)
        return target_counts

    except Exception as e:
        print(f"[ERROR] Error in target analysis: {e}")
        return None


def categorical_analysis(df):
    try:
        print("\n" + "=" * 70)
        print("CATEGORICAL ANALYSIS")
        print("=" * 70)

        categorical_cols = df.select_dtypes(include=["object"]).columns
        cat_analysis = {}

        for col in categorical_cols:
            value_counts = df[col].value_counts()
            cat_analysis[col] = value_counts

            print(f"\n{col} ({df[col].nunique()} unique values):")
            print("-" * 40)
            for val, count in value_counts.head(10).items():
                print(f"  {val}: {count} ({count/len(df)*100:.1f}%)")

        print("=" * 70)
        return cat_analysis

    except Exception as e:
        print(f"[ERROR] Error in categorical analysis: {e}")
        return {}


def group_analysis(df, group_col, target_col="layoff_risk"):
    try:
        print("\n" + "=" * 70)
        print(f"GROUP ANALYSIS: {group_col} vs {target_col}")
        print("=" * 70)

        if group_col not in df.columns or target_col not in df.columns:
            print(f"[ERROR] Column not found.")
            return None

        grouped = df.groupby(group_col)[target_col].value_counts().unstack(fill_value=0)
        print(f"\n{grouped}")

        print("=" * 70)
        return grouped

    except Exception as e:
        print(f"[ERROR] Error in group analysis: {e}")
        return None


def run_complete_eda(df):
    try:
        print("\n" + "#" * 70)
        print("# EXPLORATORY DATA ANALYSIS")
        print("#" * 70)

        results = {}

        results["basic_stats"] = basic_statistics(df)

        results["distributions"] = distribution_analysis(df)

        results["correlation"] = correlation_analysis(df)

        results["target"] = target_analysis(df)

        results["categorical"] = categorical_analysis(df)

        key_columns = ["industry", "education_level", "company_size", "ai_adoption_level"]
        results["group_analysis"] = {}
        for col in key_columns:
            if col in df.columns:
                results["group_analysis"][col] = group_analysis(df, col)

        print("\n" + "#" * 70)
        print("# EDA COMPLETED SUCCESSFULLY")
        print("#" * 70)

        return results

    except Exception as e:
        print(f"[ERROR] Error in complete EDA: {e}")
        return {}
