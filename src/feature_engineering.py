"""
Feature Engineering Module for AI Impact on Jobs & Layoff Risk Analysis.
This module provides functions to create new features from existing data.
It enhances the dataset with derived features for better analysis.
Author: Sharma Saurabh (250863131007)
Course: BE05000231 - Python for Data Science (GTU Semester-5)
Month: 2 - Analysis Phase
"""
import pandas as pd
import numpy as np


def create_experience_categories(df):
    """
    Create experience categories from years of experience.
    Parameters:
        df (pandas.DataFrame): The DataFrame to modify.
    Returns:
        pandas.DataFrame: DataFrame with new experience_category column.
    """
    try:
        print("\n[Feature Engineering] Creating experience categories...")
        if "years_of_experience" not in df.columns:
            print("[WARNING] Column 'years_of_experience' not found.")
            return df
        bins = [0, 5, 10, 20, 50]
        labels = ["Junior", "Mid-Level", "Senior", "Expert"]
        df["experience_category"] = pd.cut(
            df["years_of_experience"], bins=bins, labels=labels, include_lowest=True
        )
        print(f"  Created 'experience_category' with {df['experience_category'].nunique()} categories")
        return df
    except Exception as e:
        print(f"[ERROR] Error creating experience categories: {e}")
        return df


def create_age_groups(df):
    """
    Create age groups from age column.
    Parameters:
        df (pandas.DataFrame): The DataFrame to modify.
    Returns:
        pandas.DataFrame: DataFrame with new age_group column.
    """
    try:
        print("\n[Feature Engineering] Creating age groups...")
        if "age" not in df.columns:
            print("[WARNING] Column 'age' not found.")
            return df
        bins = [0, 25, 35, 45, 55, 100]
        labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
        df["age_group"] = pd.cut(
            df["age"], bins=bins, labels=labels, include_lowest=True
        )
        print(f"  Created 'age_group' with {df['age_group'].nunique()} groups")
        return df
    except Exception as e:
        print(f"[ERROR] Error creating age groups: {e}")
        return df


def create_ai_intensity_score(df):
    """
    Create AI intensity score based on AI-related columns.
    Parameters:
        df (pandas.DataFrame): The DataFrame to modify.
    Returns:
        pandas.DataFrame: DataFrame with new ai_intensity_score column.
    """
    try:
        print("\n[Feature Engineering] Creating AI intensity score...")
        required_cols = ["ai_usage_hours_per_week", "number_of_ai_tools_used", "tasks_automated_percentage"]
        if not all(col in df.columns for col in required_cols):
            print("[WARNING] Required columns not found for AI intensity score.")
            return df
        # Normalize columns to 0-1 scale
        ai_hours_norm = df["ai_usage_hours_per_week"] / df["ai_usage_hours_per_week"].max()
        ai_tools_norm = df["number_of_ai_tools_used"] / df["number_of_ai_tools_used"].max()
        ai_auto_norm = df["tasks_automated_percentage"] / 100
        # Calculate weighted score
        df["ai_intensity_score"] = (
            ai_hours_norm * 0.4 + ai_tools_norm * 0.3 + ai_auto_norm * 0.3
        ) * 100
        print(f"  Created 'ai_intensity_score' (range: {df['ai_intensity_score'].min():.1f} - {df['ai_intensity_score'].max():.1f})")
        return df
    except Exception as e:
        print(f"[ERROR] Error creating AI intensity score: {e}")
        return df


def create_routine_creativity_ratio(df):
    """
    Create ratio of routine tasks to creativity requirements.
    Parameters:
        df (pandas.DataFrame): The DataFrame to modify.
    Returns:
        pandas.DataFrame: DataFrame with new routine_creativity_ratio column.
    """
    try:
        print("\n[Feature Engineering] Creating routine-creativity ratio...")
        if "routine_task_percentage" not in df.columns or "creativity_requirement" not in df.columns:
            print("[WARNING] Required columns not found.")
            return df
        # Avoid division by zero
        df["routine_creativity_ratio"] = df["routine_task_percentage"] / (df["creativity_requirement"] + 1)
        print(f"  Created 'routine_creativity_ratio' (range: {df['routine_creativity_ratio'].min():.2f} - {df['routine_creativity_ratio'].max():.2f})")
        return df
    except Exception as e:
        print(f"[ERROR] Error creating routine-creativity ratio: {e}")
        return df


def create_human_ai_balance(df):
    """
    Create balance score between human interaction and AI usage.
    Parameters:
        df (pandas.DataFrame): The DataFrame to modify.
    Returns:
        pandas.DataFrame: DataFrame with new human_ai_balance column.
    """
    try:
        print("\n[Feature Engineering] Creating human-AI balance score...")
        if "human_interaction_level" not in df.columns or "ai_adoption_level" not in df.columns:
            print("[WARNING] Required columns not found.")
            return df
        # Map AI adoption to numeric
        ai_map = {"Low": 1, "Medium": 2, "High": 3}
        ai_numeric = df["ai_adoption_level"].map(ai_map).fillna(1)
        # Normalize human interaction
        human_norm = df["human_interaction_level"] / df["human_interaction_level"].max()
        # Balance score: high human + low AI = high balance
        df["human_ai_balance"] = (human_norm * 50) + ((4 - ai_numeric) / 3 * 50)
        print(f"  Created 'human_ai_balance' (range: {df['human_ai_balance'].min():.1f} - {df['human_ai_balance'].max():.1f})")
        return df
    except Exception as e:
        print(f"[ERROR] Error creating human-AI balance: {e}")
        return df


def create_risk_factors(df):
    """
    Create risk factor flags based on domain knowledge.
    Parameters:
        df (pandas.DataFrame): The DataFrame to modify.
    Returns:
        pandas.DataFrame: DataFrame with new risk factor columns.
    """
    try:
        print("\n[Feature Engineering] Creating risk factor flags...")
        # High automation risk
        if "tasks_automated_percentage" in df.columns:
            df["high_automation_risk"] = (df["tasks_automated_percentage"] > 70).astype(int)
            print(f"  Created 'high_automation_risk' ({df['high_automation_risk'].sum()} flagged)")
        # Low creativity risk
        if "creativity_requirement" in df.columns:
            df["low_creativity_risk"] = (df["creativity_requirement"] < 30).astype(int)
            print(f"  Created 'low_creativity_risk' ({df['low_creativity_risk'].sum()} flagged)")
        # High routine risk
        if "routine_task_percentage" in df.columns:
            df["high_routine_risk"] = (df["routine_task_percentage"] > 70).astype(int)
            print(f"  Created 'high_routine_risk' ({df['high_routine_risk'].sum()} flagged)")
        # Low human interaction risk
        if "human_interaction_level" in df.columns:
            df["low_interaction_risk"] = (df["human_interaction_level"] < 30).astype(int)
            print(f"  Created 'low_interaction_risk' ({df['low_interaction_risk'].sum()} flagged)")
        return df
    except Exception as e:
        print(f"[ERROR] Error creating risk factors: {e}")
        return df


def run_feature_engineering(df):
    """
    Run complete feature engineering pipeline.
    Parameters:
        df (pandas.DataFrame): The cleaned DataFrame.
    Returns:
        pandas.DataFrame: DataFrame with engineered features.
    """
    try:
        print("\n" + "#" * 70)
        print("# FEATURE ENGINEERING")
        print("#" * 70)
        initial_cols = len(df.columns)
        # Apply all feature engineering functions
        df = create_experience_categories(df)
        df = create_age_groups(df)
        df = create_ai_intensity_score(df)
        df = create_routine_creativity_ratio(df)
        df = create_human_ai_balance(df)
        df = create_risk_factors(df)
        final_cols = len(df.columns)
        print(f"\nFeatures added: {final_cols - initial_cols}")
        print(f"Total columns: {final_cols}")
        print("\n" + "#" * 70)
        print("# FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
        print("#" * 70)
        return df
    except Exception as e:
        print(f"[ERROR] Error in feature engineering: {e}")
        return df
