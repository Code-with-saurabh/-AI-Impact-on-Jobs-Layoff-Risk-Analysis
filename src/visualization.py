import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


def save_plot(fig, filename, output_dir):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filename}")

    except Exception as e:
        print(f"[ERROR] Error saving plot: {e}")


def plot_target_distribution(df, target_col="layoff_risk", output_dir="outputs/charts"):
    try:
        print("\n[Visualization] Creating target distribution plot...")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        target_counts = df[target_col].value_counts()
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        axes[0].bar(target_counts.index, target_counts.values, color=colors[:len(target_counts)])
        axes[0].set_title("Layoff Risk Distribution", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Risk Level")
        axes[0].set_ylabel("Count")
        for i, v in enumerate(target_counts.values):
            axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

        axes[1].pie(target_counts.values, labels=target_counts.index, autopct="%1.1f%%",
                    colors=colors[:len(target_counts)], startangle=90)
        axes[1].set_title("Layoff Risk Percentage", fontsize=14, fontweight="bold")

        plt.tight_layout()
        save_plot(fig, "target_distribution.png", output_dir)

    except Exception as e:
        print(f"[ERROR] Error plotting target distribution: {e}")


def plot_numerical_distributions(df, output_dir="outputs/charts"):
    try:
        print("\n[Visualization] Creating numerical distribution plots...")

        numerical_cols = ["age", "years_of_experience", "routine_task_percentage",
                         "creativity_requirement", "human_interaction_level",
                         "number_of_ai_tools_used", "ai_usage_hours_per_week",
                         "tasks_automated_percentage", "ai_training_hours"]

        available_cols = [col for col in numerical_cols if col in df.columns]

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()

        for i, col in enumerate(available_cols[:9]):
            axes[i].hist(df[col], bins=30, color="#3498db", edgecolor="black", alpha=0.7)
            axes[i].set_title(f"{col.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
            axes[i].set_xlabel(col.replace('_', ' ').title())
            axes[i].set_ylabel("Frequency")
            axes[i].axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.1f}")
            axes[i].legend()

        for j in range(len(available_cols), 9):
            axes[j].set_visible(False)

        plt.suptitle("Numerical Column Distributions", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_plot(fig, "numerical_distributions.png", output_dir)

    except Exception as e:
        print(f"[ERROR] Error plotting numerical distributions: {e}")


def plot_categorical_distributions(df, output_dir="outputs/charts"):
    try:
        print("\n[Visualization] Creating categorical distribution plots...")

        categorical_cols = ["education_level", "industry", "company_size",
                           "job_level", "ai_adoption_level"]

        available_cols = [col for col in categorical_cols if col in df.columns]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        colors = sns.color_palette("Set2", len(available_cols))

        for i, col in enumerate(available_cols[:6]):
            value_counts = df[col].value_counts()
            axes[i].barh(value_counts.index, value_counts.values, color=colors[i])
            axes[i].set_title(f"{col.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
            axes[i].set_xlabel("Count")
            for j, v in enumerate(value_counts.values):
                axes[i].text(v + 20, j, str(v), va="center", fontweight="bold")

        for j in range(len(available_cols), 6):
            axes[j].set_visible(False)

        plt.suptitle("Categorical Column Distributions", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_plot(fig, "categorical_distributions.png", output_dir)

    except Exception as e:
        print(f"[ERROR] Error plotting categorical distributions: {e}")


def plot_correlation_heatmap(df, output_dir="outputs/charts"):
    try:
        print("\n[Visualization] Creating correlation heatmap...")

        numerical_cols = df.select_dtypes(include=["number"]).columns
        corr_matrix = df[numerical_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, square=True, linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap", fontsize=16, fontweight="bold")
        plt.tight_layout()
        save_plot(fig, "correlation_heatmap.png", output_dir)

    except Exception as e:
        print(f"[ERROR] Error plotting correlation heatmap: {e}")


def plot_risk_by_category(df, category_col, target_col="layoff_risk", output_dir="outputs/charts"):
    try:
        print(f"\n[Visualization] Creating risk by {category_col} plot...")

        if category_col not in df.columns or target_col not in df.columns:
            print(f"[WARNING] Required columns not found.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        ct = pd.crosstab(df[category_col], df[target_col], normalize="index") * 100
        ct.plot(kind="bar", stacked=True, ax=ax, color=["#2ecc71", "#f39c12", "#e74c3c"])

        ax.set_title(f"Layoff Risk by {category_col.replace('_', ' ').title()}",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel(category_col.replace("_", " ").title())
        ax.set_ylabel("Percentage")
        ax.legend(title="Risk Level")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        save_plot(fig, f"risk_by_{category_col}.png", output_dir)

    except Exception as e:
        print(f"[ERROR] Error plotting risk by category: {e}")


def plot_boxplots(df, output_dir="outputs/charts"):
    try:
        print("\n[Visualization] Creating boxplots...")

        numerical_cols = ["age", "years_of_experience", "routine_task_percentage",
                         "tasks_automated_percentage"]
        target_col = "layoff_risk"

        available_cols = [col for col in numerical_cols if col in df.columns]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

        for i, col in enumerate(available_cols[:4]):
            sns.boxplot(data=df, x=target_col, y=col, ax=axes[i],
                       palette=colors, order=["Low", "Medium", "High"])
            axes[i].set_title(f"{col.replace('_', ' ').title()} by Risk Level",
                             fontsize=12, fontweight="bold")
            axes[i].set_xlabel("Layoff Risk")
            axes[i].set_ylabel(col.replace("_", " ").title())

        plt.suptitle("Boxplots: Numerical Features by Risk Level", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_plot(fig, "boxplots_by_risk.png", output_dir)

    except Exception as e:
        print(f"[ERROR] Error plotting boxplots: {e}")


def plot_ai_adoption_analysis(df, output_dir="outputs/charts"):
    try:
        print("\n[Visualization] Creating AI adoption analysis plot...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if "ai_adoption_level" in df.columns:
            ai_counts = df["ai_adoption_level"].value_counts()
            colors = ["#3498db", "#f39c12", "#e74c3c"]
            axes[0].bar(ai_counts.index, ai_counts.values, color=colors[:len(ai_counts)])
            axes[0].set_title("AI Adoption Level Distribution", fontsize=14, fontweight="bold")
            axes[0].set_xlabel("AI Adoption Level")
            axes[0].set_ylabel("Count")

        if "ai_usage_hours_per_week" in df.columns and "tasks_automated_percentage" in df.columns:
            scatter = axes[1].scatter(df["ai_usage_hours_per_week"],
                                     df["tasks_automated_percentage"],
                                     c=df["layoff_risk"].map({"Low": 0, "Medium": 1, "High": 2}),
                                     cmap="RdYlGn_r", alpha=0.5)
            axes[1].set_title("AI Usage vs Tasks Automated", fontsize=14, fontweight="bold")
            axes[1].set_xlabel("AI Usage Hours/Week")
            axes[1].set_ylabel("Tasks Automated %")

        plt.tight_layout()
        save_plot(fig, "ai_adoption_analysis.png", output_dir)

    except Exception as e:
        print(f"[ERROR] Error plotting AI adoption analysis: {e}")


def plot_feature_importance_preview(df, target_col="layoff_risk", output_dir="outputs/charts"):
    try:
        print("\n[Visualization] Creating feature importance preview...")

        numerical_cols = df.select_dtypes(include=["number"]).columns
        if len(numerical_cols) == 0:
            print("[WARNING] No numerical columns found.")
            return

        target_encoded = df[target_col].map({"Low": 0, "Medium": 1, "High": 2})

        correlations = {}
        for col in numerical_cols:
            correlations[col] = abs(df[col].corr(target_encoded))

        corr_series = pd.Series(correlations).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        corr_series.plot(kind="barh", ax=ax, color="#3498db")
        ax.set_title("Feature Correlation with Target (Absolute)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Absolute Correlation")
        ax.set_ylabel("Features")
        plt.tight_layout()
        save_plot(fig, "feature_importance_preview.png", output_dir)

    except Exception as e:
        print(f"[ERROR] Error plotting feature importance: {e}")


def run_complete_visualization(df, output_dir="outputs/charts"):
    try:
        print("\n" + "#" * 70)
        print("# DATA VISUALIZATION")
        print("#" * 70)

        plot_target_distribution(df, output_dir=output_dir)
        plot_numerical_distributions(df, output_dir=output_dir)
        plot_categorical_distributions(df, output_dir=output_dir)
        plot_correlation_heatmap(df, output_dir=output_dir)

        for col in ["industry", "education_level", "company_size", "ai_adoption_level"]:
            if col in df.columns:
                plot_risk_by_category(df, col, output_dir=output_dir)

        plot_boxplots(df, output_dir=output_dir)
        plot_ai_adoption_analysis(df, output_dir=output_dir)
        plot_feature_importance_preview(df, output_dir=output_dir)

        print("\n" + "#" * 70)
        print("# VISUALIZATION COMPLETED SUCCESSFULLY")
        print("#" * 70)

    except Exception as e:
        print(f"[ERROR] Error in visualization: {e}")
