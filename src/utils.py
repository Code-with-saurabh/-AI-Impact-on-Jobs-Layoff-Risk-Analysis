import os
import pandas as pd
from datetime import datetime


def print_section_header(title):
    print("\n" + "#" * 70)
    print(f"# {title}")
    print("#" * 70)


def print_subsection_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def display_dataframe_info(df, name="DataFrame"):
    try:
        print(f"\n--- {name} Information ---")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            print(f"  {i}. {col} ({dtype}) - {non_null} non-null")

        print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    except Exception as e:
        print(f"[ERROR] Error displaying DataFrame info: {e}")


def validate_dataframe(df, required_columns=None):
    try:
        if df is None or df.empty:
            print("[VALIDATION] DataFrame is None or empty.")
            return False

        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                print(f"[VALIDATION] Missing columns: {missing}")
                return False

        print("[VALIDATION] DataFrame is valid.")
        return True

    except Exception as e:
        print(f"[ERROR] Error validating DataFrame: {e}")
        return False


def ensure_directory(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

    except Exception as e:
        print(f"[ERROR] Error creating directory: {e}")


def get_file_size(filepath):
    try:
        if not os.path.exists(filepath):
            return "File not found"

        size_bytes = os.path.getsize(filepath)

        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.2f} MB"

    except Exception as e:
        return f"Error: {e}"


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_progress_bar(iteration, total, prefix="Progress", suffix="Complete", length=50):
    try:
        percent = f"{100 * (iteration / float(total)):.1f}"
        filled_length = int(length * iteration // total)
        bar = "█" * filled_length + "-" * (length - filled_length)
        print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")

        if iteration == total:
            print()

    except Exception as e:
        print(f"[ERROR] Error printing progress bar: {e}")


def save_text_report(content, filepath):
    try:
        ensure_directory(os.path.dirname(filepath))

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[OK] Report saved: {filepath}")

    except Exception as e:
        print(f"[ERROR] Error saving report: {e}")


def load_text_report(filepath):
    try:
        if not os.path.exists(filepath):
            print(f"[ERROR] File not found: {filepath}")
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        return content

    except Exception as e:
        print(f"[ERROR] Error loading report: {e}")
        return None


def format_dict_for_display(data, indent=2):
    try:
        lines = []
        prefix = " " * indent

        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(format_dict_for_display(value, indent + 2))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}: {', '.join(map(str, value))}")
            else:
                lines.append(f"{prefix}{key}: {value}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error formatting dict: {e}"


def create_project_summary():
    summary = {
        "Project Title": "AI Impact on Jobs & Layoff Risk Analysis",
        "Student": "Sharma Saurabh",
        "Enrollment": "250863131007",
        "Course": "BE05000231 - Python for Data Science",
        "Semester": "GTU Semester-5",
        "Duration": "3 Months",
        "Phases": {
            "Month-1": "Foundation Phase - Project Setup, Data Loading, Data Cleaning",
            "Month-2": "Analysis Phase - EDA, Visualization, Feature Engineering",
            "Month-3": "Modeling Phase - ML Training, Evaluation, Prediction"
        }
    }
    return summary
