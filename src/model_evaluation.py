import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import os


def calculate_metrics(y_true, y_pred, average="weighted"):
    try:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        return metrics

    except Exception as e:
        print(f"[ERROR] Error calculating metrics: {e}")
        return {}


def print_classification_report(y_true, y_pred, model_name="Model"):
    try:
        print(f"\n--- {model_name} Classification Report ---")
        report = classification_report(y_true, y_pred, zero_division=0)
        print(report)
        return report

    except Exception as e:
        print(f"[ERROR] Error generating classification report: {e}")
        return ""


def plot_confusion_matrix(y_true, y_pred, model_name="Model", output_dir="outputs/charts"):
    try:
        print(f"\n[Evaluation] Creating confusion matrix for {model_name}...")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                   xticklabels=["Low", "Medium", "High"],
                   yticklabels=["Low", "Medium", "High"])
        ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filename}")

    except Exception as e:
        print(f"[ERROR] Error plotting confusion matrix: {e}")


def plot_roc_curve(model, X_test, y_test, model_name="Model", output_dir="outputs/charts"):
    try:
        print(f"\n[Evaluation] Creating ROC curve for {model_name}...")

        classes = sorted(y_test.unique())
        n_classes = len(classes)

        if n_classes < 2:
            print("  Skipping ROC curve (less than 2 classes)")
            return

        y_test_bin = label_binarize(y_test, classes=classes)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        else:
            y_score = model.decision_function(X_test)

        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]

        for i, color in zip(range(n_classes), colors[:n_classes]):
            ax.plot(fpr[i], tpr[i], color=color, linewidth=2,
                   label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve - {model_name}", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {filename}")

    except Exception as e:
        print(f"[ERROR] Error plotting ROC curve: {e}")


def evaluate_model(model, X_test, y_test, model_name="Model", output_dir="outputs/charts"):
    try:
        print("\n" + "=" * 70)
        print(f"EVALUATING: {model_name}")
        print("=" * 70)

        y_pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)

        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")

        report = print_classification_report(y_test, y_pred, model_name)

        plot_confusion_matrix(y_test, y_pred, model_name, output_dir)

        plot_roc_curve(model, X_test, y_test, model_name, output_dir)

        metrics["classification_report"] = report

        print("=" * 70)

        return metrics

    except Exception as e:
        print(f"[ERROR] Error evaluating {model_name}: {e}")
        return {}


def evaluate_all_models(trained_models, X_test, y_test, output_dir="outputs/charts"):
    try:
        print("\n" + "#" * 70)
        print("# MODEL EVALUATION")
        print("#" * 70)

        all_results = {}

        for name, metadata in trained_models.items():
            model = metadata["model"]
            metrics = evaluate_model(model, X_test, y_test, name, output_dir)
            all_results[name] = metrics

        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)

        comparison_data = []
        for name, metrics in all_results.items():
            if metrics:
                comparison_data.append({
                    "Model": name,
                    "Accuracy": f"{metrics['accuracy']:.4f}",
                    "Precision": f"{metrics['precision']:.4f}",
                    "Recall": f"{metrics['recall']:.4f}",
                    "F1-Score": f"{metrics['f1_score']:.4f}"
                })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print("\n" + comparison_df.to_string(index=False))

        print("\n" + "#" * 70)
        print("# EVALUATION COMPLETED SUCCESSFULLY")
        print("#" * 70)

        return all_results

    except Exception as e:
        print(f"[ERROR] Error evaluating all models: {e}")
        return {}


def save_evaluation_report(all_results, output_path):
    try:
        print(f"\n[Evaluation] Saving report to: {output_path}")

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        with open(output_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("AI Impact on Jobs & Layoff Risk Analysis\n")
            f.write("=" * 70 + "\n\n")

            for name, metrics in all_results.items():
                f.write(f"\n{'=' * 70}\n")
                f.write(f"MODEL: {name}\n")
                f.write(f"{'=' * 70}\n\n")

                if metrics:
                    f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
                    f.write(f"Precision: {metrics['precision']:.4f}\n")
                    f.write(f"Recall:    {metrics['recall']:.4f}\n")
                    f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n\n")

                    if "classification_report" in metrics:
                        f.write("Classification Report:\n")
                        f.write(metrics["classification_report"])
                        f.write("\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 70 + "\n")

        print(f"  Report saved successfully!")

    except Exception as e:
        print(f"[ERROR] Error saving evaluation report: {e}")
