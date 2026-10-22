import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import time


def initialize_models():
    try:
        print("\n[Model Training] Initializing models...")

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        }

        for name, model in models.items():
            print(f"  - {name}: Initialized")

        return models

    except Exception as e:
        print(f"[ERROR] Error initializing models: {e}")
        return {}


def train_model(model, X_train, y_train, model_name="Model"):
    try:
        print(f"\n  Training {model_name}...")

        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()

        training_time = end_time - start_time
        print(f"  {model_name} trained in {training_time:.2f} seconds")

        return model, training_time

    except Exception as e:
        print(f"[ERROR] Error training {model_name}: {e}")
        return None, 0


def train_all_models(models, X_train, y_train):
    try:
        print("\n" + "=" * 70)
        print("TRAINING ALL MODELS")
        print("=" * 70)

        trained_models = {}

        for name, model in models.items():
            trained_model, train_time = train_model(model, X_train, y_train, name)

            if trained_model is not None:
                train_pred = trained_model.predict(X_train)
                train_accuracy = accuracy_score(y_train, train_pred)

                trained_models[name] = {
                    "model": trained_model,
                    "training_time": train_time,
                    "train_accuracy": train_accuracy
                }

                print(f"  {name} - Training Accuracy: {train_accuracy:.4f}")

        print("\n" + "=" * 70)
        print(f"Successfully trained: {len(trained_models)}/{len(models)} models")
        print("=" * 70)

        return trained_models

    except Exception as e:
        print(f"[ERROR] Error training all models: {e}")
        return {}


def save_best_model(trained_models, X_test, y_test, output_dir):
    try:
        print("\n[Model Training] Finding and saving best model...")

        best_accuracy = 0
        best_model_name = None

        for name, metadata in trained_models.items():
            model = metadata["model"]
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name

        if best_model_name:
            best_model = trained_models[best_model_name]["model"]

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            model_path = os.path.join(output_dir, "best_model.pkl")
            joblib.dump(best_model, model_path)

            print(f"\n  Best Model: {best_model_name}")
            print(f"  Test Accuracy: {best_accuracy:.4f}")
            print(f"  Saved to: {model_path}")

            return best_model_name

        return None

    except Exception as e:
        print(f"[ERROR] Error saving best model: {e}")
        return None


def compare_models(trained_models, X_test, y_test):
    try:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        results = []

        for name, metadata in trained_models.items():
            model = metadata["model"]
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            results.append({
                "Model": name,
                "Training Time (s)": f"{metadata['training_time']:.2f}",
                "Train Accuracy": f"{metadata['train_accuracy']:.4f}",
                "Test Accuracy": f"{accuracy:.4f}"
            })

        comparison_df = pd.DataFrame(results)
        print("\n" + comparison_df.to_string(index=False))

        print("\n" + "=" * 70)

        return comparison_df

    except Exception as e:
        print(f"[ERROR] Error comparing models: {e}")
        return None


def run_model_training(X_train, X_test, y_train, y_test, config):
    try:
        print("\n" + "#" * 70)
        print("# MODEL TRAINING")
        print("#" * 70)

        models = initialize_models()

        trained_models = train_all_models(models, X_train, y_train)

        comparison_df = compare_models(trained_models, X_test, y_test)

        best_model_name = save_best_model(trained_models, X_test, y_test, config.MODELS_OUTPUT_DIR)

        print("\n" + "#" * 70)
        print("# MODEL TRAINING COMPLETED SUCCESSFULLY")
        print("#" * 70)

        return trained_models, comparison_df, best_model_name

    except Exception as e:
        print(f"[ERROR] Error in model training: {e}")
        return {}, None, None
