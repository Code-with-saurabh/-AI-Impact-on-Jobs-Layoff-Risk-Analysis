import pandas as pd
import numpy as np
import joblib
import os


def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}")
            return None

        model = joblib.load(model_path)
        print(f"[OK] Model loaded from: {model_path}")
        return model

    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return None


def load_preprocessor(scaler_path, encoder_path):
    try:
        scaler = None
        encoders = {}

        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"[OK] Scaler loaded from: {scaler_path}")

        if os.path.exists(encoder_path):
            encoders = joblib.load(encoder_path)
            print(f"[OK] Encoders loaded from: {encoder_path}")

        return scaler, encoders

    except Exception as e:
        print(f"[ERROR] Error loading preprocessor: {e}")
        return None, {}


def predict_single(model, scaler, encoders, input_data, feature_columns):
    try:
        print("\n[Prediction] Making single prediction...")

        input_df = pd.DataFrame([input_data])

        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]

        for col, encoder in encoders.items():
            if col in input_df.columns and col != "layoff_risk":
                try:
                    input_df[col] = encoder.transform(input_df[col].astype(str))
                except ValueError:
                    input_df[col] = 0

        if scaler is not None:
            numerical_cols = input_df.select_dtypes(include=["number"]).columns
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        prediction = model.predict(input_df)[0]

        probabilities = {}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            classes = model.classes_
            for cls, prob in zip(classes, proba):
                probabilities[cls] = round(prob, 4)

        if "layoff_risk" in encoders:
            prediction_label = encoders["layoff_risk"].inverse_transform([prediction])[0]
        else:
            prediction_label = prediction

        result = {
            "prediction": prediction_label,
            "probabilities": probabilities
        }

        print(f"\n  Prediction: {prediction_label}")
        if probabilities:
            print("  Probabilities:")
            for cls, prob in probabilities.items():
                if "layoff_risk" in encoders:
                    cls_label = encoders["layoff_risk"].inverse_transform([cls])[0]
                else:
                    cls_label = cls
                print(f"    {cls_label}: {prob*100:.2f}%")

        return result

    except Exception as e:
        print(f"[ERROR] Error making prediction: {e}")
        return {"prediction": None, "probabilities": {}}


def predict_batch(model, scaler, encoders, df, feature_columns):
    try:
        print("\n[Prediction] Making batch predictions...")

        result_df = df.copy()

        for col in feature_columns:
            if col not in result_df.columns:
                result_df[col] = 0

        input_df = result_df[feature_columns].copy()

        for col, encoder in encoders.items():
            if col in input_df.columns and col != "layoff_risk":
                try:
                    input_df[col] = encoder.transform(input_df[col].astype(str))
                except ValueError:
                    input_df[col] = 0

        if scaler is not None:
            numerical_cols = input_df.select_dtypes(include=["number"]).columns
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        predictions = model.predict(input_df)

        if "layoff_risk" in encoders:
            predictions_labels = encoders["layoff_risk"].inverse_transform(predictions)
        else:
            predictions_labels = predictions

        result_df["predicted_risk"] = predictions_labels

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)
            classes = model.classes_
            for i, cls in enumerate(classes):
                if "layoff_risk" in encoders:
                    cls_label = encoders["layoff_risk"].inverse_transform([cls])[0]
                else:
                    cls_label = cls
                result_df[f"prob_{cls_label}"] = proba[:, i].round(4)

        print(f"  Predictions made for {len(result_df)} samples")

        return result_df

    except Exception as e:
        print(f"[ERROR] Error making batch predictions: {e}")
        return df


def save_predictions(predictions_df, output_path):
    try:
        print(f"\n[Prediction] Saving predictions to: {output_path}")

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        predictions_df.to_csv(output_path, index=False)
        print(f"  Predictions saved successfully!")

    except Exception as e:
        print(f"[ERROR] Error saving predictions: {e}")


def run_predictions(model, scaler, encoders, X_test, y_test, config):
    try:
        print("\n" + "#" * 70)
        print("# MAKING PREDICTIONS")
        print("#" * 70)

        test_df = X_test.copy()
        test_df["actual_risk"] = y_test.values

        predictions_df = predict_batch(
            model, scaler, encoders, test_df, X_test.columns.tolist()
        )

        save_predictions(predictions_df, config.PREDICTIONS_PATH)

        print("\n--- Sample Predictions ---")
        sample_cols = ["actual_risk", "predicted_risk"]
        prob_cols = [col for col in predictions_df.columns if col.startswith("prob_")]
        display_cols = sample_cols + prob_cols
        print(predictions_df[display_cols].head(10))

        print("\n" + "#" * 70)
        print("# PREDICTIONS COMPLETED SUCCESSFULLY")
        print("#" * 70)

        return predictions_df

    except Exception as e:
        print(f"[ERROR] Error in predictions: {e}")
        return None
