# API Reference

## AI Impact on Jobs & Layoff Risk Analysis

---

## src/data_loading.py

### check_file_exists(file_path)
Check if file exists at given path.
- **Parameters:** file_path (str)
- **Returns:** bool

### load_dataset(file_path)
Load CSV dataset into pandas DataFrame.
- **Parameters:** file_path (str)
- **Returns:** pandas.DataFrame

### inspect_dataset(df)
Display dataset information.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** None

### display_shape(df)
Display dataset shape.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** tuple (rows, cols)

### display_columns(df)
Display all column names.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** list

### display_data_types(df)
Display data types of columns.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** pandas.Series

### display_missing_values(df)
Display missing values analysis.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** pandas.Series

---

## src/data_cleaning.py

### handle_missing_values(df, strategy="drop", threshold=0.5)
Handle missing values in dataset.
- **Parameters:**
  - df (pandas.DataFrame)
  - strategy (str): "drop", "fill_mean", "fill_median", "fill_mode"
  - threshold (float): Max missing percentage allowed
- **Returns:** pandas.DataFrame

### remove_duplicates(df)
Remove duplicate rows.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** pandas.DataFrame

### remove_invalid_records(df)
Remove invalid records based on domain rules.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** pandas.DataFrame

### standardize_column_names(df)
Standardize column names to lowercase with underscores.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** pandas.DataFrame

### save_clean_dataset(df, output_path)
Save cleaned dataset to CSV.
- **Parameters:**
  - df (pandas.DataFrame)
  - output_path (str)
- **Returns:** bool

---

## src/exploratory_data_analysis.py

### basic_statistics(df)
Display basic statistical summary.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** pandas.DataFrame

### distribution_analysis(df)
Analyze distribution of numerical columns.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** dict

### correlation_analysis(df)
Compute correlation matrix.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** pandas.DataFrame

### target_analysis(df, target_col="layoff_risk")
Analyze target variable distribution.
- **Parameters:**
  - df (pandas.DataFrame)
  - target_col (str)
- **Returns:** pandas.Series

### categorical_analysis(df)
Analyze categorical columns.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** dict

### group_analysis(df, group_col, target_col="layoff_risk")
Analyze target across groups.
- **Parameters:**
  - df (pandas.DataFrame)
  - group_col (str)
  - target_col (str)
- **Returns:** pandas.DataFrame

### run_complete_eda(df)
Run complete EDA pipeline.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** dict

---

## src/feature_engineering.py

### create_experience_categories(df)
Create experience categories from years.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** pandas.DataFrame

### create_age_groups(df)
Create age groups from age column.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** pandas.DataFrame

### create_ai_intensity_score(df)
Create AI intensity score.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** pandas.DataFrame

### create_routine_creativity_ratio(df)
Create routine-creativity ratio.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** pandas.DataFrame

### create_human_ai_balance(df)
Create human-AI balance score.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** pandas.DataFrame

### create_risk_factors(df)
Create risk factor flags.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** pandas.DataFrame

### run_feature_engineering(df)
Run complete feature engineering pipeline.
- **Parameters:** df (pandas.DataFrame)
- **Returns:** pandas.DataFrame

---

## src/visualization.py

### plot_target_distribution(df, target_col, output_dir)
Plot target variable distribution.
- **Parameters:**
  - df (pandas.DataFrame)
  - target_col (str)
  - output_dir (str)

### plot_numerical_distributions(df, output_dir)
Plot numerical column distributions.
- **Parameters:**
  - df (pandas.DataFrame)
  - output_dir (str)

### plot_categorical_distributions(df, output_dir)
Plot categorical column distributions.
- **Parameters:**
  - df (pandas.DataFrame)
  - output_dir (str)

### plot_correlation_heatmap(df, output_dir)
Plot correlation heatmap.
- **Parameters:**
  - df (pandas.DataFrame)
  - output_dir (str)

### plot_risk_by_category(df, category_col, target_col, output_dir)
Plot risk distribution by category.
- **Parameters:**
  - df (pandas.DataFrame)
  - category_col (str)
  - target_col (str)
  - output_dir (str)

### plot_boxplots(df, output_dir)
Plot boxplots by risk level.
- **Parameters:**
  - df (pandas.DataFrame)
  - output_dir (str)

### plot_ai_adoption_analysis(df, output_dir)
Plot AI adoption analysis.
- **Parameters:**
  - df (pandas.DataFrame)
  - output_dir (str)

### plot_feature_importance_preview(df, target_col, output_dir)
Plot feature importance preview.
- **Parameters:**
  - df (pandas.DataFrame)
  - target_col (str)
  - output_dir (str)

### run_complete_visualization(df, output_dir)
Run complete visualization pipeline.
- **Parameters:**
  - df (pandas.DataFrame)
  - output_dir (str)

---

## src/preprocessing.py

### encode_categorical_columns(df, target_col)
Encode categorical columns using LabelEncoder.
- **Parameters:**
  - df (pandas.DataFrame)
  - target_col (str)
- **Returns:** tuple (DataFrame, dict)

### scale_numerical_columns(df, exclude_cols)
Scale numerical columns using StandardScaler.
- **Parameters:**
  - df (pandas.DataFrame)
  - exclude_cols (list)
- **Returns:** tuple (DataFrame, scaler)

### split_features_target(df, target_col)
Split DataFrame into features and target.
- **Parameters:**
  - df (pandas.DataFrame)
  - target_col (str)
- **Returns:** tuple (X, y)

### train_test_split_data(X, y, test_size, random_state)
Split data into train and test sets.
- **Parameters:**
  - X (pandas.DataFrame)
  - y (pandas.Series)
  - test_size (float)
  - random_state (int)
- **Returns:** tuple (X_train, X_test, y_train, y_test)

### save_preprocessor(scaler, encoders, output_dir)
Save preprocessor objects to disk.
- **Parameters:**
  - scaler (StandardScaler)
  - encoders (dict)
  - output_dir (str)

### run_preprocessing(df, config)
Run complete preprocessing pipeline.
- **Parameters:**
  - df (pandas.DataFrame)
  - config (module)
- **Returns:** tuple (X_train, X_test, y_train, y_test, scaler, encoders)

---

## src/model_training.py

### initialize_models()
Initialize ML models.
- **Returns:** dict

### train_model(model, X_train, y_train, model_name)
Train single model.
- **Parameters:**
  - model: ML model
  - X_train (pandas.DataFrame)
  - y_train (pandas.Series)
  - model_name (str)
- **Returns:** tuple (model, training_time)

### train_all_models(models, X_train, y_train)
Train all models.
- **Parameters:**
  - models (dict)
  - X_train (pandas.DataFrame)
  - y_train (pandas.Series)
- **Returns:** dict

### save_best_model(trained_models, X_test, y_test, output_dir)
Save best model to disk.
- **Parameters:**
  - trained_models (dict)
  - X_test (pandas.DataFrame)
  - y_test (pandas.Series)
  - output_dir (str)
- **Returns:** str (best model name)

### compare_models(trained_models, X_test, y_test)
Compare model performance.
- **Parameters:**
  - trained_models (dict)
  - X_test (pandas.DataFrame)
  - y_test (pandas.Series)
- **Returns:** pandas.DataFrame

### run_model_training(X_train, X_test, y_train, y_test, config)
Run complete training pipeline.
- **Parameters:**
  - X_train, X_test (pandas.DataFrame)
  - y_train, y_test (pandas.Series)
  - config (module)
- **Returns:** tuple (trained_models, comparison_df, best_model_name)

---

## src/model_evaluation.py

### calculate_metrics(y_true, y_pred, average)
Calculate classification metrics.
- **Parameters:**
  - y_true (pandas.Series)
  - y_pred (pandas.Series)
  - average (str)
- **Returns:** dict

### print_classification_report(y_true, y_pred, model_name)
Print classification report.
- **Parameters:**
  - y_true (pandas.Series)
  - y_pred (pandas.Series)
  - model_name (str)
- **Returns:** str

### plot_confusion_matrix(y_true, y_pred, model_name, output_dir)
Plot confusion matrix.
- **Parameters:**
  - y_true (pandas.Series)
  - y_pred (pandas.Series)
  - model_name (str)
  - output_dir (str)

### plot_roc_curve(model, X_test, y_test, model_name, output_dir)
Plot ROC curve.
- **Parameters:**
  - model: Trained model
  - X_test (pandas.DataFrame)
  - y_test (pandas.Series)
  - model_name (str)
  - output_dir (str)

### evaluate_model(model, X_test, y_test, model_name, output_dir)
Evaluate single model.
- **Parameters:**
  - model: Trained model
  - X_test (pandas.DataFrame)
  - y_test (pandas.Series)
  - model_name (str)
  - output_dir (str)
- **Returns:** dict

### evaluate_all_models(trained_models, X_test, y_test, output_dir)
Evaluate all models.
- **Parameters:**
  - trained_models (dict)
  - X_test (pandas.DataFrame)
  - y_test (pandas.Series)
  - output_dir (str)
- **Returns:** dict

### save_evaluation_report(all_results, output_path)
Save evaluation report.
- **Parameters:**
  - all_results (dict)
  - output_path (str)

---

## src/prediction.py

### load_model(model_path)
Load trained model from disk.
- **Parameters:** model_path (str)
- **Returns:** model

### load_preprocessor(scaler_path, encoder_path)
Load preprocessor objects.
- **Parameters:**
  - scaler_path (str)
  - encoder_path (str)
- **Returns:** tuple (scaler, encoders)

### predict_single(model, scaler, encoders, input_data, feature_columns)
Make single prediction.
- **Parameters:**
  - model: Trained model
  - scaler: Fitted scaler
  - encoders (dict)
  - input_data (dict)
  - feature_columns (list)
- **Returns:** dict

### predict_batch(model, scaler, encoders, df, feature_columns)
Make batch predictions.
- **Parameters:**
  - model: Trained model
  - scaler: Fitted scaler
  - encoders (dict)
  - df (pandas.DataFrame)
  - feature_columns (list)
- **Returns:** pandas.DataFrame

### save_predictions(predictions_df, output_path)
Save predictions to CSV.
- **Parameters:**
  - predictions_df (pandas.DataFrame)
  - output_path (str)

### run_predictions(model, scaler, encoders, X_test, y_test, config)
Run complete prediction pipeline.
- **Parameters:**
  - model: Trained model
  - scaler: Fitted scaler
  - encoders (dict)
  - X_test (pandas.DataFrame)
  - y_test (pandas.Series)
  - config (module)
- **Returns:** pandas.DataFrame

---

## src/utils.py

### print_section_header(title)
Print formatted section header.

### print_subsection_header(title)
Print formatted subsection header.

### display_dataframe_info(df, name)
Display DataFrame information.

### validate_dataframe(df, required_columns)
Validate DataFrame structure.

### ensure_directory(dir_path)
Ensure directory exists.

### get_file_size(filepath)
Get file size in human-readable format.

### get_timestamp()
Get current timestamp.

### print_progress_bar(iteration, total, prefix, suffix, length)
Print progress bar.

### save_text_report(content, filepath)
Save text content to file.

### load_text_report(filepath)
Load text content from file.

### format_dict_for_display(data, indent)
Format dictionary for display.

### create_project_summary()
Create project summary dictionary.
