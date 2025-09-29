from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# --- REGRESSION MODELS ---
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import PowerTransformer, RobustScaler
import warnings
from sklearn.ensemble import StackingRegressor
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def load_data(file_path):
    df = pd.read_csv(os.path.join(data_dir, 'parkinsons_dataset.csv'))
    return df


def preprocess_data(df):

    df = df.drop(columns=['subject#', 'test_time'], axis=1
                 )  # drop subject id and test time column as it is not a feature
    df = df.reset_index(drop=True)  # reset index after dropping rows/columns

    X = df.drop(columns=['total_UPDRS', 'motor_UPDRS'], axis=1)
    y = df[['total_UPDRS', 'motor_UPDRS']]

    return X, y


def transform_and_fit_models(X_tr):
    passthrough_cols = ['age', 'sex']  # Columns to leave unchanged
    power_transform_cols = [
        'Jitter(%)', 'Jitter(Abs)', 'Shimmer', 'Shimmer(dB)', 'NHR']
    # Remaining continuous features
    robust_scale_cols = [
        col for col in X_tr.columns
        if col not in passthrough_cols and col not in power_transform_cols
    ]
    print("Columns for Power Transformation:", power_transform_cols)
    print("Columns for Robust Scaling:", robust_scale_cols)
    # pipeline for power transform + robust scale for the most skewed acoustic features
    power_robust_pipeline = Pipeline([
        ('power_transform', PowerTransformer(method='yeo-johnson')),
        ('robust_scaler', RobustScaler())
    ])

    # Pipeline for other continuous features (needs only robust scaling)
    robust_only_pipeline = Pipeline([
        ('robust_scaler', RobustScaler())
    ])

    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('power_robust', power_robust_pipeline, power_transform_cols),
            ('robust_only', robust_only_pipeline, robust_scale_cols),
            # Pass 'age' and 'sex' columns through without transformation
            ('passthrough', 'passthrough', passthrough_cols)
        ],
        remainder='passthrough'
    )

    return preprocessor


def calculate_multioutput_metrics(Y_true, Y_pred, dataset_name):
    # R2 Score: Calculates R2 for each output and returns the mean (default behavior)
    r2 = r2_score(Y_true, Y_pred)

    # MSE: Calculates MSE for each output and returns the mean (default behavior)
    mse = mean_squared_error(Y_true, Y_pred)

    # RMSE: Calculated as the square root of the mean MSE
    rmse = np.sqrt(mse)

    print(f"\n--- {dataset_name} Metrics (Mean Across Outputs) ---")
    print(f"R-squared (R2) Score: {r2:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


if __name__ == "__main__":

    data_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    print(data_dir)
    # Load dataset
    df = load_data(data_dir)

    print('First 5 rows: \t', df.head())
    print('Dataset Shape: ', df.shape)
    # no. of unique subjects
    print('No. of Patients: ', df['subject#'].nunique())
    # record count per subject
    print('Record Count per Subject: \t',
          df['subject#'].value_counts().sort_index())

    if df.isnull().sum().any():
        print("Warning: Missing values detected in the dataset.")
    else:
        print("No missing values detected in the dataset.")
    # statistical summary of dataset
    print('Statistical Summary: \t', df.describe())
    print('Data Types and Non-Null Counts in the dataset: \t',
          df.info())  # data types and non-null counts

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print('Input Features: \n', X_train.head())
    print('Input Feature Shape: ', X_train.shape)
    print('Output Features: \n', y_train.head())
    print('Output Feature Shape: ', y_train.shape)
    # Identify key numerical features for scaling consideration.
    # Exclude subject# focusing on the core speech/UPDRS features.
    scaling_features = [
        'age', 'sex', 'motor_UPDRS', 'total_UPDRS', 'Jitter(%)', 'Jitter(Abs)', 'Shimmer', 'Shimmer(dB)',
        'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
    ]

    # Set up the plot area for boxplots
    fig, axes = plt.subplots(len(scaling_features), 1,
                             figsize=(12, 3 * len(scaling_features)))
    plt.subplots_adjust(hspace=0.5)

    for i, feature in enumerate(scaling_features):
        sns.boxplot(x=df[feature], ax=axes[i])
        axes[i].set_title(f'Box Plot of {feature}')

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'Results_Stacked')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'feature_boxplots.png'),
                bbox_inches='tight', dpi=300)

    print("Box plots saved as feature_boxplots.png")
    # Calculate skewness and kurtosis
    stats_df = df[scaling_features].agg(['skew', 'kurtosis']).transpose()

    print("Skewness and Kurtosis of Key Features:")
    print(stats_df)

    """
    ------------------------------------------------------------------------------------------------
    The acoustic features in the dataset (Jitter, Shimmer, NHR, etc.) are highly skewed and contain
    significant outliers, which is typical for biomedical data. Apply Power Transformation to reduce
    skewness and make the data more Gaussian-like. Then, use RobustScaler to scale the features, as it is
    less sensitive to outliers compared to StandardScaler or MinMaxScaler.
    ------------------------------------------------------------------------------------------------
    """

    preprocessor = transform_and_fit_models(
        X_train)

    # Define models and their best hyperparameter grids from prior tuning
    rf_estimator = RandomForestRegressor(
        n_estimators=300, max_depth=30, min_samples_split=5, min_samples_leaf=1, random_state=42)

    gb_estimator = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)

    knn_estimator = KNeighborsRegressor(
        n_neighbors=5, weights='distance', p=1)

    # --- 2. Define the Stacking Regressor ---

    estimators = [
        ('rf', rf_estimator),
        ('gb', gb_estimator),
        ('knn', knn_estimator)
    ]

    final_estimator = Ridge(alpha=1.0, random_state=42)

    # Stacking Regressor Setup
    stacked_reg_core = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,  # 5-fold cross-validation is performed internally for generating meta-features
        n_jobs=-1
    )

    # --- 3. Train and Evaluate the Stacked Model ---
    stacked_regressor_model = MultiOutputRegressor(stacked_reg_core)
    # Create the Final Pipeline: Preprocessor -> Stacking Core
    stacked_regressor = Pipeline([
        ('preprocessor', preprocessor),
        ('stacking', stacked_regressor_model)
    ])

    print("Starting training of the Stacking Regressor...")
    stacked_regressor.fit(X_train, y_train)
    print("Stacking Regressor training complete.")

    # Generate predictions
    Y_test_pred_stack = stacked_regressor.predict(X_test)
    Y_train_pred_stack = stacked_regressor.predict(X_train)

    # --- 4. Print Final Metrics ---
    print("\n--- Final Stacked Model Metrics ---")

    # Calculate Test R2
    test_r2 = r2_score(y_test, Y_test_pred_stack)
    print(f"Test Set R^2 Score (Stacked Model): {test_r2:.4f}")

    # Calculate Test RMSE
    test_mse = mean_squared_error(y_test, Y_test_pred_stack)
    test_rmse = np.sqrt(test_mse)
    print(f"Test Set RMSE (Mean Across Outputs): {test_rmse:.4f}")

    # Calculate Individual RMSE (for comparison with 2.97 and 2.39)
    rmse_by_output = np.sqrt(mean_squared_error(
        y_test, Y_test_pred_stack, multioutput='raw_values'))
    print(f"RMSE for total_UPDRS (Stacked): {rmse_by_output[0]:.4f} points")
    print(f"RMSE for motor_UPDRS (Stacked): {rmse_by_output[1]:.4f} points")

    # Overfitting check (Train R2)
    train_r2 = r2_score(y_train, Y_train_pred_stack)
    print(f"\nTraining Set R^2 Score (Stacked Model): {train_r2:.4f}")

    # Print the final results
    print("Training Complete.....")

    """
    ------------------------------------------------------------------------------------------------
     Note: In regression tasks, metrics like R² score, Mean Absolute Error (MAE), or Root Mean Squared Error (RMSE)
    are more appropriate than accuracy, confusion matrix, or AUC, which are used for classification tasks.
    ------------------------------------------------------------------------------------------------
    """

    # Function to calculate and print multi-output regression metrics
    calculate_multioutput_metrics(y_train, Y_train_pred_stack, "Training Set")
    calculate_multioutput_metrics(y_test, Y_test_pred_stack, "Test Set")

    # Optional: Print RMSE in original units for each output
    rmse_by_output = np.sqrt(mean_squared_error(
        y_test, Y_test_pred_stack, multioutput='raw_values'))
    print(f"\nRMSE for total_UPDRS: {rmse_by_output[0]:.4f} points")
    print(f"RMSE for motor_UPDRS: {rmse_by_output[1]:.4f} points")
    # Visualization: Predicted vs. True Values and Residual Plots for output features
    Y_true_total = y_test.iloc[:, 0]  # True total_UPDRS values
    Y_pred_total = Y_test_pred_stack[:, 0]  # Predicted total_UPDRS values
    residuals_total = Y_true_total - Y_pred_total

    """
    ------------------------------------------------------------------------------------------------
     Plotting the predicted values vs. true values and residual plots for total_UPDRS output feature
    ------------------------------------------------------------------------------------------------
    """
    sns.set_style("whitegrid")

    # --- Plot 1: Predicted vs. True Values (total_UPDRS) ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Y_true_total, y=Y_pred_total, color='darkred', alpha=0.6)
    # Plot the ideal Y=X line
    plt.plot([Y_true_total.min(), Y_true_total.max()],
             [Y_true_total.min(), Y_true_total.max()],
             '--k', alpha=0.7, label='Perfect Prediction')

    plt.title('Predicted vs. True total_UPDRS Scores')
    plt.xlabel('True total_UPDRS Score')
    plt.ylabel('Predicted total_UPDRS Score')
    plt.legend()
    plt.savefig(os.path.join(
        save_path, 'predicted_vs_true_total_UPDRS.png'), bbox_inches='tight', dpi=300)

    # --- Plot 2: Residual Plot (total_UPDRS) ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Y_pred_total, y=residuals_total,
                    color='darkblue', alpha=0.6)
    # Plot the zero-error line
    plt.hlines(y=0, xmin=Y_pred_total.min(), xmax=Y_pred_total.max(),
               colors='red', linestyles='--')
    plt.title('Residual Plot (Errors vs. Predicted Scores)')
    plt.xlabel('Predicted total_UPDRS Score')
    plt.ylabel('Residuals (True - Predicted)')
    plt.savefig(os.path.join(save_path, 'residuals_total_UPDRS.png'),
                bbox_inches='tight', dpi=300)

    # Feature Importance Plot (if applicable)
    multi_output_stacker = stacked_regressor.named_steps['stacking']

    try:

        feature_names = X_train.columns
    except AttributeError:

        feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]

    # The MultiOutputRegressor creates one estimator per output
    num_outputs = len(multi_output_stacker.estimators_)
    output_columns = [f'Output_{i+1}' for i in range(num_outputs)]

    # 3. Iterate through each output's StackingRegressor instance
    for i, stacker_instance in enumerate(multi_output_stacker.estimators_):
        output_name = output_columns[i]

        # Check if the final estimator has feature_importances_ (like Ridge/Linear models do not)
        final_estimator_model = stacker_instance.final_estimator_

        if hasattr(final_estimator_model, 'coef_'):
            # For linear models (like Ridge), we use the absolute coefficient magnitude as importance
            importances = np.abs(final_estimator_model.coef_)
            importance_source = 'Coefficients (Absolute)'
        elif hasattr(final_estimator_model, 'feature_importances_'):
            # For tree-based models
            importances = final_estimator_model.feature_importances_
            importance_source = 'Feature Importances'
        else:
            print(
                f"Skipping Feature Importance plot for {output_name}: Final estimator ({type(final_estimator_model).__name__}) does not support 'coef_' or 'feature_importances_'.")
            continue

        meta_feature_names = [name for name, _ in stacker_instance.estimators]

        # The feature count for the final estimator should be:
        # len(X_train.columns) + len(meta_feature_names)

        total_feature_names = list(feature_names) + meta_feature_names

        if len(importances) != len(total_feature_names):
            print(
                f"Warning: Feature count mismatch for {output_name}. Skipping plot.")
            continue  # Should not happen if StackingRegressor is used with default settings

        feature_df = pd.DataFrame({
            'Feature': total_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # 5. Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature',
                    data=feature_df.head(15),  # Plot top 15 features
                    color='teal')
        plt.title(
            f'Top 15 Feature Importance for Stacked Model ({output_name}) \n(Source: {importance_source} of Final Estimator)')
        plt.xlabel(importance_source)
        plt.savefig(os.path.join(save_path, f'feature_importances_{output_name}.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()  # Close plot to save memory

    print("Feature Importance plots (for the final estimator of the stack) saved for each output in the Results directory.")
