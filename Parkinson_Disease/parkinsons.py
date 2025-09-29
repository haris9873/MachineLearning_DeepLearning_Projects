from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# --- REGRESSION MODELS ---
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import PowerTransformer, RobustScaler
import warnings
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


def transform_and_fit_models(X_tr, X_tt):
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
    print('Input Features: \n', X.head())
    print('Output Features: \n', y.head())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

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
    save_path = os.path.join(os.path.dirname(__file__), 'Results')
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
        X_train, X_test)

    # Define the models
    models = [

        ('Random Forest Regressor', MultiOutputRegressor(RandomForestRegressor(random_state=42)), {
            'regressor__estimator__n_estimators': [300],
            'regressor__estimator__max_depth': [30],
            'regressor__estimator__min_samples_split': [5, 10],
        }),

        ('Decision Tree Regressor', MultiOutputRegressor(DecisionTreeRegressor(random_state=42)), {
            'regressor__estimator__max_depth': [5, 10, 20],
            'regressor__estimator__min_samples_split': [2, 5, 10],
        }),

        ('Gradient Boosting Regressor', MultiOutputRegressor(GradientBoostingRegressor(random_state=42)), {
            'regressor__estimator__n_estimators': [100, 200],
            'regressor__estimator__learning_rate': [0.05, 0.1, 0.2],
            'regressor__estimator__max_depth': [3, 5],
        }),

        ('AdaBoost Regressor', MultiOutputRegressor(AdaBoostRegressor(random_state=42)), {
            'regressor__estimator__n_estimators': [50, 100, 200],
            'regressor__estimator__learning_rate': [0.5, 1.0, 1.5],
        }),



        ('K-Nearest Neighbors Regressor', MultiOutputRegressor(KNeighborsRegressor()), {
            'regressor__estimator__n_neighbors': [3, 5, 7, 9],
            'regressor__estimator__weights': ['uniform', 'distance'],
            'regressor__estimator__p': [1, 2],
        }),

        ('SVR', MultiOutputRegressor(SVR()), {
            'regressor__estimator__C': [0.1, 1, 10],
            'regressor__estimator__kernel': ['rbf'],
            'regressor__estimator__gamma': ['scale', 0.01, 0.1],
        }),

        ('Neural Network Regressor', MultiOutputRegressor(MLPRegressor(random_state=42, max_iter=1000)), {
            # optimize hidden layer sizes for convergence
            'regressor__estimator__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'regressor__estimator__activation': ['relu', 'tanh'],
            'regressor__estimator__learning_rate': ['constant', 'adaptive'],
            'regressor__estimator__alpha': [0.0001, 0.001, 0.01],
        }),

        ('Ridge Regression', MultiOutputRegressor(Ridge(random_state=50)), {
            'regressor__estimator__alpha': [0.1, 1.0, 10.0, 100.0],
        })
    ]

    results = {}
    # Set up K-Fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    prev_best_r2 = -float('inf')
    bestmodel = None

    for name, model, param_grid in models:
        print(f"\n--------------------------------------------------------")
        print(f"STARTING TRAINING FOR MODEL: {name}")
        print(f"--------------------------------------------------------")

        # Create a full pipeline: Preprocessor -> Model
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # Evaluate using k-fold cross-validation on the training set
        scores = cross_val_score(
            full_pipeline, X_train, y_train, cv=kfold, scoring='r2', n_jobs=-1)

        grid_search = GridSearchCV(
            estimator=full_pipeline,
            param_grid=param_grid,
            cv=kfold,
            scoring='r2',
            verbose=1,
            n_jobs=-1  # Use all available cores
        )
        # Fit the model with error handling
        try:
            # Suppress the specific warning about max_iter
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                # If this fails, the 'except' block will handle it
                grid_search.fit(X_train, y_train)
            # Evaluate on the test set if fit was successful
            test_r2_score = grid_search.best_estimator_.score(X_test, y_test)

            results[name] = {
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'test_R2_score': test_r2_score,
            }

            if test_r2_score > prev_best_r2:
                bestmodel = grid_search.best_estimator_
                prev_best_r2 = test_r2_score

        except Exception as e:
            # if fitting fails, log the error and continue
            print(f"ERROR: Model {name} failed during fit. Skipping...")
            print(f"Details: {e}")
            # Store a placeholder result so the model is included in the final table
            results[name] = {
                'best_params': 'Failed to converge/fit',
                'best_cv_score': np.nan,
                'test_R2_score': np.nan,
            }
            # Continue to the next model in the loop
            continue

    # Print the final results
    print("Training Complete.....")

    # 1. Correct the DataFrame sorting column name
    results_df = pd.DataFrame(results).T.sort_values(
        by='test_R2_score', ascending=False)

    # 2. Extract best model info from the sorted DataFrame
    best_model_name_final = results_df.index[0]
    best_test_r2_final = results_df['test_R2_score'].iloc[0]
    best_params_final = results_df['best_params'].iloc[0]

    # 3. Print the Best Model Summary (R2 score replaces accuracy)
    print("\n--- Best Model Summary ---")
    print(f"Best Model Found: {best_model_name_final}")
    print(f"Test Set R^2 Score (Best Model): {best_test_r2_final:.4f}")
    print(f"Best parameters found: {best_params_final}")

    print("\nAll Model Results:")
    print(results_df)

    """
    ------------------------------------------------------------------------------------------------
     Note: In regression tasks, metrics like R² score, Mean Absolute Error (MAE), or Root Mean Squared Error (RMSE)
    are more appropriate than accuracy, confusion matrix, or AUC, which are used for classification tasks.
    ------------------------------------------------------------------------------------------------
    """
    # Predict on both the training and test sets using the best fitted pipeline
    Y_train_pred = bestmodel.predict(X_train)
    Y_test_pred = bestmodel.predict(X_test)

    # Function to calculate and print multi-output regression metrics
    calculate_multioutput_metrics(y_train, Y_train_pred, "Training Set")
    calculate_multioutput_metrics(y_test, Y_test_pred, "Test Set")

    # Optional: Print RMSE in original units for each output
    rmse_by_output = np.sqrt(mean_squared_error(
        y_test, Y_test_pred, multioutput='raw_values'))
    print(f"\nRMSE for total_UPDRS: {rmse_by_output[0]:.4f} points")
    print(f"RMSE for motor_UPDRS: {rmse_by_output[1]:.4f} points")
    # Visualization: Predicted vs. True Values and Residual Plots for output features
    Y_true_total = y_test.iloc[:, 0]  # True total_UPDRS values
    Y_pred_total = Y_test_pred[:, 0]  # Predicted total_UPDRS values
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

    # --- Plot 3: Feature Importance (If bestmodel is a tree ensemble) ---
    if hasattr(bestmodel.named_steps['regressor'], 'feature_importances_'):
        importances = bestmodel.named_steps['regressor'].feature_importances_
        feature_names = X_train.columns

        # Create DataFrame for plotting
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature',
                    data=feature_df.head(10), color='teal')
        plt.title('Top 10 Feature Importances')
        plt.savefig(os.path.join(save_path, 'feature_importances.png'),
                    bbox_inches='tight', dpi=300)

        print("Plots saved in the Results directory.")
