# Import necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, auc, classification_report
import lazypredict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Warnings remove alerts
import warnings
warnings.filterwarnings("ignore")


# data loading
path = 'Breast_Cancer_Detection'
data = pd.read_csv(f"{path}/breast-cancer-dataset.csv")

""" Cleaning the data
    *** Removing missing values, by finding the indexes of the missing values ***
"""
# Set the display.max_columns option to None
pd.set_option('display.max_columns', None, 'display.width',
              2000, 'display.max_colwidth', None)


# Vieweing 5 first data
print(data.head())
# Viewing 5 latest data
print(data.tail())
print(data.info())
print(data.shape)

df = pd.DataFrame(data, columns=['Year', 'Age', 'Menopause', 'Tumor Size (cm)', 'Inv-Nodes',
                  'Breast', 'Metastasis', 'Breast Quadrant', 'History', 'Diagnosis Result'])

# finding unique values

print('Age', df['Age'].unique())
print('Year', df['Year'].unique())
print('Menopause', df['Menopause'].unique())
print('Tumor Size (cm)', df['Tumor Size (cm)'].unique())
print('Inv-Nodes', df['Inv-Nodes'].unique())
print('Breast', df['Breast'].unique())
print('Metastasis', df['Metastasis'].unique())
print('Breast Quadrant', df['Breast Quadrant'].unique())
print('History', df['History'].unique())
print('Diagnosis Result', df['Diagnosis Result'].unique())

# finding missing values (#)

print('Age # Indexes', df[df['Year'] == '#'].index.values)
print('Tumor Size (cm) # Indexes',
      df[df['Tumor Size (cm)'] == '#'].index.values)
print('Inv-Nodes # Indexes', df[df['Inv-Nodes'] == '#'].index.values)
print('Metastasis # Indexes', df[df['Metastasis'] == '#'].index.values)
print('Breast # Indexes', df[df['Breast'] == '#'].index.values)
print('Metasis # Indexes', df[df['Metastasis'] == '#'].index.values)
print('Breast Quadrant # Indexes',
      df[df['Breast Quadrant'] == '#'].index.values)
print('History # Indexes', df[df['History'] == '#'].index.values)

# Clean the dataset by removing rows with missing values
dataset_cleaned = df.copy()
dataset_cleaned = dataset_cleaned.drop(
    [30, 40, 47, 67, 143, 164, 166, 178, 208])

# Descriptive statistics


def describe(df):

    features = []
    dtypes = []
    count = []
    unique = []
    missing_values = []
    min_ = []
    max_ = []

    for item in df.columns:
        features.append(item)
        dtypes.append(df[item].dtype)
        count.append(len(df[item]))
        unique.append(len(df[item].unique()))
        missing_values.append(df[item].isna().sum())

        if df[item].dtypes == 'int64' or df[item].dtypes == 'float64':

            min_.append(df[item].min())
            max_.append(df[item].max())

        else:
            min_.append('NaN')
            max_.append('NaN')

    out_put = pd.DataFrame({'Feature': features, 'Dtype': dtypes, 'Count': count, 'Unique': unique, 'Missing_value': missing_values,
                            'Min': min_, 'Max': max_})

    return out_put.T


print(describe(df))
print(dataset_cleaned.shape)

"""" Numerising the data for input to the machine learning model
    ----------------------------------------------------------
"""

dataset = dataset_cleaned.copy()
dataset['Tumor Size (cm)'] = pd.to_numeric(dataset['Tumor Size (cm)'])
dataset['Inv-Nodes'] = pd.to_numeric(dataset['Inv-Nodes'])
dataset['Metastasis'] = pd.to_numeric(dataset['Metastasis'])
dataset['History'] = pd.to_numeric(dataset['History'])
# Change malignant to 1 and benign to 0
dataset['Diagnosis Result'] = dataset['Diagnosis Result'].replace(
    'Malignant', 1)
dataset['Diagnosis Result'] = dataset['Diagnosis Result'].replace('Benign', 0)
# Change Breat left to 2 and right to 1
dataset['Breast'] = dataset['Breast'].replace('Left', 2)
dataset['Breast'] = dataset['Breast'].replace('Right', 1)
# Change Breast Quadrant to 1,2,3,4
dataset['Breast Quadrant'] = dataset['Breast Quadrant'].replace(
    'Upper inner', 1)
dataset['Breast Quadrant'] = dataset['Breast Quadrant'].replace(
    'Upper outer', 2)
dataset['Breast Quadrant'] = dataset['Breast Quadrant'].replace(
    'Lower inner', 3)
dataset['Breast Quadrant'] = dataset['Breast Quadrant'].replace(
    'Lower outer', 4)


# Input features
X = dataset.drop(columns=['Diagnosis Result'])
print("Viewing rows and columns given by X", X.shape)
# Target variable
y = dataset['Diagnosis Result']
print("Viewing rows and columns given y", y.shape)
# Training and testing division (80%, 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Viewing training data
print("Viewing rows and columns given by X train", X_train.shape)
# Viewing test data
print("Viewing rows and columns given y train", y_train.shape)


# Define the templates and parameters for GridSearchCV
models = [
    ('Naive Bayes', GaussianNB(), {}),
    ('Decision Tree', DecisionTreeClassifier(
        random_state=42), {'max_depth': [3, 5, 7]}),
    ('Logistic Regression', LogisticRegression(
        random_state=50), {'C': [0.01, 0.1, 1, 10]}),
    ('AdaBoost', AdaBoostClassifier(random_state=45),
     {'n_estimators': [50, 100, 200]}),
    ('Gradient Boosting', GradientBoostingClassifier(
        random_state=42), {'n_estimators': [50, 100, 200]}),
    ('K-Nearest Neighbors', KNeighborsClassifier(),
     {'n_neighbors': [3, 5, 7]}),
    ('SVC', SVC(probability=True, random_state=42), {
     'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    ('Neural Network', MLPClassifier(random_state=42), {
     'hidden_layer_sizes': [(50,), (100,), (50, 50)]}),
    ('Random Forest', RandomForestClassifier(
        random_state=42), {'n_estimators': [50, 100, 200]}),
]

results = {}
# Set up K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Set up and run GridSearchCV
prevbestmodelaccuracy = 0
for name,  model, param_grid in models:
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=kfold,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1  # Use all available cores
    )
    grid_search.fit(X_train, y_train)

    # Store the results
    results[name] = {
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'test_accuracy': grid_search.best_estimator_.score(X_test, y_test)
    }

    if grid_search.best_estimator_.score(X_test, y_test) > prevbestmodelaccuracy:
        bestmodel = grid_search.best_estimator_
    else:
        prevbestmodelaccuracy = grid_search.best_estimator_.score(
            X_test, y_test)

# Print the final results
results_df = pd.DataFrame(results).T.sort_values(
    by='test_accuracy', ascending=False)
print("\n--- Final Results ---")
print(results_df)
# Accuracy score and y_train y_test
train_accuracy = accuracy_score(y_train, bestmodel.predict(X_train))
test_accuracy = accuracy_score(y_test, bestmodel.predict(X_test))
print("Best Model", bestmodel)
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")


# # Using Lazypredict
# # Create and fit the LazyClassifier
# clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# # Print the results
# print(models)


""" ______________________________________________________

      Feature Importance in Cancer Diagnosis
    _________________________________________________________
"""
# Train models that support feature importances
models_with_feature_importances = [("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
                                   ("RandomForestClassifier", RandomForestClassifier(n_estimators=100,
                                                                                     random_state=42)),
                                   ("GradientBoostingClassifier",
                                    GradientBoostingClassifier(random_state=42)),
                                   ("LogisticRegression",
                                    LogisticRegression(random_state=42)),
                                   ("LinearSVC", SVC(random_state=42)),
                                   ("KNeighborsClassifier",
                                    KNeighborsClassifier()),
                                   ("GaussianNB", GaussianNB()),
                                   ("AdaBoostClassifier",
                                    AdaBoostClassifier(random_state=42)),
                                   ("NeuralNet", MLPClassifier(random_state=42))
                                   ]

# Iterate over models
for model_name, model in models_with_feature_importances:

    # Train model
    model.fit(X_train, y_train)

    # Get importance of features
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    else:
        # If the model does not have feature_importances_, continue to the next model
        print(f"\n{model_name} does not support feature importances.")
        continue

    # Create DataFrame for easier viewing
    feature_importances_df = pd.DataFrame({'Feature': X_train.columns,
                                           'Importance': feature_importances})

    # Sort by importance
    feature_importances_df = feature_importances_df.sort_values(
        by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances_df[:10])
    plt.title(f"Top 10 Features - {model_name}")
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(False)
    plt.savefig(
        f'Breast_Cancer_Detection/Results/{model_name}_feature_importances.png')
