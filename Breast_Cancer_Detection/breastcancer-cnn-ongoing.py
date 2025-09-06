import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# Load dataset
path = 'Breast_Cancer_Detection'
data = pd.read_csv(f"{path}/breast-cancer-dataset.csv")

# Set the display.max_columns option to None
pd.set_option('display.max_columns', None)
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
# numerise the data
dataset = dataset_cleaned.copy()
dataset['Tumor Size (cm)'] = pd.to_numeric(dataset['Tumor Size (cm)'])
dataset['Inv-Nodes'] = pd.to_numeric(dataset['Inv-Nodes'])
dataset['Metastasis'] = pd.to_numeric(dataset['Metastasis'])
dataset['History'] = pd.to_numeric(dataset['History'])


# Split data into train and test sets
train_data = data.sample(frac=0.8, random_state=42)  # 80% for training
test_data = data.drop(train_data.index)  # 20% for testing

# Convert data to PyTorch tensors
