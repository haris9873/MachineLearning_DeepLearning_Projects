import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.feature_selection import mutual_info_classif


dataset = pd.read_csv(
    'EMG_Classification/cleaned_emg_dataset.csv')

# Set the display.max_columns option to None
pd.set_option('display.max_columns', None, 'display.width',
              2000, 'display.max_colwidth', None)

df = pd.DataFrame(dataset, columns=[
                  'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8', 'Activity'])

print(df.head())
print(df.tail())
print(df.info())
print(df.shape)

"""
----------------------------------------------------------
  Closed Grip ----> 0
  Cylindrical Grip ----> 1
  Index Finger Extension ----> 2
  Middle Finger Extension ----> 3
  Rest ----> 5   
-----------------------------------------------------------
"""

# Input features
features = dataset.drop(columns=['Activity'])

"""
----------------------------------------------------------
Visualise Channel Patterns
-----------------------------------------------------------
"""


def visualize_class_channel_patterns():
    classes = sorted(df['Activity'].unique())
    print(classes)
    channels = features.columns

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(len(classes), len(channels), figsize=(20, 20))
    fig.suptitle(
        'EMG Patterns by Activity and Channel (200 samples per class)', fontsize=16)

    for i, class_id in enumerate(classes):
        # Get a sample of 200 consecutive data points for this class
        class_sample = df[df['Activity'] == class_id].iloc[:200]

        for j, channel in enumerate(channels):
            ax = axes[i, j]
            ax.plot(class_sample[channel], 'b-', linewidth=0.5)

            # Set y-axis format to scientific notation e-5
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.yaxis.offsetText.set_fontsize(6)

            # Only add labels on edges of the grid
            if j == 0:
                match class_id:
                    case 1:
                        ax.set_ylabel('Closed Grip')
                    case 2:
                        ax.set_ylabel(f'Cylindrical Grip')
                    case 3:
                        ax.set_ylabel(f'Index Finger Extension')
                    case 4:
                        ax.set_ylabel(f'Middle Finger Extension')
                    case 5:
                        ax.set_ylabel(f'Rest')
            if i == len(classes) - 1:
                ax.set_xlabel(channel)

            ax.set_xticks([])
            if j > 0:
                ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs('EMG_Classification/Results', exist_ok=True)
    plt.savefig('EMG_Classification/Results/class_channel_patterns.png')
    plt.show()


# Call the function
visualize_class_channel_patterns()

"""
----------------------------------------------------------
Calculate Mutual Information
-----------------------------------------------------------
"""

# Use a balanced sample to calculate mutual info
balanced_sample_size = 10000  # Limit to avoid memory issues
balanced_df = pd.DataFrame()

for class_id in df['Activity'].unique():
    class_data = df[df['Activity'] == class_id]
    # Sample with replacement for minority classes, without for majority
    sample_size = min(balanced_sample_size //
                      len(df['Activity'].unique()), len(class_data))
    class_sample = class_data.sample(
        n=sample_size, replace=(sample_size > len(class_data)))
    balanced_df = pd.concat([balanced_df, class_sample])

# Calculate mutual information on the balanced sample
mi_scores = mutual_info_classif(
    balanced_df[features.columns], balanced_df['Activity'])
mi_channel_importance = [(features.columns[i], mi_scores[i])
                         for i in range(len(features.columns))]
mi_channel_importance.sort(key=lambda x: x[1], reverse=True)

print("Mutual information channel importance:")
for channel, mi in mi_channel_importance:
    print(f"{channel}: {mi:.4f}")

"""
----------------------------------------------------------
Separation Score
-----------------------------------------------------------
"""


def calculate_separation_score(col):
    # Calculate mean for each class
    class_means = df.groupby('Activity')[col].mean()

    # Calculate standard deviation for each class
    class_stds = df.groupby('Activity')[col].std()

    # Calculate pairwise separation scores (signal-to-noise ratio between class pairs)
    separation_score = 0
    num_pairs = 0

    for i, class_i in enumerate(class_means.index):
        for j, class_j in enumerate(class_means.index):
            if i < j:  # Only consider unique pairs
                # Formula: |mean_i - mean_j| / (std_i + std_j)
                # Higher values mean better separation
                mean_diff = abs(class_means[class_i] - class_means[class_j])
                std_sum = class_stds[class_i] + class_stds[class_j]

                # Avoid division by zero
                if std_sum > 0:
                    pair_score = mean_diff / std_sum
                else:
                    pair_score = 0

                separation_score += pair_score
                num_pairs += 1

    # Average separation score across all class pairs
    if num_pairs > 0:
        return separation_score / num_pairs
    else:
        return 0


"""
----------------------------------------------------------
Channel Variance
-----------------------------------------------------------
"""

channel_variance = []
for col in features.columns:
    # Calculate variance for each class
    class_variances = df.groupby('Activity')[col].var()

    # Calculate weighted average variance based on class frequency
    weighted_var = 0
    for class_id, variance in class_variances.items():
        # Use inverse of frequency to give more weight to minority classes
        weight = 1 / df['Activity'].value_counts()[class_id]
        weighted_var += variance * weight

    channel_variance.append((col, weighted_var))

channel_variance.sort(key=lambda x: x[1], reverse=True)
print("Variance-based channel importance:")
for channel, var in channel_variance:
    print(f"{channel}: {var:.10f}")

# Calculate separation score for each channel
channel_separation = []
for col in features.columns:
    separation = calculate_separation_score(col)
    channel_separation.append((col, separation))

channel_separation.sort(key=lambda x: x[1], reverse=True)

print("Class separation channel importance:")
for channel, sep in channel_separation:
    print(f"{channel}: {sep:.4f}")

"""
----------------------------------------------------------
Variance based importance
-----------------------------------------------------------
"""
plt.figure(figsize=(15, 15))

# Variance-based importance
plt.subplot(3, 1, 1)
var_importance = [x[1] for x in channel_variance]
plt.bar(range(len(var_importance)), var_importance)
plt.xticks(range(len(var_importance)), [x[0]
           for x in channel_variance], rotation=45)
plt.title('Channel Importance by Class-Weighted Variance')
plt.ylabel('Weighted Variance')
for i, v in enumerate(var_importance):
    plt.text(i, v + 0.01*max(var_importance), f"{v:.4f}", ha='center')

# Mutual information importance
plt.subplot(3, 1, 2)
mi_importance = [x[1] for x in mi_channel_importance]
plt.bar(range(len(mi_importance)), mi_importance)
plt.xticks(range(len(mi_importance)), [x[0]
           for x in mi_channel_importance], rotation=45)
plt.title('Channel Importance by Mutual Information with Class')
plt.ylabel('Mutual Information')
for i, v in enumerate(mi_importance):
    plt.text(i, v + 0.01*max(mi_importance), f"{v:.4f}", ha='center')

# Class separation importance
plt.subplot(3, 1, 3)
sep_importance = [x[1] for x in channel_separation]
plt.bar(range(len(sep_importance)), sep_importance)
plt.xticks(range(len(sep_importance)), [x[0]
           for x in channel_separation], rotation=45)
plt.title('Channel Importance by Class Separation')
plt.ylabel('Separation Score')
for i, v in enumerate(sep_importance):
    plt.text(i, v + 0.01*max(sep_importance), f"{v:.10f}", ha='center')

plt.tight_layout()
plt.savefig('EMG_Classification/Results/channel_importance.png')
plt.show()
