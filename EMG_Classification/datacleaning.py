import pandas as pd
import os
""""

The gesture classes include Index Finger Extension, Middle Finger Extension, Cylindrical Grip,
Closed Grip, and Rest. The Myo Armband by Thalmic labs was used for data acquisition.
The armband consists of 8 Surface EMG sensor units. The 8 sensors read data every 5ms.
  
  sampling_freq = 1/5ms = 200 Samples / sec      
"""
root_dir = 'EMG_Classification/EMG_dataset'
users = os.listdir(root_dir)
print(users)
# Create an empty list to store all the data
all_data_list = []
if not os.path.exists('EMG_Classification/combined_emg_dataset.csv'):
    # Use os.walk to go through all directories and files
    for user_folder, activities, files in os.walk(root_dir):
        for activity_folder in activities:
            activity_path = os.path.join(user_folder, activity_folder)

        # Extract user ID and activity label from the folder names
            user_id = os.path.basename(user_folder)
            activity_label = activity_folder

        # Now, walk through the files within this specific activity folder
            for set_file in os.listdir(activity_path):
                if set_file.endswith(('.csv', '.txt')):  # Check for your file format
                    file_path = os.path.join(activity_path, set_file)

                # Extract the set number
                    set_number = set_file.split('-')[-1].split('.')[0]

                    try:
                        # Read the individual EMG data file
                        # Adjust pd.read_csv parameters as needed (e.g., delimiter, header)
                        emg_data = pd.read_csv(file_path)

                    # Add metadata columns to the dataframe
                        emg_data['User_ID'] = user_id
                        emg_data['Activity'] = activity_label
                        emg_data['Set_Number'] = set_number

                    # Append this dataframe to our list
                        all_data_list.append(emg_data)

                    except Exception as e:
                        print(f"Could not read {file_path}: {e}")

# Concatenate all the individual dataframes into one master dataframe
    if all_data_list:
        combined_df = pd.concat(all_data_list, ignore_index=True)

    # Save the combined dataset to a new file
        combined_df.to_csv(
            "EMG_Classification/combined_emg_dataset.csv", index=False)
        print("Dataset successfully combined and saved as 'combined_emg_dataset.csv'.")
        print(combined_df.head())  # Print the first few rows to verify
        print(combined_df.tail())  # Print the first few rows to verify
    else:
        print("No data files found. Please check your file paths.")

dataset = pd.read_csv(
    'EMG_Classification/combined_emg_dataset.csv')
print(dataset.head())
print(dataset.tail())
print(dataset.info())
print(dataset.shape)

# Set the display.max_columns option to None
pd.set_option('display.max_columns', None, 'display.width',
              2000, 'display.max_colwidth', None)

"""
----------------------------------------------------------
                 Descriptive statistics
-----------------------------------------------------------
"""
df = pd.DataFrame(dataset, columns=['slno', 'emg1', 'emg2', 'emg3', 'emg4',
                  'emg5', 'emg6', 'emg7', 'emg8', 'User_ID', 'Activity', 'Set_Number'])


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
"""
----------------------------------------------------------
                Cleaning the data
    *** Removing missing values, by finding the indexes of the missing values ***
-----------------------------------------------------------
"""


# Remove slno, User_ID and Set_Number columns as they are not required
dataset_cleaned = df.copy()
dataset_cleaned.drop(['slno', 'User_ID', 'Set_Number'], axis=1, inplace=True)
print(dataset_cleaned.head())
# checking for any data  mismatch
print(dataset_cleaned['Activity'].unique())
print(dataset_cleaned['Activity'].unique())
# replace Closed Fist and ClosedGrip with Closed Grip for consistency
dataset_cleaned['Activity'] = dataset_cleaned['Activity'].replace(
    'Closed Fist', 'Closed Grip')
dataset_cleaned['Activity'] = dataset_cleaned['Activity'].replace(
    'ClosedGrip', 'Closed Grip')

"""
----------------------------------------------------------
  Closed Grip ---->0
  Cylindrical Grip ----> 1
  Index Finger Extension ----> 2
  Middle Finger Extension ----> 3
  Rest ----> 4
-----------------------------------------------------------
"""

dataset_cleaned.replace('Closed Grip', 0, inplace=True)
dataset_cleaned.replace('Cylindrical Grip', 1, inplace=True)
dataset_cleaned.replace('Index Finger Extension', 2, inplace=True)
dataset_cleaned.replace('Middle Finger Extension', 3, inplace=True)
dataset_cleaned.replace('Rest', 4, inplace=True)

# sorting the dataset by Activity
dataset_cleaned = dataset_cleaned.sort_values(by=['Activity'])
print(dataset_cleaned.head())
print(describe(dataset_cleaned))


print('Dataset cleaned and saved as cleaned_emg_dataset.csv')
if (os.path.exists("EMG_Classification/cleaned_emg_dataset.csv")):
    os.remove("EMG_Classification/cleaned_emg_dataset.csv")
dataset_cleaned.to_csv(
    "EMG_Classification/cleaned_emg_dataset.csv", index=False)
