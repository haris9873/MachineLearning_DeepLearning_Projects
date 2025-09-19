from time import time
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn.functional as F


df = pd.read_csv(
    'EMG_Classification/cleaned_emg_dataset.csv')

# Set the display.max_columns option to None
pd.set_option('display.max_columns', None, 'display.width',
              2000, 'display.max_colwidth', None)

df.drop(columns=['User_ID'], inplace=True)
"""
----------------------------------------------------------
  Closed Grip ----> 0
  Cylindrical Grip ----> 1
  Index Finger Extension ----> 2
  Middle Finger Extension ----> 3
  Rest ----> 4   
-----------------------------------------------------------
"""


"""
----------------------------------------------------------
Splitting Data into Train and Test
-----------------------------------------------------------
"""
# Extract Activity labels

# Input features
features = df.drop(columns=['Activity'])
Activity = df["Activity"]
print("Activities:", Activity.unique())

# Convert to numpy arrays
Activity = Activity.values
features = features.values

# Address the Activity imbalance through stratified sampling

# Use stratify parameter to maintain Activity distribution
x_train, x_test, y_train, y_test = train_test_split(
    features, Activity,
    test_size=0.3,
    random_state=1,
    stratify=Activity  # This ensures the same Activity distribution in train and test sets
)

x_valid, x_test, y_valid, y_test = train_test_split(
    x_test, y_test,
    test_size=0.5,
    random_state=1,
    stratify=y_test  # This ensures the same Activity distribution in valid and test sets
)

# Confirm stratification worked
print("\nTraining set Activity distribution:")
for Activity_id in np.unique(y_train):
    print(f"Activity {Activity_id}: {(y_train == Activity_id).sum()} ({(y_train == Activity_id).sum() / len(y_train) * 100:.2f}%)")

print("\nValid set Activity distribution:")
for Activity_id in np.unique(y_valid):
    print(f"Activity {Activity_id}: {(y_valid == Activity_id).sum()} ({(y_valid == Activity_id).sum() / len(y_valid) * 100:.2f}%)")

print("\nTest set Activity distribution:")
for Activity_id in np.unique(y_test):
    print(f"Activity {Activity_id}: {(y_test == Activity_id).sum()} ({(y_test == Activity_id).sum() / len(y_test) * 100:.2f}%)")


"""

Sliding window for CNN

"""


def sliding_window(data, window_size, stride):
    windowed_data = []
    labels = []

    for i in range(0, len(data) - window_size + 1, stride):
        # Etiket sütununu hariç al ve NumPy array'e dönüştür
        windowed = data.iloc[i:i+window_size, :-1].values
        # Son sütunu etiket olarak kullan
        label = data.iloc[i+window_size-1, -1]
        windowed_data.append(windowed)
        labels.append(label)

    return np.array(windowed_data), np.array(labels)


window_size = 150
stride = 30
X, Y = sliding_window(df, 150, 30)


# Normalizing data
# Convert to float to avoid casting errors

print('Normalizing data ... \n')

x_train = x_train.astype('float')
x_valid = x_valid.astype('float')
x_test = x_test.astype('float')

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
std[std == 0] = 1e-6  # Prevent division by zero

x_train = (x_train - mean) / std
x_valid = (x_valid - mean) / std
x_test = (x_test - mean) / std

# Convert NumPy arrays to PyTorch tensors with the correct data type
print('Converting to PyTorch tensors ... \n')
x_train_tensor = torch.from_numpy(x_train).float()
x_valid_tensor = torch.from_numpy(x_valid).float()
x_test_tensor = torch.from_numpy(x_test).float()

y_train_tensor = torch.from_numpy(y_train).long()
y_valid_tensor = torch.from_numpy(y_valid).long()
y_test_tensor = torch.from_numpy(y_test).long()
# Function for plotting training history

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


class ActivityClassifier(nn.Module):
    def __init__(self, input_shape=8, num_classes=5):
        # Assuming the last column is the label and the rest are features
        super(ActivityClassifier, self).__init__()

        self.network_layers = nn.Sequential(
            # In this case, input_shape is a single integer, 8

            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),


            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),


            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),


            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25)
        )
        # Output layer for classification

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # The input is already 2D, so we don't need to flatten
        x = self.network_layers(x)
        x = self.classifier(x)
        return x


class CNN_model(nn.Module):

    def __init__(self, input_shape, num_classes):

        super(CNN_model, self).__init__()

        # Unpack the input shape to get the sequence length
        # Note: PyTorch Conv1D expects input as (batch_size, channels, sequence_length)
        # Keras's input_shape=(sequence_length, channels) is typically handled by
        # an embedding layer before the CNN. We assume the input is ready for Conv1D.
        self.sequence_length = input_shape[0]

        # First Convolutional Block
        # Keras's Conv1D(32, kernel_size=3) -> PyTorch's nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv1d(in_channels=self.sequence_length,
                               out_channels=32, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Second Convolutional Block
        # Keras's Conv1D(64, kernel_size=3) -> PyTorch's nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # The output of the last pooling layer needs to be calculated to define the
        # input size of the first fully connected (Dense) layer.
        # This is a common and often challenging part of converting between frameworks.
        # Here we will define the layers and compute the flattened size in the forward pass.

        # Fully Connected (Dense) Layers
        # Keras's Dense(128) -> PyTorch's nn.Linear(in_features, out_features)
        # Final dimension is 64 * sequence_length/4
        self.fc1 = nn.Linear(64 * (self.sequence_length // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)  # Final output layer

    def forward(self, x):
        """
        Defines the forward pass of the model.
        This method describes how data flows through the layers.
        """
        # Apply first conv and pooling block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Apply second conv and pooling block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Flatten the output for the fully connected layers
        # Keras's Flatten() -> PyTorch's x.view(batch_size, -1)
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))

        # Output layer with log_softmax for cross-entropy loss
        x = self.fc2(x)

        # It's common practice to omit the final softmax layer and use
        # nn.CrossEntropyLoss, which combines Softmax and NLLLoss.
        # If you need softmax for probabilities, you can add it separately:
        # x = F.softmax(self.fc2(x), dim=1)
        return x


model = ActivityClassifier()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
# Reduce learning rate when the loss has stopped improving for a certain number of epochs.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5)

# Create a DataLoader for the training data
# Check if GPU is available
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU.')
    model.cuda()  # move the model parameters to GPU
else:
    print('No GPU available, training on CPU.')


# 2. Set up the training loop
history = {'epoch': [], 'train_loss': [],
           'val_loss': [], 'train_acc': [], 'val_acc': []}

n_epochs = 20
valid_loss_min = np.inf  # track change in validation loss

time_start = time()
for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    val_loss = 0.0
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    ###################
    # train the model #
    ###################
    model.train()  # prep model for training
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        # Use no_grad for metrics to save memory
        with torch.no_grad():
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100 * correct_train / total_train
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)

    ######################
    # validate the model #
    ######################
    model.eval()  # prep model for evaluation
    correct_valid = 0
    total_valid = 0
    running_val_loss = 0
    with torch.no_grad():
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)

            val_loss = criterion(output, target)
            running_val_loss += val_loss.item()

            _, predicted = torch.max(output.data, 1)
            total_valid += target.size(0)
            correct_valid += (predicted == target).sum().item()

    val_loss = running_val_loss / len(test_loader.dataset)
    val_acc = 100 * correct_valid / total_valid
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print('\nValidation done .... \n')
    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.2f}% \tValidation Loss: {:.6f} \tValidation Accuracy: {:.2f}%'.format(
        epoch, train_loss, train_acc, val_loss, val_acc))
    history['epoch'].append(epoch)
    # save model if validation loss has decreased
    if val_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            val_loss))
        torch.save(model.state_dict(),
                   'Emotion_Classification/model_cnn_emotionclassifier.pt')
        valid_loss_min = val_loss
    # Update the learning rate based on validation loss
    scheduler.step(val_loss)

time_elapsed = time() - time_start

print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


def plot_history(history, epochs, valid_acc):
    loss = history.history['train_loss']
    val_loss = history.history['val_loss']
    acc = history.history['train_acc']
    val_acc = history.history['val_acc']

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('EMG_Classification/training_history.png')
    plt.show()


plot_history(history, n_epochs)

# # Save the model checkpoint
# torch.save(model.state_dict(),
#            'Emotion_Classification/model_cnn_emotionclassifier.pt')
