import time
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

Sliding window for CNN

"""


def sliding_window(data, window_size, stride):
    windowed_data = []
    labels = []

    for i in range(0, len(data) - window_size + 1, stride):
        windowed = data.iloc[i:i+window_size, :-1].values
        label = data.iloc[i+window_size-1, -1]
        windowed_data.append(windowed)
        labels.append(label)

    return np.array(windowed_data), np.array(labels)


print(df.shape)
window_size = 200
stride = 30
X, Y = sliding_window(df, window_size, stride)
print(X.shape)
print(Y.shape)

# Normalizing data


def normalized(data):
    min_value = np.min(data)
    max_value = np.max(data)
    normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data


X = normalized(X)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.3,
    random_state=1,
    stratify=Y  # This ensures the same Activity distribution in train and test sets
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

print(x_train.shape)


class CNN_model(nn.Module):

    def __init__(self, input_shape, num_classes):

        super(CNN_model, self).__init__()

        num_channels = input_shape[1]
        sequence_length = input_shape[0]

        self.conv1 = nn.Conv1d(in_channels=num_channels,
                               out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # Second Convolutional Block
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Calculate the size of the flattened layer
        # The sequence length is halved by each of the two pooling layers
        flattened_size = 128 * (sequence_length // 4)

        # Fully Connected (Dense) Layers
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(p=0.35)
        self.fc2 = nn.Linear(256, num_classes)

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
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# define the CNN model
input_shape = x_train.shape[1:]  # set input size
num_classes = len(np.unique(Y))   # set class size
print('Input shape:', input_shape)
model = CNN_model(input_shape, num_classes)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
# Reduce learning rate when the loss has stopped improving for a certain number of epochs.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5)

# Convert NumPy arrays to PyTorch tensors with the correct data type
print('Converting to PyTorch tensors ... \n')
x_train_tensor = torch.from_numpy(x_train).float()
x_valid_tensor = torch.from_numpy(x_valid).float()
x_test_tensor = torch.from_numpy(x_test).float()

y_train_tensor = torch.from_numpy(y_train).long()
y_valid_tensor = torch.from_numpy(y_valid).long()
y_test_tensor = torch.from_numpy(y_test).long()

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Check if GPU is available
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU.')
    model.cuda()  # move the model parameters to GPU
else:
    print('No GPU available, training on CPU.')


# Assuming you have already defined model, criterion, optimizer,
# train_loader, valid_loader, and have a `train_on_gpu` flag.

# 2. Set up the training loop
history = {'epoch': [], 'train_loss': [],
           'val_loss': [], 'train_acc': [], 'val_acc': []}

n_epochs = 150
valid_loss_min = np.inf  # track change in validation loss


for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
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

        # FIX: Permute the data to match the (batch, channels, sequence_length) format
        data = data.permute(0, 2, 1)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        with torch.no_grad():
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100 * correct_train / total_train
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)

    #####################
    # validate the model #
    #####################
    model.eval()  # prep model for evaluation
    correct_valid = 0
    total_valid = 0
    running_val_loss = 0
    with torch.no_grad():
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # This line was already correct
            data = data.permute(0, 2, 1)
            output = model(data)

            # FIX: Accumulate loss weighted by the number of samples in the batch
            val_loss_item = criterion(output, target).item()
            running_val_loss += val_loss_item * data.size(0)

            _, predicted = torch.max(output.data, 1)
            total_valid += target.size(0)
            correct_valid += (predicted == target).sum().item()

    # FIX: Use the correct DataLoader to get the total number of samples
    val_loss = running_val_loss / len(valid_loader.dataset)
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


def plot_history(history):
    loss = history['train_loss']
    val_loss = history['val_loss']
    acc = history['train_acc']
    val_acc = history['val_acc']

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


plot_history(history)

# # Save the model checkpoint
# torch.save(model.state_dict(),
#            'Emotion_Classification/model_cnn_emotionclassifier.pt')
