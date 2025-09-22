from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import random


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
    plt.savefig('EMG_GestureRecognition_LSTM/Results/training_history.png')
    plt.show()

# Define a bandpass filter


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data, axis=0)  # Apply filter column-wise
    return y

# Function to create sliding window sequences


def create_sequences(data, labels, window_size, stride):
    X = []
    y = []
    for i in range(0, len(data) - window_size + 1, stride):
        X.append(data[i:i + window_size, :])
        y.append(labels[i + window_size - 1])
    return np.array(X), np.array(y)

# --- 2. Custom PyTorch Dataset Class ---


class EMGDataset(Dataset):
    def __init__(self, features, classes):
        # Convert numpy arrays to PyTorch tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.classes = torch.tensor(classes, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.classes[idx]

# --- 3. Model Definition: The LSTM ---


class EMGClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.25):
        super(EMGClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_prob)

        # A fully connected block to handle the final classification
        self.fc_layers = nn.Sequential(
            # Apply dropout after the LSTM output
            nn.Dropout(dropout_prob),

            # First hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            # Final output layer
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # Pass data through the LSTM
        # The LSTM's built-in dropout is active during training
        out, (hn, cn) = self.lstm(x)

        # Take the output from the last time step
        out = out[:, -1, :]

        # Pass to the fully connected layers
        out = self.fc_layers(out)

        return out


def test_model():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in test_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_true.extend(target.cpu().numpy().tolist())
        test_acc = (np.array(all_preds) == np.array(all_true)).mean()
        print('Test Accuracy of the model on the test images: {} %'.format(
            100 * correct / total))
    return


if __name__ == '__main__':

    data_dir = 'EMG_GestureRecognition_LSTM'
    df = pd.read_csv('EMG_GestureRecognition_LSTM/EMG-data.csv')

    print(df.head())
    print(df.shape)
    print("Value Count :\n", df["class"].value_counts())
    """
    ############################################################
    Class the label of gestures:
    0  unmarked data,
    1  hand at rest,
    2  hand clenched in a fist,
    3  wrist flexion,
    4  wrist extension,
    5  radial deviations,
    6  ulnar deviations,
    7  extended palm (the gesture was not performed by all subjects).
    ############################################################
    """

    # Drop the label and time columns as not required
    df = df.drop(columns=['label', 'time'], axis=1)
    # drop unmarked data and extended palm as low sample count ruining the model
    df = df[~df['class'].isin([0, 7])]
    # Reset the index, dropping the old index to create a new one
    df = df.reset_index(drop=True)
    print(df.head())
    classes = df['class']

    emg_data = df.drop(columns=['class'], axis=1)
    print(emg_data.head())
    print(classes.value_counts())

    fs = 200  # Sampling frequency (Hz), for MYO Thalmic bracelet
    print(f"Using a sampling frequency (fs) of: {fs} Hz")

    """
    Motion Artifacts: Low-frequency noise caused by the movement of the electrodes on the skin.
    This can interfere with the signal you are trying to measure.

    Power Line Interference: High-frequency noise from nearby electrical devices, typically at 50 or 60 Hz.

    Inherent Electronic Noise: Noise from the device itself or the environment.

    """
    lowcut = 10.0
    highcut = 90.0

    # Apply the filter to each channel
    emg_filtered = pd.DataFrame()
    for col in emg_data.columns:
        emg_filtered[col] = butter_bandpass_filter(
            emg_data[col], lowcut, highcut, fs)

    # Rectify the signal by taking the absolute value
    emg_rectified = emg_filtered.abs()

    # --- 3. Normalization ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    emg_scaled = scaler.fit_transform(emg_rectified)

    # --- 4. Segmentation (Windowing) ---
    # Define window size and stride based on the fixed 3-second gesture duration
    window_size = 3 * fs  # 3 seconds * 200 samples/second = 600 samples
    stride = 100  # Example: 50% overlap
    print(f"Shape of emg_data: {emg_data.shape}")
    print(f"Shape of labels: {classes.shape}")

    X_sequences, y_labels = create_sequences(
        emg_scaled, classes, window_size, stride)

    # Check the shape to confirm it's in (samples, timesteps, features) format
    print("Shape of X_sequences (samples, timesteps, features):", X_sequences.shape)
    print("Shape of y_labels:", y_labels.shape)

    # map labels to integers
    unique_labels = np.unique(y_labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_int_labels = np.array([label_map[label] for label in y_labels])

    # --- 6. Splitting the data ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_sequences, y_int_labels, test_size=0.3, random_state=42, stratify=y_int_labels
    )

    # split the temporary set into validation and test
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print("\nFinal data shapes for the LSTM model:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

 # Instantiate the Datasets and DataLoaders
    train_dataset = EMGDataset(X_train, y_train)
    valid_dataset = EMGDataset(X_valid, y_valid)
    test_dataset = EMGDataset(X_test, y_test)

    batch_size = 64
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # --- 5. Training the Model ---

    # Hyperparameters
    input_size = X_train.shape[2]  # Number of features (channels)
    hidden_size = 256
    num_layers = 2
    num_classes = len(unique_labels)
    learning_rate = 0.0001
    num_epochs = 50

    model = EMGClassifier(input_size, hidden_size,
                          num_layers, num_classes)

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('Training on GPU.')
        device = 'cuda'
        model.to(device)
    else:
        print('No GPU available, training on CPU.')
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        # L2 regularization and Adam optimizer
        model.parameters(), lr=learning_rate, weight_decay=0.0001)
    # Reduce learning rate when the loss has stopped improving for a certain number of epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    """
    ########################################################################################
    Start the training process
    ########################################################################################
    """

    # 2. Set up the training loop
    history = {'epoch': [], 'train_loss': [],
               'val_loss': [], 'train_acc': [], 'val_acc': []}

    n_epochs = 30
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

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            with torch.no_grad():  # turn off autograd for evaluation
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
            os.makedirs('EMG_GestureRecognition_LSTM/Results', exist_ok=True)
            torch.save(model.state_dict(),
                       'EMG_GestureRecognition_LSTM/model_LSTM_emggesturerecognition.pt')
            valid_loss_min = val_loss
        # Update the learning rate based on validation loss
        scheduler.step(val_loss)

    plot_history(history)
    all_preds, all_true = [], []
    model.load_state_dict(torch.load(
        'EMG_GestureRecognition_LSTM/model_LSTM_emggesturerecognition.pt'))

    print('\nTesting the Model')
    test_model()
    class_names = ['hand at rest',
                   'hand clenched in a fist', 'wrist flexion', 'wrist extension', 'radial deviations', 'ulnar deviations']
    # Classification Report
    print("\nClassification report:\n")
    print(classification_report(all_true, all_preds, target_names=class_names))
    report = classification_report(
        all_true, all_preds, target_names=class_names, output_dict=True
    )
    # Convert the report dictionary to a pandas DataFrame
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(
        'EMG_GestureRecognition_LSTM/Results/Classification_Report.csv')

    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(6, 5))
    # Using a color map for better visualization
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig('EMG_GestureRecognition_LSTM/Results/confusion_matrix.png')
    plt.show()
