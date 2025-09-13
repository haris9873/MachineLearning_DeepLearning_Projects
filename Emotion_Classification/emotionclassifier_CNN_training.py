import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from time import time
from torch.optim.lr_scheduler import StepLR

""""

Datset publicly available through Kaggle at: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data

"""

data_path = 'Emotion_Classification/emotiondataset'

datasets = os.listdir(data_path)

# List class folders in each split
for split in datasets:
    split_path = os.path.join(data_path, split)
    classes_tt = os.listdir(split_path)
    print(f"{split} classes:", classes_tt)

    # How many images at each Class
    counts = {cls: len(os.listdir(os.path.join(split_path, cls)))
              for cls in classes_tt}
    print(f"{split} samples per class:", counts)
    print("-"*50)

# Visualize a few images
train_path = os.path.join(data_path, 'train')

classes = os.listdir(train_path)

print(len(classes), "classes in total:", classes)
# 7 Random Images
fig, axes = plt.subplots(1, len(classes), figsize=(22, 8))
plt.suptitle("7 Random Images from Training Data")
for ax in axes:
    cls = np.random.choice(classes)
    print(cls)
    img_path = os.path.join(train_path, cls, np.random.choice(
        os.listdir(os.path.join(train_path, cls))))
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.set_title(cls)
# plt.show()


train_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),  # convert to grayscale for data standardization
                                       # resize any image to standard 48x48 of the dataet (for any outliers )
                                       transforms.Resize((48, 48)),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomRotation(10),
                                       transforms.ToTensor()])


# Print the class names that ImageFolder found

train_dataset = ImageFolder(
    root=train_path, transform=train_transforms)

# split dataset into train and validation sets
print('\nSplitting dataset into 80% train and 20% validation sets\n')
# obtain training indices that will be used for validation
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 128
# percentage of training set to use as validation
valid_size = 0.2

num_train = len(train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers=num_workers)


"""

    Define the CNN Architecture

"""


class EmotionClassifier(nn.Module):
    def __init__(self):
        # call the constructor of the parent class
        super(EmotionClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            # input = 1, output = 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 2),  # Downsample: 48×48 -> 24×24

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 2),  # Downsample: 24×24 -> 12×12

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 2),  # Downsample: 12×12 -> 6×6

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2, 2)   # Downsample: 6×6 -> 3×3
        )
        """
        
        Define a Fully Connected CNN Layer of 256 * 3 * 3 (input features) -> 128 (hidden layer) -> 7 (output classes)
        
        """
        self.fc_layers = nn.Sequential(
            # input = 256*3*3, hidden layer = 128 neurons
            nn.Linear(256*3*3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            # 2nd hidden layer
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            # output layer
            nn.Linear(128, 7)  # 127 (hidden layer) → 7 (output classes)

        )

    def forward(self, x):
        x = self.conv_layers(x)  # Feature extraction
        x = x.view(x.size(0), -1)  # Flatten features
        x = self.fc_layers(x)  # Classification
        return x


model = EmotionClassifier()
print('\nModel:', model)

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU.')
    model.cuda()  # move the model parameters to GPU
else:
    print('No GPU available, training on CPU.')

"""

Define the loss function, optimizer, learning rate scheduler, and initialize variables

"""

criterion = nn.CrossEntropyLoss()  # for multiclass classification
optimizer = optim.Adam(model.parameters(), lr=10e-3)
# Reduce learning rate when the loss has stopped improving for a certain number of epochs.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5)

# Initialize variables

history = {'epoch': [], 'train_loss': [], 'valid_loss': [], 'valid_acc': []}
"""

Train the model

"""
n_epochs = 30
valid_loss_min = np.inf  # track change in validation loss

time_start = time()
for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    ###################
    # train the model #
    ###################
    model.train()  # prep model for training
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()   # clear the gradients of all optimized variables

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        loss = criterion(output, target)   # calculate the batch loss

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        optimizer.step()  # perform a single optimization step (parameter update)
        train_loss += loss.item()*data.size(0)  # update training loss

    ######################
    # validate the model #
    ######################
    model.eval()  # prep model for evaluation
    correct_valid = 0
    total_valid = 0
    with torch.no_grad():
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # Get the predicted class with the highest score
            _, predicted = torch.max(output.data, 1)
            # Count the total number of samples
            total_valid += target.size(0)
            # Count the number of correct predictions
            correct_valid += (predicted == target).sum().item()

            loss = criterion(output, target)   # calculate the batch loss
            valid_loss += loss.item()*data.size(0)  # update average validation loss

    # calculate average losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    # Calculate and print the accuracy
    valid_accuracy = 100 * correct_valid / total_valid
    print('\nValidation done .... \n')
    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.2f}%'.format(
        epoch, train_loss, valid_loss, valid_accuracy))
    history['train_loss'].append(train_loss)
    history['valid_loss'].append(valid_loss)
    history['epoch'].append(epoch)
    history['valid_acc'].append(valid_accuracy)
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(),
                   'Emotion_Classification/model_cnn_emotionclassifier.pt')
        valid_loss_min = valid_loss

    # Update the learning rate based on validation loss

    scheduler.step(valid_loss)


time_end = time()
print('Total training time: {:.0f} minutes'.format(
    (time_end-time_start)/60))

print('\t{} \t{} \t {} \t {}'.format(
    history["epoch"], history["train_loss"], history["valid_loss"], history["valid_acc"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy Curves
axes[0].plot(history["train_acc"], label="train_acc")
axes[0].plot(history["val_acc"],   label="val_acc")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Accuracy Curves")
axes[0].legend()

#  Loss Curves
axes[1].plot(history["train_loss"], label="train_loss")
axes[1].plot(history["val_loss"],   label="val_loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("Loss Curves")
axes[1].legend()

plt.tight_layout()
plt.show()
