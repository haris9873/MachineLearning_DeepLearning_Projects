from torch.utils.data import random_split
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data.sampler import SubsetRandomSampler

""""

Datset publicly available through Kaggle at: https://www.kaggle.com/datasets/uzairkhan45/breast-cancer-patients-mris/data

"""


data_path = 'Breast_Cancer_MRI/mridataset'
os.makedirs('Breast_Cancer_MRI/Results', exist_ok=True)
datasets = os.listdir(data_path)

# List class folders in each split
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

# 7 Random Images
fig, axes = plt.subplots(1, len(classes), figsize=(22, 8))
plt.suptitle("Random Images from Training Data")
for ax in axes:
    cls = np.random.choice(classes)
    print(cls)
    img_path = os.path.join(train_path, cls, np.random.choice(
        os.listdir(os.path.join(train_path, cls))))
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.set_title(cls)
plt.show()
"""

Performing Data Augmentation


"""

train_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),  # convert to grayscale for data standardization
                                       # resize any image to standard 540x250 of the dataet (for any outliers )
                                       transforms.Resize(
                                           (540, 250), antialias=True),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       #    transforms.RandomPerspective(
                                       #        distortion_scale=0.2, p=0.5),
                                       #    transforms.RandomAffine(degrees=10),

                                       transforms.RandomRotation(15),
                                       transforms.ToTensor()])


"""
##############################################
    Split Data in train and validation
##############################################
"""


train_dataset = ImageFolder(
    root=train_path, transform=train_transforms)
# split dataset into train and validation sets
print('\nSplitting dataset into 80% train and 20% validation sets\n')

# Define the split sizes
valid_size = 0.2
total_size = len(train_dataset)
valid_samples = int(valid_size * total_size)
train_samples = total_size - valid_samples

# separate datasets
train_dataset, valid_dataset = random_split(
    train_dataset, [train_samples, valid_samples])

# create the data loaders
batch_size = 32
num_workers = 0

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# The size of the datasets will now be different
print(f"Total training samples: {len(train_dataset)}")
print(f"Total validation samples: {len(valid_dataset)}")

"""
##############################################
    Define the CNN Architecture
##############################################
"""


class GaussianNoise(nn.Module):
    """
    Equivalent to Keras's GaussianNoise layer.
    """

    def __init__(self, sigma=0.1, is_training=True):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        self.is_training = is_training

    def forward(self, x):
        if self.training and self.is_training:
            # Generate noise with the same shape as the input
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


class MRI_CNN(nn.Module):
    def __init__(self):
        super(MRI_CNN, self).__init__()
        self.conv_layers = nn.Sequential(

            # input = 1 (for grayscale image), output = 32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2, 2),  # Downsample: 540x250 -> 270x125

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2),  # Downsample: 270x125 -> 135x62

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2),  # Downsample: 135x62 -> 67x31

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.35),
            nn.MaxPool2d(2, 2)   # Downsample: 67x31 -> 33x15
        )
        flattened_size = 256 * (33 * 15)

        self.fc_layers = nn.Sequential(

            GaussianNoise(0.25),
            nn.Linear(flattened_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),  # dropout to prevent overfitting

            # 2nd hidden layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            # output layer

            nn.Linear(128, 2)  # 128 (hidden layer) → 2 (output classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Feature extraction
        x = x.view(x.size(0), -1)  # Flatten features
        x = self.fc_layers(x)  # Classification
        return x


"""
#####################################################
                CNN Architecture

        Input:  1 channel (grayscale image)
        Output: 2 classes (benign and malignant)

        Architecture:
input = 2159872 ----------> hidden layer 1 = 512 neurons ------> hidden layer 2 = 128 neurons ----> output = 2 classes

#####################################################
"""
model = MRI_CNN()
print(model)


"""
########################################################################################
Define the loss function, optimizer, learning rate scheduler, and initialize variables
########################################################################################
"""

criterion = nn.CrossEntropyLoss()  # loss function
optimizer = optim.Adam(model.parameters(), lr=0.0001,
                       weight_decay=1e-5)  # L2 regularization
# Reduce learning rate when the loss has stopped improving for a certain number of epochs.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5)

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU.')
    model.cuda()  # move the model parameters to GPU
else:
    print('No GPU available, training on CPU.')

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
        os.makedirs('Breast_Cancer_MRI/Results', exist_ok=True)
        torch.save(model.state_dict(),
                   'Breast_Cancer_MRI/model_cnn_breastcancermri.pt')
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
    plt.savefig('Breast_Cancer_MRI/Results/training_history.png')
    plt.show()


plot_history(history)
