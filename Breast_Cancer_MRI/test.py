import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import random


data_path = 'Breast_Cancer_MRI/mridataset'

datasets = os.listdir(data_path)

test_path = os.path.join(data_path, 'test')

# Print the class names that ImageFolder found
norm_std_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),  # convert to grayscale for data standardization
                                      # resize any image to standard 540x250 of the dataet (for any outliers )
                                      transforms.Resize((540, 250)),
                                      transforms.ToTensor(),
                                      ])


# Print the class names that ImageFolder found
test_dataset = ImageFolder(
    root=test_path, transform=test_transforms)

class_names = test_dataset.classes

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 32

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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


model = MRI_CNN()
model.load_state_dict(torch.load(
    'Breast_Cancer_MRI/model_cnn_breastcancermri.pt'))
criterion = nn.CrossEntropyLoss()


all_preds, all_true = [], []

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Testing on GPU.')
    device = 'cuda'
    model.cuda()
else:
    print('No GPU available, training on CPU.')
    device = 'cpu'


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

        print('Test Accuracy of the model on the test images: {} %'.format(
            100 * correct / total))

    return


print('\nTesting the Model')
test_model()

# Classification Report
print("\nClassification report:\n")
print(classification_report(all_true, all_preds, target_names=class_names))
report = classification_report(
    all_true, all_preds, target_names=class_names, output_dict=True
)
# Convert the report dictionary to a pandas DataFrame
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('Breast_Cancer_MRI/Results/Classification_Report.csv')

# Confusion Matrix
cm = confusion_matrix(all_true, all_preds)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
# Add numerical values to each cell of the confusion matrix
# The threshold is used to decide if the text should be black or white for readability
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
os.makedirs('Breast_Cancer_MRI/Results', exist_ok=True)
plt.savefig('Breast_Cancer_MRI/Results/Confusion_Matrix.png')
plt.show()


def show_random_test_predictions(model, dataset, n=8):
    idxs = random.sample(range(len(dataset)), n)
    # list of (C,H,W) tensors + int labels
    imgs, labels = zip(*[dataset[i] for i in idxs])

    batch = torch.stack(imgs).to(device)              # [n,1,48,48]
    labels_t = torch.tensor(labels).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    # plot
    rows, cols = 2, n // 2 if n % 2 == 0 else (n // 3 + 1)
    cols = n // rows + (n % rows > 0)
    plt.figure(figsize=(3*cols, 3*rows))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        img = imgs[i].squeeze(0).cpu().numpy()
        plt.imshow(img, cmap="gray")
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i].item()]
        color = "green" if true_label == pred_label else "red"
        plt.title(f"P: {pred_label}\nT: {true_label}",
                  color=color, fontsize=10)
        plt.axis("off")
    plt.suptitle(
        "Random Test Images: Predictions vs Ground Truth", fontsize=14)
    plt.tight_layout()
    os.makedirs('Breast_Cancer_MRI/Results', exist_ok=True)
    plt.savefig('Breast_Cancer_MRI/Results/random_test_images.png')
    plt.show()


show_random_test_predictions(model, test_dataset)
