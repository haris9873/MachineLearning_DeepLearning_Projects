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
import random
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

model.load_state_dict(torch.load(
    'Emotion_Classification/model_cnn_emotionclassifier.pt'))

train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('Training on GPU.')
    device = 'cuda'
    model.cuda()
else:
    print('No GPU available, training on CPU.')
    device = 'cpu'

test_path = 'Emotion_Classification/emotiondataset/test'
test_transforms = transforms.Compose([
    # convert to grayscale for data standardization
    transforms.Grayscale(num_output_channels=1),
    # resize any image to standard 48x48 of the dataet
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

class_names = ['Angry', 'Disgust', 'Fear',
               'Happy', 'Neutral', 'Sad', 'Surprise']
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 128

test_dataset = ImageFolder(test_path, transform=test_transforms)

test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         num_workers=num_workers)

all_preds, all_true = [], []


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


print('\nTesting the Model')
test_model()

# Classification Report
print("\nClassification report:\n")
print(classification_report(all_true, all_preds, target_names=class_names))


# Confusion Matrix
cm = confusion_matrix(all_true, all_preds)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
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
        img = imgs[i].squeeze(0).cpu().numpy()       # (48,48)
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
    plt.show()


show_random_test_predictions(model, test_dataset)
