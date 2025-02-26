import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.signal import butter, filtfilt
from imblearn.over_sampling import RandomOverSampler
from pyts.image import GramianAngularField
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report

record = wfdb.rdrecord("mit-bih-arrhythmia-database-1.0.0/233")
annotation = wfdb.rdann("mit-bih-arrhythmia-database-1.0.0/233", "atr")

# this filter used for remove noise
def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=360, order=3):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)
filtered_signal = bandpass_filter(record.p_signal[:, 0])

# extract heartbeats
def extract_heartbeats(signal, annotations, window_size=324):
    beats, labels = [], []
    for i, r_peak in enumerate(annotations.sample):
        start, end = max(0, r_peak - 144), min(len(signal), r_peak + 180)
        heartbeat = signal[start:end]
        if len(heartbeat) == window_size:
            beats.append(heartbeat)
            labels.append(annotations.symbol[i])
    return np.array(beats), np.array(labels)
heartbeats, labels = extract_heartbeats(filtered_signal, annotation)
# print(np.unique(labels))

filtered_indices = [i for i, label in enumerate(labels) if label not in ['+', '|']]
heartbeats = heartbeats[filtered_indices]
labels = labels[filtered_indices]

# Annotate label
label_map = {'A':0, 'F':1, 'N':2, 'V':3}  

inv_label_map = {v: k for k, v in label_map.items()}
numeric_labels = np.array([label_map[label] for label in labels])

# resample the dataset
ros = RandomOverSampler(random_state=42)
heartbeats_resampled, labels_resampled = ros.fit_resample(heartbeats, numeric_labels)
labels_resampled = np.array([inv_label_map[label] for label in labels_resampled])

# convert ecg to gaf image
gaf = GramianAngularField(image_size=64)
ecg_images = gaf.fit_transform(heartbeats_resampled)
ecg_images_tensor = torch.tensor(ecg_images, dtype=torch.float32).unsqueeze(1)
labels_tensor = torch.tensor([label_map[label] for label in labels_resampled], dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(ecg_images_tensor, labels_tensor, test_size=0.2, random_state=0)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32,64, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*16*16, 128)
        self.fc2 = nn.Linear(128, 4)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(5): 
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/10], Loss: {total_loss/len(train_loader):.4f}")

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)  
        
        all_preds.extend(preds.cpu().numpy())  
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

print("Classification Report:\n", classification_report(all_labels, all_preds))

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()