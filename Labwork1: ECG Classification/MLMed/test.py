import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from pyts.image import GramianAngularField
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from scipy.signal import butter, filtfilt

def low_pass_filter(signal, cutoff=50, fs=360, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

train_data = np.loadtxt("/kaggle/input/heartbeat/mitbih_train.csv", delimiter=",")
test_data = np.loadtxt("/kaggle/input/heartbeat/mitbih_test.csv", delimiter=",")

X_train, y_train = train_data[:, :-1], train_data[:, -1].astype(int)
X_test, y_test = test_data[:, :-1], test_data[:, -1].astype(int)

X_train_filtered = np.array([low_pass_filter(sig) for sig in X_train])
X_test_filtered = np.array([low_pass_filter(sig) for sig in X_test])

X_train_raw, X_val, y_train_raw, y_val = train_test_split(
    X_train_filtered, y_train, test_size=0.2, random_state=42, stratify=y_train
)

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_raw, y_train_raw)

gaf = GramianAngularField(image_size=64)
X_train_gaf = gaf.fit_transform(X_train_resampled)
X_val_gaf = gaf.fit_transform(X_val)
X_test_gaf = gaf.fit_transform(X_test_filtered)

X_train_tensor = torch.tensor(X_train_gaf, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val_gaf, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_gaf, dtype=torch.float32).unsqueeze(1)

y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Ensure correct shape
        self.fc2 = nn.Linear(128, 5)  
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
optimizer = optim.Adam(model.parameters(), lr=0.001)

patience = 5
best_val_loss = float("inf")
patience_counter = 0

num_epochs = 50
for epoch in range(num_epochs):
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
    
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth") 
        
model.load_state_dict(torch.load("best_model.pth"))

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

label_map = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=label_map.values()))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.values(), yticklabels=label_map.values())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

