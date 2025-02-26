import numpy as np
import pandas as pd
import wfdb
import os
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

record = wfdb.rdrecord("mit-bih-arrhythmia-database-1.0.0/219")  
annotation = wfdb.rdann("mit-bih-arrhythmia-database-1.0.0/219", "atr")

def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=360, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

filtered_signal = bandpass_filter(record.p_signal[:, 0])  
# print(filtered_signal)

def extract_heartbeats(signal, annotations, window_size=324):
    beats = []
    labels = []
    
    for i, r_peak in enumerate(annotations.sample):
        start = max(0, r_peak - 144)
        end = min(len(signal), r_peak + 180)
        heartbeat = signal[start:end]
        
        if len(heartbeat) == window_size:
            beats.append(heartbeat)
            labels.append(annotations.symbol[i])  # Extract corresponding label
            
    return np.array(beats), np.array(labels)

heartbeats, labels = extract_heartbeats(filtered_signal, annotation)

# # Convert ECG signals to GAF images
gaf = GramianAngularField(image_size=64)
ecg_images = gaf.fit_transform(heartbeats) 


selected_rows = [1, 1253, 1269, 310, 10, 4]  # Indices of rows you want to plot
fig, axes = plt.subplots(1, len(selected_rows), figsize=(15, 5))

for idx, row in enumerate(selected_rows):
    axes[idx].imshow(ecg_images[row], cmap="rainbow")  
    axes[idx].set_title(f"ECG GAF Image {row}")
    axes[idx].axis("off")  

plt.show()