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

record = wfdb.rdrecord("mit-bih-arrhythmia-database-1.0.0/233")  
annotation = wfdb.rdann("mit-bih-arrhythmia-database-1.0.0/233", "atr")

def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=360, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

filtered_signal = bandpass_filter(record.p_signal[:, 0])  

def extract_heartbeats(signal, annotations, window_size=324):
    beats = []
    labels = []
    
    for i, r_peak in enumerate(annotations.sample):
        start = max(0, r_peak - 144)
        end = min(len(signal), r_peak + 180)
        heartbeat = signal[start:end]
        
        if len(heartbeat) == window_size:
            beats.append(heartbeat)
            labels.append(annotations.symbol[i]) 
            
    return np.array(beats), np.array(labels)

heartbeats, labels = extract_heartbeats(filtered_signal, annotation)
print(np.unique(labels))

df = pd.DataFrame(heartbeats)
df["Label"] = labels
df.to_csv("filtered_heartbeats.csv", index=False)

import matplotlib.pyplot as plt

num_plots = 1 # Plot up to 5 heartbeats

plt.figure(figsize=(10, 6))

for i in range(num_plots):
    plt.plot(heartbeats[i], label=f"Heartbeat {i+1} - Label: {labels[i]}")

plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.title("Extracted Heartbeats from ECG Signal")
plt.legend()
plt.grid()
plt.show()
