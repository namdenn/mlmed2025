import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
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

# annotate
filtered_indices = [i for i, label in enumerate(labels) if label not in ['+', '|']]
heartbeats = heartbeats[filtered_indices]
labels = labels[filtered_indices]

label_map = {'A':0, 'F':1, 'N':2, 'V':3}  

inv_label_map = {v: k for k, v in label_map.items()}
numeric_labels = np.array([label_map[label] for label in labels])


# # Convert ECG signals to GAF images
gaf = GramianAngularField(image_size=64)
ecg_images = gaf.fit_transform(heartbeats) 


selected_rows = [2,3,36,310]  # Indices of rows you want to plot
fig, axes = plt.subplots(1, len(selected_rows), figsize=(15, 5))

for idx, row in enumerate(selected_rows):
    axes[idx].imshow(ecg_images[row], cmap="rainbow")  
    axes[idx].set_title(f"ECG GAF Image {row}")
    axes[idx].axis("off")  

plt.show()