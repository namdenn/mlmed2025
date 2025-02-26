import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField

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

# annotate 
filtered_indices = [i for i, label in enumerate(labels) if label not in ['+', '|']]
heartbeats = heartbeats[filtered_indices]
labels = labels[filtered_indices]

label_map = {'A':0, 'F':1, 'N':2, 'V':3}  

inv_label_map = {v: k for k, v in label_map.items()}
numeric_labels = np.array([label_map[label] for label in labels])

df = pd.DataFrame(heartbeats)
df["Label"] = labels
df.to_csv("filtered_heartbeats.csv", index=False)

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
