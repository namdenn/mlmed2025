import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

record = wfdb.rdrecord("mit-bih-arrhythmia-database-1.0.0/100")
annotation = wfdb.rdann("mit-bih-arrhythmia-database-1.0.0/100", "atr")

def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=360, order=3):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)

filtered_signal = bandpass_filter(record.p_signal[:, 0])

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
print("Unique Labels:", np.unique(labels))

label_map = {'A': 0, 'N': 1, 'V': 2}
inv_label_map = {v: k for k, v in label_map.items()}

def plot_label_distribution(labels, title):
    unique_labels, counts = np.unique(labels, return_counts=True)
    df = pd.DataFrame({"Label": unique_labels, "Count": counts})

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Label", y="Count", data=df, palette="viridis")

    for index, row in df.iterrows():
        plt.text(index, row.Count + 2, str(row.Count), ha='center', fontsize=12)

    plt.xlabel("Heartbeat Type")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

plot_label_distribution(labels, "Distribution of Heartbeat Labels in ECG Dataset")

numeric_labels = np.array([label_map[label] for label in labels])
ros = RandomOverSampler(random_state=42)
heartbeats_resampled, labels_resampled = ros.fit_resample(heartbeats, numeric_labels)
labels_resampled = np.array([inv_label_map[label] for label in labels_resampled])

plot_label_distribution(labels_resampled, "Distribution of Heartbeat Labels After Resampling")

