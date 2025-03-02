import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

def low_pass_filter(signal, cutoff=50, fs=360, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

train_data = np.loadtxt("mitbih_test.csv", delimiter=",")

X_train, y_train = train_data[:, :-1], train_data[:, -1].astype(int)

X_train_filtered = np.array([low_pass_filter(sig) for sig in X_train])

label_map = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
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

plot_label_distribution(y_train, "Distribution of Heartbeat Labels in ECG Dataset")

numeric_labels = np.array([label_map[label] for label in y_train])
ros = RandomOverSampler(random_state=42)
heartbeats_resampled, labels_resampled = ros.fit_resample(X_train_filtered, numeric_labels)
labels_resampled = np.array([inv_label_map[label] for label in labels_resampled])

plot_label_distribution(labels_resampled, "Distribution of Heartbeat Labels After Resampling")

