import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
import seaborn as sns

def low_pass_filter(signal, cutoff=50, fs=360, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

train_data = np.loadtxt("mitbih_train.csv", delimiter=",")

X_train = train_data[:, :-1]

X_train_filtered = np.array([low_pass_filter(sig) for sig in X_train])

gaf = GramianAngularField(image_size=64)
ecg_images = gaf.fit_transform(X_train_filtered)

selected_rows = [2, 3, 36, 310]
fig, axes = plt.subplots(1, len(selected_rows), figsize=(15, 5))

for idx, row in enumerate(selected_rows):
    axes[idx].imshow(ecg_images[row], cmap="rainbow")
    axes[idx].set_title(f"ECG GAF Image {row}")
    axes[idx].axis("off")

plt.show()