import wfdb
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as pplt
from scipy.signal import butter, filtfilt

def low_pass_filter(signal, cutoff=50, fs=360, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

train_data = np.loadtxt("mitbih_train.csv", delimiter=",")
test_data = np.loadtxt("mitbih_test.csv", delimiter=",")

X_train, y_train = train_data[:, :-1], train_data[:, -1].astype(int)
X_test, y_test = test_data[:, :-1], test_data[:, -1].astype(int)

X_train_filtered = np.array([low_pass_filter(sig) for sig in X_train])
X_test_filtered = np.array([low_pass_filter(sig) for sig in X_test])

label_map = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
inv_label_map = {v: k for k, v in label_map.items()}

df = pd.DataFrame()
df["Label"] = y_train

df.to_csv("before_filtered_heartbeats.csv", index=False)

print(f"Total filtered heartbeats: {len(df)}")

num_plots = 1
pplt.figure(figsize=(10, 6))

for i in range(num_plots):
    pplt.plot(X_train_filtered[i][:100], label=f"Heartbeat {i+1} - Label: {y_train[i]}", color = "red")

pplt.xlabel("Time (samples)")
pplt.ylabel("Amplitude")
pplt.title("Extracted Heartbeats from ECG Signal")
pplt.legend()
pplt.grid()
pplt.show()
