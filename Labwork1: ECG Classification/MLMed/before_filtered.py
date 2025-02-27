import wfdb
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as pplt

train_data = np.loadtxt("mitbih_train.csv", delimiter=",")
test_data = np.loadtxt("mitbih_test.csv", delimiter=",")

X_train, y_train = train_data[:, :-1], train_data[:, -1].astype(int)
X_test, y_test = test_data[:, :-1], test_data[:, -1].astype(int)

label_map = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
inv_label_map = {v: k for k, v in label_map.items()}

df = pd.DataFrame()
df["Label"] = y_train

df.to_csv("before_filtered_heartbeats.csv", index=False)

print(f"Total filtered heartbeats: {len(df)}")

num_plots = 1
pplt.figure(figsize=(10, 6))

for i in range(num_plots):
    pplt.plot(X_train[i][:100], label=f"Heartbeat {i+1} - Label: {y_train[i]}", color = "red")

pplt.xlabel("Time (samples)")
pplt.ylabel("Amplitude")
pplt.title("Extracted Heartbeats from ECG Signal")
pplt.legend()
pplt.grid()
pplt.show()
