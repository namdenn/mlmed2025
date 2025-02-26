import wfdb
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as pplt

record = wfdb.rdrecord("mit-bih-arrhythmia-database-1.0.0/233")
annotation = wfdb.rdann("mit-bih-arrhythmia-database-1.0.0/233", "atr")

filtered_signal = record.p_signal[:, 0]

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

df = pd.DataFrame(heartbeats)
df["Label"] = labels  

df.to_csv("before_filtered_heartbeats.csv", index=False)

print(f"Total filtered heartbeats: {len(df)}")

num_plots = 1
pplt.figure(figsize=(10, 6))

for i in range(num_plots):
    pplt.plot(heartbeats[i], label=f"Heartbeat {i+1} - Label: {labels[i]}", color = "red")

pplt.xlabel("Time (samples)")
pplt.ylabel("Amplitude")
pplt.title("Extracted Heartbeats from ECG Signal")
pplt.legend()
pplt.grid()
pplt.show()
