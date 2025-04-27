import os
import numpy as np
import pandas as pd

# === CONFIG ===
base_path = r"C:\shared\abass"
subjects = ['sub13', 'sub11', 'sub9', 'sub8']

stimulus_labels = ['Left', 'Right', 'Up', 'Down']
metrics = ['precision', 'recall', 'f1-score', 'support']
deepconvnet_suffix = ''
eegnet_suffix = '_EEGNet'

# === Collect metric diffs ===
improvements = []

for sub in subjects:
    sub_dir = os.path.join(base_path, sub)

    # Paths to classification reports
    dc_path = os.path.join(sub_dir, f"{sub}{deepconvnet_suffix}_classification_report.csv")
    eeg_path = os.path.join(sub_dir, f"{sub}{eegnet_suffix}_classification_report.csv")

    if not (os.path.exists(dc_path) and os.path.exists(eeg_path)):
        print(f"Missing files for {sub}")
        continue

    try:
        df_dc = pd.read_csv(dc_path, index_col=0)
        df_eeg = pd.read_csv(eeg_path, index_col=0)

        # Compare per-stim metrics
        for stim in stimulus_labels:
            for metric in metrics:
                val_dc = df_dc.loc[stim, metric]
                val_eeg = df_eeg.loc[stim, metric]
                delta = val_dc - val_eeg
                improvements.append({
                    'Subject': sub,
                    'Stimulus': stim,
                    'Metric': metric,
                    'DeepConvNet': val_dc,
                    'EEGNet': val_eeg,
                    'Delta': delta
                })

        # Compare accuracy
        acc_dc = df_dc.loc['accuracy', 'f1-score']
        acc_eeg = df_eeg.loc['accuracy', 'f1-score']
        improvements.append({
            'Subject': sub,
            'Stimulus': 'All',
            'Metric': 'accuracy',
            'DeepConvNet': acc_dc,
            'EEGNet': acc_eeg,
            'Delta': acc_dc - acc_eeg
        })

    except Exception as e:
        print(f"Error processing {sub}: {e}")

# === Top 10 Improvements Table ===
df_improvements = pd.DataFrame(improvements)
df_sorted = df_improvements.sort_values(by='Delta', ascending=False).head(10)

# === Print as table ===
print("\n TOP 10 IMPROVEMENTS (DeepConvNet vs EEGNet)\n")
print(df_sorted[['Subject', 'Stimulus', 'Metric', 'EEGNet', 'DeepConvNet', 'Delta']].to_string(index=False, float_format="%.2f"))
