import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Settings ===
base_path = r"C:\shared\abass"
subs = ['sub8', 'sub9', 'sub11', 'sub13']
stimulus_labels = ['Left', 'Right', 'Up', 'Down']

# === Storage for results ===
badc_cms = []
normal_cms = []
badc_f1s = []
normal_f1s = []

# === Load and store ===
for sub in subs:
    sub_dir = os.path.join(base_path, sub)

    # Paths
    badc_report_path = os.path.join(sub_dir, f"{sub}badc_classification_report.json")
    badc_cm_path = os.path.join(sub_dir, f"{sub}badc_confusion_matrix.npy")
    normal_report_path = os.path.join(sub_dir, f"{sub}_classification_report.json")
    normal_cm_path = os.path.join(sub_dir, f"{sub}_confusion_matrix.npy")

    # Check existence
    if not (os.path.exists(badc_report_path) and os.path.exists(normal_report_path)):
        print(f"Missing classification reports for {sub}. Skipping.")
        continue
    if not (os.path.exists(badc_cm_path) and os.path.exists(normal_cm_path)):
        print(f"Missing confusion matrices for {sub}. Skipping.")
        continue

    # Load reports
    badc_report = pd.read_json(badc_report_path, orient='split')
    normal_report = pd.read_json(normal_report_path, orient='split')

    # Load confusion matrices
    badc_cm = np.load(badc_cm_path)
    normal_cm = np.load(normal_cm_path)

    badc_cms.append(badc_cm)
    normal_cms.append(normal_cm)

    # Extract F1 scores for stimulus labels
    badc_f1s.append(badc_report.loc[stimulus_labels, 'f1-score'].values)
    normal_f1s.append(normal_report.loc[stimulus_labels, 'f1-score'].values)

# Convert lists to numpy arrays
badc_cms = np.array(badc_cms)
normal_cms = np.array(normal_cms)
badc_f1s = np.array(badc_f1s)
normal_f1s = np.array(normal_f1s)

# === Compute mean confusion matrices ===
mean_badc_cm = np.mean(badc_cms, axis=0)
mean_normal_cm = np.mean(normal_cms, axis=0)

# === Plot confusion matrices side-by-side ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

vmax = max(mean_badc_cm.max(), mean_normal_cm.max())

sns.heatmap(mean_normal_cm, annot=True, fmt='.1f', cmap='Blues', vmin=0, vmax=vmax,
            xticklabels=stimulus_labels, yticklabels=stimulus_labels, ax=axes[0])
axes[0].set_title('Mean Confusion Matrix - Restricted ICA')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

sns.heatmap(mean_badc_cm, annot=True, fmt='.1f', cmap='Blues', vmin=0, vmax=vmax,
            xticklabels=stimulus_labels, yticklabels=stimulus_labels, ax=axes[1])
axes[1].set_title('Mean Confusion Matrix - Lenient ICA (badc_)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.tight_layout()
plt.show()

# === Compute mean and std F1 scores ===
mean_normal_f1 = np.mean(normal_f1s, axis=0)
std_normal_f1 = np.std(normal_f1s, axis=0)

mean_badc_f1 = np.mean(badc_f1s, axis=0)
std_badc_f1 = np.std(badc_f1s, axis=0)

# === Plot F1 score comparison across stimulus types ===
fig, ax = plt.subplots(figsize=(8,5))

x = np.arange(len(stimulus_labels))
width = 0.35

ax.bar(x - width/2, mean_normal_f1, width, yerr=std_normal_f1, capsize=5, label='Restricted ICA')
ax.bar(x + width/2, mean_badc_f1, width, yerr=std_badc_f1, capsize=5, label='Lenient ICA (badc_)')

ax.set_ylabel('F1 Score')
ax.set_title('Mean F1 Score per Stimulus Across Subjects')
ax.set_xticks(x)
ax.set_xticklabels(stimulus_labels)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
