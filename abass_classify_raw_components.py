import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.utils import resample
import pandas as pd
import seaborn as sns


# === CONFIG ===
subs = ['sub11'] #
base_path = r"C:\shared\abass"
stimulus_labels = ['Left', 'Right', 'Up', 'Down']

# Create a 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, sub in enumerate(subs):
    print(f"\n==== Subject: {sub} ====")
    data_path = os.path.join(base_path, f"{sub}/{sub}_ica_epochs_by_class.npz")
    data = np.load(data_path)
    class_data = [data['left'], data['right'], data['up'], data['down']]
    
    timepoints = class_data[0].shape[2]
    time_axis = np.linspace(-0.5, 0.5, timepoints)
    start_idx = np.argmin(np.abs(time_axis - (-0.1)))
    end_idx = np.argmin(np.abs(time_axis - 0.4))

    X_by_class = [[] for _ in range(4)]
    y_by_class = [[] for _ in range(4)]

    for label_idx, stim_data in enumerate(class_data):
      n_trials = stim_data.shape[1]
      trials = np.transpose(stim_data[:, :, start_idx:end_idx], (1, 0, 2))  # shape: [n_trials, n_components, n_timepoints]
      
      # === Normalize each trial: z-score across time for each component ===
      for i in range(n_trials):
          trials[i] = (trials[i] - np.mean(trials[i], axis=1, keepdims=True)) / np.std(trials[i], axis=1, keepdims=True)

      trials_flat = trials.reshape(n_trials, -1)
      X_by_class[label_idx] = trials_flat
      y_by_class[label_idx] = [label_idx] * n_trials


    min_trials = min(len(y_list) for y_list in y_by_class)

    X_balanced = []
    y_balanced = []

    for i in range(4):
        X_down, y_down = resample(
            X_by_class[i], y_by_class[i],
            replace=False,
            n_samples=min_trials,
            random_state=42
        )
        X_balanced.append(X_down)
        y_balanced.append(y_down)

    X = np.vstack(X_balanced)
    y = np.hstack(y_balanced)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=50),
        SVC(kernel='rbf', C=1.0, gamma='scale', probability=False)
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=stimulus_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(report_df.round(2))

    cm = confusion_matrix(y_test, y_pred)

    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt="d", cmap='jet',
                xticklabels=stimulus_labels,
                yticklabels=stimulus_labels,
                ax=ax,
                cbar=(idx == 0))  # Show colorbar only on first plot

    ax.set_title(f"{sub} - SVM Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.show()
