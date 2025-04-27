import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.utils import resample
import pandas as pd
import seaborn as sns

# === CONFIG ===
subs = ['sub13','sub11','sub8','sub9']
base_path = r"C:\shared\abass"
stimulus_labels = ['Left', 'Right', 'Up', 'Down']

for sub in subs:
    print(f"\n==== Subject: {sub} ====")
    sub_dir = os.path.join(base_path, sub)
    os.makedirs(sub_dir, exist_ok=True)
    data_path = os.path.join(sub_dir, f"{sub}_scalp_epochs_by_class.npz")
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
        trials = np.transpose(stim_data[:, :, start_idx:end_idx], (1, 0, 2))

        for i in range(n_trials):
            trials[i] = (trials[i] - np.mean(trials[i], axis=1, keepdims=True)) / np.std(trials[i], axis=1, keepdims=True)

        trials_flat = trials.reshape(n_trials, -1)
        X_by_class[label_idx] = trials_flat
        y_by_class[label_idx] = [label_idx] * n_trials

    min_trials = min(len(y_list) for y_list in y_by_class)

    X_balanced, y_balanced = [], []

    for i in range(4):
        X_down, y_down = resample(
            X_by_class[i], y_by_class[i],
            replace=False, n_samples=min_trials, random_state=42
        )
        X_balanced.append(X_down)
        y_balanced.append(y_down)

    X = np.vstack(X_balanced)
    y = np.hstack(y_balanced)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    classifiers = {
        "SVM_RBF": SVC(kernel='rbf', C=1.0, gamma='scale'),
        "Linear_SVM": LinearSVC(max_iter=5000),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic_Regression": LogisticRegression(max_iter=5000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    }

    for name, clf in classifiers.items():
        print(f"\n=== {name} ===")

        pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=100),
            clf
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        report = classification_report(y_test, y_pred, target_names=stimulus_labels, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Save classification report
        report_path = os.path.join(sub_dir, f"{sub}_{name}_classification_report.csv")
        report_df.to_csv(report_path, float_format='%.2f')
        
        # Save JSON version
        json_report_path = os.path.join(sub_dir, f"{sub}_{name}_classification_report.json")
        report_df.to_json(json_report_path, orient='split', indent=2)
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_npy_path = os.path.join(sub_dir, f"{sub}_{name}_confusion_matrix.npy")
        np.save(cm_npy_path, cm)
        
        # Optional: visual PNG
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap='jet', xticklabels=stimulus_labels, yticklabels=stimulus_labels, ax=ax_cm)
        ax_cm.set_title(f"{name} - {sub}")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        plt.tight_layout()
        
        cm_path = os.path.join(sub_dir, f"{sub}_{name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        
        print(f"Results saved for classifier '{name}' for subject {sub} at {sub_dir}")

