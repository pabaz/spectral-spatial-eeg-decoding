import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.ndimage import gaussian_filter1d
import mne

# Configs
subs = ['sub8', 'sub13', 'sub11', 'sub9']
base_path = r"C:\shared\abass"
event_id = {'left': 10, 'right': 11, 'up': 12, 'down': 13}
id_to_label = {v: k for k, v in event_id.items()}
motor_channels = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6',
                  'C5', 'C3', 'C1', 'C2', 'C4', 'C6',
                  'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6']
tmin, tmax = -0.5, 0.5
sigma = 1  # Gaussian smoothing across trials
subject_accuracies_by_class = defaultdict(dict)

fig_dist, axs_dist = plt.subplots(2, 2, figsize=(10, 8))
axs_dist = axs_dist.flatten()

for idx, sub in enumerate(subs):
    part2_folder = os.path.join(base_path, sub, "part2")
    part2_vhdr = next((os.path.join(part2_folder, f) for f in os.listdir(part2_folder) if f.endswith('.vhdr')), None)
    if part2_vhdr is None:
        continue

    raw = mne.io.read_raw_brainvision(part2_vhdr, preload=True)
    raw.filter(1.0, 30.0, fir_design='firwin')
    raw.pick_channels(motor_channels)

    events, _ = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True)

    X_trials = []
    y_labels = []

    for label, code in event_id.items():
        these_epochs = epochs[label]
        if len(these_epochs) == 0:
            continue

        n_times = these_epochs.get_data().shape[2]
        base_ep = these_epochs.copy().crop(tmin, 0.0)
        post_ep = these_epochs.copy().crop(0.0, tmax)

        psd_base = base_ep.compute_psd(method='welch', fmin=1, fmax=30,
                                       n_fft=n_times, n_per_seg=n_times)
        psd_post = post_ep.compute_psd(method='welch', fmin=1, fmax=30,
                                       n_fft=n_times, n_per_seg=n_times)

        base_log = np.log10(psd_base.get_data(picks='all'))
        post_log = np.log10(psd_post.get_data(picks='all'))

        delta = post_log - base_log
        features = delta.mean(axis=1)  # mean over channels
        X_trials.append(features)
        y_labels.extend([code] * len(features))

    if not X_trials:
        continue

    X = np.vstack(X_trials)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    label_names = le.classes_

    # Balance classes
    counts_encoded = Counter(y_encoded)
    min_count = min(counts_encoded.values())
    balanced_indices = []
    for class_id in np.unique(y_encoded):
        class_indices = np.where(y_encoded == class_id)[0]
        np.random.shuffle(class_indices)
        balanced_indices.extend(class_indices[:min_count])
    np.random.shuffle(balanced_indices)

    X_balanced = X[balanced_indices]
    y_balanced = y_encoded[balanced_indices]

    # --- Plot class distribution ---
    label_order = ['left', 'right', 'up', 'down']
    code_order = [event_id[label] for label in label_order]
    counts = Counter(y_labels)
    freqs = [counts.get(code, 0) for code in code_order]
    axs_dist[idx].bar(label_order, freqs, color='salmon')
    axs_dist[idx].set_title(f"{sub} - Class Distribution")
    axs_dist[idx].set_ylabel("Count")
    axs_dist[idx].grid(axis='y')

    # --- Cross-validate with smoothing only in training folds ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_all = np.zeros_like(y_balanced)
    for train_idx, test_idx in skf.split(X_balanced, y_balanced):
        X_train, y_train = X_balanced[train_idx], y_balanced[train_idx]
        X_test = X_balanced[test_idx]

        # Apply Gaussian smoothing only to training data (across trials)
        X_train_smoothed = gaussian_filter1d(X_train, sigma=sigma, axis=0)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_smoothed, y_train)
        y_pred_all[test_idx] = clf.predict(X_test)

    for class_id in np.unique(y_balanced):
        mask = y_balanced == class_id
        acc = accuracy_score(y_balanced[mask], y_pred_all[mask])
        trigger_code = label_names[class_id]
        trigger_name = id_to_label[trigger_code]
        subject_accuracies_by_class[sub][trigger_name] = acc

# Plot Accuracy Results
fig_dist.tight_layout()
plt.suptitle("Trigger Distribution per Subject", fontsize=16, y=1.03)
plt.show()

plt.figure(figsize=(10, 6))
bar_width = 0.2
x_labels = ['left', 'right', 'up', 'down']
x = np.arange(len(x_labels))
for i, sub in enumerate(subs):
    accs = [subject_accuracies_by_class[sub].get(label, 0) for label in x_labels]
    plt.bar(x + i * bar_width, accs, width=bar_width, label=sub)
plt.xticks(x + (len(subs) - 1) * bar_width / 2, [label.title() for label in x_labels])
plt.ylim(0, 1)
plt.ylabel("Classification Accuracy")
plt.title("Classification Accuracy per Trigger (Smoothed Within Fold, Log Power Δ, Motor Channels)")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()
