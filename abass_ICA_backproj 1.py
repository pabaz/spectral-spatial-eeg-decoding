import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import Counter, defaultdict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from mne.viz import plot_topomap
from mne.stats import f_oneway
from scipy.stats import f_oneway as scipy_anova  # Fallback
import pandas as pd


subs = ['sub9']  # Extend this list as needed (8,13,11, 9)
base_path = r"C:\shared\abass"

event_id = {
    'left': 10,
    'right': 11,
    'up': 12,
    'down': 13
}
id_to_label = {v: k for k, v in event_id.items()}

tmin, tmax = -0.5, 0.5
sigma = 4  # Gaussian smoothing strength across trials

subject_accuracies_by_class = defaultdict(dict)

# --- Plot Class Distribution ---
fig_dist, axs_dist = plt.subplots(2, 2, figsize=(10, 8))
axs_dist = axs_dist.flatten()

for idx, sub in enumerate(subs):
    print(f"Processing {sub}...")
    part2_folder = os.path.join(base_path, sub, "part2")
    part2_vhdr = next((os.path.join(part2_folder, f) for f in os.listdir(part2_folder) if f.endswith('.vhdr')), None)
    if part2_vhdr is None:
        print(f"No part2 .vhdr for {sub}")
        continue

    raw = mne.io.read_raw_brainvision(part2_vhdr, preload=True)
    raw.filter(1.0, 30.0, fir_design='firwin')
    
    from scipy.stats import zscore

    # === Find bad channels based on z-score of raw signal standard deviation ===
    eeg_data = raw.get_data(picks="eeg")  # shape: [n_channels, n_times]
    channel_stds = np.std(eeg_data, axis=1)  # standard deviation for each channel
    z_scores = zscore(channel_stds)
    
    # Threshold for bad channels
    bad_thresh = 2
    eeg_ch_names = raw.copy().pick_types(eeg=True).ch_names
    bad_chs = np.array(eeg_ch_names)[z_scores > bad_thresh].tolist()    
    if bad_chs:
        print(f"Detected bad channels (z > {bad_thresh}): {bad_chs}")
        raw.info['bads'] = bad_chs
    else:
        print("No bad channels detected.")

    # Get all EEG channel names
    all_eeg_chans = raw.copy().pick_types(eeg=True).ch_names
    
    # Determine names of channels 63–67
    excluded_indices = list(range(62, 67)) + [45]
    excluded_chans = [all_eeg_chans[i] for i in excluded_indices if i < len(all_eeg_chans)]
    
    # Combine hardcoded exclusions with detected bad channels
    channels_to_exclude = list(set(excluded_chans + bad_chs))
    
    # Drop those channels
    raw.pick_channels([ch for ch in all_eeg_chans if ch not in channels_to_exclude])
    
    print(f"Total channels removed: {channels_to_exclude}")


    # === ICA ===
    print(f"Running ICA for {sub}...")
    ica = mne.preprocessing.ICA(n_components=len(raw.ch_names), random_state=97, max_iter='auto')
    ica.fit(raw)


    # Get the ICA component topographies and layout info
    ica_data = ica.get_components()  # shape: (n_channels, n_components)
    layout = raw.info
    n_components = ica_data.shape[1]
    
    # === Topoplots for first 30 ICA components ===
    n_plot = 30
    n_rows, n_cols = 5, 6
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows), squeeze=False)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    
    for i in range(n_plot):
        ax = axes[i // n_cols][i % n_cols]
        plot_topomap(ica_data[:, i], raw.info, axes=ax, show=False, contours=4, cmap='jet')
        ax.set_title(f"IC {i}", fontsize=10)
    
    # Hide unused axes
    for j in range(n_plot, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis('off')
    
    plt.suptitle(f"{sub} - ICA Topoplots (First 30 Components)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    
    
    
    # Get ICA time series data (n_components, n_times)
    ica_data = ica.get_sources(raw).get_data()
    sfreq = raw.info['sfreq']
    n_components = ica_data.shape[0]
    
      # === Power Spectra for first 30 ICA components ===
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.5 * n_rows), squeeze=False)
    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    
    for i in range(n_plot):
        f, Pxx = welch(ica_data[i], fs=sfreq, nperseg=2048)
        mask = (f >= 1) & (f <= 30)
        f_selected = f[mask]
        Pxx_selected = Pxx[mask]
    
        f_interp = np.linspace(1, 30, 100)
        Pxx_interp = np.interp(f_interp, f_selected, 10 * np.log10(Pxx_selected))
    
        ax = axes[i // n_cols][i % n_cols]
        ax.plot(f_interp, Pxx_interp)
        ax.set_title(f"IC {i}", fontsize=12)
        ax.set_xlim(1, 30)
        ax.set_xlabel("Freq (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.grid(True)
    
    # Hide unused axes
    for j in range(n_plot, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis('off')
    
    plt.suptitle(f"ICA Power Spectra (1–30 Hz, First 30 Components)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

   
    # === Epoch the data based on the 4 triggers ===
    events, _ = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True)
    # === Get ICA source time series for the epochs ===
    ica_sources = ica.get_sources(epochs).get_data()  # shape: (n_trials, n_components, n_times)
    times = epochs.times
    
    # === Find indices for time range 0.1 to 0.25s ===
    start_idx = np.argmin(np.abs(times - 0.1))
    end_idx = np.argmin(np.abs(times - 0.25))
        
    # === Prepare data for ANOVA ===
    labels = epochs.events[:, 2]  # These are the event codes (e.g., 10, 11, 12, 13)
    unique_labels = np.unique(labels)
    print(f"Found event codes in epochs: {unique_labels}")
    
    f_values = []
    p_values = []
    
    for comp in range(ica_sources.shape[1]):
        comp_data = ica_sources[:, comp, start_idx:end_idx]  # (n_trials, time_window)
        trial_averages = comp_data.mean(axis=1)
    
        # Group by stimulus type
        group_data = {eid: [] for eid in event_id.values()}
        for val, label in zip(trial_averages, labels):
            if label in group_data:
                group_data[label].append(val)
    
        # Only do ANOVA if all 4 groups have data
        if all(len(group_data[eid]) > 1 for eid in event_id.values()):
            groups = [group_data[eid] for eid in event_id.values()]
            try:
                f, p = scipy_anova(*groups)
            except Exception:
                f, p = np.nan, np.nan
        else:
            print(f"Skipping IC {comp}: not all event types present.")
            f, p = np.nan, np.nan
    
        f_values.append(f)
        p_values.append(p)
    
    # === Report top component(s) ===
    if np.all(np.isnan(p_values)):
        print("\nANOVA failed: all p-values are NaN. Check event codes and trial counts.")
    else:
        best_idx = np.nanargmin(p_values)
        print(f"\nMost significant ICA component by ERP ANOVA (0.1–0.25s): IC {best_idx}")
        print(f"F-value: {f_values[best_idx]:.4f}, p-value: {p_values[best_idx]:.4e}")
    
        anova_df = pd.DataFrame({
            'IC': np.arange(len(f_values)),
            'F-value': f_values,
            'p-value': p_values
        }).sort_values('p-value')
    
        print("\nTop 5 components by ANOVA p-value:")
        print(anova_df.head())
    
            
       # === Mean ERP Traces for First 30 ICA Components ===
    colors = ['r', 'g', 'b', 'k']
    labels_map = {10: 'Left', 11: 'Right', 12: 'Up', 13: 'Down'}
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.5 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i in range(n_plot):
        ax = axes[i]
        ic_data = ica_sources[:, i, :]  # shape: (n_trials, n_times)
    
        for color, label_id in zip(colors, event_id.values()):
            label_trials = ic_data[labels == label_id]
            if len(label_trials) == 0:
                continue
            mean_erp = label_trials.mean(axis=0)
            ax.plot(times, mean_erp, label=labels_map[label_id], color=color)
    
        ax.set_title(f"IC {i}", fontsize=10)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.grid(True)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (a.u.)")
    
    for j in range(n_plot, len(axes)):
        axes[j].axis('off')
    
    axes[0].legend(loc='upper right')
    plt.suptitle("Mean ERP Traces (First 30 ICA Components)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    
import numpy as np
from scipy.stats import f_oneway as scipy_anova

# === Manual selection of BAD components ===
bad_components = [0, 2,3,10,17,20]
print(f"Removing {len(bad_components)} bad components: {bad_components}")

# === Extract ICA source time series for epochs ===
ica_sources = ica.get_sources(epochs).get_data()  # shape: [n_trials, n_components, n_timepoints]
labels = epochs.events[:, 2]
times = epochs.times

# === Determine good components by excluding bad ones ===
all_components = np.arange(ica_sources.shape[1])
good_components = [c for c in all_components if c not in bad_components]
print(f"Keeping {len(good_components)} components: {good_components}")

# === Select only desired components ===
sources = ica_sources[:, good_components, :]  # shape: [n_trials, n_components, n_timepoints]

# === Reconstruct using original mixing matrix without any scaling ===
sources = np.transpose(sources, (1, 0, 2))  # [n_components, n_trials, n_timepoints]
sources_flat = sources.reshape(len(good_components), -1)  # [n_components, total_samples]

mixing = ica.mixing_matrix_[:, good_components]  # [n_channels, n_components]
reconstructed_flat = mixing @ sources_flat  # [n_channels, total_samples]
reconstructed = reconstructed_flat.reshape((mixing.shape[0], -1, ica_sources.shape[2]))  # [n_channels, n_trials, n_timepoints]

# === Split into class-wise data ===
stimulus_order = [10, 11, 12, 13]
class_wise_data = []

for stim in stimulus_order:
    stim_trials = reconstructed[:, labels == stim, :]  # [n_channels, n_trials, n_timepoints]
    class_wise_data.append(stim_trials)

# === Save ===
save_path = os.path.join(base_path, sub, f"{sub}_badcs_scalp_epochs_by_class.npz")
np.savez_compressed(save_path, 
    left=class_wise_data[0], 
    right=class_wise_data[1], 
    up=class_wise_data[2], 
    down=class_wise_data[3],
    bad_components=np.array(bad_components),
    kept_components=np.array(good_components)
)

print(f"\nSaved scalp-space data (bad components removed) to:\n{save_path}")

    
    
    
    
