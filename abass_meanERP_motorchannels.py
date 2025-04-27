import mne
import os
import numpy as np
import matplotlib.pyplot as plt

subs = ['sub8', 'sub13', 'sub11', 'sub9']
base_path = r"C:\shared\abass"

# Stimulus mapping
event_id = {
    'left_arrow': 10,
    'right_arrow': 11,
    'up_arrow': 12,
    'down_arrow': 13
}

tmin, tmax = -0.5, 0.5
motor_channels = ['FC5', 'FC3', 'FC1', 'C5', 'C3', 'C1']

# Storage for grand averages
all_subject_means = {key: [] for key in event_id}

# Plot setup for individual subject plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, sub in enumerate(subs):
    part2_folder = os.path.join(base_path, sub, "part2")
    part2_vhdr = next((os.path.join(part2_folder, f) for f in os.listdir(part2_folder) if f.endswith('.vhdr')), None)
    if part2_vhdr is None:
        print(f"No part2 .vhdr for {sub}")
        continue

    raw = mne.io.read_raw_brainvision(part2_vhdr, preload=True)
    raw.pick_channels(motor_channels)
    raw.filter(1.0, 30.0, fir_design='firwin')

    events, _ = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=(None, 0), preload=True)

    ax = axes[i]
    times = epochs.times

    for stim_name in event_id:
        if stim_name not in epochs.event_id:
            continue
        evoked = epochs[stim_name].average()
        mean_signal = evoked.data.mean(axis=0)
        ax.plot(times, mean_signal, label=stim_name.replace('_', ' ').title())

        # Save this subject's mean ERP for the grand average
        all_subject_means[stim_name].append(mean_signal)

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(f"{sub}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.legend(loc='upper right')
    ax.grid(True)

plt.tight_layout()
plt.suptitle("Mean ERP per Subject (1–30 Hz, Motor Channels)", fontsize=16, y=1.03)
plt.show()

# --- Plot Grand Average across Subjects ---
plt.figure(figsize=(10, 5))
for stim_name in event_id:
    if all_subject_means[stim_name]:  # ensure there's data
        # Stack and average across subjects
        subject_data = np.vstack(all_subject_means[stim_name])
        grand_avg = subject_data.mean(axis=0)
        plt.plot(times, grand_avg, label=stim_name.replace('_', ' ').title())

plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.title("Grand Average ERP Across Subjects (1–30 Hz, Motor Channels)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (µV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
