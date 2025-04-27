import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === CONFIG ===
base_path = r"C:\shared\abass"
subjects = ['sub13', 'sub11', 'sub9', 'sub8']

classifiers = ['DeepConvNet', 'EEGNet']
suffix_map = {
    'DeepConvNet': '',
    'EEGNet': '_EEGNet',
}
display_names = {
    'DeepConvNet': 'DeepConvNet (ours)',
    'EEGNet': 'EEGNet',
}

stimulus_labels = ['Left', 'Right', 'Up', 'Down']
metrics = ['precision', 'recall', 'f1-score', 'support']

# === Load per-class metrics ===
metric_data = []

for sub in subjects:
    sub_dir = os.path.join(base_path, sub)
    for clf in classifiers:
        suffix = suffix_map[clf]
        clf_label = display_names[clf]
        csv_path = os.path.join(sub_dir, f"{sub}{suffix}_classification_report.csv")

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, index_col=0)
                for stim in stimulus_labels:
                    for metric in metrics:
                        val = df.loc[stim, metric]
                        metric_data.append({
                            'Stimulus': stim,
                            'Metric': metric,
                            'Value': val,
                            'Classifier': clf_label,
                            'Subject': sub
                        })
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
        else:
            print(f"Missing: {csv_path}")

df_metrics = pd.DataFrame(metric_data)

# === Print group means ± std for console reporting ===
print("\n=== Mean ± Std for Each Metric by Stimulus and Classifier ===")
summary = df_metrics.groupby(['Stimulus', 'Metric', 'Classifier'])['Value'].agg(['mean', 'std']).reset_index()
summary = summary.round(3)
print(summary.to_string(index=False))

# === Plot ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.3)
fig.suptitle("EEGNet vs DeepConvNet (ours): Per-Class Metrics ± Std", fontsize=18, fontweight='bold')

metric_titles = {
    'precision': 'Precision',
    'recall': 'Recall',
    'f1-score': 'F1-Score',
    'support': 'Support'
}

for ax, metric in zip(axes.flat, metrics):
    sns.barplot(
        data=df_metrics[df_metrics['Metric'] == metric],
        x='Stimulus',
        y='Value',
        hue='Classifier',
        ci='sd',
        capsize=0.1,
        palette='Set2',
        ax=ax
    )
    ax.set_title(f"{metric_titles[metric]} per Stimulus", fontsize=14)
    ax.set_ylabel("Score" if metric != 'support' else "Count")
    ax.set_ylim(0, 1.05 if metric != 'support' else None)
    ax.legend(title='Classifier')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
