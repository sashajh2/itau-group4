"""
Plot the ensemble deepfake-detection results from
notebooks/embedding_experiments/Ensemble_Setup.ipynb.

Two meta-learners (Hard-OR voting and a Random-Forest meta-learner) are
compared against the three single-modality logistic baselines on the same
70/15/15 split (n=941 clips). Metrics are taken directly from the notebook
cell outputs (cells 18 and 20).
"""
import numpy as np
import matplotlib.pyplot as plt

# ---- Results copied from notebook cell outputs -------------------------------
# Random-Forest meta-learner + its per-modality RF baselines (cell 20)
rf_results = {
    "Audio (RF)":    {"Accuracy": 0.8169, "F1": 0.8796, "ROC AUC": 0.8125},
    "Video (RF)":    {"Accuracy": 0.8873, "F1": 0.9273, "ROC AUC": 0.9497},
    "Forensic (RF)": {"Accuracy": 0.8451, "F1": 0.8991, "ROC AUC": 0.8415},
    "RF Ensemble":   {"Accuracy": 0.8944, "F1": 0.9289, "ROC AUC": 0.9527},
}
# Hard-OR voting ensemble (cell 18)
or_results = {"Hard-OR Ensemble": {"Accuracy": 0.8732, "F1": 0.9151, "ROC AUC": 0.8540}}

# Assemble in display order: single modalities -> ensembles
order = ["Audio (RF)", "Video (RF)", "Forensic (RF)", "Hard-OR Ensemble", "RF Ensemble"]
all_results = {**rf_results, **or_results}

metrics = ["Accuracy", "F1", "ROC AUC"]
# colour-blind-safe qualitative palette
metric_colors = {"Accuracy": "#4C72B0", "F1": "#DD8452", "ROC AUC": "#55A868"}
is_ensemble = {name: ("Ensemble" in name) for name in order}

# ---- Grouped bar chart -------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(order))
width = 0.26

for i, metric in enumerate(metrics):
    vals = [all_results[name][metric] for name in order]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width, label=metric,
                  color=metric_colors[metric], edgecolor="white", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5, color="#333333")

# shade the ensemble group to set it apart from the single-modality baselines
for j, name in enumerate(order):
    if is_ensemble[name]:
        ax.axvspan(j - 0.5, j + 0.5, color="#000000", alpha=0.04, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(order, fontsize=10)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("Ensemble vs. single-modality deepfake detection\n(test set, n=141 clips of 941 total)",
             fontsize=13, fontweight="bold")
ax.legend(title="Metric", loc="upper left", bbox_to_anchor=(1.01, 1.0),
          framealpha=0.95)
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(0.0, 0.505, "chance", ha="left", va="bottom",
        fontsize=8, color="grey")
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
out = "results/classifiers/ensemble/ensemble_results.png"
import os
os.makedirs("results/classifiers/ensemble", exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
