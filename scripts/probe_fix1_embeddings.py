"""
Linear-probe evaluation of the disentanglement (fix1) z_auth embeddings.

Question: do the fix1a / fix1b authenticity embeddings actually let a simple
logistic classifier separate real vs. fake audio segments?

The fix1 experiments only ever reported clustering metrics (silhouette, etc.)
on z_auth. Here we instead train a logistic-regression probe on the frozen
z_auth embeddings and test it on held-out videos.

Setup
-----
- Data:   exports/avdeepfake_20pct_embeddings.npz  (AVDeepFake1M 20% subset,
          61,181 HuBERT segments, 768-d, ~1.1% fake).
- Split:  video-level 80/20, seed 42 — identical to the fix1 training split,
          so probe-test videos were held out during model training.
- Models: DisentangledProjector checkpoints (768->256->128, L2-normalized z_auth).
- Probe:  sklearn LogisticRegression(class_weight="balanced") trained on the
          train split, evaluated on the val split.

Because the split is heavily imbalanced (98.9% real) we report ROC-AUC, PR-AUC,
balanced accuracy and fake-class F1 — plain accuracy is ~0.99 for "predict all
real" and is not informative.
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    balanced_accuracy_score, accuracy_score, confusion_matrix, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/ricardocarrillo/Desktop/Itau_UROP/itau-group4"
NPZ_PATH = os.path.join(REPO, "exports/avdeepfake_20pct_embeddings.npz")
SEED = 42
VAL_FRAC = 0.20
SAVE_DIR = "results/fix1_probe"

CHECKPOINTS = {
    # label -> checkpoint path (None = raw HuBERT input, no projection)
    "Raw HuBERT":       None,
    "Baseline z_auth":  os.path.join(REPO, "results/fix1_variants/baseline_model.pt"),
    "Fix1a z_auth":     os.path.join(REPO, "results/fix1a_repulsion/fix1a_model.pt"),
    "Fix1b z_auth":     os.path.join(REPO, "results/fix1_variants/fix1b_model.pt"),
}


# ── Model (matches the saved checkpoints) ────────────────────────────────────
class DisentangledProjector(nn.Module):
    def __init__(self, input_dim=768, output_dim=128):
        super().__init__()
        self.f_auth = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, output_dim))
        self.f_id   = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, output_dim))

    def forward(self, z):
        return F.normalize(self.f_auth(z), dim=-1), F.normalize(self.f_id(z), dim=-1)


def load_split():
    """Reproduce the fix1 video-level 80/20 split (seed 42)."""
    data = np.load(NPZ_PATH, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    is_real = data["is_real"]
    vid_ids = data["vid_ids"]

    y = (~is_real).astype(int)  # 1 = fake, 0 = real

    unique_vids = list(dict.fromkeys(vid_ids.tolist()))
    rng = random.Random(SEED)
    rng.shuffle(unique_vids)
    n_val = max(1, int(len(unique_vids) * VAL_FRAC))
    val_vid_set = set(unique_vids[:n_val])

    train_mask = np.array([v not in val_vid_set for v in vid_ids])
    val_mask = ~train_mask
    return embeddings, y, train_mask, val_mask


@torch.no_grad()
def project(embeddings, ckpt_path):
    """Return z_auth for every row (or the raw input if ckpt_path is None)."""
    if ckpt_path is None:
        return embeddings
    model = DisentangledProjector(input_dim=embeddings.shape[1], output_dim=128)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    out = []
    for i in range(0, len(embeddings), 4096):
        z_auth, _ = model(torch.from_numpy(embeddings[i:i + 4096]))
        out.append(z_auth.numpy())
    return np.concatenate(out).astype(np.float32)


def evaluate(name, X, y, train_mask, val_mask):
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[val_mask], y[val_mask]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, ytr)

    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    res = {
        "roc_auc":  float(roc_auc_score(yte, prob)),
        "pr_auc":   float(average_precision_score(yte, prob)),
        "bal_acc":  float(balanced_accuracy_score(yte, pred)),
        "f1_fake":  float(f1_score(yte, pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(yte, pred)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    fpr, tpr, _ = roc_curve(yte, prob)
    print(f"[{name:16s}] ROC-AUC={res['roc_auc']:.3f}  PR-AUC={res['pr_auc']:.3f}  "
          f"BalAcc={res['bal_acc']:.3f}  F1(fake)={res['f1_fake']:.3f}  "
          f"Acc={res['accuracy']:.3f}  (tp={tp}, fn={fn}, fp={fp}, tn={tn})")
    return res, (fpr, tpr)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    embeddings, y, train_mask, val_mask = load_split()
    print(f"Total {len(y):,} segments | train {int(train_mask.sum()):,} | val {int(val_mask.sum()):,}")
    print(f"Fake rate — train {y[train_mask].mean()*100:.2f}%  val {y[val_mask].mean()*100:.2f}%\n")

    results, roc_curves = {}, {}
    for name, ckpt in CHECKPOINTS.items():
        X = project(embeddings, ckpt)
        results[name], roc_curves[name] = evaluate(name, X, y, train_mask, val_mask)

    with open(os.path.join(SAVE_DIR, "probe_results.json"), "w") as f:
        json.dump({"config": {"seed": SEED, "val_frac": VAL_FRAC,
                              "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum()),
                              "fake_rate_val": float(y[val_mask].mean())},
                   "results": results}, f, indent=2)

    plot_results(results, roc_curves)
    print(f"\nSaved results + plots to {SAVE_DIR}/")


def plot_results(results, roc_curves):
    names = list(results.keys())
    metrics = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
               ("bal_acc", "Balanced Acc"), ("f1_fake", "F1 (fake)")]
    palette = {"Raw HuBERT": "#8C8C8C", "Baseline z_auth": "#4C72B0",
               "Fix1a z_auth": "#DD8452", "Fix1b z_auth": "#55A868"}
    colors = [palette[n] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # Grouped bar chart of the four imbalance-robust metrics
    x = np.arange(len(metrics))
    width = 0.2
    for i, name in enumerate(names):
        vals = [results[name][k] for k, _ in metrics]
        offset = (i - (len(names) - 1) / 2) * width
        bars = ax1.bar(x + offset, vals, width, label=name,
                       color=colors[i], edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([lbl for _, lbl in metrics], fontsize=10)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.5, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax1.text(len(metrics) - 0.5, 0.505, "chance (AUC/BalAcc)", ha="right",
             va="bottom", fontsize=7, color="grey")
    ax1.set_title("Logistic probe on fix1 z_auth embeddings\n"
                  "AVDeepFake1M 20% subset — held-out videos", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", ls=":", alpha=0.4)
    ax1.set_axisbelow(True)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)

    # ROC curves
    for name in names:
        fpr, tpr = roc_curves[name]
        ax2.plot(fpr, tpr, color=palette[name], lw=1.8,
                 label=f"{name} (AUC={results[name]['roc_auc']:.2f})")
    ax2.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.set_title("ROC curves", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(ls=":", alpha=0.4)
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    out = os.path.join(SAVE_DIR, "fix1_probe_results.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
