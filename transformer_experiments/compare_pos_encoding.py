"""Comparison: Vanilla Transformer (with pos encoding) vs Transformer (no pos encoding).

Dataset: AVDeepFake1M only.
  Train: 200 real + 200 fake = 400 samples
  Test:  100 real + 100 fake = 200 samples

NOTE on video repetition: AVDeepFake1M has only 121 fake videos. After the 75/25
pool split (~91 train, ~30 test), sampling 200 fake train samples requires each
fake video to appear ~2.2x on average; 100 fake test samples from 30 videos means
~3.3x repetition. Results should be interpreted with this in mind.

Embeddings tested: HuBERT (768-dim), OpenL3 (512-dim).
Loss: BCEWithLogitsLoss for both models.
Metrics: accuracy, FNR, ROC-AUC, confusion matrix.

Usage:
    python -m transformer_experiments.compare_pos_encoding
"""

import os
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from transformer_experiments.dataset import DeepfakeSequenceDataset, collate_fn
from transformer_experiments.model import VanillaTransformerClassifier

# ── Config ────────────────────────────────────────────────────────────────────
HDF5_PATH       = "exports/deepfake_embeddings.h5"
DATASET_FILTER  = "avdeepfake1m"

N_TRAIN_REAL = 200
N_TRAIN_FAKE = 200
N_TEST_REAL  = 100
N_TEST_FAKE  = 100

SEED         = 42
EPOCHS       = 30
BATCH_SIZE   = 16
LR           = 1e-4
WEIGHT_DECAY = 1e-4
MAX_SEQ_LEN  = 256

D_MODEL    = 256
NHEAD      = 8
NUM_LAYERS = 4
DIM_FF     = 1024
DROPOUT    = 0.1

EMBEDDING_CONFIGS = [
    {"key": "hubert",  "dim": 768},
    {"key": "openl3",  "dim": 512},
]
# ─────────────────────────────────────────────────────────────────────────────


class TransformerNoPosEncoding(nn.Module):
    """Identical to VanillaTransformerClassifier but with no positional encoding.

    The CLS token + self-attention operate purely on content, making this a
    permutation-invariant (bag-of-embeddings) transformer.
    """

    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier  = nn.Linear(d_model, 1)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, embeddings: torch.Tensor, attention_mask=None) -> torch.Tensor:
        b, t, _ = embeddings.shape
        x   = self.input_proj(embeddings)
        cls = self.cls_token.expand(b, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        # no positional encoding added here
        if attention_mask is not None:
            cls_mask = torch.ones(b, 1, device=attention_mask.device)
            full_mask = torch.cat([cls_mask, attention_mask], dim=1)
            src_key_padding_mask = (full_mask == 0)
        else:
            src_key_padding_mask = None
        x       = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        cls_out = x[:, 0, :]
        return self.classifier(cls_out)


def sample_avdeepfake_only(
    hdf5_path: str,
    embedding_key: str,
    n_train_real: int,
    n_train_fake: int,
    n_test_real: int,
    n_test_fake: int,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Sample train/test exclusively from AVDeepFake1M with video-level no-overlap."""
    rng = random.Random(seed)

    real_videos: List[Tuple[str, List[int]]] = []
    fake_videos: List[Tuple[str, List[int]]] = []

    with h5py.File(hdf5_path, "r") as f:
        for safe_id in f["videos"].keys():
            vid = f["videos"][safe_id]
            ds  = vid.attrs.get("dataset", b"")
            if isinstance(ds, bytes):
                ds = ds.decode()
            if ds != DATASET_FILTER:
                continue
            aug_types = [
                t.decode() if isinstance(t, bytes) else t
                for t in vid["augmentation_info"]["types"][:]
            ]
            real_idx = [i for i, t in enumerate(aug_types) if t in ("source", "real")]
            fake_idx = [i for i, t in enumerate(aug_types) if t == "fake"]
            if real_idx:
                real_videos.append((safe_id, real_idx))
            if fake_idx:
                fake_videos.append((safe_id, fake_idx))

    rng.shuffle(real_videos)
    rng.shuffle(fake_videos)

    n_real_train_pool = max(1, round(len(real_videos) * n_train_real / (n_train_real + n_test_real)))
    n_fake_train_pool = max(1, round(len(fake_videos) * n_train_fake / (n_train_fake + n_test_fake)))

    train_real_pool = real_videos[:n_real_train_pool]
    test_real_pool  = real_videos[n_real_train_pool:]
    train_fake_pool = fake_videos[:n_fake_train_pool]
    test_fake_pool  = fake_videos[n_fake_train_pool:]

    print(f"  AVDeepFake1M video pools:")
    print(f"    Real — total: {len(real_videos)}, train pool: {len(train_real_pool)}, test pool: {len(test_real_pool)}")
    print(f"    Fake — total: {len(fake_videos)}, train pool: {len(train_fake_pool)}, test pool: {len(test_fake_pool)}")
    avg_rep_train = n_train_fake / len(train_fake_pool)
    avg_rep_test  = n_test_fake  / len(test_fake_pool)
    print(f"    Fake repetition — train: {avg_rep_train:.1f}x per video, test: {avg_rep_test:.1f}x per video")

    def collect(pool, n) -> List[Dict]:
        samples = []
        with h5py.File(hdf5_path, "r") as f:
            for _ in range(n):
                safe_id, aug_indices = rng.choice(pool)
                aug_idx = rng.choice(aug_indices)
                vid = f["videos"][safe_id]
                emb = np.array(vid["embeddings"][embedding_key][aug_idx], dtype=np.float32)
                labels = vid["labels"]["audio"][aug_idx]
                samples.append({
                    "embeddings": emb,
                    "label": 1 if np.any(labels > 0) else 0,
                    "video_id": safe_id,
                    "aug_idx": int(aug_idx),
                })
        return samples

    train_samples = collect(train_real_pool, n_train_real) + collect(train_fake_pool, n_train_fake)
    test_samples  = collect(test_real_pool,  n_test_real)  + collect(test_fake_pool,  n_test_fake)
    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(float)
    labs  = labels.numpy()
    tp = ((preds == 1) & (labs == 1)).sum()
    fp = ((preds == 1) & (labs == 0)).sum()
    fn = ((preds == 0) & (labs == 1)).sum()
    tn = ((preds == 0) & (labs == 0)).sum()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    fnr = fn / max(fn + tp, 1)
    try:
        auc = roc_auc_score(labs, probs)
    except ValueError:
        auc = float("nan")
    return {"acc": float(acc), "fnr": float(fnr), "auc": float(auc),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_logits, all_labels = 0.0, [], []
    for batch in loader:
        emb    = batch["embeddings"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(emb, mask).squeeze(-1)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    return torch.cat(all_logits), torch.cat(all_labels), total_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        logits = model(batch["embeddings"].to(device),
                       batch["attention_mask"].to(device)).squeeze(-1)
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"])
    return torch.cat(all_logits), torch.cat(all_labels)


def plot_confusion_matrix(cm, title, ax, metrics):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"], fontsize=10)
    ax.set_yticklabels(["Real", "Fake"], fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    total = cm.sum()
    tags  = {(0,0):"TN",(0,1):"FP",(1,0):"FN",(1,1):"TP"}
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            color = "white" if count > total / 4 else "black"
            if (i, j) == (1, 0):
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                             linewidth=3, edgecolor="red", facecolor="none"))
            ax.text(j, i, f"{tags[(i,j)]}\n{count}\n({count/total*100:.0f}%)",
                    ha="center", va="center", color=color, fontsize=11, fontweight="bold")
    summary = f"Acc: {metrics['acc']:.3f}  FNR: {metrics['fnr']:.3f}  AUC: {metrics['auc']:.3f}"
    ax.text(0.5, -0.20, summary, transform=ax.transAxes, ha="center", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                                   edgecolor="orange", linewidth=1.5))
    return im


def run_experiment(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[dict, dict, np.ndarray, np.ndarray]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1, best_state = 0.0, None
    t0 = time.time()

    print(f"\n  Training {model_name} — {EPOCHS} epochs")
    print(f"  {'-'*60}")
    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        _, _, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_logits, te_labels = evaluate(model, test_loader, device)
        te_m = compute_metrics(te_logits, te_labels)

        prec = te_m["tp"] / max(te_m["tp"] + te_m["fp"], 1)
        rec  = te_m["tp"] / max(te_m["tp"] + te_m["fn"], 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        tag  = ""
        if f1 > best_f1:
            best_f1   = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tag = " *"

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{EPOCHS}  ({time.time()-t_ep:.1f}s)"
                  f"  test acc={te_m['acc']:.3f} fnr={te_m['fnr']:.3f} auc={te_m['auc']:.3f}{tag}")

    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s ({train_time/60:.1f} min)  Best F1: {best_f1:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    tr_logits, tr_labels = evaluate(model, train_loader, device)
    te_logits, te_labels = evaluate(model, test_loader,  device)
    train_m = compute_metrics(tr_logits, tr_labels)
    test_m  = compute_metrics(te_logits, te_labels)
    train_cm = np.array([[train_m["tn"], train_m["fp"]], [train_m["fn"], train_m["tp"]]])
    test_cm  = np.array([[test_m["tn"],  test_m["fp"]],  [test_m["fn"],  test_m["tp"]]])
    return train_m, test_m, train_cm, test_cm


def save_plot(results_row: list, embedding_key: str, results_dir: str):
    """One figure per embedding: 2 columns (with/without pos enc) × 2 rows (train/test)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"Transformer Comparison — {embedding_key.upper()}  |  AVDeepFake1M only\n"
        f"Train: {N_TRAIN_REAL+N_TRAIN_FAKE} samples  Test: {N_TEST_REAL+N_TEST_FAKE} samples",
        fontsize=13, fontweight="bold",
    )
    col_labels = ["With Positional Encoding", "Without Positional Encoding"]
    for col, res in enumerate(results_row):
        plot_confusion_matrix(res["train_cm"], f"{col_labels[col]}\nTrain",
                              axes[0][col], res["train_m"])
        plot_confusion_matrix(res["test_cm"],  f"{col_labels[col]}\nTest",
                              axes[1][col], res["test_m"])
    fig.tight_layout(pad=2.5)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"compare_{embedding_key}_{timestamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: AVDeepFake1M only  |  Train: {N_TRAIN_REAL+N_TRAIN_FAKE}  |  Test: {N_TEST_REAL+N_TEST_FAKE}")

    results_dir = os.path.join("results", "compare_pos_encoding")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    t_total = time.time()

    for cfg in EMBEDDING_CONFIGS:
        emb_key = cfg["key"]
        input_dim = cfg["dim"]

        print(f"\n{'═'*70}")
        print(f"  EMBEDDING: {emb_key.upper()}  (dim={input_dim})")
        print(f"{'═'*70}")

        t_sample = time.time()
        train_samples, test_samples = sample_avdeepfake_only(
            HDF5_PATH, emb_key,
            N_TRAIN_REAL, N_TRAIN_FAKE, N_TEST_REAL, N_TEST_FAKE,
            seed=SEED,
        )
        print(f"  Sampling took {time.time()-t_sample:.1f}s")

        train_ds = DeepfakeSequenceDataset(train_samples, max_seq_len=MAX_SEQ_LEN)
        test_ds  = DeepfakeSequenceDataset(test_samples,  max_seq_len=MAX_SEQ_LEN)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        emb_results = []
        model_variants = [
            ("With Pos Encoding",    VanillaTransformerClassifier(
                input_dim=input_dim, d_model=D_MODEL, nhead=NHEAD,
                num_layers=NUM_LAYERS, dim_feedforward=DIM_FF,
                dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN,
            )),
            ("Without Pos Encoding", TransformerNoPosEncoding(
                input_dim=input_dim, d_model=D_MODEL, nhead=NHEAD,
                num_layers=NUM_LAYERS, dim_feedforward=DIM_FF,
                dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN,
            )),
        ]

        for model_name, model in model_variants:
            model = model.to(device)
            train_m, test_m, train_cm, test_cm = run_experiment(
                model_name, model, train_loader, test_loader, device
            )
            print(f"\n  [{model_name}] TRAIN — acc={train_m['acc']:.3f} fnr={train_m['fnr']:.3f} auc={train_m['auc']:.3f}")
            print(f"  [{model_name}] TEST  — acc={test_m['acc']:.3f}  fnr={test_m['fnr']:.3f}  auc={test_m['auc']:.3f}")
            emb_results.append({"model": model_name, "train_m": train_m, "test_m": test_m,
                                 "train_cm": train_cm, "test_cm": test_cm})

        save_plot(emb_results, emb_key, results_dir)
        all_results.append({"embedding": emb_key, "results": emb_results})

    total_time = time.time() - t_total

    print(f"\n{'═'*70}")
    print(f"  FINAL SUMMARY  (Test set — {N_TEST_REAL+N_TEST_FAKE} samples)")
    print(f"{'═'*70}")
    print(f"  {'Embedding':<10}  {'Model':<25}  {'Accuracy':>10}  {'FNR':>8}  {'ROC-AUC':>8}")
    print(f"  {'-'*65}")
    for entry in all_results:
        for res in entry["results"]:
            m = res["test_m"]
            print(f"  {entry['embedding']:<10}  {res['model']:<25}  "
                  f"{m['acc']:>10.3f}  {m['fnr']:>8.3f}  {m['auc']:>8.3f}")
    print(f"\n  Total wall time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
