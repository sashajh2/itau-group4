"""
Diagnostic: is the fix1 z_auth real/fake boundary non-linear (radial)?

For each representation we compare three probes on the same held-out videos:
  - linear     : LogisticRegression on the full embedding
  - non-linear : kNN and RandomForest (can carve curved / closed boundaries)
  - radial     : LogisticRegression on ONE feature — Euclidean distance from
                 each point to the real-training centroid.

If disentanglement pushed fakes into a shell *around* the real core (which is
what a centroid-repulsion term does), then:
  * linear ROC-AUC drops (a hyperplane can't split a core from a shell),
  * the radial 1-D probe and the non-linear probes recover/beat it.
"""
import os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

REPO = "/Users/ricardocarrillo/Desktop/Itau_UROP/itau-group4"
NPZ = os.path.join(REPO, "exports/avdeepfake_20pct_embeddings.npz")
SEED, VAL_FRAC = 42, 0.20
CKPTS = {
    "Raw HuBERT":      None,
    "Baseline z_auth": os.path.join(REPO, "results/disentangled/fix1_variants/baseline_model.pt"),
    "Fix1a z_auth":    os.path.join(REPO, "results/disentangled/fix1a_repulsion/fix1a_model.pt"),
    "Fix1b z_auth":    os.path.join(REPO, "results/disentangled/fix1_variants/fix1b_model.pt"),
}


class DisentangledProjector(nn.Module):
    def __init__(self, input_dim=768, output_dim=128):
        super().__init__()
        self.f_auth = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, output_dim))
        self.f_id   = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, output_dim))
    def forward(self, z):
        return F.normalize(self.f_auth(z), dim=-1), F.normalize(self.f_id(z), dim=-1)


def load_split():
    d = np.load(NPZ, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    y = (~d["is_real"]).astype(int)
    vids = d["vid_ids"]
    uniq = list(dict.fromkeys(vids.tolist()))
    random.Random(SEED).shuffle(uniq)
    val_set = set(uniq[:max(1, int(len(uniq) * VAL_FRAC))])
    tr = np.array([v not in val_set for v in vids]); return emb, y, tr, ~tr


@torch.no_grad()
def project(emb, ckpt):
    if ckpt is None: return emb
    m = DisentangledProjector(emb.shape[1], 128); m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
    return np.concatenate([m(torch.from_numpy(emb[i:i+4096]))[0].numpy() for i in range(0, len(emb), 4096)]).astype(np.float32)


def auc(clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def radial_auc(Xtr, ytr, Xte, yte):
    """1-D probe: distance of each point to the REAL training centroid."""
    mu = Xtr[ytr == 0].mean(0, keepdims=True)
    dtr = np.linalg.norm(Xtr - mu, axis=1).reshape(-1, 1)
    dte = np.linalg.norm(Xte - mu, axis=1).reshape(-1, 1)
    return auc(LogisticRegression(max_iter=1000, class_weight="balanced"), dtr, ytr, dte, yte)


def main():
    emb, y, tr, te = load_split()
    print(f"train {tr.sum():,} | val {te.sum():,} | val fake rate {y[te].mean()*100:.1f}%\n")
    hdr = f"{'representation':16s} {'linear':>8s} {'kNN':>8s} {'RF':>8s} {'radial-1D':>10s}"
    print(hdr); print("-" * len(hdr))
    for name, ck in CKPTS.items():
        X = project(emb, ck)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        lin = auc(LogisticRegression(max_iter=2000, class_weight="balanced"), Xtr, ytr, Xte, yte)
        knn = auc(KNeighborsClassifier(n_neighbors=25, weights="distance"), Xtr, ytr, Xte, yte)
        rf  = auc(RandomForestClassifier(n_estimators=200, max_depth=None,
                                         class_weight="balanced", random_state=SEED, n_jobs=-1),
                  Xtr, ytr, Xte, yte)
        rad = radial_auc(Xtr, ytr, Xte, yte)
        print(f"{name:16s} {lin:8.3f} {knn:8.3f} {rf:8.3f} {rad:10.3f}")


if __name__ == "__main__":
    main()
