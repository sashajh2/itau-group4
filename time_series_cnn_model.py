"""
1D CNN Temporal Model for Multimodal Deepfake Detection

Self-contained module: 1D CNN encoders for audio + video embeddings, FusionHead,
dataset, collate, and training/test loops. No dependency on time_series_model.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ============================================================================
# DATASET
# ============================================================================

class AVH5Dataset(Dataset):
    """
    PyTorch Dataset for loading audio + video embeddings from HDF5.
    Each augmentation is a separate example. Returns (T, d) sequences.
    """

    def __init__(
        self,
        hdf5_path: str,
        audio_embedding_type: str = "openl3",
        video_embedding_type: str = "senet",
        use_audio_labels: bool = True,
        video_ids: Optional[List[str]] = None,
        filter_dataset: Optional[str] = None,
    ):
        self.hdf5_path = hdf5_path
        self.audio_embedding_type = audio_embedding_type
        self.video_embedding_type = video_embedding_type
        self.use_audio_labels = use_audio_labels
        self.filter_dataset = filter_dataset
        self.samples = []

        with h5py.File(hdf5_path, "r") as f:
            if video_ids is None:
                video_ids = [vid.decode() if isinstance(vid, bytes) else vid for vid in f["metadata/video_ids"][:]]
            videos_grp = f["videos"]
            for video_id in video_ids:
                safe_video_id = video_id.replace("/", "_")
                if safe_video_id not in videos_grp:
                    continue
                vid_grp = videos_grp[safe_video_id]
                dataset = vid_grp.attrs.get("dataset", "avdeepfake1m")
                if isinstance(dataset, bytes):
                    dataset = dataset.decode()
                if self.filter_dataset is not None:
                    fl = self.filter_dataset.lower()
                    dl = dataset.lower()
                    if fl not in dl and dl not in fl:
                        continue
                for aug_idx in range(vid_grp.attrs["num_augmentations"]):
                    self.samples.append((video_id, aug_idx, dataset))

        suf = f" (filtered: {filter_dataset})" if filter_dataset else ""
        print(f"Loaded {len(self.samples)} samples from {hdf5_path}{suf}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id, aug_idx, dataset = self.samples[idx]
        safe_video_id = video_id.replace("/", "_")
        with h5py.File(self.hdf5_path, "r") as f:
            vid_grp = f["videos"][safe_video_id]
            audio_emb = vid_grp[f"embeddings/{self.audio_embedding_type}"][aug_idx]
            video_emb = vid_grp[f"embeddings/{self.video_embedding_type}"][aug_idx]
            labels = vid_grp["labels/audio"][aug_idx] if self.use_audio_labels else vid_grp["labels/video"][aug_idx]
        audio_seq = torch.from_numpy(audio_emb).float()
        video_seq = torch.from_numpy(video_emb).float()
        label_seq = (torch.from_numpy(labels).float() <= 0.5).float()  # 1=real, 0=fake
        return {
            "audio_seq": audio_seq,
            "video_seq": video_seq,
            "label_seq": label_seq,
            "video_id": video_id,
            "aug_idx": aug_idx,
            "dataset": dataset,
        }


# ============================================================================
# COLLATE
# ============================================================================

def collate_fn(batch):
    """Collate for variable-length sequences."""
    audio_seqs = [b["audio_seq"] for b in batch]
    video_seqs = [b["video_seq"] for b in batch]
    label_seqs = [b["label_seq"] for b in batch]
    max_len = max(s.shape[0] for s in audio_seqs)
    valid_lengths = torch.tensor([s.shape[0] for s in audio_seqs], dtype=torch.long)

    def pad(seq, L):
        if seq.shape[0] < L:
            return torch.cat([seq, torch.zeros(L - seq.shape[0], seq.shape[1])], dim=0)
        return seq

    return {
        "audio_seq": torch.stack([pad(s, max_len) for s in audio_seqs]),
        "video_seq": torch.stack([pad(s, max_len) for s in video_seqs]),
        "label_seq": torch.stack([pad(s.unsqueeze(-1), max_len).squeeze(-1) for s in label_seqs]),
        "valid_length": valid_lengths,
        "video_id": [b["video_id"] for b in batch],
        "aug_idx": [b["aug_idx"] for b in batch],
        "dataset": [b.get("dataset", "unknown") for b in batch],
    }


# ============================================================================
# FUSION HEAD
# ============================================================================

class FusionHead(nn.Module):
    """Fuses audio and video tokens; outputs per-step logits (B, N)."""

    def __init__(self, model_dim: int, hidden_dim: int = 512, dropout: float = 0.1, use_fusion_mlp: bool = True):
        super().__init__()
        if use_fusion_mlp:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(2 * model_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.fusion_mlp = nn.Linear(2 * model_dim, 1)

    def forward(self, audio_tokens: torch.Tensor, video_tokens: torch.Tensor) -> torch.Tensor:
        return self.fusion_mlp(torch.cat([audio_tokens, video_tokens], dim=-1)).squeeze(-1)


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class CNNModelConfig:
    """Configuration for AVTemporalModelCNN."""
    audio_emb_dim: int = 512
    video_emb_dim: int = 2048
    model_dim: int = 256
    num_conv_blocks: int = 4
    kernel_size: int = 5
    hidden_channels: int = 256
    dropout: float = 0.1
    fusion_mlp_hidden: int = 512
    fusion_dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 16
    num_epochs: int = 50
    num_heads: int = 0
    num_layers: int = 0
    audio_embedding_type: str = "openl3"
    video_embedding_type: str = "senet"
    use_audio_labels: bool = True
    filter_dataset: Optional[str] = None


# ============================================================================
# CNN ENCODER & MODEL
# ============================================================================

class ModalityEncoderCNN(nn.Module):
    """1D CNN encoder: (B, T, d) -> (B, T, model_dim)."""

    def __init__(self, input_dim: int, model_dim: int, num_blocks: int = 4, kernel_size: int = 5,
                 hidden_channels: int = 256, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        layers = []
        in_c = input_dim
        for i in range(num_blocks):
            out_c = model_dim if i == num_blocks - 1 else hidden_channels
            layers += [nn.Conv1d(in_c, out_c, kernel_size, padding=pad), nn.BatchNorm1d(out_c)]
            if i < num_blocks - 1:
                layers += [nn.GELU(), nn.Dropout(dropout)]
            in_c = out_c
        self.conv_net = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv_net(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class AVTemporalModelCNN(nn.Module):
    """CNN-based temporal model: segment-level (B,T,d) -> (B,T) logits. forward(..., valid_length=...) -> (logits, {})."""

    def __init__(self, config: CNNModelConfig):
        super().__init__()
        self.config = config
        self.audio_encoder = ModalityEncoderCNN(
            config.audio_emb_dim, config.model_dim, config.num_conv_blocks,
            config.kernel_size, config.hidden_channels, config.dropout,
        )
        self.video_encoder = ModalityEncoderCNN(
            config.video_emb_dim, config.model_dim, config.num_conv_blocks,
            config.kernel_size, config.hidden_channels, config.dropout,
        )
        self.fusion_head = FusionHead(config.model_dim, config.fusion_mlp_hidden, config.fusion_dropout, use_fusion_mlp=True)

    def forward(self, audio_seq: torch.Tensor, video_seq: torch.Tensor,
                valid_length: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        a = self.audio_encoder(audio_seq)
        v = self.video_encoder(video_seq)
        logits = self.fusion_head(a, v)
        return logits, {}


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def _metrics(all_logits: List, all_labels: List) -> Dict[str, float]:
    log_flat = np.concatenate([a.flatten() for a in all_logits])
    lab_flat = np.concatenate([a.flatten() for a in all_labels])
    lab_bin = (lab_flat > 0.5).astype(int)
    uniq = np.unique(lab_bin)
    cnt = {int(u): int(np.sum(lab_bin == u)) for u in uniq}
    probs = 1 / (1 + np.exp(-log_flat))
    preds = (probs > 0.5).astype(int)
    acc = np.mean(preds == lab_bin)
    try:
        auroc = roc_auc_score(lab_bin, log_flat) if len(uniq) >= 2 else 0.0
    except Exception:
        auroc = 0.0
    print(f"  Labels: {cnt}, Accuracy: {acc:.4f}, AUROC: {auroc:.4f}")
    return {"loss": 0.0, "auroc": auroc, "accuracy": acc}


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        audio = batch["audio_seq"].to(device)
        video = batch["video_seq"].to(device)
        label = batch["label_seq"].to(device)
        valid = batch.get("valid_length")
        if valid is not None:
            valid = valid.to(device)
        logits, _ = model(audio, video, valid_length=valid)
        mask = torch.zeros_like(label, dtype=torch.bool)
        if valid is not None:
            for b in range(label.shape[0]):
                mask[b, :valid[b]] = True
        else:
            mask.fill_(True)
        log_m = logits.masked_fill(~mask, 0.0)
        lab_m = label.masked_fill(~mask, 0.0)
        loss = criterion(log_m, lab_m)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        for b in range(label.shape[0]):
            ln = int(valid[b].item()) if valid is not None else label.shape[1]
            all_logits.append(logits[b, :ln].detach().cpu().numpy())
            all_labels.append(label[b, :ln].detach().cpu().numpy())
    avg = total_loss / len(dataloader)
    m = _metrics(all_logits, all_labels)
    m["loss"] = avg
    return m


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device,
                   epoch: int) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Val {epoch}"):
            audio = batch["audio_seq"].to(device)
            video = batch["video_seq"].to(device)
            label = batch["label_seq"].to(device)
            valid = batch.get("valid_length")
            if valid is not None:
                valid = valid.to(device)
            logits, _ = model(audio, video, valid_length=valid)
            mask = torch.zeros_like(label, dtype=torch.bool)
            if valid is not None:
                for b in range(label.shape[0]):
                    mask[b, :valid[b]] = True
            else:
                mask.fill_(True)
            log_m = logits.masked_fill(~mask, 0.0)
            lab_m = label.masked_fill(~mask, 0.0)
            loss = criterion(log_m, lab_m)
            total_loss += loss.item()
            for b in range(label.shape[0]):
                ln = int(valid[b].item()) if valid is not None else label.shape[1]
                all_logits.append(logits[b, :ln].cpu().numpy())
                all_labels.append(label[b, :ln].cpu().numpy())
    avg = total_loss / len(dataloader)
    m = _metrics(all_logits, all_labels)
    m["loss"] = avg
    return m


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader], config: CNNModelConfig,
                device: torch.device, save_dir: str = "./checkpoints") -> Dict:
    os.makedirs(save_dir, exist_ok=True)
    history = {
        "train_loss": [], "train_auroc": [], "train_accuracy": [],
        "val_loss": [], "val_auroc": [], "val_accuracy": [],
        "epochs": [], "best_epoch": None, "best_val_auroc": 0.0,
        "config": {
            "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
            "batch_size": config.batch_size, "num_epochs": config.num_epochs,
            "model_dim": config.model_dim, "num_heads": config.num_heads, "num_layers": config.num_layers,
        },
    }
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_val_auroc = 0.0

    for epoch in range(1, config.num_epochs + 1):
        tr = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Epoch {epoch} Train - Loss: {tr['loss']:.4f}, AUROC: {tr['auroc']:.4f}, Acc: {tr['accuracy']:.4f}")
        history["train_loss"].append(float(tr["loss"]))
        history["train_auroc"].append(float(tr["auroc"]))
        history["train_accuracy"].append(float(tr["accuracy"]))
        history["epochs"].append(epoch)

        if val_loader is not None:
            va = validate_epoch(model, val_loader, criterion, device, epoch)
            print(f"Epoch {epoch} Val   - Loss: {va['loss']:.4f}, AUROC: {va['auroc']:.4f}, Acc: {va['accuracy']:.4f}")
            history["val_loss"].append(float(va["loss"]))
            history["val_auroc"].append(float(va["auroc"]))
            history["val_accuracy"].append(float(va["accuracy"]))
            if va["auroc"] > best_val_auroc:
                best_val_auroc = va["auroc"]
                history["best_epoch"] = epoch
                history["best_val_auroc"] = float(best_val_auroc)
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                           "optimizer_state_dict": optimizer.state_dict(), "val_auroc": best_val_auroc},
                          os.path.join(save_dir, "best_model.pt"))
                print(f"Saved best model with AUROC: {best_val_auroc:.4f}")

        with open(os.path.join(save_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n✅ Training history saved to {save_dir}/training_history.json")
    return history


def test_model(model: nn.Module, test_loader: DataLoader, device: torch.device,
               checkpoint_path: Optional[str] = None) -> Dict[str, float]:
    if checkpoint_path and os.path.exists(checkpoint_path):
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print(f"✓ Loaded {checkpoint_path} (epoch {ck.get('epoch','?')}, val_auroc {ck.get('val_auroc',0):.4f})")
    criterion = nn.BCEWithLogitsLoss()
    m = validate_epoch(model, test_loader, criterion, device, 0)
    print("\n" + "=" * 60 + "\nTEST: Loss {:.4f}, AUROC {:.4f}, Acc {:.4f}\n".format(m["loss"], m["auroc"], m["accuracy"]) + "=" * 60)
    return m


# ============================================================================
# MAIN & TEST
# ============================================================================

def main():
    config = CNNModelConfig(
        audio_emb_dim=512, video_emb_dim=2048, model_dim=256, num_conv_blocks=4, kernel_size=5,
        hidden_channels=256, dropout=0.1, fusion_mlp_hidden=512, fusion_dropout=0.1,
        learning_rate=1e-4, weight_decay=1e-5, batch_size=16, num_epochs=50,
        audio_embedding_type="openl3", video_embedding_type="senet", use_audio_labels=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hdf5_path = "/Users/jerrysheng/Desktop/Lab/deepfake_embeddings_2.h5"

    av = AVH5Dataset(hdf5_path, config.audio_embedding_type, config.video_embedding_type, config.use_audio_labels, None, "avdeepfake1m")
    sh = AVH5Dataset(hdf5_path, config.audio_embedding_type, config.video_embedding_type, config.use_audio_labels, None, "shareveo3")
    from torch.utils.data import ConcatDataset
    combined = ConcatDataset([av, sh])
    tr_sz = int(0.8 * len(combined))
    train_ds, val_ds = torch.utils.data.random_split(combined, [tr_sz, len(combined) - tr_sz])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=(device.type == "cuda"))

    model = AVTemporalModelCNN(config).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"AVTemporalModelCNN: {n:,} params")

    train_model(model, train_loader, val_loader, config, device, save_dir="./checkpoints")


def test_main(hdf5_path: str, checkpoint_path: str, filter_dataset: Optional[str] = None,
              audio_embedding_type: str = "openl3", video_embedding_type: str = "senet",
              use_audio_labels: bool = True, batch_size: int = 16) -> Dict[str, float]:
    config = CNNModelConfig(audio_emb_dim=512, video_emb_dim=2048, model_dim=256, num_conv_blocks=4, kernel_size=5,
                           hidden_channels=256, dropout=0.1, batch_size=batch_size, num_epochs=1,
                           audio_embedding_type=audio_embedding_type, video_embedding_type=video_embedding_type,
                           use_audio_labels=use_audio_labels, filter_dataset=filter_dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AVH5Dataset(hdf5_path, config.audio_embedding_type, config.video_embedding_type, config.use_audio_labels, None, filter_dataset)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=(device.type == "cuda"))
    model = AVTemporalModelCNN(config).to(device)
    return test_model(model, loader, device, checkpoint_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) < 4:
            print("Usage: python time_series_cnn_model.py test <hdf5_path> <checkpoint_path> [filter_dataset]")
            sys.exit(1)
        test_main(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        main()
