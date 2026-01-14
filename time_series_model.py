"""
PatchTST-style Temporal Transformer for Multimodal Deepfake Detection

This module implements a complete time-series model that processes audio + video embeddings
across time, constructs temporal patches, encodes them with separate Transformer encoders
(one per modality), fuses the two streams, and outputs per-segment real/fake logits.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for AVTemporalModel"""
    # Embedding dimensions
    audio_emb_dim: int = 512  # openl3 default
    video_emb_dim: int = 2048  # senet default
    
    # Patch parameters
    patch_size: int = 8  # Number of segments per patch
    patch_stride: int = 4  # Stride for overlapping patches
    
    # Model architecture
    model_dim: int = 256  # Hidden dimension D
    num_heads: int = 8  # Multi-head attention heads
    num_layers: int = 4  # Transformer encoder layers per modality
    dim_feedforward: int = 1024  # FFN dimension
    dropout: float = 0.1
    activation: str = "gelu"
    
    # Fusion
    fusion_mlp_hidden: int = 512  # Hidden dim for fusion MLP
    fusion_dropout: float = 0.1
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 16
    num_epochs: int = 50
    
    # Dataset
    audio_embedding_type: str = "openl3"  # or "hubert"
    video_embedding_type: str = "senet"
    use_audio_labels: bool = True  # Use audio labels, else video labels
    filter_dataset: Optional[str] = None  # "avdeepfake1m" or "shareveo3" or "sora2" or None for all


# ============================================================================
# DATASET
# ============================================================================

class AVH5Dataset(Dataset):
    """
    PyTorch Dataset for loading audio + video embeddings from HDF5.
    
    Each augmentation is treated as a separate example.
    Returns sequences of shape (T, d) where T is the number of segments.
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
        """
        Args:
            hdf5_path: Path to HDF5 file
            audio_embedding_type: Which audio embedding to use ("openl3" or "hubert")
            video_embedding_type: Which video embedding to use ("senet")
            use_audio_labels: If True, use audio labels; else use video labels
            video_ids: Optional list of video_ids to load (None = load all)
            filter_dataset: Filter by dataset ("avdeepfake1m" or "shareveo3" or None for both)
        """
        self.hdf5_path = hdf5_path
        self.audio_embedding_type = audio_embedding_type
        self.video_embedding_type = video_embedding_type
        self.use_audio_labels = use_audio_labels
        self.filter_dataset = filter_dataset
        
        # Build index: list of (video_id, aug_idx, dataset) tuples
        self.samples = []
        
        with h5py.File(hdf5_path, "r") as f:
            # Get video_ids to process
            if video_ids is None:
                video_ids = [vid.decode() if isinstance(vid, bytes) else vid 
                            for vid in f["metadata/video_ids"][:]]
            
            # Build sample index
            videos_grp = f["videos"]
            for video_id in video_ids:
                safe_video_id = video_id.replace("/", "_")
                if safe_video_id not in videos_grp:
                    continue
                
                vid_grp = videos_grp[safe_video_id]
                
                # Check dataset filter
                dataset = vid_grp.attrs.get("dataset", "avdeepfake1m")
                if isinstance(dataset, bytes):
                    dataset = dataset.decode()
                
                # Handle case-insensitive matching and alternative names
                if self.filter_dataset is not None:
                    filter_lower = self.filter_dataset.lower()
                    dataset_lower = dataset.lower()
                    # Support "sora2", "Sora2", "SORA2" variations
                    if filter_lower not in dataset_lower and dataset_lower not in filter_lower:
                        continue
                
                num_augmentations = vid_grp.attrs["num_augmentations"]
                
                # Add each augmentation as a separate sample
                for aug_idx in range(num_augmentations):
                    self.samples.append((video_id, aug_idx, dataset))
        
        dataset_str = f" (filtered: {filter_dataset})" if filter_dataset else ""
        print(f"Loaded {len(self.samples)} samples from {hdf5_path}{dataset_str}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
            - "audio_seq": Tensor(T, d_a)
            - "video_seq": Tensor(T, d_v)
            - "label_seq": Tensor(T) - binary labels (0=fake, 1=real)
            - "video_id": str
            - "aug_idx": int
        """
        video_id, aug_idx, dataset = self.samples[idx]
        safe_video_id = video_id.replace("/", "_")
        
        with h5py.File(self.hdf5_path, "r") as f:
            vid_grp = f["videos"][safe_video_id]
            
            # Load embeddings: shape [num_augs, num_segs, emb_dim]
            audio_emb = vid_grp[f"embeddings/{self.audio_embedding_type}"][aug_idx]  # (T, d_a)
            video_emb = vid_grp[f"embeddings/{self.video_embedding_type}"][aug_idx]  # (T, d_v)
            
            # Load labels: shape [num_augs, num_segs]
            if self.use_audio_labels:
                labels = vid_grp["labels/audio"][aug_idx]  # (T,)
            else:
                labels = vid_grp["labels/video"][aug_idx]  # (T,)
        
        # Convert to tensors
        audio_seq = torch.from_numpy(audio_emb).float()
        video_seq = torch.from_numpy(video_emb).float()
        label_seq = torch.from_numpy(labels).float()
        
        # Convert soft labels to binary (threshold at 0.5)
        # Original convention: 1=fake, 0=real
        # We flip to standard convention: 1=real, 0=fake
        # Labels > 0.5 (original fake) → 0 (fake), Labels <= 0.5 (original real) → 1 (real)
        label_seq_binary = (label_seq <= 0.5).float()
        
        return {
            "audio_seq": audio_seq,
            "video_seq": video_seq,
            "label_seq": label_seq_binary,  # Use binary labels
            "label_seq_soft": label_seq,  # Keep soft labels for reference
            "video_id": video_id,
            "aug_idx": aug_idx,
            "dataset": dataset,
        }


# ============================================================================
# PATCH TOKENIZER
# ============================================================================

class PatchTokenizer(nn.Module):
    """
    Converts time-series sequences into patches.
    
    Input: (B, T, d) where T is number of segments, d is embedding dimension
    Output: (B, N, P*d) where N is number of patches, P is patch size
    """
    
    def __init__(self, patch_size: int, patch_stride: int, use_mean_pooling: bool = False):
        """
        Args:
            patch_size: Number of segments per patch (P)
            patch_stride: Stride for overlapping patches (S)
            use_mean_pooling: If True, output (B, N, d) by mean-pooling patches
                             If False, output (B, N, P*d) by flattening patches
        """
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.use_mean_pooling = use_mean_pooling
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (B, T, d) input sequence
        
        Returns:
            patches: (B, N, P*d) or (B, N, d) if mean_pooling
            patch_info: Dict with mapping information for upsampling
                - "num_patches": int
                - "patch_size": int
                - "patch_stride": int
                - "segment_to_patches": List[List[int]] - for each segment, which patches cover it
        """
        B, T, d = x.shape
        
        # Create patches using unfold
        # We'll use unfold on the time dimension
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)  # (B, N, d, P)
        N = patches.shape[1]
        
        # Rearrange: (B, N, d, P) -> (B, N, P, d)
        patches = patches.permute(0, 1, 3, 2)
        
        if self.use_mean_pooling:
            # Mean pool over patch dimension: (B, N, P, d) -> (B, N, d)
            patches = patches.mean(dim=2)
        else:
            # Flatten: (B, N, P, d) -> (B, N, P*d)
            patches = patches.reshape(B, N, self.patch_size * d)
        
        # Build segment-to-patches mapping for upsampling
        segment_to_patches = []
        for seg_idx in range(T):
            covering_patches = []
            for patch_idx in range(N):
                patch_start = patch_idx * self.patch_stride
                patch_end = patch_start + self.patch_size
                if patch_start <= seg_idx < patch_end:
                    covering_patches.append(patch_idx)
            segment_to_patches.append(covering_patches)
        
        patch_info = {
            "num_patches": N,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "segment_to_patches": segment_to_patches,
            "original_length": T,
        }
        
        return patches, patch_info


# ============================================================================
# MODALITY ENCODER
# ============================================================================

class ModalityEncoder(nn.Module):
    """
    Encodes patches for a single modality.
    
    Components:
    1. Patch projection: projects patch tokens to model dimension D
    2. Positional encoding: adds learnable positional embeddings
    3. Transformer encoder: contextualizes tokens
    """
    
    def __init__(
        self,
        patch_dim: int,  # P*d or d (depending on mean_pooling)
        model_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.model_dim = model_dim
        
        # Patch projection
        self.patch_projection = nn.Linear(patch_dim, model_dim)
        
        # Positional encoding (learnable)
        # We'll use a max sequence length of 1000 patches (can be adjusted)
        max_len = 1000
        self.pos_encoder = nn.Parameter(torch.randn(max_len, model_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,  # Use batch_first=True for (B, N, D) format
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, N, patch_dim) patch tokens
        
        Returns:
            encoded: (B, N, model_dim) contextualized tokens
        """
        B, N, _ = patches.shape
        
        # Project patches to model dimension
        x = self.patch_projection(patches)  # (B, N, model_dim)
        
        # Add positional encoding
        pos_emb = self.pos_encoder[:N].unsqueeze(0)  # (1, N, model_dim)
        x = x + pos_emb
        
        # Apply dropout
        x = self.dropout(x)
        
        # Transformer encoding
        encoded = self.transformer(x)  # (B, N, model_dim)
        
        return encoded


# ============================================================================
# FUSION HEAD
# ============================================================================

class FusionHead(nn.Module):
    """
    Fuses audio and video tokens and outputs per-patch logits.
    """
    
    def __init__(
        self,
        model_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_fusion_mlp: bool = True,
    ):
        """
        Args:
            model_dim: Dimension of each modality's tokens (D)
            hidden_dim: Hidden dimension for fusion MLP
            dropout: Dropout rate
            use_fusion_mlp: If True, apply MLP after concatenation
        """
        super().__init__()
        self.use_fusion_mlp = use_fusion_mlp
        
        if use_fusion_mlp:
            # Concatenate audio + video: 2*model_dim -> hidden_dim -> 1
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
            # Simple linear projection
            self.fusion_mlp = nn.Linear(2 * model_dim, 1)
    
    def forward(self, audio_tokens: torch.Tensor, video_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_tokens: (B, N, model_dim)
            video_tokens: (B, N, model_dim)
        
        Returns:
            logits: (B, N) per-patch logits
        """
        # Concatenate along feature dimension
        fused = torch.cat([audio_tokens, video_tokens], dim=-1)  # (B, N, 2*model_dim)
        
        # Apply fusion MLP
        logits = self.fusion_mlp(fused).squeeze(-1)  # (B, N)
        
        return logits


# ============================================================================
# UPSAMPLING
# ============================================================================

def upsample_patch_logits(
    patch_logits: torch.Tensor,
    patch_info: Dict,
    method: str = "average",
    valid_length: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convert patch-level logits to segment-level logits.
    
    Args:
        patch_logits: (B, N) patch logits
        patch_info: Dict from PatchTokenizer with segment_to_patches mapping
        method: "average" or "max" for handling overlapping patches
        valid_length: Optional (B,) tensor with actual sequence lengths (for padding handling)
    
    Returns:
        segment_logits: (B, T) segment logits
    """
    B, N = patch_logits.shape
    T = patch_info["original_length"]
    segment_to_patches = patch_info["segment_to_patches"]
    
    segment_logits = torch.zeros(B, T, device=patch_logits.device, dtype=patch_logits.dtype)
    
    # For each segment, collect logits from all covering patches
    for seg_idx in range(T):
        covering_patches = segment_to_patches[seg_idx]
        if len(covering_patches) == 0:
            continue
        
        # Gather logits from covering patches
        patch_logits_for_seg = patch_logits[:, covering_patches]  # (B, num_covering_patches)
        
        if method == "average":
            segment_logits[:, seg_idx] = patch_logits_for_seg.mean(dim=1)
        elif method == "max":
            segment_logits[:, seg_idx] = patch_logits_for_seg.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown upsampling method: {method}")
    
    # Mask out padding if valid_length is provided
    if valid_length is not None:
        for b in range(B):
            if valid_length[b] < T:
                segment_logits[b, valid_length[b]:] = float('-inf')
    
    return segment_logits


# ============================================================================
# FULL MODEL
# ============================================================================

class AVTemporalModel(nn.Module):
    """
    Complete PatchTST-style temporal Transformer for multimodal deepfake detection.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Patch tokenizers
        audio_patch_dim = config.audio_emb_dim if config.patch_size == 1 else config.patch_size * config.audio_emb_dim
        video_patch_dim = config.video_emb_dim if config.patch_size == 1 else config.patch_size * config.video_emb_dim
        
        # Use mean pooling if patch_size > 1 to keep dimensions manageable
        use_mean_pooling = config.patch_size > 1
        if use_mean_pooling:
            audio_patch_dim = config.audio_emb_dim
            video_patch_dim = config.video_emb_dim
        
        self.audio_tokenizer = PatchTokenizer(
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            use_mean_pooling=use_mean_pooling,
        )
        self.video_tokenizer = PatchTokenizer(
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            use_mean_pooling=use_mean_pooling,
        )
        
        # Modality encoders
        self.audio_encoder = ModalityEncoder(
            patch_dim=audio_patch_dim,
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.video_encoder = ModalityEncoder(
            patch_dim=video_patch_dim,
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
        )
        
        # Fusion head
        self.fusion_head = FusionHead(
            model_dim=config.model_dim,
            hidden_dim=config.fusion_mlp_hidden,
            dropout=config.fusion_dropout,
            use_fusion_mlp=True,
        )
    
    def forward(
        self,
        audio_seq: torch.Tensor,
        video_seq: torch.Tensor,
        valid_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            audio_seq: (B, T, d_a) audio embeddings
            video_seq: (B, T, d_v) video embeddings
            valid_length: Optional (B,) tensor with actual sequence lengths (for padding)
        
        Returns:
            segment_logits: (B, T) per-segment logits
            patch_info: Dict with patch information (from audio tokenizer)
        """
        # Tokenize into patches
        audio_patches, audio_patch_info = self.audio_tokenizer(audio_seq)
        video_patches, _ = self.video_tokenizer(video_seq)
        
        # Encode patches
        audio_tokens = self.audio_encoder(audio_patches)  # (B, N, D)
        video_tokens = self.video_encoder(video_patches)  # (B, N, D)
        
        # Fuse and get patch logits
        patch_logits = self.fusion_head(audio_tokens, video_tokens)  # (B, N)
        
        # Upsample to segment level
        segment_logits = upsample_patch_logits(
            patch_logits, audio_patch_info, method="average", valid_length=valid_length
        )
        
        return segment_logits, audio_patch_info


# ============================================================================
# COLLATE FUNCTION
# ============================================================================

def collate_fn(batch):
    """Collate function for variable-length sequences"""
    audio_seqs = [item["audio_seq"] for item in batch]
    video_seqs = [item["video_seq"] for item in batch]
    label_seqs = [item["label_seq"] for item in batch]
    video_ids = [item["video_id"] for item in batch]
    aug_indices = [item["aug_idx"] for item in batch]
    datasets = [item.get("dataset", "unknown") for item in batch]
    
    # Pad to same length (use max length in batch)
    max_len = max(seq.shape[0] for seq in audio_seqs)
    valid_lengths = torch.tensor([seq.shape[0] for seq in audio_seqs], dtype=torch.long)
    
    def pad_sequence(seq, max_len):
        if seq.shape[0] < max_len:
            padding = torch.zeros(max_len - seq.shape[0], seq.shape[1])
            return torch.cat([seq, padding], dim=0)
        return seq
    
    audio_padded = torch.stack([pad_sequence(seq, max_len) for seq in audio_seqs])
    video_padded = torch.stack([pad_sequence(seq, max_len) for seq in video_seqs])
    label_padded = torch.stack([pad_sequence(seq.unsqueeze(-1), max_len).squeeze(-1) for seq in label_seqs])
    
    return {
        "audio_seq": audio_padded,
        "video_seq": video_padded,
        "label_seq": label_padded,
        "valid_length": valid_lengths,
        "video_id": video_ids,
        "aug_idx": aug_indices,
        "dataset": datasets,
    }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model: AVTemporalModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        audio_seq = batch["audio_seq"].to(device)  # (B, T, d_a)
        video_seq = batch["video_seq"].to(device)  # (B, T, d_v)
        label_seq = batch["label_seq"].to(device)  # (B, T)
        valid_length = batch.get("valid_length", None)
        if valid_length is not None:
            valid_length = valid_length.to(device)
        
        # Forward pass
        segment_logits, _ = model(audio_seq, video_seq, valid_length=valid_length)  # (B, T)
        
        # Create mask for valid segments
        if valid_length is not None:
            mask = torch.zeros_like(label_seq, dtype=torch.bool)
            for b in range(label_seq.shape[0]):
                mask[b, :valid_length[b]] = True
        else:
            mask = torch.ones_like(label_seq, dtype=torch.bool)
        
        # Mask out padding in loss computation
        segment_logits_masked = segment_logits.masked_fill(~mask, 0.0)
        label_seq_masked = label_seq.masked_fill(~mask, 0.0)
        
        # Compute loss
        loss = criterion(segment_logits_masked, label_seq_masked)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        
        # Store logits and labels (only valid segments)
        for b in range(label_seq.shape[0]):
            if valid_length is not None:
                valid_len = valid_length[b].item()
                all_logits.append(segment_logits[b, :valid_len].detach().cpu().numpy())
                all_labels.append(label_seq[b, :valid_len].detach().cpu().numpy())
            else:
                all_logits.append(segment_logits[b].detach().cpu().numpy())
                all_labels.append(label_seq[b].detach().cpu().numpy())
        
        pbar.set_postfix({"loss": loss.item()})
    
    # Compute metrics
    all_logits_flat = np.concatenate([arr.flatten() for arr in all_logits])
    all_labels_flat = np.concatenate([arr.flatten() for arr in all_labels])
    
    # Convert labels to binary for AUROC (should already be binary, but ensure)
    all_labels_binary = (all_labels_flat > 0.5).astype(int)
    
    # Diagnostic info
    unique_labels = np.unique(all_labels_binary)
    label_counts = {int(label): int(np.sum(all_labels_binary == label)) for label in unique_labels}
    mean_logit = np.mean(all_logits_flat)
    std_logit = np.std(all_logits_flat)
    
    # Calculate predictions for accuracy
    all_probs = 1 / (1 + np.exp(-all_logits_flat))  # Sigmoid
    all_preds = (all_probs > 0.5).astype(int)
    accuracy = np.mean(all_preds == all_labels_binary)
    
    avg_loss = total_loss / len(dataloader)
    try:
        # Check if we have both classes
        if len(unique_labels) < 2:
            print(f"  Warning: Only one class present in labels: {unique_labels}")
            auroc = 0.0
        else:
            auroc = roc_auc_score(all_labels_binary, all_logits_flat)
    except ValueError as e:
        print(f"  Warning: AUROC computation failed: {e}")
        auroc = 0.0
    
    # Print diagnostics every epoch (to track loss vs metrics relationship)
    # Calculate confidence metrics
    correct_mask = (all_preds == all_labels_binary)
    correct_probs = all_probs[correct_mask]
    incorrect_probs = all_probs[~correct_mask] if np.sum(~correct_mask) > 0 else np.array([])
    
    print(f"  Label distribution (binary): {label_counts}")
    print(f"  Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}")
    print(f"  Logit stats: mean={mean_logit:.4f}, std={std_logit:.4f}")
    print(f"  Prediction range: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
    if len(correct_probs) > 0:
        print(f"  Correct predictions: mean_prob={correct_probs.mean():.4f}, median={np.median(correct_probs):.4f}")
    if len(incorrect_probs) > 0:
        print(f"  Incorrect predictions: mean_prob={incorrect_probs.mean():.4f}, median={np.median(incorrect_probs):.4f}")
    
    # Explain loss vs metrics
    avg_prob_correct = correct_probs.mean() if len(correct_probs) > 0 else 0.5
    theoretical_loss_at_confidence = -np.log(avg_prob_correct) if avg_prob_correct > 0 else 0.693
    print(f"  Loss explanation: avg_loss={avg_loss:.4f}, theoretical_min_loss_at_current_confidence≈{theoretical_loss_at_confidence:.4f}")
    print(f"    (Lower loss requires higher confidence in correct predictions)")
    
    return {
        "loss": avg_loss,
        "auroc": auroc,
        "accuracy": accuracy,
    }


def validate_epoch(
    model: AVTemporalModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Val {epoch}")
        for batch in pbar:
            audio_seq = batch["audio_seq"].to(device)
            video_seq = batch["video_seq"].to(device)
            label_seq = batch["label_seq"].to(device)
            valid_length = batch.get("valid_length", None)
            if valid_length is not None:
                valid_length = valid_length.to(device)
            
            # Forward pass
            segment_logits, _ = model(audio_seq, video_seq, valid_length=valid_length)
            
            # Create mask for valid segments
            if valid_length is not None:
                mask = torch.zeros_like(label_seq, dtype=torch.bool)
                for b in range(label_seq.shape[0]):
                    mask[b, :valid_length[b]] = True
            else:
                mask = torch.ones_like(label_seq, dtype=torch.bool)
            
            # Mask out padding in loss computation
            segment_logits_masked = segment_logits.masked_fill(~mask, 0.0)
            label_seq_masked = label_seq.masked_fill(~mask, 0.0)
            
            # Compute loss
            loss = criterion(segment_logits_masked, label_seq_masked)
            
            # Accumulate metrics
            total_loss += loss.item()
            
            # Store logits and labels (only valid segments)
            for b in range(label_seq.shape[0]):
                if valid_length is not None:
                    valid_len = valid_length[b].item()
                    all_logits.append(segment_logits[b, :valid_len].cpu().numpy())
                    all_labels.append(label_seq[b, :valid_len].cpu().numpy())
                else:
                    all_logits.append(segment_logits[b].cpu().numpy())
                    all_labels.append(label_seq[b].cpu().numpy())
            
            pbar.set_postfix({"loss": loss.item()})
    
    # Compute metrics
    all_logits_flat = np.concatenate([arr.flatten() for arr in all_logits])
    all_labels_flat = np.concatenate([arr.flatten() for arr in all_labels])
    
    # Convert labels to binary for AUROC (should already be binary, but ensure)
    all_labels_binary = (all_labels_flat > 0.5).astype(int)
    
    # Diagnostic info
    unique_labels = np.unique(all_labels_binary)
    label_counts = {int(label): int(np.sum(all_labels_binary == label)) for label in unique_labels}
    mean_logit = np.mean(all_logits_flat)
    std_logit = np.std(all_logits_flat)
    
    # Calculate predictions for accuracy
    all_probs = 1 / (1 + np.exp(-all_logits_flat))  # Sigmoid
    all_preds = (all_probs > 0.5).astype(int)
    accuracy = np.mean(all_preds == all_labels_binary)
    
    avg_loss = total_loss / len(dataloader)
    try:
        # Check if we have both classes
        if len(unique_labels) < 2:
            print(f"  Warning: Only one class present in labels: {unique_labels}")
            auroc = 0.0
        else:
            auroc = roc_auc_score(all_labels_binary, all_logits_flat)
    except ValueError as e:
        print(f"  Warning: AUROC computation failed: {e}")
        auroc = 0.0
    
    # Print diagnostics every epoch (to track loss vs metrics relationship)
    # Calculate confidence metrics
    correct_mask = (all_preds == all_labels_binary)
    correct_probs = all_probs[correct_mask]
    incorrect_probs = all_probs[~correct_mask] if np.sum(~correct_mask) > 0 else np.array([])
    
    print(f"  Label distribution (binary): {label_counts}")
    print(f"  Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}")
    print(f"  Logit stats: mean={mean_logit:.4f}, std={std_logit:.4f}")
    print(f"  Prediction range: [{all_probs.min():.4f}, {all_probs.max():.4f}]")
    if len(correct_probs) > 0:
        print(f"  Correct predictions: mean_prob={correct_probs.mean():.4f}, median={np.median(correct_probs):.4f}")
    if len(incorrect_probs) > 0:
        print(f"  Incorrect predictions: mean_prob={incorrect_probs.mean():.4f}, median={np.median(incorrect_probs):.4f}")
    
    # Explain loss vs metrics
    avg_prob_correct = correct_probs.mean() if len(correct_probs) > 0 else 0.5
    theoretical_loss_at_confidence = -np.log(avg_prob_correct) if avg_prob_correct > 0 else 0.693
    print(f"  Loss explanation: avg_loss={avg_loss:.4f}, theoretical_min_loss_at_current_confidence≈{theoretical_loss_at_confidence:.4f}")
    print(f"    (Lower loss requires higher confidence in correct predictions)")
    
    return {
        "loss": avg_loss,
        "auroc": auroc,
        "accuracy": accuracy,
    }


def train_model(
    model: AVTemporalModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: ModelConfig,
    device: torch.device,
    save_dir: str = "./checkpoints",
):
    """
    Full training loop with per-epoch history tracking.
    
    Saves training history to training_history.json alongside the checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_auroc': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_auroc': [],
        'val_accuracy': [],
        'epochs': [],
        'best_epoch': None,
        'best_val_auroc': 0.0,
        'config': {
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'batch_size': config.batch_size,
            'num_epochs': config.num_epochs,
            'model_dim': config.model_dim,
            'num_heads': config.num_heads,
            'num_layers': config.num_layers,
        }
    }
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    best_val_auroc = 0.0
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Epoch {epoch} Train - Loss: {train_metrics['loss']:.4f}, AUROC: {train_metrics['auroc']:.4f}, Acc: {train_metrics.get('accuracy', 0.0):.4f}")
        
        # Store training metrics
        history['train_loss'].append(float(train_metrics['loss']))
        history['train_auroc'].append(float(train_metrics['auroc']))
        history['train_accuracy'].append(float(train_metrics.get('accuracy', 0.0)))
        history['epochs'].append(epoch)
        
        # Validate
        if val_loader is not None:
            val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
            print(f"Epoch {epoch} Val   - Loss: {val_metrics['loss']:.4f}, AUROC: {val_metrics['auroc']:.4f}, Acc: {val_metrics.get('accuracy', 0.0):.4f}")
            
            # Store validation metrics
            history['val_loss'].append(float(val_metrics['loss']))
            history['val_auroc'].append(float(val_metrics['auroc']))
            history['val_accuracy'].append(float(val_metrics.get('accuracy', 0.0)))
            
            # Save best model
            if val_metrics['auroc'] > best_val_auroc:
                best_val_auroc = val_metrics['auroc']
                history['best_epoch'] = epoch
                history['best_val_auroc'] = float(best_val_auroc)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auroc': best_val_auroc,
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"Saved best model with AUROC: {best_val_auroc:.4f}")
        
        # Save history after each epoch (so you don't lose progress if training stops)
        history_file = os.path.join(save_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f"\n✅ Training history saved to: {history_file}")
    return history


# ============================================================================
# TESTING FUNCTION
# ============================================================================

def test_model(
    model: AVTemporalModel,
    test_loader: DataLoader,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Test/evaluate the model on a test dataset.
    
    Args:
        model: The model to test
        test_loader: DataLoader for test data
        device: Device to run on
        checkpoint_path: Optional path to checkpoint to load
    
    Returns:
        Dictionary with test metrics (loss, auroc, accuracy)
    """
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
        if 'val_auroc' in checkpoint:
            print(f"  Checkpoint validation AUROC: {checkpoint['val_auroc']:.4f}")
    
    # Use validation epoch function for testing
    criterion = nn.BCEWithLogitsLoss()
    test_metrics = validate_epoch(model, test_loader, criterion, device, epoch=0)
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print("="*60)
    
    # Additional diagnostics for AUROC=0.0 issue
    if test_metrics['auroc'] == 0.0:
        print("\n⚠️  WARNING: AUROC is 0.0 - this usually means:")
        print("   1. All labels in the test set are the same class (all 0 or all 1)")
        print("   2. All model predictions are the same class")
        print("   Check the diagnostic output above for label/prediction distribution")
    
    return test_metrics


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    """Example training script"""
    # Configuration
    config = ModelConfig(
        audio_emb_dim=512,  # openl3
        video_emb_dim=2048,  # senet (updated to match your embeddings)
        patch_size=8,
        patch_stride=4,
        model_dim=256,
        num_heads=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        learning_rate=1e-4,
        weight_decay=1e-5,
        batch_size=16,
        num_epochs=50,
        audio_embedding_type="openl3",
        video_embedding_type="senet",
        use_audio_labels=True,
    )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset paths
    hdf5_path = "/Users/jerrysheng/Desktop/Lab/deepfake_embeddings_2.h5"
    
    # Create separate datasets for AVDeepfake and ShareVeo3
    print("\n" + "="*60)
    print("Loading AVDeepfake1M dataset...")
    print("="*60)
    avdeepfake_dataset = AVH5Dataset(
        hdf5_path=hdf5_path,
        audio_embedding_type=config.audio_embedding_type,
        video_embedding_type=config.video_embedding_type,
        use_audio_labels=config.use_audio_labels,
        video_ids=None,
        filter_dataset="avdeepfake1m",
    )
    
    print("\n" + "="*60)
    print("Loading ShareVeo3 dataset...")
    print("="*60)
    shareveo3_dataset = AVH5Dataset(
        hdf5_path=hdf5_path,
        audio_embedding_type=config.audio_embedding_type,
        video_embedding_type=config.video_embedding_type,
        use_audio_labels=config.use_audio_labels,
        video_ids=None,
        filter_dataset="shareveo3",
    )
    
    # Combine datasets (you can also train separately if desired)
    # For now, we'll use only AVDeepfake for training, ShareVeo3 for validation
    # Or combine them - you can adjust this strategy
    print(f"\nDataset sizes:")
    print(f"  AVDeepfake1M: {len(avdeepfake_dataset)} samples")
    print(f"  ShareVeo3: {len(shareveo3_dataset)} samples")
    
    # Strategy: Use AVDeepfake for train/val split, ShareVeo3 as separate test set
    # Or combine both - let's combine for now
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset([avdeepfake_dataset, shareveo3_dataset])
    
    # Split into train/val (simple 80/20 split)
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, val_size]
    )
    
    print(f"\nTrain/Val split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    
    # Create dataloaders
    # Note: collate_fn is defined at module level to support multiprocessing
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Create model
    model = AVTemporalModel(config).to(device)
    
    # Print model info with breakdown
    def count_parameters(model):
        """Count parameters by module"""
        total = 0
        breakdown = {}
        for name, param in model.named_parameters():
            count = param.numel()
            total += count
            module_name = name.split('.')[0]
            if module_name not in breakdown:
                breakdown[module_name] = 0
            breakdown[module_name] += count
        return total, breakdown
    
    total_params, param_breakdown = count_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print("MODEL PARAMETER BREAKDOWN")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nBy module:")
    for module, count in sorted(param_breakdown.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_params
        print(f"  {module:20s}: {count:>10,} ({pct:5.1f}%)")
    
    # Detailed breakdown
    print(f"\nDetailed breakdown:")
    audio_enc_params = sum(p.numel() for p in model.audio_encoder.parameters())
    video_enc_params = sum(p.numel() for p in model.video_encoder.parameters())
    fusion_params = sum(p.numel() for p in model.fusion_head.parameters())
    print(f"  Audio Encoder:  {audio_enc_params:>10,} ({100*audio_enc_params/total_params:5.1f}%)")
    print(f"  Video Encoder:  {video_enc_params:>10,} ({100*video_enc_params/total_params:5.1f}%)")
    print(f"  Fusion Head:    {fusion_params:>10,} ({100*fusion_params/total_params:5.1f}%)")
    print(f"{'='*60}\n")
    
    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir="./checkpoints",
    )


def test_main(
    hdf5_path: str,
    checkpoint_path: str,
    filter_dataset: Optional[str] = None,
    audio_embedding_type: str = "openl3",
    video_embedding_type: str = "senet",
    use_audio_labels: bool = True,
    batch_size: int = 16,
):
    """
    Test script for evaluating a trained model on a new dataset.
    
    Args:
        hdf5_path: Path to HDF5 file containing test data
        checkpoint_path: Path to model checkpoint (.pt file)
        filter_dataset: Optional dataset filter ("sora2", "avdeepfake1m", etc.)
        audio_embedding_type: Type of audio embedding to use
        video_embedding_type: Type of video embedding to use
        use_audio_labels: Whether to use audio labels
        batch_size: Batch size for testing
    """
    # Configuration (same as training config)
    config = ModelConfig(
        audio_emb_dim=512,  # openl3
        video_emb_dim=2048,  # senet
        patch_size=8,
        patch_stride=4,
        model_dim=256,
        num_heads=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        learning_rate=1e-4,
        weight_decay=1e-5,
        batch_size=batch_size,
        num_epochs=1,  # Not used for testing
        audio_embedding_type=audio_embedding_type,
        video_embedding_type=video_embedding_type,
        use_audio_labels=use_audio_labels,
        filter_dataset=filter_dataset,
    )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test dataset
    print("\n" + "="*60)
    filter_str = f" (filtered: {filter_dataset})" if filter_dataset else ""
    print(f"Loading test dataset from: {hdf5_path}{filter_str}")
    print("="*60)
    
    test_dataset = AVH5Dataset(
        hdf5_path=hdf5_path,
        audio_embedding_type=config.audio_embedding_type,
        video_embedding_type=config.video_embedding_type,
        use_audio_labels=config.use_audio_labels,
        video_ids=None,
        filter_dataset=filter_dataset,
    )
    
    print(f"\nTest dataset size: {len(test_dataset)} samples")
    
    # Check label distribution in the dataset
    print("\nChecking label distribution...")
    all_labels_check = []
    for i in range(min(100, len(test_dataset))):  # Sample first 100 samples
        sample = test_dataset[i]
        labels = sample['label_seq'].numpy()
        all_labels_check.extend(labels.flatten())
    
    if len(all_labels_check) > 0:
        unique_labels, counts = np.unique(np.array(all_labels_check), return_counts=True)
        print(f"  Label distribution (from first {min(100, len(test_dataset))} samples):")
        for label, count in zip(unique_labels, counts):
            pct = 100 * count / len(all_labels_check)
            label_name = "real" if label > 0.5 else "fake"
            print(f"    {label_name} (value={label:.2f}): {count:,} ({pct:.1f}%)")
        
        if len(unique_labels) == 1:
            print("\n  ⚠️  WARNING: Only one class found in labels!")
            print("     AUROC will be 0.0 because AUROC requires both classes.")
            print("     This might be normal if sora2 dataset only contains one class (all fake or all real).")
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Create model
    model = AVTemporalModel(config).to(device)
    
    # Test model
    test_metrics = test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    
    return test_metrics


if __name__ == "__main__":
    import sys
    
    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode: python time_series_model.py test <hdf5_path> <checkpoint_path> [filter_dataset]
        if len(sys.argv) < 4:
            print("Usage: python time_series_model.py test <hdf5_path> <checkpoint_path> [filter_dataset]")
            print("\nExample:")
            print("  python time_series_model.py test sora2_embeddings.h5 checkpoints/best_model.pt sora2")
            sys.exit(1)
        
        hdf5_path = sys.argv[2]
        checkpoint_path = sys.argv[3]
        filter_dataset = sys.argv[4] if len(sys.argv) > 4 else None
        
        test_main(
            hdf5_path=hdf5_path,
            checkpoint_path=checkpoint_path,
            filter_dataset=filter_dataset,
        )
    else:
        # Training mode (default)
        main()

