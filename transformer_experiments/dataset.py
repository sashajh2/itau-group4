"""
Dataset utilities for the vanilla transformer deepfake detector.

Handles sampling from the HDF5 file, building PyTorch datasets,
and train/test splitting with stratification.
"""

import random
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def sample_dataset(
    hdf5_path: str,
    n_real: int = 50,
    n_fake: int = 50,
    seed: int = 42,
) -> List[Dict]:
    """Sample a balanced dataset of real and fake video-augmentation pairs.

    For each sample, loads HuBERT embeddings (768-dim) for all segments
    of a single augmentation, ordered by segment index.

    Args:
        hdf5_path: Path to HDF5 embeddings file.
        n_real: Number of real samples to collect.
        n_fake: Number of fake samples to collect.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts, each with:
            - 'embeddings': np.ndarray (T, 768) HuBERT sequence
            - 'label': int (0=real, 1=fake)
            - 'video_id': str
            - 'aug_idx': int
    """
    rng = random.Random(seed)

    # Index videos by augmentation type availability
    real_videos = []  # (safe_id, list of real aug indices)
    fake_videos = []  # (safe_id, list of fake aug indices)

    with h5py.File(hdf5_path, "r") as f:
        videos_grp = f["videos"]

        for safe_id in videos_grp.keys():
            vid = videos_grp[safe_id]
            aug_types = vid["augmentation_info"]["types"][:]
            aug_types = [t.decode() if isinstance(t, bytes) else t for t in aug_types]

            real_indices = [i for i, t in enumerate(aug_types) if t in ("source", "real")]
            fake_indices = [i for i, t in enumerate(aug_types) if t == "fake"]

            if real_indices:
                real_videos.append((safe_id, real_indices))
            if fake_indices:
                fake_videos.append((safe_id, fake_indices))

    if len(real_videos) == 0:
        raise ValueError("No videos with real augmentations found in HDF5 file")
    if len(fake_videos) == 0:
        raise ValueError("No videos with fake augmentations found in HDF5 file")

    print(f"Index: {len(real_videos)} videos with real augs, {len(fake_videos)} videos with fake augs")

    samples = []

    with h5py.File(hdf5_path, "r") as f:
        # Collect real samples
        for _ in range(n_real):
            safe_id, aug_indices = rng.choice(real_videos)
            aug_idx = rng.choice(aug_indices)
            emb = f["videos"][safe_id]["embeddings"]["hubert"][aug_idx]  # (T, 768)
            labels = f["videos"][safe_id]["labels"]["audio"][aug_idx]  # (T,)

            # Video-level label: 0 if all segment labels are 0, else 1
            video_label = 1 if np.any(labels > 0) else 0
            # For real augmentations this should be 0
            samples.append({
                "embeddings": np.array(emb, dtype=np.float32),
                "label": video_label,
                "video_id": safe_id,
                "aug_idx": int(aug_idx),
            })

        # Collect fake samples
        for _ in range(n_fake):
            safe_id, aug_indices = rng.choice(fake_videos)
            aug_idx = rng.choice(aug_indices)
            emb = f["videos"][safe_id]["embeddings"]["hubert"][aug_idx]  # (T, 768)
            labels = f["videos"][safe_id]["labels"]["audio"][aug_idx]  # (T,)

            video_label = 1 if np.any(labels > 0) else 0
            # For fake augmentations this should be 1
            samples.append({
                "embeddings": np.array(emb, dtype=np.float32),
                "label": video_label,
                "video_id": safe_id,
                "aug_idx": int(aug_idx),
            })

    return samples


class DeepfakeSequenceDataset(Dataset):
    """PyTorch dataset wrapping sampled video-augmentation pairs.

    All sequences are padded (or truncated) to a fixed max_seq_len so that
    every sample has identical shape, enabling simple stacking in collate_fn.
    """

    def __init__(self, samples: List[Dict], max_seq_len: Optional[int] = None):
        """
        Args:
            samples: List from sample_dataset(), each with 'embeddings' and 'label'.
            max_seq_len: Fixed sequence length. Shorter sequences are zero-padded,
                         longer ones are truncated. If None, uses the max length
                         found in samples.
        """
        self.samples = samples
        if max_seq_len is None:
            max_seq_len = max(s["embeddings"].shape[0] for s in samples)
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (embeddings, attention_mask, label), all fixed-length."""
        s = self.samples[idx]
        raw = torch.from_numpy(s["embeddings"]).float()  # (T, 768)
        t = raw.shape[0]
        d = raw.shape[1]

        embeddings = torch.zeros(self.max_seq_len, d)
        mask = torch.zeros(self.max_seq_len)
        length = min(t, self.max_seq_len)

        embeddings[:length] = raw[:length]
        mask[:length] = 1.0

        label = torch.tensor(s["label"], dtype=torch.float32)
        return embeddings, mask, label


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack fixed-length sequences into a batch.

    Args:
        batch: List of (embeddings (T, 768), attention_mask (T,), label) tuples.

    Returns:
        Dict with:
            - 'embeddings': (B, T, 768)
            - 'attention_mask': (B, T) binary mask (1=real token, 0=padding)
            - 'labels': (B,)
    """
    embeddings_list, mask_list, labels_list = zip(*batch)
    return {
        "embeddings": torch.stack(embeddings_list),
        "attention_mask": torch.stack(mask_list),
        "labels": torch.stack(labels_list),
    }


def make_train_test_split(
    samples: List[Dict],
    train_size: int = 80,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Stratified train/test split.

    Splits so that the train set has equal real/fake counts, and
    similarly for the test set.

    Args:
        samples: Full list of samples from sample_dataset().
        train_size: Number of training samples (rest go to test).
        seed: Random seed.

    Returns:
        (train_samples, test_samples)
    """
    rng = random.Random(seed)

    real_samples = [s for s in samples if s["label"] == 0]
    fake_samples = [s for s in samples if s["label"] == 1]

    rng.shuffle(real_samples)
    rng.shuffle(fake_samples)

    # Split each class proportionally
    n_total = len(samples)
    test_size = n_total - train_size
    train_per_class = train_size // 2
    test_per_class = test_size // 2

    train_real = real_samples[:train_per_class]
    test_real = real_samples[train_per_class:train_per_class + test_per_class]

    train_fake = fake_samples[:train_per_class]
    test_fake = fake_samples[train_per_class:train_per_class + test_per_class]

    train_samples = train_real + train_fake
    test_samples = test_real + test_fake

    # Shuffle within splits
    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    return train_samples, test_samples


def sample_train_test_no_overlap(
    hdf5_path: str,
    n_train_real: int,
    n_train_fake: int,
    n_test_real: int,
    n_test_fake: int,
    seed: int = 42,
    embedding_keys: Tuple[str, ...] = ("hubert",),
) -> Tuple[List[Dict], List[Dict]]:
    """Sample train and test sets with strict video-level separation.

    Splits the video pool before sampling augmentations, so no video_id
    can appear in both train and test.

    Returns:
        (train_samples, test_samples)
    """
    rng = random.Random(seed)

    real_videos = []
    fake_videos = []

    with h5py.File(hdf5_path, "r") as f:
        videos_grp = f["videos"]
        for safe_id in videos_grp.keys():
            vid = videos_grp[safe_id]
            aug_types = vid["augmentation_info"]["types"][:]
            aug_types = [t.decode() if isinstance(t, bytes) else t for t in aug_types]

            real_indices = [i for i, t in enumerate(aug_types) if t in ("source", "real")]
            fake_indices = [i for i, t in enumerate(aug_types) if t == "fake"]

            if real_indices:
                real_videos.append((safe_id, real_indices))
            if fake_indices:
                fake_videos.append((safe_id, fake_indices))

    rng.shuffle(real_videos)
    rng.shuffle(fake_videos)

    # Split video pools proportionally so no video appears in both train and test.
    # Sampling within each pool is with replacement, so pool size can be smaller
    # than the number of samples requested.
    real_train_ratio = n_train_real / (n_train_real + n_test_real)
    n_real_train_pool = max(1, round(len(real_videos) * real_train_ratio))
    train_real_pool = real_videos[:n_real_train_pool]
    test_real_pool  = real_videos[n_real_train_pool:]

    fake_train_ratio = n_train_fake / (n_train_fake + n_test_fake)
    n_fake_train_pool = max(1, round(len(fake_videos) * fake_train_ratio))
    train_fake_pool = fake_videos[:n_fake_train_pool]
    test_fake_pool  = fake_videos[n_fake_train_pool:]

    print(f"Index: {len(real_videos)} videos with real augs, {len(fake_videos)} videos with fake augs")
    print(f"  Real pool  — train: {len(train_real_pool)}, test: {len(test_real_pool)}")
    print(f"  Fake pool  — train: {len(train_fake_pool)}, test: {len(test_fake_pool)}")

    def collect(pool: list, n: int) -> List[Dict]:
        samples = []
        with h5py.File(hdf5_path, "r") as f:
            for _ in range(n):
                safe_id, aug_indices = rng.choice(pool)
                aug_idx = rng.choice(aug_indices)
                parts = [
                    np.array(f["videos"][safe_id]["embeddings"][k][aug_idx], dtype=np.float32)
                    for k in embedding_keys
                ]
                emb = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
                labels = f["videos"][safe_id]["labels"]["audio"][aug_idx]
                video_label = 1 if np.any(labels > 0) else 0
                samples.append({
                    "embeddings": emb,
                    "label": video_label,
                    "video_id": safe_id,
                    "aug_idx": int(aug_idx),
                })
        return samples

    train_samples = collect(train_real_pool, n_train_real) + collect(train_fake_pool, n_train_fake)
    test_samples  = collect(test_real_pool, n_test_real)   + collect(test_fake_pool, n_test_fake)

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    return train_samples, test_samples


def sample_fixed_test(
    hdf5_path: str,
    dataset: str,
    n: int = 200,
    seed: int = 42,
    embedding_keys: Tuple[str, ...] = ("hubert",),
) -> List[Dict]:
    """Sample a fixed test set from a specific dataset (e.g. shareveo3).

    All sampled videos are fake. Used as a cross-dataset generalization test
    that is held constant across all folds.
    """
    rng = random.Random(seed)
    fake_videos = []

    with h5py.File(hdf5_path, "r") as f:
        for safe_id in f["videos"].keys():
            vid = f["videos"][safe_id]
            ds = vid.attrs.get("dataset", b"")
            if isinstance(ds, bytes):
                ds = ds.decode()
            if ds != dataset:
                continue
            aug_types = [
                t.decode() if isinstance(t, bytes) else t
                for t in vid["augmentation_info"]["types"][:]
            ]
            fake_idx = [i for i, t in enumerate(aug_types) if t == "fake"]
            if fake_idx:
                fake_videos.append((safe_id, fake_idx))

    rng.shuffle(fake_videos)
    samples = []
    with h5py.File(hdf5_path, "r") as f:
        for _ in range(n):
            safe_id, aug_indices = rng.choice(fake_videos)
            aug_idx = rng.choice(aug_indices)
            parts = [
                np.array(f["videos"][safe_id]["embeddings"][k][aug_idx], dtype=np.float32)
                for k in embedding_keys
            ]
            emb = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
            labels = f["videos"][safe_id]["labels"]["audio"][aug_idx]
            video_label = 1 if np.any(labels > 0) else 0
            duration = float(f["videos"][safe_id].attrs.get("duration", 0.0))
            samples.append({
                "embeddings": emb,
                "label": video_label,
                "video_id": safe_id,
                "aug_idx": int(aug_idx),
                "duration": duration,
                "dataset": dataset,
            })
    return samples


def sample_kfold_splits(
    hdf5_path: str,
    k: int = 5,
    n_train_per_class: int = 400,
    n_test_per_class: int = 100,
    seed: int = 42,
    embedding_keys: Tuple[str, ...] = ("hubert",),
    dataset_filter: Optional[str] = None,
) -> List[Tuple[List[Dict], List[Dict]]]:
    """Build k (train, test) splits with strict video-level separation per fold.

    Videos are partitioned into k groups. For fold i, test videos = group i,
    train videos = all other groups. No video ID ever appears in both the train
    and test set of the same fold.

    Args:
        dataset_filter: If set, only include videos from this dataset.

    Returns:
        List of k (train_samples, test_samples) tuples.
    """
    rng = random.Random(seed)

    real_videos: List[Tuple[str, List[int]]] = []
    fake_videos: List[Tuple[str, List[int]]] = []

    with h5py.File(hdf5_path, "r") as f:
        for safe_id in f["videos"].keys():
            vid = f["videos"][safe_id]
            if dataset_filter is not None:
                ds = vid.attrs.get("dataset", b"")
                if isinstance(ds, bytes):
                    ds = ds.decode()
                if ds != dataset_filter:
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

    # Split video IDs into k equal-ish folds
    def split_into_folds(videos, k):
        folds = [[] for _ in range(k)]
        for i, v in enumerate(videos):
            folds[i % k].append(v)
        return folds

    real_folds = split_into_folds(real_videos, k)
    fake_folds = split_into_folds(fake_videos, k)

    print(f"K-fold setup (k={k}):")
    print(f"  Real videos: {len(real_videos)} → ~{len(real_folds[0])} per fold")
    print(f"  Fake videos: {len(fake_videos)} → ~{len(fake_folds[0])} per fold")

    def collect(pool: list, n: int) -> List[Dict]:
        samples = []
        with h5py.File(hdf5_path, "r") as f:
            for _ in range(n):
                safe_id, aug_indices = rng.choice(pool)
                aug_idx = rng.choice(aug_indices)
                parts = [
                    np.array(f["videos"][safe_id]["embeddings"][key][aug_idx], dtype=np.float32)
                    for key in embedding_keys
                ]
                emb = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
                labels = f["videos"][safe_id]["labels"]["audio"][aug_idx]
                video_label = 1 if np.any(labels > 0) else 0
                duration = float(f["videos"][safe_id].attrs.get("duration", 0.0))
                samples.append({
                    "embeddings": emb,
                    "label": video_label,
                    "video_id": safe_id,
                    "aug_idx": int(aug_idx),
                    "duration": duration,
                })
        return samples

    splits = []
    for fold_i in range(k):
        test_real_pool  = real_folds[fold_i]
        train_real_pool = [v for j, fold in enumerate(real_folds) if j != fold_i for v in fold]
        test_fake_pool  = fake_folds[fold_i]
        train_fake_pool = [v for j, fold in enumerate(fake_folds) if j != fold_i for v in fold]

        train = collect(train_real_pool, n_train_per_class) + collect(train_fake_pool, n_train_per_class)
        test  = collect(test_real_pool,  n_test_per_class)  + collect(test_fake_pool,  n_test_per_class)
        rng.shuffle(train)
        rng.shuffle(test)
        splits.append((train, test))

    return splits


def sample_cross_dataset_split(
    hdf5_path: str,
    n_train_real: int,
    n_train_fake: int,
    n_test_real: int,
    n_test_fake: int,
    train_dataset: str = "avdeepfake1m",
    test_fake_dataset: str = "shareveo3",
    seed: int = 42,
    embedding_keys: Tuple[str, ...] = ("hubert",),
) -> Tuple[List[Dict], List[Dict]]:
    """Cross-dataset train/test split.

    Train real + train fake come from train_dataset.
    Test real comes from held-out videos of train_dataset.
    Test fake comes entirely from test_fake_dataset.

    No video ID appears in both train and test.
    """
    rng = random.Random(seed)

    train_real_videos = []
    train_fake_videos = []
    test_fake_videos  = []

    with h5py.File(hdf5_path, "r") as f:
        for safe_id in f["videos"].keys():
            vid = f["videos"][safe_id]
            ds = vid.attrs.get("dataset", b"")
            if isinstance(ds, bytes):
                ds = ds.decode()

            aug_types = vid["augmentation_info"]["types"][:]
            aug_types = [t.decode() if isinstance(t, bytes) else t for t in aug_types]
            real_indices = [i for i, t in enumerate(aug_types) if t in ("source", "real")]
            fake_indices = [i for i, t in enumerate(aug_types) if t == "fake"]

            if ds == train_dataset:
                if real_indices:
                    train_real_videos.append((safe_id, real_indices))
                if fake_indices:
                    train_fake_videos.append((safe_id, fake_indices))
            elif ds == test_fake_dataset:
                if fake_indices:
                    test_fake_videos.append((safe_id, fake_indices))

    rng.shuffle(train_real_videos)
    rng.shuffle(train_fake_videos)
    rng.shuffle(test_fake_videos)

    # Split train_dataset real videos into train pool and test real pool
    real_train_ratio = n_train_real / (n_train_real + n_test_real)
    n_real_train_pool = max(1, round(len(train_real_videos) * real_train_ratio))
    train_real_pool = train_real_videos[:n_real_train_pool]
    test_real_pool  = train_real_videos[n_real_train_pool:]

    print(f"Train dataset ({train_dataset}):")
    print(f"  Real pool  — train: {len(train_real_pool)}, test: {len(test_real_pool)}")
    print(f"  Fake pool  — train: {len(train_fake_videos)}")
    print(f"Test fake dataset ({test_fake_dataset}): {len(test_fake_videos)} videos")

    def collect(pool: list, n: int) -> List[Dict]:
        samples = []
        with h5py.File(hdf5_path, "r") as f:
            for _ in range(n):
                safe_id, aug_indices = rng.choice(pool)
                aug_idx = rng.choice(aug_indices)
                parts = [
                    np.array(f["videos"][safe_id]["embeddings"][k][aug_idx], dtype=np.float32)
                    for k in embedding_keys
                ]
                emb = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]
                labels = f["videos"][safe_id]["labels"]["audio"][aug_idx]
                video_label = 1 if np.any(labels > 0) else 0
                samples.append({
                    "embeddings": emb,
                    "label": video_label,
                    "video_id": safe_id,
                    "aug_idx": int(aug_idx),
                    "dataset": f["videos"][safe_id].attrs.get("dataset", b"").decode()
                    if isinstance(f["videos"][safe_id].attrs.get("dataset", b""), bytes)
                    else f["videos"][safe_id].attrs.get("dataset", ""),
                })
        return samples

    train_samples = collect(train_real_pool, n_train_real) + collect(train_fake_videos, n_train_fake)
    test_samples  = collect(test_real_pool, n_test_real)   + collect(test_fake_videos, n_test_fake)

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)

    return train_samples, test_samples
