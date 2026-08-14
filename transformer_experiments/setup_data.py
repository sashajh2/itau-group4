"""
Data setup script for the vanilla transformer deepfake detector.

Samples the dataset, creates train/test splits, and prints summary stats.

Usage:
    python -m transformer_experiments.setup_data
"""

import numpy as np

from transformer_experiments.dataset import (
    DeepfakeSequenceDataset,
    collate_fn,
    make_train_test_split,
    sample_dataset,
)

HDF5_PATH = "exports/deepfake_embeddings.h5"


def main():
    print("=" * 60)
    print("Transformer Experiments — Data Setup")
    print("=" * 60)

    # 1. Sample dataset
    print("\n[1/3] Sampling dataset...")
    samples = sample_dataset(HDF5_PATH, n_real=50, n_fake=50, seed=42)
    print(f"  Total samples: {len(samples)}")

    n_real = sum(1 for s in samples if s["label"] == 0)
    n_fake = sum(1 for s in samples if s["label"] == 1)
    print(f"  Real (label=0): {n_real}")
    print(f"  Fake (label=1): {n_fake}")

    # 2. Train/test split
    print("\n[2/3] Splitting train/test...")
    train_samples, test_samples = make_train_test_split(samples, train_size=80, seed=42)

    train_real = sum(1 for s in train_samples if s["label"] == 0)
    train_fake = sum(1 for s in train_samples if s["label"] == 1)
    test_real = sum(1 for s in test_samples if s["label"] == 0)
    test_fake = sum(1 for s in test_samples if s["label"] == 1)

    print(f"  Train: {len(train_samples)} total ({train_real} real, {train_fake} fake)")
    print(f"  Test:  {len(test_samples)} total ({test_real} real, {test_fake} fake)")

    # 3. Sequence length statistics
    print("\n[3/3] Sequence length statistics...")
    all_lengths = [s["embeddings"].shape[0] for s in samples]
    print(f"  Min T:    {min(all_lengths)}")
    print(f"  Max T:    {max(all_lengths)}")
    print(f"  Mean T:   {np.mean(all_lengths):.1f}")
    print(f"  Median T: {np.median(all_lengths):.1f}")

    # Print a few examples
    print("\n  First 5 samples:")
    for i, s in enumerate(samples[:5]):
        label_str = "FAKE" if s["label"] == 1 else "REAL"
        print(
            f"    {i+1}. [{label_str}] video={s['video_id'][:30]:30s} "
            f"aug={s['aug_idx']} T={s['embeddings'].shape[0]} "
            f"shape={s['embeddings'].shape}"
        )

    # Verify datasets work
    max_t = max(all_lengths)
    print(f"\n  Padding all sequences to max_seq_len={max_t}...")
    train_ds = DeepfakeSequenceDataset(train_samples, max_seq_len=max_t)
    test_ds = DeepfakeSequenceDataset(test_samples, max_seq_len=max_t)

    # Test collate on a small batch
    batch_items = [train_ds[i] for i in range(min(4, len(train_ds)))]
    batch = collate_fn(batch_items)
    print(f"    Batch embeddings shape: {batch['embeddings'].shape}")
    print(f"    Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"    Batch labels shape: {batch['labels'].shape}")
    print(f"    Batch labels: {batch['labels'].tolist()}")

    print("\n" + "=" * 60)
    print("Data setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
