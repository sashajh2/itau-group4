"""
Balanced sampling utilities for the encoder-MLP pipeline.

Handles class imbalance by sampling Real/Fake with equal probability,
and further balances real samples between fully-real videos and
mixed (real+fake) videos.

Sampling strategy:
    1. Coin flip: 50% Real, 50% Fake
    2. If Real:
        - 80% from fully-real-only videos (source + real augmentations)
        - 20% from real augmentations/sources of videos that also have fakes
    3. If Fake:
        - Sample a fake augmentation (from mixed videos or ShareVeo3)

Returns concatenated embeddings (hubert + openl3 + senet) as a temporal
sequence: (T, 768 + 512 + 2048) = (T, 3328).
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch


@dataclass
class VideoIndex:
    """Index entry for a single video."""
    video_id: str
    safe_id: str  # HDF5-safe key (slashes replaced)
    dataset: str  # 'avdeepfake1m' or 'shareveo3'
    num_segments: int
    real_aug_indices: List[int]  # indices of source + real augmentations
    fake_aug_indices: List[int]  # indices of fake augmentations
    is_real_only: bool  # True if video has no fake augmentations


class BalancedSampler:
    """Balanced sampler for temporal embedding sequences.

    Pre-indexes the HDF5 file to categorize videos into:
        - real_only_videos: videos with only source/real augmentations
        - mixed_videos: videos with both real and fake augmentations
        - fake_only_videos: videos with only fake augmentations (e.g., ShareVeo3)

    Sampling probabilities:
        - P(Real) = P(Fake) = 0.5
        - P(real_only | Real) = 0.8, P(mixed_real | Real) = 0.2
    """

    def __init__(
        self,
        hdf5_path: str,
        embedding_types: Tuple[str, ...] = ("hubert", "openl3", "senet"),
        use_audio_labels: bool = True,
        filter_dataset: Optional[str] = None,
        real_only_prob: float = 0.8,
        seed: Optional[int] = None,
    ):
        """
        Args:
            hdf5_path: Path to HDF5 embeddings file.
            embedding_types: Which embeddings to concatenate (order matters).
            use_audio_labels: If True, use audio labels; else video labels.
            filter_dataset: Optional filter ('avdeepfake1m' or 'shareveo3').
            real_only_prob: Probability of sampling from real-only videos
                            when sampling a real sample. Default 0.8.
            seed: Random seed for reproducibility.
        """
        self.hdf5_path = hdf5_path
        self.embedding_types = embedding_types
        self.use_audio_labels = use_audio_labels
        self.real_only_prob = real_only_prob

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Build index
        self.real_only_videos: List[VideoIndex] = []
        self.mixed_videos: List[VideoIndex] = []
        self.fake_only_videos: List[VideoIndex] = []

        self._build_index(filter_dataset)

    def _build_index(self, filter_dataset: Optional[str]) -> None:
        """Scan HDF5 and categorize videos."""
        with h5py.File(self.hdf5_path, "r") as f:
            videos_grp = f["videos"]

            for safe_id in videos_grp.keys():
                vid = videos_grp[safe_id]

                # Get dataset
                dataset = vid.attrs.get("dataset", "avdeepfake1m")
                if isinstance(dataset, bytes):
                    dataset = dataset.decode()

                # Apply filter
                if filter_dataset is not None:
                    if filter_dataset.lower() not in dataset.lower():
                        continue

                # Get augmentation types
                aug_types = vid["augmentation_info"]["types"][:]
                aug_types = [t.decode() if isinstance(t, bytes) else t for t in aug_types]

                # Categorize augmentations
                real_indices = [i for i, t in enumerate(aug_types) if t in ("source", "real")]
                fake_indices = [i for i, t in enumerate(aug_types) if t == "fake"]

                original_id = vid.attrs.get("original_video_id", safe_id)
                if isinstance(original_id, bytes):
                    original_id = original_id.decode()

                idx = VideoIndex(
                    video_id=original_id,
                    safe_id=safe_id,
                    dataset=dataset,
                    num_segments=vid.attrs["num_segments"],
                    real_aug_indices=real_indices,
                    fake_aug_indices=fake_indices,
                    is_real_only=(len(fake_indices) == 0),
                )

                if len(real_indices) > 0 and len(fake_indices) == 0:
                    self.real_only_videos.append(idx)
                elif len(real_indices) > 0 and len(fake_indices) > 0:
                    self.mixed_videos.append(idx)
                elif len(fake_indices) > 0:
                    self.fake_only_videos.append(idx)

        print(f"BalancedSampler index built:")
        print(f"  Real-only videos: {len(self.real_only_videos)}")
        print(f"  Mixed videos:     {len(self.mixed_videos)}")
        print(f"  Fake-only videos: {len(self.fake_only_videos)}")

    def _load_embeddings(
        self, f: h5py.File, safe_id: str, aug_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and concatenate embeddings for a single augmentation.

        Returns:
            embeddings: (T, concat_dim) concatenated embeddings
            labels: (T,) soft labels
        """
        vid = f["videos"][safe_id]

        # Concatenate embeddings along feature dimension
        emb_list = []
        for etype in self.embedding_types:
            emb = vid[f"embeddings/{etype}"][aug_idx]  # (T, d)
            emb_list.append(emb)
        embeddings = np.concatenate(emb_list, axis=-1)  # (T, sum of dims)

        # Get labels
        label_key = "audio" if self.use_audio_labels else "video"
        labels = vid[f"labels/{label_key}"][aug_idx]  # (T,)

        return embeddings, labels

    def sample(self) -> Dict[str, torch.Tensor]:
        """Sample a single video sequence with balanced Real/Fake probability.

        Returns:
            Dict with:
                - 'embeddings': (T, D) tensor of concatenated embeddings
                - 'labels': (T,) tensor of soft labels (0=real, 1=fake)
                - 'is_fake': bool, True if this is a fake augmentation
                - 'video_id': str
                - 'aug_idx': int
                - 'source': str ('real_only', 'mixed_real', 'mixed_fake', 'fake_only')
        """
        # Step 1: Coin flip for Real vs Fake
        sample_fake = random.random() < 0.5

        if sample_fake:
            # Sample a fake augmentation
            # Prefer mixed videos (they have real counterparts), but also use fake-only
            if len(self.mixed_videos) > 0 and (
                len(self.fake_only_videos) == 0 or random.random() < 0.5
            ):
                video = random.choice(self.mixed_videos)
                aug_idx = random.choice(video.fake_aug_indices)
                source = "mixed_fake"
            else:
                video = random.choice(self.fake_only_videos)
                aug_idx = random.choice(video.fake_aug_indices)
                source = "fake_only"
        else:
            # Sample a real augmentation
            if random.random() < self.real_only_prob and len(self.real_only_videos) > 0:
                # 80%: from real-only videos
                video = random.choice(self.real_only_videos)
                aug_idx = random.choice(video.real_aug_indices)
                source = "real_only"
            else:
                # 20%: from mixed videos (real augmentations)
                if len(self.mixed_videos) > 0:
                    video = random.choice(self.mixed_videos)
                    aug_idx = random.choice(video.real_aug_indices)
                    source = "mixed_real"
                else:
                    # Fallback to real-only if no mixed videos
                    video = random.choice(self.real_only_videos)
                    aug_idx = random.choice(video.real_aug_indices)
                    source = "real_only"

        # Load embeddings
        with h5py.File(self.hdf5_path, "r") as f:
            embeddings, labels = self._load_embeddings(f, video.safe_id, aug_idx)

        return {
            "embeddings": torch.from_numpy(embeddings).float(),
            "labels": torch.from_numpy(labels).float(),
            "is_fake": sample_fake,
            "video_id": video.video_id,
            "aug_idx": aug_idx,
            "source": source,
            "num_segments": video.num_segments,
        }

    def sample_batch(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Sample a batch of sequences."""
        return [self.sample() for _ in range(batch_size)]

    def get_video_sequence(
        self, video_id: str, aug_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Get a specific video/augmentation sequence by ID.

        Args:
            video_id: The video ID (will try both original and safe versions)
            aug_idx: Augmentation index

        Returns:
            Same dict format as sample()
        """
        safe_id = video_id.replace("/", "_")

        with h5py.File(self.hdf5_path, "r") as f:
            if safe_id not in f["videos"]:
                raise KeyError(f"Video {video_id} not found in HDF5")

            embeddings, labels = self._load_embeddings(f, safe_id, aug_idx)

            vid = f["videos"][safe_id]
            aug_types = vid["augmentation_info"]["types"][:]
            aug_type = aug_types[aug_idx]
            if isinstance(aug_type, bytes):
                aug_type = aug_type.decode()

        is_fake = aug_type == "fake"

        return {
            "embeddings": torch.from_numpy(embeddings).float(),
            "labels": torch.from_numpy(labels).float(),
            "is_fake": is_fake,
            "video_id": video_id,
            "aug_idx": aug_idx,
            "source": "direct_access",
            "num_segments": embeddings.shape[0],
        }

    def stats(self) -> Dict[str, int]:
        """Return statistics about the indexed data."""
        total_real_augs = sum(len(v.real_aug_indices) for v in self.real_only_videos)
        total_real_augs += sum(len(v.real_aug_indices) for v in self.mixed_videos)
        total_fake_augs = sum(len(v.fake_aug_indices) for v in self.mixed_videos)
        total_fake_augs += sum(len(v.fake_aug_indices) for v in self.fake_only_videos)

        return {
            "real_only_videos": len(self.real_only_videos),
            "mixed_videos": len(self.mixed_videos),
            "fake_only_videos": len(self.fake_only_videos),
            "total_real_augmentations": total_real_augs,
            "total_fake_augmentations": total_fake_augs,
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def get_concatenated_sequence(
    hdf5_path: str,
    video_id: str,
    aug_idx: int,
    embedding_types: Tuple[str, ...] = ("hubert", "openl3", "senet"),
    use_audio_labels: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load concatenated embeddings for a single video/augmentation.

    This is a simple utility without the sampling logic.

    Args:
        hdf5_path: Path to HDF5 file
        video_id: Video ID
        aug_idx: Augmentation index
        embedding_types: Which embeddings to concatenate
        use_audio_labels: Use audio or video labels

    Returns:
        embeddings: (T, D) tensor
        labels: (T,) tensor
    """
    safe_id = video_id.replace("/", "_")

    with h5py.File(hdf5_path, "r") as f:
        vid = f["videos"][safe_id]

        emb_list = []
        for etype in embedding_types:
            emb = vid[f"embeddings/{etype}"][aug_idx]
            emb_list.append(emb)
        embeddings = np.concatenate(emb_list, axis=-1)

        label_key = "audio" if use_audio_labels else "video"
        labels = vid[f"labels/{label_key}"][aug_idx]

    return torch.from_numpy(embeddings).float(), torch.from_numpy(labels).float()


if __name__ == "__main__":
    # Quick test
    sampler = BalancedSampler(
        hdf5_path="exports/deepfake_embeddings.h5",
        filter_dataset="avdeepfake1m",
        seed=42,
    )

    print("\nStats:", sampler.stats())

    print("\nSampling 10 examples:")
    for i in range(10):
        sample = sampler.sample()
        label_str = "FAKE" if sample["is_fake"] else "REAL"
        n_fake_segs = (sample["labels"] > 0.5).sum().item()
        print(
            f"  {i+1}. [{label_str:4s}] {sample['source']:12s} | "
            f"vid={sample['video_id'][:20]:20s} aug={sample['aug_idx']} | "
            f"T={sample['num_segments']:3d} | fake_segs={n_fake_segs}"
        )
