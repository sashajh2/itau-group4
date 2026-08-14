"""
Export AVDeepFake1M 20% subset to a .npz for Google Colab.

Run from repo root:
    python scripts/export_embeddings_npz.py

Output: exports/avdeepfake_20pct_embeddings.npz  (~60-90 MB compressed)
  Keys:
    embeddings  float32 [N, 768]  — HuBERT segment embeddings
    is_real     bool    [N]       — True = real audio segment
    cg_src      int32   [N]       — content-group source index
    cg_seg      int32   [N]       — content-group segment index
    vid_ids     object  [N]       — video ID string (used for train/val split)
"""

import os
import random
import h5py
import numpy as np
from tqdm import tqdm

HDF5_PATH   = "exports/deepfake_embeddings.h5"
ENCODER     = "hubert"
SUBSET_FRAC = 0.20
SEED        = 42
OUT_PATH    = "exports/avdeepfake_20pct_embeddings.npz"


def main():
    rng = random.Random(SEED)

    with h5py.File(HDF5_PATH, "r") as f:
        all_vids = list(f["videos"].keys())
        av_vids  = [v for v in all_vids
                    if f["videos"][v].attrs.get("dataset", "") == "avdeepfake1m"]

    n_subset = max(1, int(len(av_vids) * SUBSET_FRAC))
    selected = sorted(rng.sample(av_vids, n_subset))
    print(f"Exporting {len(selected)} videos ({SUBSET_FRAC*100:.0f}% of {len(av_vids)})...")

    all_embs, all_real, all_cg_src, all_cg_seg, all_vid = [], [], [], [], []
    source_counter = 0

    with h5py.File(HDF5_PATH, "r") as f:
        for video_id in tqdm(selected, desc="Reading HDF5"):
            v = f["videos"][video_id]
            if f"embeddings/{ENCODER}" not in v:
                continue
            if "labels" not in v or "audio" not in v["labels"]:
                continue

            embs_vid  = v[f"embeddings/{ENCODER}"][:]   # (num_augs, num_segs, D)
            audio_lbl = v["labels/audio"][:]              # (num_augs, num_segs)
            num_augs, num_segs, _ = embs_vid.shape

            for aug_idx in range(num_augs):
                for seg_idx in range(num_segs):
                    all_embs.append(embs_vid[aug_idx, seg_idx])
                    all_real.append(audio_lbl[aug_idx, seg_idx] == 0.0)
                    all_cg_src.append(source_counter)
                    all_cg_seg.append(seg_idx)
                    all_vid.append(video_id)

            source_counter += 1

    embeddings = np.stack(all_embs).astype(np.float32)
    is_real    = np.array(all_real,    dtype=bool)
    cg_src     = np.array(all_cg_src,  dtype=np.int32)
    cg_seg     = np.array(all_cg_seg,  dtype=np.int32)
    vid_ids    = np.array(all_vid)                       # object dtype (strings)

    real_n = int(is_real.sum())
    fake_n = len(is_real) - real_n
    print(f"Segments: {len(embeddings):,}  |  "
          f"real: {real_n:,} ({100*real_n/len(embeddings):.1f}%)  |  "
          f"fake: {fake_n:,} ({100*fake_n/len(embeddings):.1f}%)  |  "
          f"{embeddings.nbytes / 1e6:.0f} MB uncompressed")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        embeddings=embeddings,
        is_real=is_real,
        cg_src=cg_src,
        cg_seg=cg_seg,
        vid_ids=vid_ids,
    )

    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"\nSaved → {OUT_PATH}  ({size_mb:.1f} MB compressed)")
    print("Upload this file to Google Colab and run experiments/fix1a_colab.py")


if __name__ == "__main__":
    main()
