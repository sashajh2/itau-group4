import argparse
import os
import tempfile
from typing import Tuple

import faiss  # type: ignore
import numpy as np  # type: ignore

from dropbox_utils.dropbox_utils import download_file


def load_faiss_index_and_dim(dropbox_index_path: str) -> Tuple[faiss.Index, int]:
    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
        local_index_path = tmp.name
    # Use shared helper that handles auth/config
    download_file(dropbox_index_path, local_index_path)
    index = faiss.read_index(local_index_path)
    try:
        os.unlink(local_index_path)
    except Exception:
        pass
    return index, index.d


def main():
    parser = argparse.ArgumentParser(description="Compare Dropbox FAISS index with local numpy embeddings")
    parser.add_argument(
        "--dropbox-index-path",
        type=str,
        required=True,
        help="Dropbox path to .index file (e.g., /embedding_store/AVDeepfake1M/raw/audio/hubert.index)",
    )
    parser.add_argument(
        "--local-npy-path",
        type=str,
        default="embeddings/audio/hubert/unified_hubert_embeddings.npy",
        help="Local .npy embeddings file to compare",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="Number of random samples for nearest-neighbor similarity check",
    )
    args = parser.parse_args()

    # Load FAISS index from Dropbox
    print(f"Downloading FAISS index from Dropbox: {args.dropbox_index_path}")
    faiss_index, dim = load_faiss_index_and_dim(args.dropbox_index_path)
    n_index = faiss_index.ntotal
    print(f"Index vectors: {n_index}, dim: {dim}")

    # Load local embeddings
    print(f"Loading local embeddings: {args.local_npy_path}")
    local_embs = np.load(args.local_npy_path)
    if local_embs.ndim != 2:
        raise ValueError("Local embeddings must be a 2D array of shape (N, D)")
    n_local, d_local = local_embs.shape
    print(f"Local vectors: {n_local}, dim: {d_local}")

    # Basic checks
    same_dim = (dim == d_local)
    same_len = (n_index == n_local)
    print(f"Same dimension: {same_dim}")
    print(f"Same length:   {same_len}")

    # Similarity check via nearest neighbor search
    sample_size = min(args.sample, n_local)
    if sample_size == 0:
        print("No vectors to compare. Exiting.")
        return

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_local, size=sample_size, replace=False)
    queries = local_embs[sample_idx].astype(np.float32)

    # search in FAISS index (L2). Dists are squared L2 distances for IndexFlatL2
    distances, _ = faiss_index.search(queries, 1)
    distances = distances.reshape(-1)

    mean_dist = float(np.mean(distances))
    p95_dist = float(np.percentile(distances, 95))
    print(f"Nearest-neighbor L2^2 distance stats on {sample_size} samples:")
    print(f"  mean: {mean_dist:.6f}")
    print(f"  p95:  {p95_dist:.6f}")

    # Simple pass/fail heuristic: dims must match; length match is ideal but optional.
    # For similarity, flag if mean squared L2 distance is very large relative to typical unit norms (~1.0-4.0).
    ok = same_dim and (mean_dist < 4.0)
    print(f"PASS: {ok}")


if __name__ == "__main__":
    main()


