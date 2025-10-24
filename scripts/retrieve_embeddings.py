#!/usr/bin/env python3
"""
Retrieve embeddings and labels from Neon Postgres (pgvector) for a specific model and version.

This script:
1. Queries Neon directly for a given model/version (and optional filters)
2. Joins to segments to fetch labels
3. Saves either:
   - Two arrays (embeddings.npy, labels.npy) in a single NPZ file (default), or
   - One combined N x (D+1) .npy file with last column = label (if --combine)

Usage:
    python scripts/retrieve_embeddings.py --model hubert --version 2025-09-12
"""

import argparse
import os
from typing import Optional, Tuple
import numpy as np
import psycopg2

from utils.config_loader import load_config


_SPACE_TO_TABLE = {
    ("audio", "hubert"): ("embeddings_audio_hubert", "audio_label"),
    ("audio", "openl3"): ("embeddings_audio_openl3", "audio_label"),
    ("video", "senet"): ("embeddings_video_senet", "video_label"),
}


def _connect_neon():
    cfg = load_config()
    dsn = cfg["database"]["postgres"]["neon_database_url"]
    return psycopg2.connect(dsn)


def _fetch_embeddings_and_labels(model_name: str, version: str, mode: Optional[str], noise: Optional[str], denoiser: Optional[str]):
    # Determine table and label column
    if mode is None:
        # Infer mode from model
        if model_name in ("hubert", "openl3"):
            mode = "audio"
        elif model_name == "senet":
            mode = "video"
        else:
            raise ValueError("Provide --mode for unknown model")
    key = (mode.lower(), model_name.lower())
    if key not in _SPACE_TO_TABLE:
        raise ValueError(f"Unsupported model/mode: {key}")
    table, label_col = _SPACE_TO_TABLE[key]

    where = ["e.version = %s"]
    params = [version]
    if noise is not None:
        where.append("e.noise = %s")
        params.append(noise)
    if denoiser is not None:
        where.append("e.denoiser_name = %s")
        params.append(denoiser)
    where_sql = " AND ".join(where)

    sql = f"""
      SELECT e.segment_id,
             e.embedding::float4[] AS emb,
             s.{label_col} AS label,
             s.video_id AS video_id
      FROM {table} e
      JOIN segments s USING (segment_id)
      WHERE {where_sql}
      ORDER BY e.segment_id
    """

    with _connect_neon() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    # rows: (segment_id, list_of_floats, label, video_id)
    seg_ids = [r[0] for r in rows]
    embs = [np.array(r[1], dtype=np.float32) for r in rows]
    labels = np.array([int(r[2]) for r in rows], dtype=np.int64)
    video_ids = [r[3] for r in rows]
    return seg_ids, np.vstack(embs) if embs else np.zeros((0, 0), dtype=np.float32), labels, video_ids, table


def retrieve_embeddings_and_labels(
    model_name: str,
    version: str,
    mode: Optional[str] = None,
    noise: Optional[str] = None,
    denoiser_name: Optional[str] = None,
    output_dir: str = "./embeddings/retrieved",
    combine: bool = False,
) -> Tuple[str, str]:
    """
    Retrieve all embeddings and labels for a specific model and version.
    
    Args:
        model_name: Name of the model (e.g., 'hubert', 'openl3', 'senet')
        version: Version string (e.g., '2025-09-12')
        mode: Optional mode filter ('audio' or 'video')
        noise: Optional noise filter ('none', 'denoised', 'noisy')
        denoiser_name: Optional denoiser filter ('demucs', 'voicefixer', etc.)
        output_dir: Directory to save the output files
    
    Returns:
        Tuple of (embeddings_file_path, labels_file_path)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Retrieving embeddings for model={model_name}, version={version}")
    if mode:
        print(f"  Mode filter: {mode}")
    if noise:
        print(f"  Noise filter: {noise}")
    if denoiser_name:
        print(f"  Denoiser filter: {denoiser_name}")
    
    # Fetch from Neon directly
    seg_ids, all_embeddings, all_labels, video_ids, table = _fetch_embeddings_and_labels(
        model_name, version, mode, noise, denoiser_name
    )
    if all_embeddings.size == 0:
        raise ValueError("No embeddings found")
    print(f"Rows: {len(seg_ids)} | dim: {all_embeddings.shape[1]} | table: {table}")
    print(f"Label distribution: {np.bincount(all_labels)}")
    
    # Create output filenames
    filename_parts = [model_name, version]
    if mode:
        filename_parts.append(mode)
    if noise:
        filename_parts.append(noise)
    if denoiser_name:
        filename_parts.append(denoiser_name)
    
    base_filename = "_".join(filename_parts)
    if combine:
        combined = np.concatenate([all_embeddings, all_labels.reshape(-1, 1).astype(np.float32)], axis=1)
        out_path = os.path.join(output_dir, f"{base_filename}_combined.npy")
        np.save(out_path, combined)
        print(f"\n✅ Saved combined array to: {out_path} (shape={combined.shape})")
        return out_path, out_path
    else:
        out_path = os.path.join(output_dir, f"{base_filename}.npz")
        np.savez(out_path,
                 embeddings=all_embeddings,
                 labels=all_labels,
                 segment_ids=np.array(seg_ids),
                 video_ids=np.array(video_ids),
                 model_name=model_name,
                 version=version,
                 mode=mode,
                 noise=noise,
                 denoiser_name=denoiser_name)
        print(f"\n✅ Saved to: {out_path} (embeddings shape={all_embeddings.shape})")
        return out_path, out_path


def main():
    parser = argparse.ArgumentParser(description="Retrieve embeddings+labels from Neon for a specific model/version")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., hubert, openl3, senet)")
    parser.add_argument("--version", type=str, required=True, help="Version string (e.g., 2025-09-12)")
    parser.add_argument("--mode", type=str, choices=['audio', 'video'], help="Mode filter (optional)")
    parser.add_argument("--noise", type=str, choices=['none', 'denoised', 'noisy'], help="Noise filter (optional)")
    parser.add_argument("--denoiser", type=str, help="Denoiser filter (optional, e.g., demucs, voicefixer)")
    parser.add_argument("--output-dir", type=str, default="./embeddings/retrieved", help="Output directory")
    parser.add_argument("--combine", action="store_true", help="Save single combined .npy with last column = label")
    
    args = parser.parse_args()
    
    try:
        embeddings_file, labels_file = retrieve_embeddings_and_labels(
            model_name=args.model,
            version=args.version,
            mode=args.mode,
            noise=args.noise,
            denoiser_name=args.denoiser,
            output_dir=args.output_dir,
            combine=args.combine
        )
        
        print(f"\n🎉 Files saved successfully at {labels_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
