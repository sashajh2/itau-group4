#!/usr/bin/env python3
"""
Retrieve embeddings and labels from Neon Postgres (pgvector).

This module provides functions to:
1. Query Neon directly for a given model/version
2. Join to segments to fetch labels
3. Return embeddings, labels, and video_ids as numpy arrays

Usage:
    from retriever.retriever import load_embedding_data
    
    embeddings, labels, video_ids, seg_ids = load_embedding_data(
        model_name="openl3",
        version="2025-09-12"
    )
"""

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


def load_embedding_data(
    model_name: str,
    version: str,
    mode: Optional[str] = None,
    noise: Optional[str] = None,
    denoiser_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Load embeddings, labels, video_ids, and segment_ids from Neon Postgres.
    
    Args:
        model_name: Name of the model (e.g., 'hubert', 'openl3', 'senet')
        version: Version string (e.g., '2025-09-12')
        mode: Optional mode filter ('audio' or 'video')
        noise: Optional noise filter ('none', 'denoised', 'noisy')
        denoiser_name: Optional denoiser filter ('demucs', 'voicefixer', etc.)
    
    Returns:
        Tuple of (embeddings, labels, video_ids, segment_ids)
        - embeddings: [N, D] numpy array
        - labels: [N] numpy array
        - video_ids: [N] list of video IDs
        - segment_ids: [N] list of segment IDs
    """
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
    if denoiser_name is not None:
        where.append("e.denoiser_name = %s")
        params.append(denoiser_name)
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

    print(f"Loading embeddings for model={model_name}, version={version}, mode={mode}")
    
    with _connect_neon() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    
    if not rows:
        raise ValueError(f"No embeddings found for {model_name} version {version}")
    
    # Parse results
    seg_ids = [r[0] for r in rows]
    embs = [np.array(r[1], dtype=np.float32) for r in rows]
    labels = np.array([int(r[2]) for r in rows], dtype=np.int64)
    video_ids = [r[3] for r in rows]
    
    all_embeddings = np.vstack(embs) if embs else np.zeros((0, 0), dtype=np.float32)
    
    print(f"Loaded {len(seg_ids)} samples | dim: {all_embeddings.shape[1]} | table: {table}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    return all_embeddings, labels, video_ids, seg_ids
