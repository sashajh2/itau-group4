#!/usr/bin/env python3
"""
Export segments and embeddings from Neon Postgres for analysis.

This script:
1. Retrieves all segments from a given created_at batch
2. Joins with embedding tables (hubert, openl3, senet)
3. Creates a long-format DataFrame with segment_idx for alignment
4. Saves as CSV (metadata only) and .npz files (embeddings + metadata)

Usage:
    python scripts/export_batch_for_analysis.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import psycopg2
from typing import List
from tqdm import tqdm

from utils.config_loader import load_config

# Configuration
BATCH_CREATED_AT = "2025-11-05 17:31:18.485225+00"
EMBEDDING_TABLES = {
    "hubert": "embeddings_audio_hubert",  # Audio embeddings from Hubert model
    "openl3": "embeddings_audio_openl3",  # Audio embeddings from OpenL3 model
    "senet": "embeddings_video_senet",    # Video embeddings from SENet model
}
OUTPUT_DIR = "exports"


def connect_neon():
    """Connect to Neon Postgres database."""
    cfg = load_config()
    dsn = cfg["database"]["postgres"]["neon_database_url"]
    return psycopg2.connect(dsn)


def fetch_segments_with_embeddings(
    conn: psycopg2.extensions.connection,
    embedding_table: str,
    created_at: str,
) -> pd.DataFrame:
    """
    Query segments joined with embeddings for a given table and created_at.
    
    Returns:
        DataFrame with columns: segment_id, source, video_id, video_path, start_time,
        duration, video_label, audio_label, audio_model, video_model, created_at, embedding
    """
    query = """
        SELECT
            s.segment_id,
            s.source,
            s.video_id,
            s.video_path,
            s.start_time,
            s.duration,
            s.video_label,
            s.audio_label,
            s.audio_model,
            s.video_model,
            s.created_at,
            e.embedding::float4[] AS embedding
        FROM segments s
        JOIN {table} e
          ON e.segment_id = s.segment_id
        WHERE s.created_at = %s
        ORDER BY s.video_id, s.video_path, s.start_time
    """.format(table=embedding_table)
    
    # Use psycopg2 connection directly (pandas warns but it works fine)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='.*pandas only supports SQLAlchemy.*')
        df = pd.read_sql_query(query, conn, params=(created_at,))
    
    # Convert embedding from list to numpy array
    # Use tqdm for progress if there are many rows
    if len(df) > 1000:
        tqdm.pandas()
        df['embedding'] = df['embedding'].progress_apply(lambda x: np.array(x, dtype=np.float32))
    else:
        df['embedding'] = df['embedding'].apply(lambda x: np.array(x, dtype=np.float32))
    
    return df


def add_segment_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add segment_idx column for alignment across augmentations.
    
    Groups by (video_id, video_path) and assigns sequential indices.
    """
    df = df.copy()
    df['segment_idx'] = df.groupby(['video_id', 'video_path']).cumcount()
    return df


def main():
    """Main export function."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Sanitize timestamp for filename
    safe_ts = BATCH_CREATED_AT.replace(" ", "_").replace(":", "-").replace("+", "p")
    
    print(f"📊 Exporting batch: {BATCH_CREATED_AT}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🔗 Connecting to Neon...")
    
    # Connect to database
    conn = connect_neon()
    
    try:
        # Fetch data for each embedding model
        all_dfs: List[pd.DataFrame] = []
        
        for model_name, table_name in tqdm(EMBEDDING_TABLES.items(), desc="Fetching embeddings"):
            print(f"\n📥 Fetching {model_name} embeddings from {table_name}...")
            
            try:
                df = fetch_segments_with_embeddings(conn, table_name, BATCH_CREATED_AT)
                
                if len(df) == 0:
                    print(f"  ⚠️  No data found for {model_name}")
                    continue
                
                # Add segment_idx
                df = add_segment_idx(df)
                
                # Add embedding_model column
                df['embedding_model'] = model_name
                
                # Reorder columns
                cols = [
                    'segment_id', 'source', 'video_id', 'video_path', 'start_time', 'duration',
                    'video_label', 'audio_label', 'audio_model', 'video_model', 'created_at',
                    'embedding', 'segment_idx', 'embedding_model'
                ]
                df = df[cols]
                
                all_dfs.append(df)
                print(f"  ✅ Loaded {len(df)} rows")
                
            except Exception as e:
                print(f"  ❌ Error fetching {model_name} from {table_name}: {e}")
                print(f"     (Table might not exist or have different name)")
                continue
        
        if len(all_dfs) == 0:
            print("\n❌ No data found for any embedding model!")
            return
        
        # Concatenate all DataFrames
        print(f"\n🔗 Concatenating {len(all_dfs)} model DataFrames...")
        df_all = pd.concat(all_dfs, ignore_index=True)
        print(f"✅ Total rows: {len(df_all)}")
        
        # Save combined CSV file (metadata only, no embeddings)
        # CSV: Native pandas format, no dependencies required
        # - Easy to inspect, filter, and join in Excel/pandas
        # - Embeddings excluded (too large, use NPZ files instead)
        csv_path = os.path.join(OUTPUT_DIR, f"all_models_batch_{safe_ts}_metadata.csv")
        print(f"\n💾 Saving metadata (without embeddings) to {csv_path}...")
        df_metadata = df_all.drop(columns=['embedding'])  # Remove embeddings column
        df_metadata.to_csv(csv_path, index=False)
        print(f"✅ Saved {len(df_metadata)} rows to CSV")
        
        # Save individual .npz files per model
        # NPZ: NumPy's native compressed format
        # - Embeddings pre-stacked into 2D arrays (N x D) for fast access
        # - All metadata as separate arrays, easy to index together
        # - Best for NumPy-based ML workflows (PCA, UMAP, clustering, distance calculations)
        # - More memory-efficient when loading all embeddings at once
        print(f"\n💾 Saving individual .npz files per model...")
        for model_name in tqdm(EMBEDDING_TABLES.keys(), desc="Saving .npz files"):
            df_model = df_all[df_all['embedding_model'] == model_name]
            
            if len(df_model) == 0:
                print(f"  ⚠️  Skipping {model_name} (no data)")
                continue
            
            # Stack embeddings into numpy array
            # This converts list of arrays into a single 2D array (N x D)
            embeddings = np.stack(df_model['embedding'].to_numpy(), axis=0)
            
            npz_path = os.path.join(OUTPUT_DIR, f"{model_name}_batch_{safe_ts}.npz")
            np.savez_compressed(
                npz_path,
                embeddings=embeddings,
                segment_ids=df_model['segment_id'].astype(str).to_numpy(),
                source=df_model['source'].astype(str).to_numpy(),
                video_ids=df_model['video_id'].astype(str).to_numpy(),
                video_paths=df_model['video_path'].astype(str).to_numpy(),
                segment_idx=df_model['segment_idx'].to_numpy(),
                start_time=df_model['start_time'].to_numpy(),
                duration=df_model['duration'].to_numpy(),
                video_label=df_model['video_label'].to_numpy(),
                audio_label=df_model['audio_label'].to_numpy(),
                audio_model=df_model['audio_model'].fillna('').astype(str).to_numpy(),
                video_model=df_model['video_model'].fillna('').astype(str).to_numpy(),
                created_at=df_model['created_at'].astype(str).to_numpy()
            )
            print(f"  ✅ Saved {model_name}: {len(df_model)} samples, shape {embeddings.shape} to {npz_path}")
        
        # Print summary statistics
        print(f"\n📊 Summary Statistics:")
        print(f"   Total segments: {df_all['segment_id'].nunique()}")
        print(f"   Total videos: {df_all['video_id'].nunique()}")
        print(f"   Total video_paths: {df_all['video_path'].nunique()}")
        print(f"   Embedding models: {df_all['embedding_model'].unique().tolist()}")
        print(f"\n   Label distributions:")
        print(f"     video_label: {df_all['video_label'].value_counts().sort_index().to_dict()}")
        print(f"     audio_label: {df_all['audio_label'].value_counts().sort_index().to_dict()}")
        
    finally:
        conn.close()
        print(f"\n✅ Export complete!")


if __name__ == "__main__":
    main()

