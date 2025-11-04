from utils.config_loader import load_config
from ..generators.embedding_generator import embed_segments
from ..generators.embedding_saver import (
    save_embeddings_to_files,
    insert_embeddings_to_db,
)
from ..storage.shard_writer import ShardWriter
from ..storage.neon_writer import NeonEmbeddingWriter
from ..storage.dropbox_storage import create_faiss_index_and_upload
import argparse
import os
import psycopg2
from typing import Optional

def get_segments_by_created_at_neon(created_at: str) -> list[dict]:
    """Query segments from Neon Postgres by created_at timestamp."""
    config = load_config()
    dsn = config["database"]["postgres"]["neon_database_url"]
    
    conn = psycopg2.connect(dsn)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT segment_id, source, video_id, video_path, start_time, duration,
               video_label, audio_label, audio_model, video_model, created_at
        FROM segments
        WHERE created_at = %s
        ORDER BY segment_id
    """, (created_at,))
    
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    segments = [dict(zip(columns, row)) for row in rows]
    
    cursor.close()
    conn.close()
    
    return segments


def generate_for_created_at(
    created_at: str, 
    output_dir: str = "./embeddings/generated", 
    shard_writer: Optional[ShardWriter] = None, 
    neon_writer: Optional[NeonEmbeddingWriter] = None
) -> tuple[int, int]:
    """
    Generate embeddings for all segments with the given created_at.

    Args:
        created_at: ISO8601 timestamp to filter segments
        output_dir: Directory for output (currently unused but kept for compatibility)
        shard_writer: Optional ShardWriter for file-based storage
        neon_writer: Optional NeonEmbeddingWriter for Neon Postgres storage

    Returns (num_segments, num_embeddings_written)
    """
    print(f"Getting segments for {created_at}")
    segments = get_segments_by_created_at_neon(created_at)
    print(f"Found {len(segments)} segments")
    
    if len(segments) == 0:
        print("❌ No segments found for the specified created_at timestamp")
        return 0, 0

    print("🔄 Generating embeddings for all segments...")
    accumulator = embed_segments(segments)
    print("✅ Embedding generation complete")

    attempted = 0
    if shard_writer is None and neon_writer is None:
        raise ValueError("Provide either shard_writer or neon_writer")
    
    for (mode, model, noise, denoiser_name), data in accumulator.items():
        embs = data["embeddings"]
        seg_ids = data["segment_ids"]
        for seg_id, emb in zip(seg_ids, embs):
            if neon_writer is not None:
                neon_writer.add(model, mode, noise, denoiser_name, seg_id, emb)
            else:
                shard_writer.add(model, mode, noise, denoiser_name, seg_id, emb)
            attempted += 1

    # Note: Don't flush/close neon_writer here - let the caller manage it
    # This allows batch_pipeline to flush at appropriate times
    return len(segments), attempted


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for segments with a specific created_at")
    parser.add_argument("--created-at", type=str, required=True, help="ISO8601 created_at partition to process")
    parser.add_argument("--output-dir", type=str, default="./embeddings/generated", help="Directory to save embeddings")
    parser.add_argument("--shard-writer", type=ShardWriter, required=True, help="Shard writer")
    args = parser.parse_args()

    num_segments, num_uploaded = generate_for_created_at(args.created_at, args.output_dir, args.shard_writer)
    if num_segments > 0:
        print("\n🎉 Complete! Summary:")
        print(f"  - Processed {num_segments} segments")
        print(f"  - Uploaded {num_uploaded} shards to Dropbox")

if __name__ == "__main__":
    main()
