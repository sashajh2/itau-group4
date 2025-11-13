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
    print(f"📥 Queried {len(segments)} segments from Neon for created_at={created_at}")
    
    cursor.close()
    conn.close()
    
    return segments


def get_segments_with_existing_embeddings(created_at: str, version: str) -> set[str]:
    """
    Get set of segment_ids that already have embeddings for the given version.
    
    Checks all embedding tables (hubert, openl3, senet) for the given version.
    Returns a set of segment_ids that have at least one embedding.
    """
    config = load_config()
    dsn = config["database"]["postgres"]["neon_database_url"]
    
    conn = psycopg2.connect(dsn)
    cursor = conn.cursor()
    
    # Check all embedding tables
    embedding_tables = [
        "embeddings_audio_hubert",
        "embeddings_audio_openl3",
        "embeddings_video_senet",
    ]
    
    all_segment_ids = set()
    
    for table in embedding_tables:
        cursor.execute(f"""
            SELECT DISTINCT e.segment_id
            FROM {table} e
            JOIN segments s ON e.segment_id = s.segment_id
            WHERE s.created_at = %s AND e.version = %s
        """, (created_at, version))
        
        segment_ids = {row[0] for row in cursor.fetchall()}
        all_segment_ids.update(segment_ids)
        print(f"  📊 Found {len(segment_ids)} segments with embeddings in {table}")
    
    cursor.close()
    conn.close()
    
    print(f"📊 Total: {len(all_segment_ids)} unique segments already have embeddings")
    return all_segment_ids


def generate_for_created_at(
    created_at: str, 
    output_dir: str = "./embeddings/generated", 
    shard_writer: Optional[ShardWriter] = None, 
    neon_writer: Optional[NeonEmbeddingWriter] = None,
    skip_existing: bool = True,
) -> tuple[int, int]:
    """
    Generate embeddings for all segments with the given created_at.

    Args:
        created_at: ISO8601 timestamp to filter segments
        output_dir: Directory for output (currently unused but kept for compatibility)
        shard_writer: Optional ShardWriter for file-based storage
        neon_writer: Optional NeonEmbeddingWriter for Neon Postgres storage
        skip_existing: If True, skip segments that already have embeddings (saves computation)

    Returns (num_segments, num_embeddings_written)
    """
    print(f"🔍 Getting segments for created_at={created_at}")
    segments = get_segments_by_created_at_neon(created_at)
    print(f"Found {len(segments)} segments")
    
    if len(segments) == 0:
        print("❌ No segments found for the specified created_at timestamp")
        return 0, 0

    if shard_writer is None and neon_writer is None:
        raise ValueError("Provide either shard_writer or neon_writer")
    
    # Filter out segments that already have embeddings (if requested and neon_writer provided)
    original_count = len(segments)
    if skip_existing and neon_writer is not None:
        print(f"\n🔍 Checking for existing embeddings (version={neon_writer.version})...")
        existing_segment_ids = get_segments_with_existing_embeddings(created_at, neon_writer.version)
        
        if existing_segment_ids:
            segments = [s for s in segments if s['segment_id'] not in existing_segment_ids]
            skipped_count = original_count - len(segments)
            print(f"⏭️  Skipping {skipped_count} segments that already have embeddings")
            print(f"📝 Will process {len(segments)} remaining segments")
        
        if len(segments) == 0:
            print("✅ All segments already have embeddings! Nothing to do.")
            return original_count, 0
    
    # Process segments in batches to flush incrementally
    batch_size = 1000  # Process 1000 segments at a time
    total_attempted = 0
    total_segments = len(segments)
    
    print(f"🔄 Processing {total_segments} segments in batches of {batch_size}...")
    
    for batch_start in range(0, total_segments, batch_size):
        batch_end = min(batch_start + batch_size, total_segments)
        batch_segments = segments[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total_segments + batch_size - 1) // batch_size
        
        print(f"\n📦 Batch {batch_num}/{total_batches}: Processing segments {batch_start+1}-{batch_end} ({len(batch_segments)} segments)")
        
        # Generate embeddings for this batch
        print(f"  🔄 Generating embeddings...")
        accumulator = embed_segments(batch_segments)
        print(f"  ✅ Generated embeddings for batch {batch_num}")
        
        # Write embeddings to Neon immediately
        batch_attempted = 0
        for (mode, model, noise, denoiser_name), data in accumulator.items():
            embs = data["embeddings"]
            seg_ids = data["segment_ids"]
            for seg_id, emb in zip(seg_ids, embs):
                if neon_writer is not None:
                    neon_writer.add(model, mode, noise, denoiser_name, seg_id, emb)
                else:
                    shard_writer.add(model, mode, noise, denoiser_name, seg_id, emb)
                batch_attempted += 1
                total_attempted += 1
        
        # Flush immediately after each batch
        if neon_writer is not None:
            print(f"  💾 Flushing {batch_attempted} embeddings from batch {batch_num} to Neon...")
            try:
                neon_writer.flush_all()
                print(f"  ✅ Successfully flushed batch {batch_num} to Neon")
            except Exception as e:
                print(f"  ❌ ERROR flushing batch {batch_num}: {e}")
                raise
        print(f"  ✅ Batch {batch_num} complete ({batch_attempted} embeddings written)")
    
    print(f"\n✅ All batches complete! Total: {total_segments} segments, {total_attempted} embeddings")
    
    # Note: Don't flush/close neon_writer here - let the caller manage it
    # This allows batch_pipeline to flush at appropriate times
    return total_segments, total_attempted


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
