from db.embedding_store_utils import get_segments_by_created_at
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

def generate_for_created_at(created_at: str, output_dir: str = "./embeddings/generated", shard_writer: ShardWriter = None, neon_version: str | None = None) -> tuple[int, int]:
    """
    Generate embeddings for all segments with the given created_at.

    Returns (num_segments, num_uploaded_indices)
    """
    config = load_config()
    db_path = config["database"]["embedding_db_path"]
    print(f"Getting segments for {created_at}")
    segments = get_segments_by_created_at(db_path, created_at)
    print(f"Found {len(segments)} segments")
    
    # Debug: Check if there are any segments with this created_at
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM segments WHERE created_at = ?", (created_at,))
        total_count = cursor.fetchone()[0]
        print(f"🔍 Debug: Total segments in DB with created_at={created_at}: {total_count}")
    if len(segments) == 0:
        print("❌ No segments found for the specified created_at timestamp")
        return 0, 0

    print("🔄 Generating embeddings for all segments...")
    accumulator = embed_segments(segments)
    print("✅ Embedding generation complete")

    attempted = 0
    neon_writer = None
    if shard_writer is None and neon_version is None:
        raise ValueError("Provide either shard_writer or neon_version for Neon insertion")
    if neon_version is not None:
        neon_writer = NeonEmbeddingWriter(version=neon_version)
    
    for (mode, model, noise, denoiser_name), data in accumulator.items():
        embs = data["embeddings"]
        seg_ids = data["segment_ids"]
        for seg_id, emb in zip(seg_ids, embs):
            if neon_writer is not None:
                neon_writer.add(model, mode, noise, denoiser_name, seg_id, emb)
            else:
                shard_writer.add(model, mode, noise, denoiser_name, seg_id, emb)
            attempted += 1

    if neon_writer is not None:
        neon_writer.flush_all()
        neon_writer.close()
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
