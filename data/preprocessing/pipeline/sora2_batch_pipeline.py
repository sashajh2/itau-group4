import argparse
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.preprocessing.pipeline.embedding_pipeline import generate_for_created_at
from data.preprocessing.storage.neon_writer import NeonEmbeddingWriter


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for Sora2 segments")
    parser.add_argument("--created-at", type=str, default="2025-11-21 16:27:37.670504+00", 
                       help="PostgreSQL timestamp for segments to process (default: 2025-11-21 16:27:37.670504+00)")
    parser.add_argument("--version", type=str, default="2025-09-12", help="Version string for embedding writer")
    args = parser.parse_args()

    # One Neon version tag for the whole batch
    neon_version = args.version
    created_at = args.created_at
    
    # Create embedding writer
    embedding_writer = NeonEmbeddingWriter(version=neon_version, batch_size=1000)

    print(f"🚀 Starting Sora2 embedding generation")
    print(f"📅 Created at timestamp: {created_at}")
    print(f"🔖 Embedding version: {neon_version}")

    try:
        # # Step 1: Extract segments into Neon with a shared created_at
        # print(f"\n{'='*60}")
        # print(f"🔍 Step 1: Extracting segments and inserting into database...")
        # print(f"{'='*60}")
        # created_at = datetime.now(timezone.utc).isoformat()
        # num_segments = extract_and_insert_sora2_segments(
        #     args.video_dir, 
        #     created_at, 
        #     args.segment_duration,
        #     segment_writer=segment_writer
        # )
        # print(f"📝 Inserted {num_segments} segments into Neon")
        # print(f"📅 Created at timestamp: {created_at}")
        # 
        # # Flush segments to ensure they're persisted
        # segment_writer.flush_all()
        # print("✅ Flushed segment buffer to Neon")
        
        # Step 2: Generate embeddings for this created_at (Neon write)
        print(f"\n{'='*60}")
        print(f"🧠 Step 2: Generating embeddings...")
        print(f"{'='*60}")
        num_segments_processed, num_embeddings = generate_for_created_at(
            created_at, 
            "./embeddings/generated", 
            shard_writer=None, 
            neon_writer=embedding_writer
        )
        print(f"🧠 Read {num_segments_processed} segments from Neon for embeddings")
        print(f"🧮 Generated {num_embeddings} embeddings (attempted writes)")
        
        # Flush embeddings to persist
        embedding_writer.flush_all()
        print("✅ Flushed embedding buffer to Neon")

        print(f"\n{'='*60}")
        print(f"🎉 Sora2 embedding generation completed successfully!")
        print(f"📊 Processed {num_segments_processed} segments")
        print(f"🧮 Generated {num_embeddings} embeddings")
        print(f"📅 Created at timestamp: {created_at}")
        print(f"{'='*60}")
    
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Final flush for any remaining data
        embedding_writer.flush_all()
        embedding_writer.close()

    return 0


if __name__ == "__main__":
    exit(main())

