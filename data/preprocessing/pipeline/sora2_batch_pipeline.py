import argparse
import os
import sys
from datetime import datetime, timezone

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.preprocessing.extractors.sora2_segment_extractor import extract_and_insert_sora2_segments
from data.preprocessing.pipeline.embedding_pipeline import generate_for_created_at
from data.preprocessing.storage.neon_writer import NeonSegmentWriter, NeonEmbeddingWriter


def main():
    parser = argparse.ArgumentParser(description="Batch process Sora2 videos from a folder")
    parser.add_argument("--video-dir", type=str, required=True, help="Directory containing Sora2 video files")
    parser.add_argument("--segment-duration", type=float, default=0.15, help="Duration of each segment in seconds")
    parser.add_argument("--version", type=str, default="2025-09-12", help="Version string for embedding writer")
    args = parser.parse_args()

    # Validate video directory
    if not os.path.exists(args.video_dir):
        print(f"❌ Video directory not found: {args.video_dir}")
        return 1
    
    if not os.path.isdir(args.video_dir):
        print(f"❌ Path is not a directory: {args.video_dir}")
        return 1

    # One Neon version tag for the whole batch
    neon_version = args.version
    
    # Create writers for the batch
    segment_writer = NeonSegmentWriter(batch_size=1000)
    embedding_writer = NeonEmbeddingWriter(version=neon_version, batch_size=1000)

    print(f"🚀 Starting Sora2 batch processing")
    print(f"📁 Video directory: {args.video_dir}")
    print(f"⏱️ Segment duration: {args.segment_duration} seconds")
    print(f"🔖 Embedding version: {neon_version}")

    try:
        # Step 1: Extract segments into Neon with a shared created_at
        print(f"\n{'='*60}")
        print(f"🔍 Step 1: Extracting segments and inserting into database...")
        print(f"{'='*60}")
        created_at = datetime.now(timezone.utc).isoformat()
        num_segments = extract_and_insert_sora2_segments(
            args.video_dir, 
            created_at, 
            args.segment_duration,
            segment_writer=segment_writer
        )
        print(f"📝 Inserted {num_segments} segments into Neon")
        print(f"📅 Created at timestamp: {created_at}")
        
        # Flush segments to ensure they're persisted
        segment_writer.flush_all()
        print("✅ Flushed segment buffer to Neon")
        
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
        print(f"🎉 Sora2 batch processing completed successfully!")
        print(f"📊 Processed {num_segments} segments")
        print(f"📅 Created at timestamp: {created_at}")
        print(f"{'='*60}")
    
    except Exception as e:
        print(f"❌ Error processing Sora2 videos: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Final flush for any remaining data
        embedding_writer.flush_all()
        embedding_writer.close()
        segment_writer.close()

    return 0


if __name__ == "__main__":
    exit(main())

