import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from tqdm import tqdm

from data.loaders.share_veo3 import download_and_extract_part, cleanup_files
from data.preprocessing.extractors.share_veo3_segment_extractor import extract_and_insert_share_veo3_segments
from data.preprocessing.pipeline.embedding_pipeline import generate_for_created_at
from utils.config_loader import load_config
from data.preprocessing.storage.shard_writer import ShardWriter


def main():
    parser = argparse.ArgumentParser(description="Batch process ShareVeo3 parts 1-50")
    parser.add_argument("--start", type=int, default=1, help="Start part (inclusive), e.g., 1 for part 1")
    parser.add_argument("--end", type=int, default=50, help="End part (inclusive), e.g., 50 for part 50")
    parser.add_argument("--base-dir", type=str, default="./data/temp_video_extracted/ShareVeo3", help="Base local dir for downloads")
    parser.add_argument("--cleanup", action="store_true", help="Clean up tar files and extracted directories after processing")
    parser.add_argument("--segment-duration", type=float, default=0.25, help="Duration of each segment in seconds")
    parser.add_argument("--version", type=str, default="2025-09-12", help="Version string")
    args = parser.parse_args()

    # Validate part numbers
    if args.start < 1 or args.start > 50:
        print("❌ Start part must be between 1 and 50")
        return 1
    
    if args.end < 1 or args.end > 50:
        print("❌ End part must be between 1 and 50")
        return 1
    
    if args.start > args.end:
        print("❌ Start part must be less than or equal to end part")
        return 1

    print(f"🚀 Starting ShareVeo3 batch processing from part {args.start} to {args.end}")
    print(f"📁 Base directory: {args.base_dir}")
    print(f"🧹 Cleanup after processing: {args.cleanup}")
    print(f"⏱️ Segment duration: {args.segment_duration} seconds")
    print(f"🏷️ Version: {args.version}")

    # Initialize ShardWriter for the entire batch
    config = load_config()
    db_path = config["database"]["embedding_db_path"]
    shard_writer = ShardWriter(
        dropbox_root="/embedding_store", 
        db_path=db_path, 
        source="ShareVeo3", 
        version=args.version
    )
    
    total_segments = 0
    total_attempted = 0

    for part in tqdm(range(args.start, args.end + 1), desc="Processing parts", unit="part"):
        part_str = f"{part:02d}"
        print(f"\n{'='*60}")
        print(f"🔄 Processing ShareVeo3 part {part_str}")
        print(f"{'='*60}")

        try:
            # Step 1: Download and extract this part
            print(f"📥 Step 1: Downloading and extracting part {part_str}...")
            tar_path, part_out_dir, log_path = download_and_extract_part(
                part=part,
                local_dir=args.base_dir,
            )

            # Print verification info
            print(f"✅ Download and extraction completed")
            print(f"TAR_PATH={tar_path}")
            print(f"EXTRACTED_PART_DIR={part_out_dir}")
            if log_path:
                print(f"EXTRACTION_LOG={log_path}")

            # Step 2: Extract segments into DB with a shared created_at for this part
            print(f"🔍 Step 2: Extracting segments and inserting into database...")
            created_at = datetime.now(timezone.utc).isoformat()
            num_segments = extract_and_insert_share_veo3_segments(
                part_out_dir, 
                created_at, 
                args.segment_duration
            )
            print(f"✅ Inserted {num_segments} segments for part {part_str}")
            print(f"📅 Created at timestamp: {created_at}")

            # Step 3: Generate embeddings for this created_at
            print(f"🧠 Step 3: Generating embeddings...")
            num_segments_processed, num_attempted = generate_for_created_at(created_at, "./embeddings/generated", shard_writer)
            print(f"✅ Embeddings: processed={num_segments_processed}, attempted={num_attempted}")
            total_segments += num_segments_processed
            total_attempted += num_attempted

            # Step 4: Clean up if requested
            if args.cleanup:
                print(f"🧹 Step 4: Cleaning up files...")
                cleanup_files(tar_path, part_out_dir)
                print(f"✅ Cleanup completed for part {part_str}")
            else:
                print(f"💾 Step 4: Skipping cleanup (files preserved)")

            print(f"🎉 Part {part_str} processing completed successfully!")

        except Exception as e:
            print(f"❌ Error processing part {part_str}: {e}")
            print(f"⚠️ Continuing with next part...")
            continue

    # Finalize the ShardWriter
    shard_writer.finalize()
    
    print(f"\n{'='*60}")
    print(f"🏁 ShareVeo3 batch processing completed!")
    print(f"📊 Processed parts {args.start} to {args.end}")
    print(f"📈 Total segments: {total_segments}")
    print(f"📈 Total embeddings attempted: {total_attempted}")
    print(f"📈 Uploaded shards: {getattr(shard_writer, 'uploaded_shards', 'n/a')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    exit(main()) 
