import argparse
import os
import shutil
from datetime import datetime, timezone
from tqdm import tqdm

from ...loaders.avdeepfake import download_and_extract_part
from ..extractors.segment_extractor import extract_and_insert_segments
from .embedding_pipeline import generate_for_created_at
from utils.config_loader import load_config
from ..storage.shard_writer import ShardWriter
from ..storage.neon_writer import NeonSegmentWriter, NeonEmbeddingWriter


def main():
    parser = argparse.ArgumentParser(description="Batch process AVDeepfake parts 002-050")
    parser.add_argument("--start", type=int, default=2, help="Start part (inclusive), e.g., 2 for 002")
    parser.add_argument("--end", type=int, default=50, help="End part (inclusive), e.g., 50 for 050")
    parser.add_argument("--base-dir", type=str, default="./data/temp_video_extracted/AV1M", help="Base local dir for downloads")
    parser.add_argument("--version", type=str, default="2025-09-12", help="Version string")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit number of segments per part (for testing)")
    args = parser.parse_args()

    config = load_config()
    
    # One Neon version tag for the whole batch (use pooler URL in config)
    neon_version = args.version
    
    # Create writers for the entire batch
    segment_writer = NeonSegmentWriter(batch_size=1000)
    embedding_writer = NeonEmbeddingWriter(version=neon_version, batch_size=1000)
    
    total_segments = 0
    total_attempted = 0

    try:
        for part in tqdm(range(args.start, args.end + 1), desc="Processing parts", unit="part"):
            part_str = f"{part:03d}"
            print(f"\n===== Processing part {part_str} =====")

            # Step 1: Download and extract this part
            zip_path, part_out_dir, log_path = download_and_extract_part(
                part=part_str,
                local_dir=args.base_dir,
            )

            # Print statements verifying step 1 is done
            print(f"ZIP_PATH={zip_path}")
            print(f"EXTRACTED_PART_DIR={part_out_dir}")
            if log_path:
                print(f"EXTRACTION_LOG={log_path}")

            # Where files are extracted
            extracted_part_dir = os.path.join(args.base_dir, "extracted", f"part_{part_str}")
            lrs3_root = os.path.join(extracted_part_dir, "train", "lrs3")

            # Step 2: Extract segments into Neon with a shared created_at for this part
            created_at = datetime.now(timezone.utc).isoformat()
            num_segments = extract_and_insert_segments(lrs3_root, created_at, segment_writer=segment_writer, limit=args.test_limit)
            print(f"Inserted {num_segments} segments for part {part_str}")
            # Flush segments after each part to ensure they're persisted
            segment_writer.flush_all()

            # Step 3: Generate embeddings for this created_at; write to Neon (batched)
            num_segments_processed, num_embeddings = generate_for_created_at(
                created_at, 
                "./embeddings/generated", 
                shard_writer=None, 
                neon_writer=embedding_writer
            )
            print(f"Processed: segments={num_segments_processed}, embeddings={num_embeddings}")
            total_segments += num_segments_processed
            total_attempted += num_embeddings

            # Step 4: Clean up local zip and extracted directory for this part
            zip_file = os.path.join(args.base_dir, "train", f"train.zip.{part_str}")
            try:
                if os.path.exists(zip_file):
                    os.remove(zip_file)
                    print(f"🗑️ Deleted {zip_file}")
            except Exception as e:
                print(f"⚠️ Could not delete {zip_file}: {e}")

            try:
                if os.path.exists(extracted_part_dir):
                    shutil.rmtree(extracted_part_dir)
                    print(f"🗑️ Deleted extracted dir {extracted_part_dir}")
            except Exception as e:
                print(f"⚠️ Could not delete {extracted_part_dir}: {e}")
    
    finally:
        # Final flush for any remaining embeddings
        embedding_writer.flush_all()
        embedding_writer.close()
        segment_writer.close()

    # Summary (Neon write path)
    print(f"Batch summary: segments={total_segments}, vectors_attempted={total_attempted}")

if __name__ == "__main__":
    main()