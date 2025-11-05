#!/usr/bin/env python3
"""
Command-line interface for the data package.
"""

import argparse
import sys
from datetime import datetime, timezone

from preprocessing.pipeline.embedding_pipeline import generate_for_created_at
from preprocessing.pipeline.av_deepfake_batch_pipeline import main as batch_process_main
from loaders.avdeepfake import download_and_extract_part
from preprocessing.extractors.av_deepfake_segment_extractor import extract_and_insert_segments


def main():
    parser = argparse.ArgumentParser(description="Data processing CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Embedding generation command
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings")
    embed_parser.add_argument("--created-at", type=str, required=True, help="ISO8601 created_at partition")
    embed_parser.add_argument("--output-dir", type=str, default="./embeddings/generated", help="Output directory")
    
    # Batch processing command
    batch_parser = subparsers.add_parser("batch", help="Batch process AVDeepfake parts")
    batch_parser.add_argument("--start", type=int, default=2, help="Start part (inclusive)")
    batch_parser.add_argument("--end", type=int, default=50, help="End part (inclusive)")
    batch_parser.add_argument("--base-dir", type=str, default="./data/temp_video_extracted/AV1M", help="Base directory")
    
    # Download and extract command
    download_parser = subparsers.add_parser("download", help="Download and extract a zip part")
    download_parser.add_argument("--part", type=str, required=True, help="Part number (e.g., 001)")
    download_parser.add_argument("--local-dir", type=str, default="./data/temp_video_extracted/AV1M", help="Local directory")
    
    # Segment extraction command
    segment_parser = subparsers.add_parser("extract", help="Extract segments from videos")
    segment_parser.add_argument("--video-root", type=str, required=True, help="Root directory containing videos")
    segment_parser.add_argument("--created-at", type=str, help="Created timestamp (defaults to now)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "embed":
            num_segments, num_uploaded = generate_for_created_at(args.created_at, args.output_dir)
            print(f"✅ Processed {num_segments} segments, uploaded {num_uploaded} indices")
            
        elif args.command == "batch":
            # Set sys.argv for the batch processor
            sys.argv = [sys.argv[0], "--start", str(args.start), "--end", str(args.end), "--base-dir", args.base_dir]
            batch_process_main()
            
        elif args.command == "download":
            zip_path, part_out_dir, log_path = download_and_extract_part(args.part, args.local_dir)
            print(f"✅ Downloaded: {zip_path}")
            print(f"✅ Extracted to: {part_out_dir}")
            if log_path:
                print(f"✅ Log saved to: {log_path}")
                
        elif args.command == "extract":
            created_at = args.created_at or datetime.now(timezone.utc).isoformat()
            num_segments = extract_and_insert_segments(args.video_root, created_at)
            print(f"✅ Extracted {num_segments} segments with created_at: {created_at}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
