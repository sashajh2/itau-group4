#!/usr/bin/env python3
"""
Main entry point for the itau-group4 project.
Provides easy access to key functions for data processing.
"""

import sys
import os

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.preprocessing.pipeline.embedding_pipeline import generate_for_created_at
from data.preprocessing.pipeline.av_deepfake_batch_pipeline import main as batch_process_main
from data.loaders.avdeepfake import download_and_extract_part
from data.preprocessing.extractors.av_deepfake_segment_extractor import extract_and_insert_segments

def main():
    """Main entry point for the CLI."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [args...]")
        print("\nAvailable commands:")
        print("  embed <created_at> [output_dir]  - Generate embeddings for segments")
        print("  batch [start] [end] [base_dir]  - Process AVDeepfake parts in batch")
        print("  download <part> [local_dir]     - Download and extract a zip part")
        print("  extract <video_root> [created_at] - Extract segments from videos")
        print("\nExamples:")
        print("  python main.py embed 2024-01-01T00:00:00Z")
        print("  python main.py batch 2 10")
        print("  python main.py download 001")
        print("  python main.py extract ./videos/")
        return
    
    command = sys.argv[1]
    
    try:
        if command == "embed":
            if len(sys.argv) < 3:
                print("Error: embed command requires created_at argument")
                return
            created_at = sys.argv[2]
            output_dir = sys.argv[3] if len(sys.argv) > 3 else "./embeddings/generated"
            
            num_segments, num_uploaded = generate_for_created_at(created_at, output_dir)
            print(f"✅ Processed {num_segments} segments, uploaded {num_uploaded} indices")
            
        elif command == "batch":
            start = int(sys.argv[2]) if len(sys.argv) > 2 else 2
            end = int(sys.argv[3]) if len(sys.argv) > 3 else 50
            base_dir = sys.argv[4] if len(sys.argv) > 4 else "./data/temp_video_extracted/AV1M"
            
            # Set sys.argv for the batch processor
            sys.argv = [sys.argv[0], "--start", str(start), "--end", str(end), "--base-dir", base_dir]
            batch_process_main()
            
        elif command == "download":
            if len(sys.argv) < 3:
                print("Error: download command requires part argument")
                return
            part = sys.argv[2]
            local_dir = sys.argv[3] if len(sys.argv) > 3 else "./data/temp_video_extracted/AV1M"
            
            zip_path, part_out_dir, log_path = download_and_extract_part(part, local_dir)
            print(f"✅ Downloaded: {zip_path}")
            print(f"✅ Extracted to: {part_out_dir}")
            if log_path:
                print(f"✅ Log saved to: {log_path}")
                
        elif command == "extract":
            if len(sys.argv) < 3:
                print("Error: extract command requires video_root argument")
                return
            video_root = sys.argv[2]
            created_at = sys.argv[3] if len(sys.argv) > 3 else None
            
            from datetime import datetime, timezone
            if created_at is None:
                created_at = datetime.now(timezone.utc).isoformat()
                
            num_segments = extract_and_insert_segments(video_root, created_at)
            print(f"✅ Extracted {num_segments} segments with created_at: {created_at}")
            
        else:
            print(f"Unknown command: {command}")
            print("Use 'python main.py' to see available commands")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
