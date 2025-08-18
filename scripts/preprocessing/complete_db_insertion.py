#!/usr/bin/env python3
"""
Script to complete database insertion for existing embeddings.
This is useful when embeddings were generated and saved but the DB insertion failed.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.preprocessing.embedding_saver import insert_embeddings_to_db
from scripts.preprocessing.embedding_generator import embed_segments
from utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description='Complete database insertion for existing embeddings')
    parser.add_argument('--output-dir', 
                       default='./embeddings/generated',
                       help='Directory containing the generated embeddings (default: ./embeddings/generated)')
    parser.add_argument('--db-path',
                       default='./db/embeddings.sqlite3',
                       help='Path to the embeddings database')
    parser.add_argument('--created-at',
                       help='Timestamp for the embedding generation (default: current time)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    db_path = args.db_path
    
    if not output_dir.exists():
        print(f"❌ Output directory does not exist: {output_dir}")
        return 1
    
    if not os.path.exists(db_path):
        print(f"❌ Database does not exist: {db_path}")
        return 1
    
    # Check if there are any .npy files
    npy_files = list(output_dir.glob("*.npy"))
    if not npy_files:
        print(f"❌ No .npy files found in {output_dir}")
        return 1
    
    print(f"📁 Found {len(npy_files)} embedding files in {output_dir}")
    
    # Use provided created_at or current time
    if args.created_at:
        created_at = args.created_at
    else:
        created_at = datetime.now(timezone.utc).isoformat()
    
    print(f"🕐 Using created_at: {created_at}")
    
    # We need to reconstruct the accumulator structure from the saved files
    # This is a bit hacky but necessary since we don't have the original accumulator
    print("🔧 Reconstructing accumulator from saved files...")
    
    accumulator = {}
    
    for npy_file in npy_files:
        # Parse filename to get model and mode
        base_name = npy_file.stem  # Remove .npy extension
        
        # Extract model and mode from filename
        parts = base_name.split("_")
        
        # Find the mode
        mode = None
        model_name = None
        
        if "audio_denoised" in base_name:
            mode = "audio_denoised"
            mode_start = base_name.find("audio_denoised")
            model_name = base_name[:mode_start-1]
        elif "audio_noise" in base_name:
            mode = "audio_noise"
            mode_start = base_name.find("audio_noise")
            model_name = base_name[:mode_start-1]
        elif "audio" in base_name and "audio_denoised" not in base_name and "audio_noise" not in base_name:
            mode = "audio"
            mode_start = base_name.find("audio")
            model_name = base_name[:mode_start-1]
        elif "video" in base_name:
            mode = "video"
            mode_start = base_name.find("video")
            model_name = base_name[:mode_start-1]
        else:
            print(f"⚠️ Could not parse mode from filename: {npy_file.name}")
            continue
        
        print(f"  📝 Parsed: model='{model_name}', mode='{mode}' from '{npy_file.name}'")
        
        # Load the mapping file to get segment IDs
        mapping_file = npy_file.parent / f"{npy_file.stem}_mapping.json"
        if not mapping_file.exists():
            print(f"⚠️ Mapping file not found for {npy_file.name}")
            continue
        
        with open(mapping_file, 'r') as f:
            import json
            mapping = json.load(f)
        
        # Extract segment IDs in order
        segment_ids = [mapping["index_to_segment"][str(idx)] for idx in range(len(mapping["index_to_segment"]))]
        
        # Create accumulator entry
        accumulator[(model_name, mode)] = {
            "embeddings": [],  # We don't need the actual embeddings for DB insertion
            "segment_ids": segment_ids
        }
    
    print(f"✅ Reconstructed accumulator with {len(accumulator)} entries")
    
    # Now insert into database
    print("🗄️ Inserting embeddings into database...")
    try:
        insert_embeddings_to_db(accumulator, db_path, created_at, str(output_dir))
        print("✅ Database insertion complete!")
    except Exception as e:
        print(f"❌ Database insertion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 