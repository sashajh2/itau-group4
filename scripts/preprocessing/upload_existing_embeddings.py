#!/usr/bin/env python3
"""
Script to upload existing embeddings from the generated folder to Dropbox.
This is useful when the embeddings were generated but the Dropbox upload failed.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.preprocessing.dropbox_uploader import create_faiss_index_and_upload

def main():
    parser = argparse.ArgumentParser(description='Upload existing embeddings to Dropbox')
    parser.add_argument('--output-dir', 
                       default='./embeddings/generated',
                       help='Directory containing the generated embeddings (default: ./embeddings/generated)')
    parser.add_argument('--dropbox-base-path',
                       default='/embedding_store/AVDeepfake1M/raw/',
                       help='Base path in Dropbox for uploading indices')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        print(f"❌ Output directory does not exist: {output_dir}")
        return
    
    # Check if there are any .npy files
    npy_files = list(output_dir.glob("*.npy"))
    if not npy_files:
        print(f"❌ No .npy files found in {output_dir}")
        return
    
    print(f"📁 Found {len(npy_files)} embedding files in {output_dir}")
    print("📋 Files to upload:")
    for npy_file in npy_files:
        print(f"  - {npy_file.name}")
    
    print(f"\n🚀 Starting Dropbox upload...")
    print(f"📤 Dropbox base path: {args.dropbox_base_path}")
    
    try:
        uploaded_files = create_faiss_index_and_upload(
            str(output_dir), 
            args.dropbox_base_path
        )
        
        print(f"\n✅ Upload complete!")
        print(f"📊 Summary:")
        for file_info in uploaded_files:
            status = "📥 Appended" if file_info.get("appended", False) else "🆕 Created"
            print(f"  {status}: {file_info['model']}_{file_info['mode']} "
                  f"({file_info['shape'][0]} embeddings)")
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 