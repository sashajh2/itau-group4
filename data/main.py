#!/usr/bin/env python3
"""
Main entry point for the data package.
Provides easy access to key functions for data processing.
"""

from data.preprocessing.pipeline.embedding_pipeline import generate_for_created_at
from data.preprocessing.pipeline.batch_pipeline import main as batch_process_main
from data.loaders.avdeepfake import download_and_extract_part
from data.preprocessing.extractors.segment_extractor import extract_and_insert_segments

# Export main functions for easy access
__all__ = [
    'generate_for_created_at',
    'batch_process_main', 
    'download_and_extract_part',
    'extract_and_insert_segments'
]

if __name__ == "__main__":
    print("Data package main module. Import specific functions as needed.")
    print("Available functions:")
    print("  - generate_for_created_at: Generate embeddings for segments")
    print("  - batch_process_main: Process AVDeepfake parts in batch")
    print("  - download_and_extract_part: Download and extract a zip part")
    print("  - extract_and_insert_segments: Extract segments and insert into DB")
