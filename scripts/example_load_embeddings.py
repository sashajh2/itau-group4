#!/usr/bin/env python3
"""
Example script showing how to load the retrieved embeddings and labels.

This demonstrates how to load the .npy and .pkl files created by retrieve_embeddings.py
and use them for analysis or machine learning tasks.
"""

import numpy as np
import pickle
import argparse
from pathlib import Path


def load_embeddings_and_labels(embeddings_file: str, labels_file: str):
    """
    Load embeddings and labels from the files created by retrieve_embeddings.py
    
    Args:
        embeddings_file: Path to the .npy file containing embeddings
        labels_file: Path to the .pkl file containing labels and metadata
    
    Returns:
        Tuple of (embeddings, labels, metadata)
    """
    # Load embeddings
    print(f"Loading embeddings from: {embeddings_file}")
    embeddings = np.load(embeddings_file)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Load labels and metadata
    print(f"Loading labels from: {labels_file}")
    with open(labels_file, 'rb') as f:
        labels_data = pickle.load(f)
    
    labels = labels_data['labels']
    segment_ids = labels_data['segment_ids']
    metadata = {k: v for k, v in labels_data.items() if k not in ['labels', 'segment_ids']}
    
    print(f"Labels shape: {labels.shape}")
    print(f"Number of segments: {len(segment_ids)}")
    print(f"Metadata: {metadata}")
    
    return embeddings, labels, segment_ids, metadata


def analyze_embeddings(embeddings, labels, metadata):
    """
    Perform basic analysis on the loaded embeddings and labels.
    """
    print("\n" + "="*50)
    print("EMBEDDING ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print(f"Model: {metadata.get('model_name', 'Unknown')}")
    print(f"Version: {metadata.get('version', 'Unknown')}")
    print(f"Mode: {metadata.get('mode', 'Unknown')}")
    print(f"Noise: {metadata.get('noise', 'Unknown')}")
    print(f"Denoiser: {metadata.get('denoiser_name', 'None')}")
    
    print(f"\nEmbedding Statistics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {np.mean(embeddings):.4f}")
    print(f"  Std: {np.std(embeddings):.4f}")
    print(f"  Min: {np.min(embeddings):.4f}")
    print(f"  Max: {np.max(embeddings):.4f}")
    
    print(f"\nLabel Statistics:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(labels)) * 100
        print(f"  Label {label}: {count} samples ({percentage:.1f}%)")
    
    # Check for any missing data
    if np.any(np.isnan(embeddings)):
        print("⚠️  Warning: Found NaN values in embeddings")
    
    if np.any(np.isinf(embeddings)):
        print("⚠️  Warning: Found infinite values in embeddings")


def main():
    parser = argparse.ArgumentParser(description="Load and analyze retrieved embeddings")
    parser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings .npy file")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels .pkl file")
    parser.add_argument("--analyze", action="store_true", help="Perform basic analysis")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.embeddings).exists():
        print(f"❌ Embeddings file not found: {args.embeddings}")
        return
    
    if not Path(args.labels).exists():
        print(f"❌ Labels file not found: {args.labels}")
        return
    
    try:
        # Load the data
        embeddings, labels, segment_ids, metadata = load_embeddings_and_labels(
            args.embeddings, args.labels
        )
        
        if args.analyze:
            analyze_embeddings(embeddings, labels, metadata)
        
        print(f"\n✅ Successfully loaded data!")
        print(f"You can now use 'embeddings' and 'labels' variables in your code.")
        
        return embeddings, labels, segment_ids, metadata
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


if __name__ == "__main__":
    main()
