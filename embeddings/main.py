import argparse
from audio_embedder import AudioEmbedder
from video_embedder import VideoEmbedder
from forensic_embedder import ForensicEmbedder
from faiss_indexer import FaissIndexer
# from data.loaders.audio_loader import get_audio_loader  # Uncomment and implement this
import os
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import pickle


def main():
    parser = argparse.ArgumentParser(description="Embedding Extraction and Indexing")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--dataset_filepath', type=str, required=True, help='Path to dataset DataFrame (e.g., .pkl)')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for embeddings and index')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--faiss', action='store_true', help='Whether to build a FAISS index')
    parser.add_argument('--mode', type=str, required=True, choices=['audio', 'video', 'forensic'], help='Embedding mode: audio, video, or forensic')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_pickle(args.dataset_filepath)

    if args.mode == 'audio':
        embedder = AudioEmbedder(args.model_name, args.checkpoint_path, args.device)
        embedding_fn = embedder.get_embedding_fn(args.model_name)
    elif args.mode == 'video':
        embedder = VideoEmbedder(args.model_name, args.checkpoint_path, args.device)
        embedding_fn = embedder.embedding_fn  # TODO: add get_embedding_fn for video if needed
    elif args.mode == 'forensic':
        embedder = ForensicEmbedder(args.model_name, args.checkpoint_path, args.device)
        embedding_fn = embedder.embedding_fn  # TODO: add get_embedding_fn for forensic if needed
    else:
        raise ValueError("Invalid mode. Must be one of: audio, video, forensic.")

    results_df = embedder.embed_dataset(df, embedding_fn, mode=args.mode)
    embeddings = results_df['embedding'].to_list()
    labels = results_df['label'].to_list()

    # Save embeddings
    embeddings = np.stack(embeddings)
    np.save(f"{args.out_dir}/embeddings.npy", embeddings)

    # Save labels
    with open(f"{args.out_dir}/labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    # Save metadata: all columns from the original DataFrame, aligned with embeddings
    metadata = [row._asdict() if hasattr(row, '_asdict') else row.to_dict() for _, row in df.iterrows()]
    with open(f"{args.out_dir}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata for {len(metadata)} samples, aligned with embeddings.")

    if args.faiss:
        indexer = FaissIndexer(dim=embeddings.shape[1])
        indexer.build_index(embeddings)
        indexer.save_index(os.path.join(args.out_dir, 'index.faiss'))
        print(f"FAISS index saved to {os.path.join(args.out_dir, 'index.faiss')}")

    print(f"Embeddings, labels, and metadata saved to {args.out_dir}")

if __name__ == "__main__":
    main()
