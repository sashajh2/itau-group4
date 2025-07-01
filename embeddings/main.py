import argparse
from audio_embedder import AudioEmbedder
from video_embedder import VideoEmbedder
from forensic_embedder import ForensicEmbedder
from faiss_indexer import FaissIndexer
# from data.loaders.audio_loader import get_audio_loader  # Uncomment and implement this
import os


def main():
    parser = argparse.ArgumentParser(description="Embedding Extraction and Indexing")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for embeddings and index')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding extraction')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--faiss', action='store_true', help='Whether to build a FAISS index')
    parser.add_argument('--mode', type=str, required=True, choices=['audio', 'video', 'forensic'], help='Embedding mode: audio, video, or forensic')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # loader = get_audio_loader(args.data_path, batch_size=args.batch_size)  # Implement this
    loader = []  # Placeholder: replace with actual loader

    if args.mode == 'audio':
        embedder = AudioEmbedder(args.model_name, args.checkpoint_path, args.device)
        embedding_fn = embedder.embedding_fn
    elif args.mode == 'video':
        embedder = VideoEmbedder(args.model_name, args.checkpoint_path, args.device)
        embedding_fn = embedder.embedding_fn
    elif args.mode == 'forensic':
        embedder = ForensicEmbedder(args.model_name, args.checkpoint_path, args.device)
        embedding_fn = embedder.embedding_fn
    else:
        raise ValueError("Invalid mode. Must be one of: audio, video, forensic.")

    # The following assumes loader is a DataFrame and uses embed_dataset
    # Replace with actual loader logic as needed
    import pandas as pd  # type: ignore
    df = pd.DataFrame(loader)  # Placeholder: replace with actual DataFrame
    results_df = embedder.embed_dataset(df, embedding_fn, mode=args.mode)
    embeddings = results_df['embedding'].to_list()
    metadata = results_df['label'].to_list()

    import numpy as np  # type: ignore
    embeddings = np.stack(embeddings)

    # Save embeddings and metadata
    np.save(f"{args.out_dir}/embeddings.npy", embeddings)
    import pickle
    with open(f"{args.out_dir}/meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

    if args.faiss:
        indexer = FaissIndexer(dim=embeddings.shape[1])
        indexer.build_index(embeddings)
        indexer.save_index(os.path.join(args.out_dir, 'index.faiss'))
        print(f"FAISS index saved to {os.path.join(args.out_dir, 'index.faiss')}")

    print(f"Embeddings and metadata saved to {args.out_dir}")

if __name__ == "__main__":
    main()
