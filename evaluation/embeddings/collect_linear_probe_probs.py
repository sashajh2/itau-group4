import argparse
import numpy as np
import pandas as pd
import pickle
import os
from embedding_utils import get_oof_linear_probe_probs

def main():
    parser = argparse.ArgumentParser(description="Collect out-of-fold linear probe probabilities for multiple embeddings.")
    parser.add_argument('--embeddings', type=str, nargs='+', required=True, help='List of embedding .npy files (same order as names)')
    parser.add_argument('--names', type=str, nargs='+', required=True, help='List of names for each embedding (e.g., hubert mfcc wav2vec2)')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels.pkl')
    parser.add_argument('--output', type=str, required=True, help='Output path for DataFrame (without extension)')
    args = parser.parse_args()

    if len(args.embeddings) != len(args.names):
        raise ValueError('Number of embedding files must match number of names')

    # Load labels
    with open(args.labels, 'rb') as f:
        labels = np.array(pickle.load(f))

    # Collect probabilities for each embedding
    prob_dict = {}
    for emb_file, name in zip(args.embeddings, args.names):
        print(f"Processing {name} from {emb_file}")
        emb = np.load(emb_file)
        probs = get_oof_linear_probe_probs(emb, labels)
        prob_dict[f'{name}_prob'] = probs

    # Build DataFrame
    df = pd.DataFrame(prob_dict)
    df['label'] = labels

    # Save
    df.to_pickle(f'{args.output}.pkl')
    df.to_csv(f'{args.output}.csv', index=False)
    print(f"Saved DataFrame with columns: {df.columns.tolist()} to {args.output}.pkl and .csv")

if __name__ == "__main__":
    main() 