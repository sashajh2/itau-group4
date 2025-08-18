import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Compute Pearson correlation matrix between model probabilities in a CSV file and plot a heatmap.")
    parser.add_argument('--input', type=str, required=True, help='Path to CSV file (e.g., audio_linear_probe_probs.csv)')
    parser.add_argument('--output', type=str, default=None, help='Optional: path to save correlation matrix as CSV')
    parser.add_argument('--output_heatmap', type=str, default=None, help='Optional: path to save heatmap as PNG')
    args = parser.parse_args()

    # Load DataFrame
    df = pd.read_csv(args.input)

    # Select only probability columns (exclude label)
    prob_cols = [col for col in df.columns if col.endswith('_prob')]
    if not prob_cols:
        raise ValueError("No columns ending with '_prob' found in the input CSV.")
    if len(prob_cols) < 2:
        raise ValueError("Need at least two probability columns to compute a correlation matrix.")
    prob_df = pd.DataFrame(df[prob_cols])
    corr = prob_df.corr(method='pearson')

    print("Pearson correlation matrix:")
    print(corr)

    if args.output:
        corr.to_csv(args.output)
        print(f"Saved correlation matrix to {args.output}")

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Pearson Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.show()

    if args.output_heatmap:
        plt.savefig(args.output_heatmap)
        print(f"Saved heatmap to {args.output_heatmap}")

if __name__ == "__main__":
    main()
 