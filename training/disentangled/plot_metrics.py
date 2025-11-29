"""
Plot metrics over training.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def plot_metrics_history(metrics_path='./checkpoints/disentangled/metrics_history.json', 
                        output_path='./checkpoints/disentangled/metrics_plot.png'):
    """
    Plot AMI, ARI, and Silhouette over training epochs.
    
    Args:
        metrics_path: Path to metrics_history.json
        output_path: Path to save the plot
    """
    if not os.path.exists(metrics_path):
        print(f"❌ Metrics file not found: {metrics_path}")
        return
    
    with open(metrics_path, 'r') as f:
        history = json.load(f)
    
    before = history['before_training']
    epochs_data = history['epochs']
    
    if len(epochs_data) == 0:
        print("❌ No epoch data found in metrics history")
        return
    
    # Extract metrics per epoch
    epochs = [e['epoch'] for e in epochs_data]
    ami = [e['metrics']['ami'] for e in epochs_data]
    ari = [e['metrics']['ari'] for e in epochs_data]
    sil_gt = [e['metrics']['silhouette_gt'] for e in epochs_data]
    sil_km = [e['metrics']['silhouette_clusters'] for e in epochs_data]
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Embedding Quality Metrics Over Training', fontsize=16, fontweight='bold')
    
    # AMI
    ax = axes[0, 0]
    ax.axhline(before['ami'], color='red', linestyle='--', alpha=0.5, label='Before Training')
    ax.plot(epochs, ami, marker='o', linewidth=2, markersize=4, label='After Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AMI')
    ax.set_title('Adjusted Mutual Information')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # ARI
    ax = axes[0, 1]
    ax.axhline(before['ari'], color='red', linestyle='--', alpha=0.5, label='Before Training')
    ax.plot(epochs, ari, marker='o', linewidth=2, markersize=4, label='After Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ARI')
    ax.set_title('Adjusted Rand Index')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Silhouette (Ground Truth)
    ax = axes[1, 0]
    ax.axhline(before['silhouette_gt'], color='red', linestyle='--', alpha=0.5, label='Before Training')
    ax.plot(epochs, sil_gt, marker='o', linewidth=2, markersize=4, label='After Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Coefficient (Ground Truth Labels)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1, 1])
    
    # Silhouette (K-means)
    ax = axes[1, 1]
    ax.axhline(before['silhouette_clusters'], color='red', linestyle='--', alpha=0.5, label='Before Training')
    ax.plot(epochs, sil_km, marker='o', linewidth=2, markersize=4, label='After Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Coefficient (K-means Clusters)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1, 1])
    
    plt.tight_layout()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved metrics plot to {output_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument('--metrics-path', type=str, 
                       default='./checkpoints/disentangled/metrics_history.json',
                       help='Path to metrics_history.json')
    parser.add_argument('--output-path', type=str,
                       default='./checkpoints/disentangled/metrics_plot.png',
                       help='Path to save the plot')
    
    args = parser.parse_args()
    plot_metrics_history(args.metrics_path, args.output_path)

