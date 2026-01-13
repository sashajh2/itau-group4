#!/usr/bin/env python3
"""
Simple MLP classifier for deepfake detection using pre-computed embeddings.

Trains an MLP with logistic output on HuBERT embeddings from the AVDeepfake1M dataset.
Uses 70-30 train/val split and visualizes results.
"""

import os
import sys
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class MLPClassifier(nn.Module):
    """Simple MLP with logistic output for binary classification."""

    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Final logistic layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_features(self, x):
        """Get penultimate layer features for visualization."""
        for layer in list(self.network.children())[:-1]:
            x = layer(x)
        return x


def load_embeddings_from_hdf5(hdf5_path, embedding_type='hubert'):
    """
    Load embeddings and labels from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        embedding_type: 'hubert', 'openl3', or 'senet'

    Returns:
        embeddings: numpy array of shape (n_samples, embed_dim)
        labels: numpy array of shape (n_samples,) with 0=real, 1=fake
    """
    print(f"Loading {embedding_type} embeddings from {hdf5_path}...")

    all_embeddings = []
    all_labels = []

    with h5py.File(hdf5_path, 'r') as f:
        videos = f['videos']

        for vid_id in videos.keys():
            vid = videos[vid_id]

            # Get embeddings for this video
            embeddings_group = vid['embeddings']
            if embedding_type not in embeddings_group:
                continue

            embeddings = embeddings_group[embedding_type][:]  # shape: (n_augs, n_segments, embed_dim)

            # Get augmentation types to determine labels
            aug_info = vid['augmentation_info']
            aug_types = aug_info['types'][:]

            # Flatten and create labels
            n_augs, n_segments, embed_dim = embeddings.shape

            for aug_idx, aug_type in enumerate(aug_types):
                aug_type_str = aug_type.decode() if isinstance(aug_type, bytes) else aug_type

                # Label: 0 for real/source, 1 for fake
                label = 0 if aug_type_str in ['real', 'source'] else 1

                # Add all segments for this augmentation
                aug_embeddings = embeddings[aug_idx]  # (n_segments, embed_dim)
                all_embeddings.append(aug_embeddings)
                all_labels.extend([label] * n_segments)

    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)

    print(f"Loaded {len(embeddings):,} samples")
    print(f"  Real: {np.sum(labels == 0):,}")
    print(f"  Fake: {np.sum(labels == 1):,}")
    print(f"  Embedding dim: {embeddings.shape[1]}")

    return embeddings, labels


def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=1e-3):
    """Train the MLP classifier."""

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    best_val_auc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * len(X_batch)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        val_auc = roc_auc_score(all_labels, all_probs)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()

        print(f"Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val AUC: {val_auc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)

    return history


def plot_results(history, probs, labels, features, output_dir):
    """Create visualization plots."""

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Training curves
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss', color='blue')
    ax.plot(history['val_loss'], label='Val Loss', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. ROC Curve
    ax = axes[0, 1]
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Confusion Matrix
    ax = axes[1, 0]
    preds = (probs > 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Real', 'Fake'])
    ax.set_yticklabels(['Real', 'Fake'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                   color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)

    # 4. PCA Projection
    ax = axes[1, 1]
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # Subsample for visualization if too many points
    n_plot = min(5000, len(features_2d))
    idx = np.random.choice(len(features_2d), n_plot, replace=False)

    scatter = ax.scatter(features_2d[idx, 0], features_2d[idx, 1],
                        c=labels[idx], cmap='coolwarm', alpha=0.5, s=10)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA Projection of Learned Features')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Real', 'Fake'])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mlp_classifier_results.png'), dpi=150)
    plt.show()
    print(f"Plot saved to {output_dir}/mlp_classifier_results.png")

    # Print final metrics
    print("\n=== Final Metrics ===")
    print(f"Accuracy:  {accuracy_score(labels, preds):.4f}")
    print(f"Precision: {precision_score(labels, preds):.4f}")
    print(f"Recall:    {recall_score(labels, preds):.4f}")
    print(f"F1 Score:  {f1_score(labels, preds):.4f}")
    print(f"AUC-ROC:   {auc:.4f}")


def main():
    # Configuration
    hdf5_path = 'exports/deepfake_embeddings_2.h5'
    embedding_type = 'hubert'  # 'hubert', 'openl3', or 'senet'
    output_dir = 'results/mlp_classifier'

    hidden_dims = [256, 128]
    batch_size = 256
    num_epochs = 20
    lr = 1e-3
    test_size = 0.3
    random_state = 42

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Load data
    embeddings, labels = load_embeddings_from_hdf5(hdf5_path, embedding_type)

    # Train/val split (70-30)
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    print(f"\nTrain set: {len(X_train):,} samples")
    print(f"Val set:   {len(X_val):,} samples")

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = embeddings.shape[1]
    model = MLPClassifier(input_dim, hidden_dims=hidden_dims).to(device)
    print(f"\nModel architecture:")
    print(model)

    # Train
    print("\n=== Training ===")
    history = train_model(model, train_loader, val_loader, device, num_epochs, lr)

    # Evaluate and visualize
    print("\n=== Evaluation ===")
    probs, labels_eval, features = evaluate_model(model, val_loader, device)
    plot_results(history, probs, labels_eval, features, output_dir)

    # Save model
    model_path = os.path.join(output_dir, 'mlp_classifier.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'history': history,
    }, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    main()
