"""
Concatenated Embeddings MLP Classifier Experiment

This experiment:
1. Loads segment embeddings from HDF5 (hubert + openl3 + senet)
2. Concatenates all three embedding types per segment (768 + 512 + 2048 = 3328 dim)
3. Trains a small MLP with BCE loss for real/fake classification
4. Uses 70-30 train/val split with class balance
5. Visualizes results and penultimate layer projections

Handles class imbalance with two approaches:
- BALANCE_METHOD = "class_weights": Use weighted BCE loss
- BALANCE_METHOD = "oversampling": Use WeightedRandomSampler

Run in Colab with GPU for faster training.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# Configuration
# ==============================================================================
HDF5_PATH = "exports/deepfake_embeddings_2.h5"  # Adjust path as needed or Colab
RESULTS_DIR = "results/classifiers/concat_mlp"
BATCH_SIZE = 256
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Cross-validation settings
N_FOLDS = 4  # 4-fold CV with 75-25 splits

# Class imbalance handling: "none", "class_weights", or "oversampling"
BALANCE_METHOD = "oversampling"

# ==============================================================================
# 1. Load and Concatenate Embeddings
# ==============================================================================
def load_concatenated_embeddings(hdf5_path):
    """
    Load and concatenate all embedding types (hubert, openl3, senet) per segment.

    Returns:
        embeddings: np.array of shape [N, 3328] (768 + 512 + 2048)
        labels: np.array of shape [N] (0 = real, 1 = fake)
        metadata: list of dicts with video_id, aug_idx, seg_idx
    """
    all_embeddings = []
    all_labels = []
    metadata = []

    with h5py.File(hdf5_path, 'r') as f:
        video_ids = list(f['videos'].keys())
        print(f"Loading {len(video_ids)} videos...")

        for video_id in tqdm(video_ids, desc="Loading videos"):
            video = f['videos'][video_id]

            # Load all embeddings
            hubert = video['embeddings/hubert'][:]      # [num_augs, num_segs, 768]
            openl3 = video['embeddings/openl3'][:]      # [num_augs, num_segs, 512]
            senet = video['embeddings/senet'][:]        # [num_augs, num_segs, 2048]

            # Load labels (0 = real, 1 = fake)
            audio_labels = video['labels/audio'][:]     # [num_augs, num_segs]
            video_labels = video['labels/video'][:]     # [num_augs, num_segs]

            num_augs, num_segs = audio_labels.shape

            for aug_idx in range(num_augs):
                for seg_idx in range(num_segs):
                    # Concatenate embeddings
                    concat_emb = np.concatenate([
                        hubert[aug_idx, seg_idx],
                        openl3[aug_idx, seg_idx],
                        senet[aug_idx, seg_idx]
                    ])
                    all_embeddings.append(concat_emb)

                    # Label: fake if either audio or video is fake
                    is_fake = (audio_labels[aug_idx, seg_idx] == 1) or (video_labels[aug_idx, seg_idx] == 1)
                    all_labels.append(1 if is_fake else 0)

                    metadata.append({
                        'video_id': video_id,
                        'aug_idx': aug_idx,
                        'seg_idx': seg_idx,
                        'audio_label': int(audio_labels[aug_idx, seg_idx]),
                        'video_label': int(video_labels[aug_idx, seg_idx])
                    })

    embeddings = np.array(all_embeddings, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    print(f"\nLoaded {len(embeddings):,} samples")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print(f"Real samples: {(labels == 0).sum():,} ({100 * (labels == 0).mean():.1f}%)")
    print(f"Fake samples: {(labels == 1).sum():,} ({100 * (labels == 1).mean():.1f}%)")

    return embeddings, labels, metadata


# ==============================================================================
# 2. Dataset Class
# ==============================================================================
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.from_numpy(embeddings)
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ==============================================================================
# 3. MLP Model
# ==============================================================================
class MLPClassifier(nn.Module):
    """
    Small MLP for binary classification.
    Architecture: input -> 512 -> 128 (penultimate) -> 1
    """
    def __init__(self, input_dim=3328, hidden_dim=1024, penultimate_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, penultimate_dim)
        self.bn2 = nn.BatchNorm1d(penultimate_dim)
        self.fc3 = nn.Linear(penultimate_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, return_penultimate=False):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        penultimate = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(penultimate)
        logits = self.fc3(x)

        if return_penultimate:
            return logits, penultimate
        return logits


# ==============================================================================
# 4. Metrics
# ==============================================================================
def compute_metrics(preds, labels):
    """Compute accuracy, precision, recall, F1."""
    preds_binary = (preds > 0.5).float()

    tp = ((preds_binary == 1) & (labels == 1)).sum().float()
    fp = ((preds_binary == 1) & (labels == 0)).sum().float()
    fn = ((preds_binary == 0) & (labels == 1)).sum().float()
    tn = ((preds_binary == 0) & (labels == 0)).sum().float()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return accuracy.item(), precision.item(), recall.item(), f1.item()


# ==============================================================================
# 5. Training
# ==============================================================================
def train_model(model, train_loader, val_loader, device, num_epochs=30, lr=1e-3, weight_decay=1e-4, pos_weight=None):
    """Train the model and return history.

    Args:
        pos_weight: Weight for positive class in BCE loss (for class_weights method)
    """
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
        print(f"Using class-weighted BCE with pos_weight={pos_weight:.2f}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': []
    }

    best_val_f1 = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x).squeeze()
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(torch.sigmoid(logits).detach().cpu())
            train_labels.append(batch_y.cpu())

        scheduler.step()

        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)
        train_acc, train_prec, train_rec, train_f1 = compute_metrics(train_preds, train_labels)

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels_list = [], []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x).squeeze()
                loss = criterion(logits, batch_y)

                val_loss += loss.item()
                val_preds.append(torch.sigmoid(logits).cpu())
                val_labels_list.append(batch_y.cpu())

        val_preds = torch.cat(val_preds)
        val_labels_tensor = torch.cat(val_labels_list)
        val_acc, val_prec, val_rec, val_f1 = compute_metrics(val_preds, val_labels_tensor)

        # Record history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_precision'].append(train_prec)
        history['val_precision'].append(val_prec)
        history['train_recall'].append(train_rec)
        history['val_recall'].append(val_rec)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {history['train_loss'][-1]:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
                  f"Val Loss: {history['val_loss'][-1]:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    print(f"\nBest Val F1: {best_val_f1:.4f}")
    return history, best_model_state, best_val_f1


# ==============================================================================
# 6. Plotting Functions
# ==============================================================================
def plot_training_curves(history, save_path=None):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('BCE Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.5, 1.0])

    # Precision/Recall
    axes[1, 0].plot(epochs, history['train_precision'], 'b--', label='Train Prec', linewidth=2)
    axes[1, 0].plot(epochs, history['val_precision'], 'r--', label='Val Prec', linewidth=2)
    axes[1, 0].plot(epochs, history['train_recall'], 'b:', label='Train Rec', linewidth=2)
    axes[1, 0].plot(epochs, history['val_recall'], 'r:', label='Val Rec', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # F1 Score
    axes[1, 1].plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(epochs, history['val_f1'], 'r-', label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0.5, 1.0])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_penultimate_projections(embeddings, labels, save_path=None, max_samples=10000):
    """Plot PCA of penultimate layer embeddings."""
    # Subsample if needed
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        emb_subset = embeddings[indices]
        labels_subset = labels[indices]
    else:
        emb_subset = embeddings
        labels_subset = labels

    print(f"Using {len(emb_subset)} samples for visualization")

    # PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(emb_subset)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = ['#2ecc71' if l == 0 else '#e74c3c' for l in labels_subset]

    # PCA plot
    ax.scatter(emb_pca[:, 0], emb_pca[:, 1], c=colors, alpha=0.5, s=10)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'PCA of Penultimate Layer\n(Explained var: {pca.explained_variance_ratio_.sum():.1%})')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Real'),
        Patch(facecolor='#e74c3c', label='Fake')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

    return emb_pca


def plot_evaluation(labels, preds, save_path=None):
    """Plot confusion matrix and ROC curve."""
    preds_binary = (preds > 0.5).astype(int)
    cm = confusion_matrix(labels, preds_binary)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion matrix
    im = axes[0].imshow(cm, cmap='Blues')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['Real', 'Fake'])
    axes[0].set_yticklabels(['Real', 'Fake'])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix')

    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f'{cm[i, j]:,}\n({100*cm[i,j]/cm.sum():.1f}%)',
                        ha='center', va='center', fontsize=12,
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.colorbar(im, ax=axes[0])

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

    print("\nClassification Report:")
    print(classification_report(labels, preds_binary, target_names=['Real', 'Fake']))

    return roc_auc


# ==============================================================================
# 7. Main with K-Fold Cross Validation
# ==============================================================================
def main():
    # Create results directory (include balance method in path)
    results_dir = f"{RESULTS_DIR}_{BALANCE_METHOD}_{N_FOLDS}fold"
    os.makedirs(results_dir, exist_ok=True)

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    embeddings, labels, metadata = load_concatenated_embeddings(HDF5_PATH)

    # Setup K-Fold cross validation
    print("\n" + "="*60)
    print(f"Setting up {N_FOLDS}-Fold Cross Validation (75-25 splits)")
    print("="*60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Store results from each fold
    fold_results = []
    all_fold_histories = []
    best_overall_f1 = 0
    best_overall_model_state = None
    best_fold = -1

    # Collect all predictions for final evaluation
    all_val_labels = []
    all_val_preds = []
    all_val_penultimate = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, labels)):
        print("\n" + "="*60)
        print(f"FOLD {fold + 1}/{N_FOLDS}")
        print("="*60)

        # Split data
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        print(f"Train set: {len(X_train):,} samples")
        print(f"  Real: {(y_train == 0).sum():,} ({100 * (y_train == 0).mean():.1f}%)")
        print(f"  Fake: {(y_train == 1).sum():,} ({100 * (y_train == 1).mean():.1f}%)")
        print(f"Val set: {len(X_val):,} samples")
        print(f"  Real: {(y_val == 0).sum():,} ({100 * (y_val == 0).mean():.1f}%)")
        print(f"  Fake: {(y_val == 1).sum():,} ({100 * (y_val == 1).mean():.1f}%)")

        # Calculate class weights for imbalance handling
        num_real = (y_train == 0).sum()
        num_fake = (y_train == 1).sum()
        pos_weight = num_real / num_fake

        print(f"Class imbalance ratio: {num_real/num_fake:.1f}:1 (real:fake)")
        print(f"Balance method: {BALANCE_METHOD}")

        # Create datasets
        train_dataset = EmbeddingDataset(X_train, y_train)
        val_dataset = EmbeddingDataset(X_val, y_val)

        # Create dataloaders based on balance method
        if BALANCE_METHOD == "oversampling":
            sample_weights = np.where(y_train == 1, pos_weight, 1.0)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Create fresh model for each fold
        model = MLPClassifier(input_dim=3328).to(device)

        # Train
        train_pos_weight = pos_weight if BALANCE_METHOD == "class_weights" else None
        history, best_model_state, best_val_f1 = train_model(
            model, train_loader, val_loader, device,
            num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            pos_weight=train_pos_weight
        )

        all_fold_histories.append(history)

        # Track best model across all folds
        if best_val_f1 > best_overall_f1:
            best_overall_f1 = best_val_f1
            best_overall_model_state = best_model_state
            best_fold = fold + 1

        # Extract predictions and embeddings from best model
        model.load_state_dict(best_model_state)
        model.eval()

        fold_penultimate = []
        fold_labels = []
        fold_preds = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits, penultimate = model(batch_x, return_penultimate=True)

                fold_penultimate.append(penultimate.cpu().numpy())
                fold_labels.append(batch_y.numpy())
                fold_preds.append(torch.sigmoid(logits).squeeze().cpu().numpy())

        fold_penultimate = np.concatenate(fold_penultimate)
        fold_labels = np.concatenate(fold_labels)
        fold_preds = np.concatenate(fold_preds)

        # Compute fold metrics
        fold_preds_binary = (fold_preds > 0.5).astype(int)
        fold_tp = ((fold_preds_binary == 1) & (fold_labels == 1)).sum()
        fold_fp = ((fold_preds_binary == 1) & (fold_labels == 0)).sum()
        fold_fn = ((fold_preds_binary == 0) & (fold_labels == 1)).sum()
        fold_tn = ((fold_preds_binary == 0) & (fold_labels == 0)).sum()

        fold_acc = (fold_tp + fold_tn) / (fold_tp + fold_tn + fold_fp + fold_fn)
        fold_precision = fold_tp / (fold_tp + fold_fp + 1e-8)
        fold_recall = fold_tp / (fold_tp + fold_fn + 1e-8)
        fold_f1 = 2 * fold_precision * fold_recall / (fold_precision + fold_recall + 1e-8)

        fpr, tpr, _ = roc_curve(fold_labels, fold_preds)
        fold_auc = auc(fpr, tpr)

        fold_results.append({
            'fold': fold + 1,
            'best_val_f1': best_val_f1,
            'accuracy': fold_acc,
            'precision': fold_precision,
            'recall': fold_recall,
            'f1': fold_f1,
            'auc': fold_auc
        })

        print(f"\nFold {fold + 1} Results: Acc={fold_acc:.4f}, F1={fold_f1:.4f}, AUC={fold_auc:.4f}")

        # Collect for aggregate evaluation
        all_val_labels.append(fold_labels)
        all_val_preds.append(fold_preds)
        all_val_penultimate.append(fold_penultimate)

        # Save fold-specific training curves
        plot_training_curves(history, save_path=f"{results_dir}/training_curves_fold{fold+1}.png")

    # Aggregate results across folds
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)

    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    cv_summary = {}

    for metric in metrics_names:
        values = [r[metric] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv_summary[f'{metric}_mean'] = mean_val
        cv_summary[f'{metric}_std'] = std_val
        print(f"{metric.capitalize():12s}: {mean_val:.4f} +/- {std_val:.4f}")

    print(f"\nBest fold: {best_fold} with F1={best_overall_f1:.4f}")

    # Concatenate all validation predictions for overall evaluation
    all_val_labels = np.concatenate(all_val_labels)
    all_val_preds = np.concatenate(all_val_preds)
    all_val_penultimate = np.concatenate(all_val_penultimate)

    # Plot aggregate evaluation
    print("\n" + "="*60)
    print("Plotting Aggregate Results")
    print("="*60)

    roc_auc = plot_evaluation(
        all_val_labels, all_val_preds,
        save_path=f"{results_dir}/evaluation_aggregate.png"
    )

    # Plot penultimate projections (subsample from all folds)
    plot_penultimate_projections(
        all_val_penultimate, all_val_labels,
        save_path=f"{results_dir}/penultimate_projections.png"
    )

    # Save best model and metrics
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)

    torch.save({
        'model_state_dict': best_overall_model_state,
        'best_fold': best_fold,
        'best_val_f1': best_overall_f1,
        'fold_results': fold_results,
        'config': {
            'input_dim': 3328,
            'hidden_dim': 1024,
            'penultimate_dim': 512,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'num_epochs': NUM_EPOCHS,
            'n_folds': N_FOLDS,
            'balance_method': BALANCE_METHOD
        }
    }, f"{results_dir}/best_model.pt")

    # Save detailed metrics
    metrics_summary = {
        'n_folds': N_FOLDS,
        'balance_method': BALANCE_METHOD,
        'best_fold': best_fold,
        'best_fold_f1': best_overall_f1,
        'aggregate_auc': roc_auc,
        **cv_summary,
        'fold_results': fold_results
    }

    with open(f"{results_dir}/metrics.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nResults saved to {results_dir}/")
    print("\nFinal Cross-Validation Summary:")
    print(f"  F1 Score:  {cv_summary['f1_mean']:.4f} +/- {cv_summary['f1_std']:.4f}")
    print(f"  Accuracy:  {cv_summary['accuracy_mean']:.4f} +/- {cv_summary['accuracy_std']:.4f}")
    print(f"  Precision: {cv_summary['precision_mean']:.4f} +/- {cv_summary['precision_std']:.4f}")
    print(f"  Recall:    {cv_summary['recall_mean']:.4f} +/- {cv_summary['recall_std']:.4f}")
    print(f"  AUC:       {cv_summary['auc_mean']:.4f} +/- {cv_summary['auc_std']:.4f}")


if __name__ == "__main__":
    main()
