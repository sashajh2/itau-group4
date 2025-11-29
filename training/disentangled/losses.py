"""
Loss functions for disentangled representation learning.

Implements:
- Variance minimization loss (Equation 3.6)
- Prototypical contrastive loss (Equation 3.5)
- Orthogonality constraint loss (Equation 3.3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def variance_loss(z_auth: torch.Tensor, is_real: torch.Tensor) -> torch.Tensor:
    """
    Minimize variance of real samples around their centroid (Equation 3.6).
    
    Args:
        z_auth: Authenticity embeddings, shape [batch_size, emb_dim]
        is_real: Boolean mask, shape [batch_size]
    
    Returns:
        scalar loss
    """
    z_auth_real = z_auth[is_real]
    
    if z_auth_real.shape[0] < 2:
        return torch.tensor(0.0, device=z_auth.device, requires_grad=True)
    
    # Compute real centroid (mean of all real samples in batch)
    mu_real = z_auth_real.mean(dim=0, keepdim=True)  # [1, emb_dim]
    
    # Minimize squared distances to centroid
    # L_var = (1/N_real) * sum(||z_i^auth - mu_real||^2)
    loss = ((z_auth_real - mu_real) ** 2).sum(dim=1).mean()
    
    return loss


def prototypical_contrastive_loss(
    z_id: torch.Tensor, 
    content_groups: torch.Tensor, 
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Cluster samples by content group using prototypical contrastive learning (Equation 3.5).
    
    Args:
        z_id: Identity embeddings, shape [batch_size, emb_dim]
        content_groups: Content group IDs, shape [batch_size] (integer IDs)
        temperature: Temperature hyperparameter (default: 0.1)
    
    Returns:
        scalar loss
    """
    device = z_id.device
    unique_groups = torch.unique(content_groups)
    
    if len(unique_groups) < 2:
        # Need at least 2 groups for contrastive learning
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute prototypes for each content group
    # c_k = (1/|G_k|) * sum_{i in G_k} z_i^id
    prototypes = []
    prototype_labels = []
    
    for group_id in unique_groups:
        mask = content_groups == group_id
        group_embeddings = z_id[mask]
        prototype = group_embeddings.mean(dim=0)  # [emb_dim]
        prototypes.append(prototype)
        prototype_labels.append(group_id.item())
    
    prototypes = torch.stack(prototypes)  # [num_groups, emb_dim]
    
    # Compute distances from each sample to all prototypes
    # Using negative Euclidean distance (higher = more similar)
    # d(a, b) = ||a - b||_2
    distances = -torch.cdist(z_id, prototypes, p=2)  # [batch_size, num_groups]
    
    # Create target indices (which prototype each sample belongs to)
    targets = torch.zeros(z_id.shape[0], dtype=torch.long, device=device)
    for i, group_id in enumerate(prototype_labels):
        targets[content_groups == group_id] = i
    
    # Apply temperature scaling and compute cross-entropy
    # L_proto = (-1/N) * sum_i log[exp(-d(z_i^id, c_k)/τ) / sum_j exp(-d(z_i^id, c_j)/τ)]
    logits = distances / temperature
    loss = F.cross_entropy(logits, targets)
    
    return loss


def orthogonality_loss(z_id: torch.Tensor, z_auth: torch.Tensor) -> torch.Tensor:
    """
    Penalize correlation between identity and authenticity embeddings (Equation 3.3).
    
    Args:
        z_id: Identity embeddings, shape [batch_size, emb_dim]
        z_auth: Authenticity embeddings, shape [batch_size, emb_dim]
    
    Returns:
        scalar loss
    """
    batch_size = z_id.shape[0]
    
    # Compute pairwise cosine similarities
    # sim_matrix[i,j] = cos_sim(z_id[i], z_auth[j])
    # Since embeddings are normalized, this is just dot product
    sim_matrix = torch.matmul(z_id, z_auth.t())  # [batch_size, batch_size]
    
    # Average absolute similarity
    # L_orth = (1/N^2) * sum_{i,j} |sim(z_i^id, z_j^auth)|
    loss = sim_matrix.abs().sum() / (batch_size ** 2)
    
    return loss


def compute_total_loss(
    z_id: torch.Tensor,
    z_auth: torch.Tensor,
    is_real: torch.Tensor,
    content_groups: torch.Tensor,
    lambda_var: float = 0.5,
    lambda_orth: float = 0.1,
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combine all three loss components (Equation 3.7).
    
    Args:
        z_id: Identity embeddings
        z_auth: Authenticity embeddings
        is_real: Boolean mask for real samples
        content_groups: Content group IDs
        lambda_var: Weight for variance loss (default: 0.5)
        lambda_orth: Weight for orthogonality loss (default: 0.1)
        temperature: Temperature for prototypical loss (default: 0.1)
    
    Returns:
        total_loss: Combined loss
        losses_dict: Dictionary with individual loss values
    """
    L_proto = prototypical_contrastive_loss(z_id, content_groups, temperature=temperature)
    L_var = variance_loss(z_auth, is_real)
    L_orth = orthogonality_loss(z_id, z_auth)
    
    # L_total = L_proto + λ_var * L_var + λ_orth * L_orth
    total_loss = L_proto + lambda_var * L_var + lambda_orth * L_orth
    
    losses_dict = {
        'total': total_loss.item(),
        'proto': L_proto.item(),
        'var': L_var.item(),
        'orth': L_orth.item(),
    }
    
    return total_loss, losses_dict

