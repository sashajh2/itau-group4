"""
Loss functions for disentangled representation learning.

Implements:
- Variance minimization loss (Equation 3.6)
- Prototypical contrastive loss (Equation 3.5)
- Orthogonality constraint loss (Equation 3.3)
- Adaptive loss scaler for balanced training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def variance_loss(z_auth: torch.Tensor, is_real: torch.Tensor, 
                 min_variance: float = 0.01, regularization_weight: float = 0.1) -> torch.Tensor:
    """
    Minimize variance of real samples around their centroid (Equation 3.6).
    Includes regularization to prevent collapse.
    
    Args:
        z_auth: Authenticity embeddings, shape [batch_size, emb_dim]
        is_real: Boolean mask, shape [batch_size]
        min_variance: Minimum variance threshold to prevent collapse
        regularization_weight: Weight for regularization term that encourages minimum spread
    
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
    variance = ((z_auth_real - mu_real) ** 2).sum(dim=1).mean()
    
    # Regularization: encourage variance to stay above minimum threshold
    # Use a combination of quadratic and linear penalties for better gradient signal
    # L_reg_quad = max(0, min_variance - variance)^2  (smooth, but weak for small values)
    # L_reg_linear = max(0, min_variance - variance)  (stronger gradient when variance is low)
    regularization_quad = torch.clamp(min_variance - variance, min=0.0) ** 2
    regularization_linear = torch.clamp(min_variance - variance, min=0.0)
    
    # Combine: quadratic for smoothness, linear for strength when variance is very low
    regularization = regularization_quad + 5.0 * regularization_linear
    
    loss = variance + regularization_weight * regularization
    
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


class AdaptiveLossScaler:
    """
    Adaptive loss scaler using exponential moving average.
    Normalizes losses to similar magnitudes before weighting.
    """
    def __init__(self, alpha: float = 0.99, initial_scale: float = 1.0, warmup_steps: int = 100):
        """
        Args:
            alpha: Smoothing factor for exponential moving average (0.99 = very smooth)
            initial_scale: Initial scaling factor for each loss
            warmup_steps: Number of steps to use smaller alpha for faster adaptation
        """
        self.alpha = alpha
        self.alpha_warmup = 0.9  # Faster adaptation during warmup
        self.warmup_steps = warmup_steps
        self.running_means = {
            'proto': initial_scale,
            'var': initial_scale,
            'orth': initial_scale,
        }
        self.step_count = 0
    
    def scale_losses(self, L_proto: torch.Tensor, L_var: torch.Tensor, 
                     L_orth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Scale losses based on running averages.
        
        Returns:
            Scaled losses (L_proto_scaled, L_var_scaled, L_orth_scaled)
        """
        self.step_count += 1
        
        # Use faster adaptation during warmup
        current_alpha = self.alpha_warmup if self.step_count < self.warmup_steps else self.alpha
        
        # Update running averages (exponential moving average)
        # Use max(actual_loss, small_value) to prevent division by tiny numbers
        proto_val = max(L_proto.item(), 1e-6)
        var_val = max(L_var.item(), 1e-6)  # Prevent tiny variance from causing huge scales
        orth_val = max(L_orth.item(), 1e-6)
        
        self.running_means['proto'] = (
            current_alpha * self.running_means['proto'] + 
            (1 - current_alpha) * proto_val
        )
        self.running_means['var'] = (
            current_alpha * self.running_means['var'] + 
            (1 - current_alpha) * var_val
        )
        self.running_means['orth'] = (
            current_alpha * self.running_means['orth'] + 
            (1 - current_alpha) * orth_val
        )
        
        # Normalize by running averages (so each loss has similar magnitude)
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        L_proto_scaled = L_proto / (self.running_means['proto'] + eps)
        L_var_scaled = L_var / (self.running_means['var'] + eps)
        L_orth_scaled = L_orth / (self.running_means['orth'] + eps)
        
        return L_proto_scaled, L_var_scaled, L_orth_scaled
    
    def get_scales(self) -> Dict[str, float]:
        """Get current scaling factors (inverse of running means)."""
        eps = 1e-8
        return {
            'proto': 1.0 / (self.running_means['proto'] + eps),
            'var': 1.0 / (self.running_means['var'] + eps),
            'orth': 1.0 / (self.running_means['orth'] + eps),
        }


def compute_total_loss(
    z_id: torch.Tensor,
    z_auth: torch.Tensor,
    is_real: torch.Tensor,
    content_groups: torch.Tensor,
    lambda_var: float = 0.5,
    lambda_orth: float = 0.1,
    temperature: float = 0.1,
    min_variance: float = 0.01,
    variance_reg_weight: float = 0.1,
    adaptive_scaler: Optional[AdaptiveLossScaler] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combine all three loss components (Equation 3.7).
    Supports adaptive scaling for balanced training.
    
    Args:
        z_id: Identity embeddings
        z_auth: Authenticity embeddings
        is_real: Boolean mask for real samples
        content_groups: Content group IDs
        lambda_var: Weight for variance loss (default: 0.5)
        lambda_orth: Weight for orthogonality loss (default: 0.1)
        temperature: Temperature for prototypical loss (default: 0.1)
        min_variance: Minimum variance threshold for regularization
        variance_reg_weight: Weight for variance regularization term
        adaptive_scaler: Optional adaptive loss scaler
    
    Returns:
        total_loss: Combined loss
        losses_dict: Dictionary with individual loss values and scales
    """
    L_proto = prototypical_contrastive_loss(z_id, content_groups, temperature=temperature)
    L_var = variance_loss(z_auth, is_real, min_variance=min_variance, 
                         regularization_weight=variance_reg_weight)
    L_orth = orthogonality_loss(z_id, z_auth)
    
    # Apply adaptive scaling if provided
    if adaptive_scaler is not None:
        L_proto_scaled, L_var_scaled, L_orth_scaled = adaptive_scaler.scale_losses(
            L_proto, L_var, L_orth
        )
        scales = adaptive_scaler.get_scales()
        
        # L_total = L_proto_scaled + λ_var * L_var_scaled + λ_orth * L_orth_scaled
        total_loss = L_proto_scaled + lambda_var * L_var_scaled + lambda_orth * L_orth_scaled
        
        losses_dict = {
            'total': total_loss.item(),
            'proto': L_proto.item(),
            'var': L_var.item(),
            'orth': L_orth.item(),
            'proto_scaled': L_proto_scaled.item(),
            'var_scaled': L_var_scaled.item(),
            'orth_scaled': L_orth_scaled.item(),
            'scale_proto': scales['proto'],
            'scale_var': scales['var'],
            'scale_orth': scales['orth'],
        }
    else:
        # Standard loss combination without scaling
        total_loss = L_proto + lambda_var * L_var + lambda_orth * L_orth
        
        losses_dict = {
            'total': total_loss.item(),
            'proto': L_proto.item(),
            'var': L_var.item(),
            'orth': L_orth.item(),
        }
    
    return total_loss, losses_dict

