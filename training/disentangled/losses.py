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
from typing import Dict, Tuple, Optional, List


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
    regularization = regularization_quad + 5.0 * regularization_linear # Where did this 5.0 come from?
    
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


def orthogonality_loss(z_id: torch.Tensor, z_auth: torch.Tensor, min_orth: float = 0.001) -> torch.Tensor:
    """
    Penalize correlation between identity and authenticity embeddings (Equation 3.3).
    Includes minimum threshold to prevent collapse.
    
    Args:
        z_id: Identity embeddings, shape [batch_size, emb_dim]
        z_auth: Authenticity embeddings, shape [batch_size, emb_dim]
        min_orth: Minimum orthogonality loss value to prevent collapse
    
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
    
    # Prevent collapse: ensure minimum orthogonality loss
    # This ensures the model maintains some separation between z_id and z_auth
    loss = torch.clamp(loss, min=min_orth)
    
    return loss


class AdaptiveLossScaler:
    """
    Adaptive loss scaler using exponential moving average.
    Normalizes losses to similar magnitudes before weighting.
    Prevents collapse by maintaining minimum loss contributions.
    """
    def __init__(self, alpha: float = 0.99, initial_scale: float = 1.0, warmup_steps: int = 100,
                 min_loss_ratio: float = 0.1):
        """
        Args:
            alpha: Smoothing factor for exponential moving average (0.99 = very smooth)
            initial_scale: Initial scaling factor for each loss
            warmup_steps: Number of steps to use smaller alpha for faster adaptation
            min_loss_ratio: Minimum ratio of each loss to max loss (prevents collapse)
        """
        self.alpha = alpha
        self.alpha_warmup = 0.9  # Faster adaptation during warmup
        self.warmup_steps = warmup_steps
        self.min_loss_ratio = min_loss_ratio
        self.running_means = {
            'proto': initial_scale,
            'var': initial_scale,
            'orth': initial_scale,
        }
        self.initial_losses = {}  # Track initial loss values for normalization
        self.step_count = 0
    
    def scale_losses(self, L_proto: torch.Tensor, L_var: torch.Tensor, 
                     L_orth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Scale losses based on running averages with collapse prevention.
        
        Returns:
            Scaled losses (L_proto_scaled, L_var_scaled, L_orth_scaled)
        """
        self.step_count += 1
        
        # Store initial losses for normalization
        if self.step_count == 1:
            self.initial_losses = {
                'proto': L_proto.item(),
                'var': L_var.item(),
                'orth': L_orth.item(),
            }
        
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
        eps = 1e-8
        L_proto_scaled = L_proto / (self.running_means['proto'] + eps)
        L_var_scaled = L_var / (self.running_means['var'] + eps)
        L_orth_scaled = L_orth / (self.running_means['orth'] + eps)
        
        # Prevent collapse: ensure each loss contributes at least min_loss_ratio of max loss
        max_scaled = max(L_proto_scaled.item(), L_var_scaled.item(), L_orth_scaled.item())
        min_allowed = max_scaled * self.min_loss_ratio
        
        # Clamp losses to minimum contribution
        L_proto_scaled = torch.clamp(L_proto_scaled, min=min_allowed)
        L_var_scaled = torch.clamp(L_var_scaled, min=min_allowed)
        L_orth_scaled = torch.clamp(L_orth_scaled, min=min_allowed)
        
        return L_proto_scaled, L_var_scaled, L_orth_scaled
    
    def get_scales(self) -> Dict[str, float]:
        """Get current scaling factors (inverse of running means)."""
        eps = 1e-8
        return {
            'proto': 1.0 / (self.running_means['proto'] + eps),
            'var': 1.0 / (self.running_means['var'] + eps),
            'orth': 1.0 / (self.running_means['orth'] + eps),
        }


class GradientBalancer:
    """
    Gradient balancing (GradNorm-style) to automatically balance loss contributions.
    Instead of weighting losses, we balance their gradient magnitudes.
    This ensures all losses contribute equally to learning.
    """
    def __init__(self, alpha: float = 0.16, initial_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            alpha: Restoring force hyperparameter (0.16 is standard from GradNorm paper)
            initial_weights: Initial weights for each loss (default: equal weights)
        """
        self.alpha = alpha
        self.initial_weights = initial_weights or {'proto': 1.0, 'var': 1.0, 'orth': 1.0}
        self.weights = self.initial_weights.copy()
        self.initial_losses = {}
        self.running_grad_norms = {'proto': 1.0, 'var': 1.0, 'orth': 1.0}
        self.step_count = 0
    
    def compute_balanced_losses(
        self,
        L_proto: torch.Tensor,
        L_var: torch.Tensor,
        L_orth: torch.Tensor,
        model: torch.nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Compute balanced losses using gradient balancing.
        
        Args:
            L_proto: Prototypical loss
            L_var: Variance loss
            L_orth: Orthogonality loss
            model: Model to compute gradients for
        
        Returns:
            (L_proto_balanced, L_var_balanced, L_orth_balanced, weights_dict)
        """
        self.step_count += 1
        
        # Store initial losses (first step only)
        if self.step_count == 1:
            self.initial_losses = {
                'proto': L_proto.item(),
                'var': L_var.item(),
                'orth': L_orth.item(),
            }
        
        # Compute weighted losses
        L_proto_weighted = self.weights['proto'] * L_proto
        L_var_weighted = self.weights['var'] * L_var
        L_orth_weighted = self.weights['orth'] * L_orth
        
        # Compute gradients for each loss (without creating graph to avoid double backward)
        # We compute gradients separately, then use them to balance weights
        grad_proto = torch.autograd.grad(
            L_proto_weighted, model.parameters(), retain_graph=True, create_graph=False
        )
        grad_var = torch.autograd.grad(
            L_var_weighted, model.parameters(), retain_graph=True, create_graph=False
        )
        grad_orth = torch.autograd.grad(
            L_orth_weighted, model.parameters(), retain_graph=False, create_graph=False
        )
        
        # Compute gradient norms
        grad_norm_proto = torch.norm(torch.cat([g.flatten() for g in grad_proto]))
        grad_norm_var = torch.norm(torch.cat([g.flatten() for g in grad_var]))
        grad_norm_orth = torch.norm(torch.cat([g.flatten() for g in grad_orth]))
        
        # Update running averages
        self.running_grad_norms['proto'] = 0.99 * self.running_grad_norms['proto'] + 0.01 * grad_norm_proto.item()
        self.running_grad_norms['var'] = 0.99 * self.running_grad_norms['var'] + 0.01 * grad_norm_var.item()
        self.running_grad_norms['orth'] = 0.99 * self.running_grad_norms['orth'] + 0.01 * grad_norm_orth.item()
        
        # Compute relative inverse training rates (how fast each loss is decreasing)
        if self.step_count > 1:
            # Normalize by initial loss values
            rel_rate_proto = L_proto.item() / (self.initial_losses['proto'] + 1e-8)
            rel_rate_var = L_var.item() / (self.initial_losses['var'] + 1e-8)
            rel_rate_orth = L_orth.item() / (self.initial_losses['orth'] + 1e-8)
            
            # Average relative rate (target for all losses)
            avg_rel_rate = (rel_rate_proto + rel_rate_var + rel_rate_orth) / 3.0
            
            # Target gradient norms (proportional to relative rates)
            target_norm_proto = self.running_grad_norms['proto'] * (rel_rate_proto / avg_rel_rate) ** self.alpha
            target_norm_var = self.running_grad_norms['var'] * (rel_rate_var / avg_rel_rate) ** self.alpha
            target_norm_orth = self.running_grad_norms['orth'] * (rel_rate_orth / avg_rel_rate) ** self.alpha
            
            # Update weights to match target norms
            # w_i = w_i * (target_norm_i / current_norm_i)
            self.weights['proto'] = self.weights['proto'] * (target_norm_proto / (grad_norm_proto.item() + 1e-8))
            self.weights['var'] = self.weights['var'] * (target_norm_var / (grad_norm_var.item() + 1e-8))
            self.weights['orth'] = self.weights['orth'] * (target_norm_orth / (grad_norm_orth.item() + 1e-8))
            
            # Clamp weights to reasonable range
            self.weights['proto'] = max(0.1, min(10.0, self.weights['proto']))
            self.weights['var'] = max(0.1, min(10.0, self.weights['var']))
            self.weights['orth'] = max(0.1, min(10.0, self.weights['orth']))
        
        # Return balanced losses
        return (
            self.weights['proto'] * L_proto,
            self.weights['var'] * L_var,
            self.weights['orth'] * L_orth,
            self.weights.copy()
        )


class EqualWeightNormalizer:
    """
    Simple loss normalizer: normalize each loss by its initial value, then use equal weights.
    This ensures all losses start at similar scales and contribute equally.
    """
    def __init__(self):
        self.initial_losses = {}
        self.step_count = 0
    
    def normalize_losses(
        self,
        L_proto: torch.Tensor,
        L_var: torch.Tensor,
        L_orth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Normalize losses by their initial values.
        
        Returns:
            (L_proto_norm, L_var_norm, L_orth_norm, scales_dict)
        """
        self.step_count += 1
        
        # Store initial losses (first step only)
        if self.step_count == 1:
            self.initial_losses = {
                'proto': L_proto.item(),
                'var': L_var.item(),
                'orth': L_orth.item(),
            }
        
        # Normalize by initial values (so all start at ~1.0)
        eps = 1e-8
        L_proto_norm = L_proto / (self.initial_losses['proto'] + eps)
        L_var_norm = L_var / (self.initial_losses['var'] + eps)
        L_orth_norm = L_orth / (self.initial_losses['orth'] + eps)
        
        scales = {
            'proto': 1.0 / (self.initial_losses['proto'] + eps),
            'var': 1.0 / (self.initial_losses['var'] + eps),
            'orth': 1.0 / (self.initial_losses['orth'] + eps),
        }
        
        return L_proto_norm, L_var_norm, L_orth_norm, scales


def compute_total_loss(
    z_id: torch.Tensor,
    z_auth: torch.Tensor,
    is_real: torch.Tensor,
    content_groups: torch.Tensor,
    lambda_var: float = 0, # test without the effect of the variance loss
    lambda_orth: float = 0.1,
    temperature: float = 0.1,
    min_variance: float = 0.01,
    variance_reg_weight: float = 0.1,
    min_orth: float = 0.001,
    adaptive_scaler: Optional[AdaptiveLossScaler] = None,
    equal_weight_normalizer: Optional[EqualWeightNormalizer] = None,
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
    L_orth = orthogonality_loss(z_id, z_auth, min_orth=min_orth)
    
    # Apply equal-weight normalization if provided (simplest approach)
    if equal_weight_normalizer is not None:
        L_proto_norm, L_var_norm, L_orth_norm, scales = equal_weight_normalizer.normalize_losses(
            L_proto, L_var, L_orth
        )
        
        # Total loss: equal weights (1.0 each) after normalization
        total_loss = L_proto_norm + L_var_norm + L_orth_norm
        
        losses_dict = {
            'total': total_loss.item(),
            'proto': L_proto.item(),
            'var': L_var.item(),
            'orth': L_orth.item(),
            'proto_norm': L_proto_norm.item(),
            'var_norm': L_var_norm.item(),
            'orth_norm': L_orth_norm.item(),
            'scale_proto': scales['proto'],
            'scale_var': scales['var'],
            'scale_orth': scales['orth'],
        }
        
        return total_loss, losses_dict
    
    # Apply adaptive scaling if provided
    elif adaptive_scaler is not None:
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

