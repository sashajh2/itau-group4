"""
Dual projection heads for disentangling identity and authenticity.

Architecture:
- Shared encoder (pretrained embeddings as input)
- Two 2-layer MLP projection heads: f_auth and f_id
- Both outputs are L2-normalized
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DisentangledProjector(nn.Module):
    """
    Dual projection heads for disentangling identity and authenticity.
    
    Args:
        input_dim: Dimension of input embeddings (e.g., 768 for Hubert)
        output_dim: Dimension of projected embeddings (default: 128)
    """
    
    def __init__(self, input_dim: int = 768, output_dim: int = 128):
        super().__init__()
        
        # Authenticity projection head (2-layer MLP with ReLU)
        self.f_auth = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
        # Identity projection head (2-layer MLP with ReLU)
        self.f_id = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both projection heads.
        
        Args:
            z: Input embeddings, shape [batch_size, input_dim]
        
        Returns:
            z_auth: Authenticity embeddings, shape [batch_size, output_dim] (L2-normalized)
            z_id: Identity embeddings, shape [batch_size, output_dim] (L2-normalized)
        """
        z_auth = self.f_auth(z)
        z_id = self.f_id(z)
        
        # L2-normalize both outputs
        z_auth = F.normalize(z_auth, dim=-1)
        z_id = F.normalize(z_id, dim=-1)
        
        return z_auth, z_id

