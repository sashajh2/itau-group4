"""
Base class for encoder + MLP pipeline models.

Provides freeze/unfreeze utilities and a common interface
for the three subclass variants.
"""

from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn

from pipeline.encoder import BaseEncoder


class BaseEncoderMLPModel(ABC, nn.Module):
    """Abstract base for all encoder-MLP combinations.

    Subclasses define the ordering (encoder first vs MLP first)
    and whether the encoder is frozen.
    """

    def __init__(self, encoder: BaseEncoder, mlp: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.mlp = mlp

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        ...

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (no gradient updates)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_mlp(self) -> None:
        """Freeze all MLP parameters."""
        for param in self.mlp.parameters():
            param.requires_grad = False

    def unfreeze_mlp(self) -> None:
        """Unfreeze all MLP parameters."""
        for param in self.mlp.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Parameter inspection
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return only parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> str:
        """One-line summary of parameter counts."""
        total = self.num_total_params()
        trainable = self.num_trainable_params()
        frozen = total - trainable
        return (
            f"{self.__class__.__name__}: "
            f"{total:,} total params | "
            f"{trainable:,} trainable | "
            f"{frozen:,} frozen"
        )
