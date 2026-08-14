"""
Subclass 2: Encoder -> Neural Network (Joint Training)

The encoder processes the input first, then the MLP classifies
the encoded representation. All parameters are trained jointly.

Use case: end-to-end training from scratch where both the
encoder and the MLP learn together.
"""

import torch
import torch.nn as nn

from pipeline.base_model import BaseEncoderMLPModel
from pipeline.encoder import BaseEncoder


class EncoderThenNN(BaseEncoderMLPModel):
    """Encoder first, then MLP — all trained jointly.

    Data flow:
        input -> Encoder (trainable) -> MLP (trainable) -> output

    Both encoder and MLP parameters receive gradient updates.
    """

    def __init__(self, encoder: BaseEncoder, mlp: nn.Module):
        super().__init__(encoder, mlp)
        # Everything trainable by default — no freezing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor matching the encoder's expected input shape.
        Returns:
            MLP output tensor.
        """
        h = self.encoder(x)
        out = self.mlp(h)
        return out
