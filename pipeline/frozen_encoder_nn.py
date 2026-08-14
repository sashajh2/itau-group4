"""
Subclass 3: Frozen Encoder -> Neural Network

The encoder processes the input first (frozen), then the MLP
classifies the encoded representation. Only the MLP is trained.

Use case: fine-tuning a downstream classifier on top of a
pre-trained encoder whose representations are already good.
"""

import torch
import torch.nn as nn

from pipeline.base_model import BaseEncoderMLPModel
from pipeline.encoder import BaseEncoder


class FrozenEncoderThenNN(BaseEncoderMLPModel):
    """Frozen encoder first, then trainable MLP.

    Data flow:
        input -> Encoder (frozen, no_grad) -> MLP (trainable) -> output

    The encoder is frozen at construction time and forward passes
    use ``torch.no_grad()`` for efficiency. Call ``unfreeze_encoder()``
    if you later want to fine-tune it.
    """

    def __init__(self, encoder: BaseEncoder, mlp: nn.Module):
        super().__init__(encoder, mlp)
        self.freeze_encoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor matching the encoder's expected input shape.
        Returns:
            MLP output tensor.
        """
        with torch.no_grad():
            h = self.encoder(x)
        out = self.mlp(h)
        return out
