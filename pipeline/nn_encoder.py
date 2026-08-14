"""
Subclass 1: Neural Network -> Frozen Encoder

The MLP processes the input first, then the frozen encoder refines
the representation. Only the MLP parameters are trained.

Use case: learn a projection/preprocessing step while leveraging
a pre-trained encoder whose weights should not change.
"""

import torch
import torch.nn as nn

from pipeline.base_model import BaseEncoderMLPModel
from pipeline.encoder import BaseEncoder


class NNThenEncoder(BaseEncoderMLPModel):
    """MLP first, then frozen encoder.

    Data flow:
        input -> MLP (trainable) -> Encoder (frozen) -> output

    The encoder is frozen at construction time. Call
    ``unfreeze_encoder()`` if you later want to fine-tune it.
    """

    def __init__(self, encoder: BaseEncoder, mlp: nn.Module):
        super().__init__(encoder, mlp)
        self.freeze_encoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor matching the MLP's expected input shape.
        Returns:
            Encoder output tensor.
        """
        h = self.mlp(x)
        with torch.no_grad():
            out = self.encoder(h)
        return out
