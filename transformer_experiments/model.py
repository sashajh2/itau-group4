"""
Vanilla Transformer classifier for deepfake detection on HuBERT sequences.

Architecture:
    Input (B, T, 768)
      -> Linear projection: 768 -> d_model
      -> Prepend learnable [CLS] token -> (B, T+1, d_model)
      -> Add learnable positional encoding
      -> TransformerEncoder (N layers, H heads)
      -> Extract CLS output -> (B, d_model)
      -> Linear classifier -> (B, 1) logit
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class VanillaTransformerClassifier(nn.Module):
    """Transformer encoder with [CLS] token for binary sequence classification."""

    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.d_model = d_model

        # Project input features to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Learnable positional encoding (max_seq_len + 1 for CLS)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len + 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Classification head
        self.classifier = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, T, 768) input HuBERT sequences.
            attention_mask: (B, T) binary mask (1=real, 0=padding).

        Returns:
            logits: (B, 1) raw logits for binary classification.
        """
        b, t, _ = embeddings.shape

        # Project to d_model
        x = self.input_proj(embeddings)  # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, : t + 1, :]

        # Build attention mask for transformer (True = ignore)
        if attention_mask is not None:
            # Add 1 for CLS token (always attended)
            cls_mask = torch.ones(b, 1, device=attention_mask.device)
            full_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, T+1)
            # PyTorch TransformerEncoder uses True = ignore
            src_key_padding_mask = (full_mask == 0)
        else:
            src_key_padding_mask = None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, T+1, d_model)

        # Extract CLS output
        cls_out = x[:, 0, :]  # (B, d_model)

        # Classify
        logits = self.classifier(cls_out)  # (B, 1)
        return logits
