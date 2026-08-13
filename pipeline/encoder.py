"""
Encoder abstractions for the pipeline.

Provides a BaseEncoder interface, a generic transformer encoder,
and an adapter that wraps the existing ModalityEncoder from
models/time_series/time_series_model.py.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class BaseEncoder(ABC, nn.Module):
    """Abstract encoder interface.

    All encoders must expose ``output_dim`` so downstream MLPs know
    what input dimension to expect.
    """

    def __init__(self, output_dim: int):
        super().__init__()
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Encode input and return a tensor of shape (..., output_dim)."""
        ...


# ---------------------------------------------------------------------------
# Generic transformer encoder
# ---------------------------------------------------------------------------

class GenericTransformerEncoder(BaseEncoder):
    """Transformer encoder that accepts any (B, T, input_dim) sequence.

    Architecture:
        Linear(input_dim -> model_dim) -> PosEmbed -> TransformerEncoder

    Output shape depends on ``pool``:
        - ``None``:  (B, T, model_dim)  — full sequence
        - ``"mean"``: (B, model_dim)    — mean-pooled
        - ``"cls"``:  (B, model_dim)    — CLS-token pooling
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_len: int = 1000,
        pool: Optional[str] = None,
    ):
        super().__init__(output_dim=model_dim)
        self.pool = pool

        # Input projection
        self.input_projection = nn.Linear(input_dim, model_dim)

        # Learnable positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(max_len, model_dim) * 0.02)

        # Optional CLS token
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            Tensor of shape (B, T, model_dim), (B, model_dim), or (B, model_dim)
            depending on ``pool``.
        """
        B, T, _ = x.shape

        # Project to model dimension
        x = self.input_projection(x)  # (B, T, model_dim)

        # Prepend CLS token if using CLS pooling
        if self.pool == "cls":
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, model_dim)
            x = torch.cat([cls, x], dim=1)  # (B, T+1, model_dim)
            pos = self.pos_encoder[: T + 1].unsqueeze(0)
        else:
            pos = self.pos_encoder[:T].unsqueeze(0)

        x = self.dropout(x + pos)
        x = self.transformer(x)  # (B, T[+1], model_dim)

        if self.pool == "cls":
            return x[:, 0]  # (B, model_dim)
        elif self.pool == "mean":
            return x.mean(dim=1)  # (B, model_dim)
        return x  # (B, T, model_dim)


# ---------------------------------------------------------------------------
# Adapter for the existing ModalityEncoder from time_series_model.py
# ---------------------------------------------------------------------------

class ModalityEncoderAdapter(BaseEncoder):
    """Wraps the existing PatchTokenizer + ModalityEncoder pair.

    This adapter accepts *separate* audio and video sequences, tokenises them,
    runs each through its own ModalityEncoder, and concatenates the outputs
    along the feature dimension.

    Input:
        audio_seq: (B, T, audio_emb_dim)
        video_seq: (B, T, video_emb_dim)
    Output:
        (B, N, 2 * model_dim)   — concatenated encoded tokens
        plus ``patch_info`` dict for optional upsampling.
    """

    def __init__(
        self,
        audio_emb_dim: int = 512,
        video_emb_dim: int = 2048,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        patch_size: int = 8,
        patch_stride: int = 4,
    ):
        super().__init__(output_dim=2 * model_dim)

        # Import here to avoid circular imports
        from models.time_series.time_series_model import (
            ModalityEncoder,
            PatchTokenizer,
        )

        # Shared tokenizer config
        self.audio_tokenizer = PatchTokenizer(patch_size, patch_stride)
        self.video_tokenizer = PatchTokenizer(patch_size, patch_stride)

        audio_patch_dim = patch_size * audio_emb_dim
        video_patch_dim = patch_size * video_emb_dim

        self.audio_encoder = ModalityEncoder(
            patch_dim=audio_patch_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.video_encoder = ModalityEncoder(
            patch_dim=video_patch_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

    def forward(
        self,
        audio_seq: torch.Tensor,
        video_seq: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Args:
            audio_seq: (B, T, audio_emb_dim)
            video_seq: (B, T, video_emb_dim)
        Returns:
            encoded: (B, N, 2*model_dim) concatenated audio+video tokens
            patch_info: dict from the tokenizer (for optional upsampling)
        """
        audio_patches, patch_info = self.audio_tokenizer(audio_seq)
        video_patches, _ = self.video_tokenizer(video_seq)

        audio_encoded = self.audio_encoder(audio_patches)  # (B, N, model_dim)
        video_encoded = self.video_encoder(video_patches)  # (B, N, model_dim)

        encoded = torch.cat([audio_encoded, video_encoded], dim=-1)  # (B, N, 2*model_dim)
        return encoded, patch_info
