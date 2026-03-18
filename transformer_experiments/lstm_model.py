"""
Bidirectional LSTM classifier for deepfake detection on HuBERT sequences.

Architecture:
    Input (B, T, 768)
      -> pack_padded_sequence (skip padding)
      -> BiLSTM (num_layers, hidden_dim, bidirectional)
      -> Concatenate final forward + backward hidden states -> (B, hidden_dim*2)
      -> Linear classifier -> (B, 1) logit
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class VideoLSTM(nn.Module):
    """Bidirectional LSTM for binary sequence classification."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, T, 768) padded HuBERT sequences.
            lengths: (B,) actual sequence lengths (derived from attention_mask).

        Returns:
            logits: (B,) raw logits for binary classification.
        """
        self.lstm.flatten_parameters()

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (h_n, _) = self.lstm(packed)

        # Concatenate last forward and backward hidden states
        video_repr = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, hidden_dim*2)
        logits = self.classifier(video_repr).squeeze(-1)    # (B,)
        return logits
