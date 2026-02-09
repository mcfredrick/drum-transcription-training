"""Kick drum detection branch (binary onset detection)."""

import torch
import torch.nn as nn


class KickBranch(nn.Module):
    """
    Binary kick drum detection branch.

    Frequency focus: 20-100 Hz
    Output: Binary kick/no_kick predictions
    Training weight: HIGH (1.5x) - Critical for rhythm game
    """

    def __init__(self, hidden_size: int = 512, dropout: float = 0.3):
        """
        Initialize kick branch.

        Args:
            hidden_size: Size of shared encoder output (bidirectional, so 256*2=512)
            dropout: Dropout rate
        """
        super().__init__()

        self.detector = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Binary output (logits, no sigmoid)
        )

    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            shared_features: (batch, time, hidden_size) from shared encoder

        Returns:
            Kick predictions: (batch, time, 1) - logits for binary classification
        """
        return self.detector(shared_features)
