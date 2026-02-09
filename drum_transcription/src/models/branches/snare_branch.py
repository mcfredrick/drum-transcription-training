"""Snare drum detection and classification branch."""

import torch
import torch.nn as nn


class SnareBranch(nn.Module):
    """
    3-class snare detection and classification branch.

    Frequency focus: 200-1000 Hz (shell) + broadband (wire rattle)
    Output: 3-class (none/head/rim)
    Training weight: HIGH (1.5x) - Critical for rhythm game
    """

    def __init__(self, hidden_size: int = 512, dropout: float = 0.3):
        """
        Initialize snare branch.

        Args:
            hidden_size: Size of shared encoder output
            dropout: Dropout rate
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # 3-class output (none/head/rim) - logits
        )

    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            shared_features: (batch, time, hidden_size) from shared encoder

        Returns:
            Snare predictions: (batch, time, 3) - logits for 3-class classification
        """
        return self.classifier(shared_features)
