"""Crash cymbal detection branch (accent detector)."""

import torch
import torch.nn as nn


class CrashBranch(nn.Module):
    """
    Binary crash cymbal detection branch (accent detector).

    Frequency focus: 4-8 kHz (broadband, chaotic spectral content)
    Characteristics: Explosive onset, very long decay (1-2 sec)
    Frequency: Rare events (~1-5 per 30-second section)
    Output: Binary crash/no_crash predictions
    Training weight: MEDIUM (0.6x) - Lower priority than rhythm elements
    Strategy: Higher precision priority (avoid false positives)
    """

    def __init__(self, hidden_size: int = 512, dropout: float = 0.3):
        """
        Initialize crash branch.

        Args:
            hidden_size: Size of shared encoder output
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
            Crash predictions: (batch, time, 1) - logits for binary classification
        """
        return self.detector(shared_features)
