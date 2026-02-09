"""Rhythm cymbal detection and classification branch (hihat and ride only)."""

import torch
import torch.nn as nn
from typing import Dict


class CymbalBranch(nn.Module):
    """
    Hierarchical rhythm cymbal detection branch (hihat and ride only).
    Crash cymbal is handled by separate crash branch.

    Frequency focus: 2-8 kHz
    Primary output: none/hihat/ride (3-class)
    Hihat variation: closed/open (2-class, conditional on hihat)
    Ride variation: body/bell (2-class, conditional on ride)
    Training weight: HIGH primary (1.2x), MEDIUM variations (0.5x)
    """

    def __init__(self, hidden_size: int = 512, dropout: float = 0.3):
        """
        Initialize cymbal branch.

        Args:
            hidden_size: Size of shared encoder output
            dropout: Dropout rate
        """
        super().__init__()

        # Primary: Which rhythm cymbal?
        self.primary_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)  # none/hihat/ride
        )

        # Hihat variation (conditional on hihat detection)
        self.hihat_variation = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # closed/open
        )

        # Ride variation (conditional on ride detection)
        self.ride_variation = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # body/bell
        )

    def forward(self, shared_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            shared_features: (batch, time, hidden_size) from shared encoder

        Returns:
            Dictionary with:
                'primary': (batch, time, 3) - logits for none/hihat/ride
                'hihat_variation': (batch, time, 2) - logits for closed/open
                'ride_variation': (batch, time, 2) - logits for body/bell
        """
        primary = self.primary_classifier(shared_features)
        hihat_var = self.hihat_variation(shared_features)
        ride_var = self.ride_variation(shared_features)

        return {
            'primary': primary,
            'hihat_variation': hihat_var,
            'ride_variation': ride_var
        }
