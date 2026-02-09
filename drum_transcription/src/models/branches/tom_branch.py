"""Tom detection and classification branch (hierarchical)."""

import torch
import torch.nn as nn
from typing import Dict


class TomBranch(nn.Module):
    """
    Hierarchical tom detection and classification branch.

    Frequency focus: 80-400 Hz
    Primary output: tom/no_tom (binary detection)
    Variation output: floor/high/mid (3-class, conditional on primary)
    Training weight: MEDIUM (1.0x primary, 0.7x variation)
    """

    def __init__(self, hidden_size: int = 512, dropout: float = 0.3):
        """
        Initialize tom branch.

        Args:
            hidden_size: Size of shared encoder output
            dropout: Dropout rate
        """
        super().__init__()

        # Primary: Is there a tom hit?
        self.primary_detector = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)  # Binary: tom/no_tom (2-class for consistency)
        )

        # Variation: Which tom? (conditional on tom detection)
        self.variation_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 3)  # floor/high/mid
        )

    def forward(self, shared_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            shared_features: (batch, time, hidden_size) from shared encoder

        Returns:
            Dictionary with:
                'primary': (batch, time, 2) - logits for tom/no_tom
                'variation': (batch, time, 3) - logits for floor/high/mid
        """
        primary = self.primary_detector(shared_features)
        variation = self.variation_classifier(shared_features)

        return {
            'primary': primary,
            'variation': variation
        }
