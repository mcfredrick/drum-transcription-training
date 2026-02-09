"""Hierarchical CRNN model for drum transcription with specialized branches."""

import torch
import torch.nn as nn
import lightning as L
from typing import Dict, Optional, Any
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from src.models.branches import KickBranch, SnareBranch, TomBranch, CymbalBranch, CrashBranch


class HierarchicalDrumCRNN(L.LightningModule):
    """
    Hierarchical Convolutional Recurrent Neural Network for drum transcription.

    Architecture:
        Input: Log-mel spectrogram (batch, 1, n_mels, time)
        ↓
        Shared CNN Encoder: Extract spatial features
        ↓
        Shared LSTM Encoder: Capture temporal dependencies
        ↓
        5 Specialized Branches:
          1. Kick Branch (binary)
          2. Snare Branch (3-class)
          3. Tom Branch (hierarchical: primary + variation)
          4. Cymbal Branch (hierarchical: primary + variations)
          5. Crash Branch (binary accent detector)
    """

    def __init__(
        self,
        n_mels: int = 128,
        conv_filters: list = [32, 64, 128, 256],
        conv_kernel_size: int = 3,
        pool_size: int = 2,
        dropout_cnn: float = 0.25,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        dropout_lstm: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        branch_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize hierarchical CRNN model.

        Args:
            n_mels: Number of mel frequency bins
            conv_filters: List of CNN filter counts per block
            conv_kernel_size: Kernel size for convolutions
            pool_size: Pooling size (frequency dimension only)
            dropout_cnn: Dropout rate for CNN blocks
            hidden_size: LSTM hidden size (will be doubled with bidirectional)
            num_lstm_layers: Number of LSTM layers
            dropout_lstm: Dropout rate for LSTM
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            scheduler_patience: Patience for learning rate scheduler
            scheduler_factor: Factor for learning rate scheduler
            branch_weights: Weights for each branch in loss computation
        """
        super().__init__()
        self.save_hyperparameters()

        # Store hyperparameters
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.pool_size = pool_size
        self.num_pools = len(conv_filters)

        # Branch weights for loss computation
        if branch_weights is None:
            self.branch_weights = {
                'kick': 1.5,           # High priority
                'snare': 1.5,          # High priority
                'tom_primary': 1.0,    # Medium priority
                'tom_variation': 0.7,  # Lower priority (conditional)
                'cymbal_primary': 1.2, # High priority
                'hihat_variation': 0.5, # Lower priority (conditional)
                'ride_variation': 0.5,  # Lower priority (conditional)
                'crash': 0.6           # Medium-low priority (rare events)
            }
        else:
            self.branch_weights = branch_weights

        # ===== SHARED ENCODER =====

        # CNN encoder (reused from baseline CRNN)
        self.conv_blocks = self._build_cnn_encoder(
            conv_filters, conv_kernel_size, pool_size, dropout_cnn
        )

        # Calculate CNN output size
        cnn_output_freq = n_mels
        for _ in conv_filters:
            cnn_output_freq = cnn_output_freq // pool_size
        self.cnn_output_size = conv_filters[-1] * cnn_output_freq

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_lstm if num_lstm_layers > 1 else 0
        )

        # Shared features output size (bidirectional)
        self.shared_output_size = hidden_size * 2

        # ===== SPECIALIZED BRANCHES =====

        self.kick_branch = KickBranch(self.shared_output_size, dropout_lstm)
        self.snare_branch = SnareBranch(self.shared_output_size, dropout_lstm)
        self.tom_branch = TomBranch(self.shared_output_size, dropout_lstm)
        self.cymbal_branch = CymbalBranch(self.shared_output_size, dropout_lstm)
        self.crash_branch = CrashBranch(self.shared_output_size, dropout_lstm)

        # Storage for epoch-level metrics (AUC computation)
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _build_cnn_encoder(
        self,
        filters: list,
        kernel_size: int,
        pool_size: int,
        dropout: float
    ) -> nn.Sequential:
        """
        Build CNN encoder blocks.
        Reused from baseline CRNN architecture.
        """
        layers = []
        in_channels = 1

        for out_channels in filters:
            layers.extend([
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((pool_size, 1)),  # Pool in frequency only, not time
                nn.Dropout(dropout)
            ])
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through shared encoder and all branches.

        Args:
            x: Input spectrogram (batch, 1, n_mels, time)

        Returns:
            Dictionary with predictions from all branches
        """
        # Shared CNN encoding
        features = self.conv_blocks(x)  # (batch, channels, freq, time)

        # Reshape for LSTM: (batch, time, features)
        batch_size = features.size(0)
        features = features.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        features = features.reshape(batch_size, features.size(1), -1)

        # Shared LSTM encoding
        shared_features, _ = self.lstm(features)  # (batch, time, hidden*2)

        # Branch predictions
        predictions = {
            'kick': self.kick_branch(shared_features),
            'snare': self.snare_branch(shared_features),
            'tom': self.tom_branch(shared_features),
            'cymbal': self.cymbal_branch(shared_features),
            'crash': self.crash_branch(shared_features)
        }

        return predictions

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        lengths: torch.Tensor
    ) -> tuple:
        """
        Compute weighted multi-branch loss.

        Args:
            predictions: Dictionary of predictions from all branches
            labels: Dictionary of ground truth labels (hierarchical format)
            lengths: Original sequence lengths (before pooling)

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Adjust lengths for pooling
        adjusted_lengths = self._adjust_lengths_for_pooling(lengths)

        total_loss = 0.0
        loss_components = {}

        # 1. Kick loss (binary)
        kick_loss = self._compute_binary_loss(
            predictions['kick'],
            labels['kick'],
            adjusted_lengths,
            name='kick'
        )
        total_loss += self.branch_weights['kick'] * kick_loss
        loss_components['kick'] = kick_loss.item()

        # 2. Snare loss (3-class)
        snare_loss = self._compute_multiclass_loss(
            predictions['snare'],
            labels['snare'],
            adjusted_lengths,
            name='snare'
        )
        total_loss += self.branch_weights['snare'] * snare_loss
        loss_components['snare'] = snare_loss.item()

        # 3. Tom primary loss (2-class: tom/no_tom)
        tom_primary_loss = self._compute_multiclass_loss(
            predictions['tom']['primary'],
            labels['tom_primary'],
            adjusted_lengths,
            name='tom_primary'
        )
        total_loss += self.branch_weights['tom_primary'] * tom_primary_loss
        loss_components['tom_primary'] = tom_primary_loss.item()

        # 4. Tom variation loss (conditional: only where tom detected)
        tom_var_loss = self._compute_conditional_loss(
            predictions['tom']['variation'],
            labels['tom_variation'],
            labels['tom_primary'],  # Mask: only where tom==1
            adjusted_lengths,
            name='tom_variation'
        )
        total_loss += self.branch_weights['tom_variation'] * tom_var_loss
        loss_components['tom_variation'] = tom_var_loss.item()

        # 5. Cymbal primary loss (3-class: none/hihat/ride)
        cymbal_primary_loss = self._compute_multiclass_loss(
            predictions['cymbal']['primary'],
            labels['cymbal_primary'],
            adjusted_lengths,
            name='cymbal_primary'
        )
        total_loss += self.branch_weights['cymbal_primary'] * cymbal_primary_loss
        loss_components['cymbal_primary'] = cymbal_primary_loss.item()

        # 6. Hihat variation loss (conditional: only where hihat detected)
        hihat_var_loss = self._compute_conditional_loss(
            predictions['cymbal']['hihat_variation'],
            labels['hihat_variation'],
            labels['cymbal_primary'],  # Mask: only where cymbal_primary==1 (hihat)
            adjusted_lengths,
            name='hihat_variation',
            mask_value=1  # Hihat is class 1
        )
        total_loss += self.branch_weights['hihat_variation'] * hihat_var_loss
        loss_components['hihat_variation'] = hihat_var_loss.item()

        # 7. Ride variation loss (conditional: only where ride detected)
        ride_var_loss = self._compute_conditional_loss(
            predictions['cymbal']['ride_variation'],
            labels['ride_variation'],
            labels['cymbal_primary'],  # Mask: only where cymbal_primary==2 (ride)
            adjusted_lengths,
            name='ride_variation',
            mask_value=2  # Ride is class 2
        )
        total_loss += self.branch_weights['ride_variation'] * ride_var_loss
        loss_components['ride_variation'] = ride_var_loss.item()

        # 8. Crash loss (binary)
        crash_loss = self._compute_binary_loss(
            predictions['crash'],
            labels['crash'],
            adjusted_lengths,
            name='crash'
        )
        total_loss += self.branch_weights['crash'] * crash_loss
        loss_components['crash'] = crash_loss.item()

        loss_components['total'] = total_loss.item()

        return total_loss, loss_components

    def _adjust_lengths_for_pooling(self, lengths: torch.Tensor) -> torch.Tensor:
        """Adjust lengths to account for pooling in time dimension."""
        # Note: We only pool in frequency dimension (pool_size, 1)
        # So time dimension is preserved - no adjustment needed
        return lengths

    def _compute_binary_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor,
        name: str
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss with masking."""
        # predictions: (batch, time, 1)
        # labels: (batch, time) - binary

        predictions = predictions.squeeze(-1)  # (batch, time)
        labels = labels.float()  # Ensure float

        # Create mask for valid frames
        batch_size, max_time = predictions.shape
        mask = torch.arange(max_time, device=predictions.device)[None, :] < lengths[:, None]

        # Compute loss only on valid frames
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        loss = loss_fn(predictions, labels)
        loss = loss * mask.float()

        # Average over valid frames
        loss = loss.sum() / mask.sum().clamp(min=1)

        return loss

    def _compute_multiclass_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor,
        name: str
    ) -> torch.Tensor:
        """Compute cross-entropy loss with masking."""
        # predictions: (batch, time, n_classes)
        # labels: (batch, time) - class indices

        batch_size, max_time, n_classes = predictions.shape

        # Create mask for valid frames
        mask = torch.arange(max_time, device=predictions.device)[None, :] < lengths[:, None]

        # Flatten for cross-entropy
        predictions_flat = predictions.reshape(-1, n_classes)
        labels_flat = labels.reshape(-1)
        mask_flat = mask.reshape(-1)

        # Compute loss only on valid frames
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(predictions_flat, labels_flat)
        loss = loss * mask_flat.float()

        # Average over valid frames
        loss = loss.sum() / mask_flat.sum().clamp(min=1)

        return loss

    def _compute_conditional_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        condition_labels: torch.Tensor,
        lengths: torch.Tensor,
        name: str,
        mask_value: int = 1
    ) -> torch.Tensor:
        """
        Compute loss conditional on another branch's prediction.
        Only compute loss where condition_labels == mask_value.
        """
        # predictions: (batch, time, n_classes)
        # labels: (batch, time) - class indices
        # condition_labels: (batch, time) - class indices for masking

        batch_size, max_time, n_classes = predictions.shape

        # Create mask: valid frames AND condition met
        valid_mask = torch.arange(max_time, device=predictions.device)[None, :] < lengths[:, None]
        condition_mask = (condition_labels == mask_value)
        combined_mask = valid_mask & condition_mask

        # If no frames meet condition, return zero loss
        if combined_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Flatten for cross-entropy
        predictions_flat = predictions.reshape(-1, n_classes)
        labels_flat = labels.reshape(-1)
        mask_flat = combined_mask.reshape(-1)

        # Compute loss only on masked frames
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(predictions_flat, labels_flat)
        loss = loss * mask_flat.float()

        # Average over masked frames
        loss = loss.sum() / mask_flat.sum().clamp(min=1)

        return loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        specs, labels, lengths = batch

        # Forward pass
        predictions = self(specs)

        # Compute loss
        loss, loss_components = self.compute_loss(predictions, labels, lengths)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for name, value in loss_components.items():
            if name != 'total':
                self.log(f'train_{name}_loss', value, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        specs, labels, lengths = batch

        # Forward pass
        predictions = self(specs)

        # Compute loss
        loss, loss_components = self.compute_loss(predictions, labels, lengths)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, value in loss_components.items():
            if name != 'total':
                self.log(f'val_{name}_loss', value, on_step=False, on_epoch=True)

        # Store for epoch-end AUC computation
        # Detach from computation graph and move to CPU to prevent memory leak
        predictions_detached = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else {
                kk: vv.detach().cpu() for kk, vv in v.items()
            }
            for k, v in predictions.items()
        }
        labels_detached = {k: v.detach().cpu() for k, v in labels.items()}

        self.validation_step_outputs.append({
            'predictions': predictions_detached,
            'labels': labels_detached,
            'lengths': lengths.detach().cpu()
        })

        return loss

    def on_validation_epoch_end(self):
        """Compute AUC metrics at end of validation epoch."""
        if not self.validation_step_outputs:
            return

        # Compute per-branch AUC metrics
        auc_metrics = self._compute_branch_auc_metrics(self.validation_step_outputs)

        # Log AUC metrics
        for name, value in auc_metrics.items():
            # Show kick ROC-AUC in progress bar (primary metric)
            prog_bar = (name == 'kick_roc_auc')
            self.log(f'val_{name}', value, prog_bar=prog_bar)

        # Clear outputs
        self.validation_step_outputs.clear()

    def _compute_branch_auc_metrics(self, outputs: list) -> Dict[str, float]:
        """Compute ROC-AUC and PR-AUC for each branch."""
        metrics = {}

        # Collect all predictions and labels
        all_predictions = {
            'kick': [], 'snare': [], 'tom_primary': [],
            'cymbal_primary': [], 'crash': []
        }
        all_labels = {
            'kick': [], 'snare': [], 'tom_primary': [],
            'cymbal_primary': [], 'crash': []
        }

        for output in outputs:
            predictions = output['predictions']
            labels = output['labels']
            lengths = output['lengths']

            # Extract valid frames for each sample
            for i in range(lengths.size(0)):
                length = lengths[i].item()

                # Kick (binary)
                kick_pred = torch.sigmoid(predictions['kick'][i, :length, 0]).cpu().numpy()
                kick_label = labels['kick'][i, :length].cpu().numpy()
                all_predictions['kick'].append(kick_pred)
                all_labels['kick'].append(kick_label)

                # Snare (3-class -> binary for AUC: any snare vs none)
                snare_pred = torch.softmax(predictions['snare'][i, :length], dim=-1).cpu().numpy()
                snare_pred_binary = 1 - snare_pred[:, 0]  # P(head or rim)
                snare_label = (labels['snare'][i, :length] > 0).cpu().numpy().astype(float)
                all_predictions['snare'].append(snare_pred_binary)
                all_labels['snare'].append(snare_label)

                # Tom (2-class -> binary)
                tom_pred = torch.softmax(predictions['tom']['primary'][i, :length], dim=-1).cpu().numpy()
                tom_pred_binary = tom_pred[:, 1]  # P(tom)
                tom_label = (labels['tom_primary'][i, :length] > 0).cpu().numpy().astype(float)
                all_predictions['tom_primary'].append(tom_pred_binary)
                all_labels['tom_primary'].append(tom_label)

                # Cymbal (3-class -> binary: any cymbal vs none)
                cymbal_pred = torch.softmax(predictions['cymbal']['primary'][i, :length], dim=-1).cpu().numpy()
                cymbal_pred_binary = 1 - cymbal_pred[:, 0]  # P(hihat or ride)
                cymbal_label = (labels['cymbal_primary'][i, :length] > 0).cpu().numpy().astype(float)
                all_predictions['cymbal_primary'].append(cymbal_pred_binary)
                all_labels['cymbal_primary'].append(cymbal_label)

                # Crash (binary)
                crash_pred = torch.sigmoid(predictions['crash'][i, :length, 0]).cpu().numpy()
                crash_label = labels['crash'][i, :length].cpu().numpy()
                all_predictions['crash'].append(crash_pred)
                all_labels['crash'].append(crash_label)

        # Concatenate all frames
        for branch in all_predictions.keys():
            all_predictions[branch] = np.concatenate(all_predictions[branch])
            all_labels[branch] = np.concatenate(all_labels[branch])

        # Compute AUC metrics for each branch
        for branch in all_predictions.keys():
            preds = all_predictions[branch]
            labels_arr = all_labels[branch]

            # Check if we have both classes
            if len(np.unique(labels_arr)) > 1:
                try:
                    roc_auc = roc_auc_score(labels_arr, preds)
                    pr_auc = average_precision_score(labels_arr, preds)
                    metrics[f'{branch}_roc_auc'] = roc_auc
                    metrics[f'{branch}_pr_auc'] = pr_auc
                except Exception as e:
                    print(f"Warning: Could not compute AUC for {branch}: {e}")
                    metrics[f'{branch}_roc_auc'] = 0.0
                    metrics[f'{branch}_pr_auc'] = 0.0
            else:
                metrics[f'{branch}_roc_auc'] = 0.0
                metrics[f'{branch}_pr_auc'] = 0.0

        return metrics

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
