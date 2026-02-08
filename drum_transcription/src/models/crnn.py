"""CRNN model for drum transcription using PyTorch Lightning."""

import torch
import torch.nn as nn
import lightning as L
from typing import Optional, Dict, Any


class DrumTranscriptionCRNN(L.LightningModule):
    """
    Convolutional Recurrent Neural Network for drum transcription.
    
    Architecture:
        Input: Log-mel spectrogram (batch, 1, n_mels, time)
        ↓
        CNN Encoder: Extract spatial features
        ↓
        Bidirectional GRU: Capture temporal dependencies
        ↓
        Dense Layer: Frame-level predictions (batch, time, n_classes)
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        n_classes: int = 8,
        conv_filters: list = [32, 64, 128],
        conv_kernel_size: int = 3,
        pool_size: int = 2,
        dropout_cnn: float = 0.25,
        hidden_size: int = 128,
        num_gru_layers: int = 3,
        dropout_gru: float = 0.3,
        bidirectional: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        class_weights: Optional[list] = None
    ):
        """
        Initialize CRNN model.
        
        Args:
            n_mels: Number of mel frequency bins
            n_classes: Number of drum classes (8 for our task)
            conv_filters: List of CNN filter counts per block
            conv_kernel_size: Kernel size for convolutions
            pool_size: Pooling size
            dropout_cnn: Dropout rate for CNN blocks
            hidden_size: GRU hidden size
            num_gru_layers: Number of GRU layers
            dropout_gru: Dropout rate for GRU
            bidirectional: Use bidirectional GRU
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            scheduler_patience: Patience for learning rate scheduler
            scheduler_factor: Factor for learning rate scheduler
            class_weights: Optional class weights for BCE loss
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store hyperparameters
        self.n_mels = n_mels
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.pool_size = pool_size
        self.num_pools = len(conv_filters)  # Number of pooling operations
        
        # Build CNN encoder
        self.conv_blocks = self._build_cnn_encoder(
            conv_filters, conv_kernel_size, pool_size, dropout_cnn
        )
        
        # Calculate CNN output size
        cnn_output_freq = n_mels
        for _ in conv_filters:
            cnn_output_freq = cnn_output_freq // pool_size
        self.cnn_output_size = conv_filters[-1] * cnn_output_freq
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_gru if num_gru_layers > 1 else 0
        )
        
        # Output layer (no sigmoid - will use BCEWithLogitsLoss)
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_gru),
            nn.Linear(64, n_classes)
            # No Sigmoid here - BCEWithLogitsLoss applies it
        )
        
        # Loss function (BCEWithLogitsLoss is safe for mixed precision)
        if class_weights is not None:
            pos_weight = torch.FloatTensor(class_weights)
        else:
            pos_weight = None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def _build_cnn_encoder(
        self, 
        filters: list, 
        kernel_size: int, 
        pool_size: int, 
        dropout: float
    ) -> nn.Sequential:
        """Build CNN encoder blocks."""
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
                nn.MaxPool2d((pool_size, pool_size)),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input spectrogram (batch, 1, n_mels, time)
            
        Returns:
            Frame-level predictions (batch, time, n_classes)
        """
        batch_size = x.size(0)
        
        # CNN encoding
        x = self.conv_blocks(x)  # (batch, channels, freq, time)
        
        # Reshape for RNN: (batch, time, features)
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, time, channels*freq)
        
        # RNN processing
        x, _ = self.gru(x)  # (batch, time, hidden*2)
        
        # Frame-level predictions
        x = self.fc(x)  # (batch, time, n_classes)
        
        return x
    
    def _adjust_lengths_for_pooling(self, lengths: torch.Tensor) -> torch.Tensor:
        """Adjust lengths to account for pooling in time dimension."""
        # After pooling, time dimension is reduced by pool_size^num_pools
        time_reduction = self.pool_size ** self.num_pools
        adjusted_lengths = lengths // time_reduction
        return adjusted_lengths
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        specs, labels, lengths = batch
        
        # Forward pass
        predictions = self(specs)  # (batch, time_reduced, n_classes)
        
        # Adjust lengths and labels for pooling
        adjusted_lengths = self._adjust_lengths_for_pooling(lengths)
        
        # Downsample labels to match predictions
        labels_downsampled = self._downsample_labels(labels, predictions.size(1))
        
        # Compute loss (only on valid frames)
        loss = self._compute_masked_loss(predictions, labels_downsampled, adjusted_lengths)
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Compute metrics (only log epoch-level to avoid slowdown)
        if self.trainer.state.stage == 'train':
            with torch.no_grad():
                metrics = self._compute_metrics(predictions, labels_downsampled, adjusted_lengths)
                for name, value in metrics.items():
                    # Log to tensorboard but not progress bar to avoid clutter
                    self.log(f'train_{name}', value, on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def _downsample_labels(self, labels: torch.Tensor, target_time: int) -> torch.Tensor:
        """Downsample labels to match model output size."""
        # labels: (batch, time, n_classes)
        # We use max pooling to preserve positive labels
        batch_size, orig_time, n_classes = labels.shape
        
        if orig_time == target_time:
            return labels
        
        # Use interpolation to downsample
        # Transpose to (batch, n_classes, time) for interpolation
        labels_transposed = labels.transpose(1, 2)  # (batch, n_classes, time)
        
        # Interpolate using nearest neighbor to preserve binary labels
        labels_downsampled = torch.nn.functional.interpolate(
            labels_transposed.float(),
            size=target_time,
            mode='nearest'
        )
        
        # Transpose back to (batch, time, n_classes)
        labels_downsampled = labels_downsampled.transpose(1, 2)
        
        return labels_downsampled
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        specs, labels, lengths = batch
        
        # Forward pass
        predictions = self(specs)
        
        # Adjust lengths and labels for pooling
        adjusted_lengths = self._adjust_lengths_for_pooling(lengths)
        labels_downsampled = self._downsample_labels(labels, predictions.size(1))
        
        # Compute loss
        loss = self._compute_masked_loss(predictions, labels_downsampled, adjusted_lengths)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Compute per-class metrics
        metrics = self._compute_metrics(predictions, labels_downsampled, adjusted_lengths)
        for name, value in metrics.items():
            # Show F1 in progress bar, all metrics in logs
            prog_bar = (name == 'f1')
            self.log(f'val_{name}', value, on_step=False, on_epoch=True, prog_bar=prog_bar)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        specs, labels, lengths = batch
        
        # Forward pass
        predictions = self(specs)
        
        # Adjust lengths and labels for pooling
        adjusted_lengths = self._adjust_lengths_for_pooling(lengths)
        labels_downsampled = self._downsample_labels(labels, predictions.size(1))
        
        # Compute loss
        loss = self._compute_masked_loss(predictions, labels_downsampled, adjusted_lengths)
        
        # Log metrics
        self.log('test_loss', loss)
        
        # Compute per-class metrics
        metrics = self._compute_metrics(predictions, labels_downsampled, adjusted_lengths)
        for name, value in metrics.items():
            self.log(f'test_{name}', value)
        
        return loss
    
    def _compute_masked_loss(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor, 
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss only on valid (non-padded) frames.
        
        Args:
            predictions: (batch, time, n_classes)
            labels: (batch, time, n_classes)
            lengths: (batch,) - actual lengths before padding
            
        Returns:
            Masked loss
        """
        batch_size = predictions.size(0)
        total_loss = 0
        
        for i in range(batch_size):
            length = lengths[i]
            pred = predictions[i, :length]  # (length, n_classes)
            label = labels[i, :length]
            
            loss = self.criterion(pred, label)
            total_loss += loss
        
        return total_loss / batch_size
    
    def _compute_metrics(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor, 
        lengths: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute frame-level metrics.
        
        Args:
            predictions: (batch, time, n_classes) - logits
            labels: (batch, time, n_classes)
            lengths: (batch,)
            threshold: Threshold for binary predictions
            
        Returns:
            Dictionary of metrics
        """
        batch_size = predictions.size(0)
        
        # Apply sigmoid to convert logits to probabilities
        predictions_prob = torch.sigmoid(predictions)
        
        # Binarize predictions
        pred_binary = (predictions_prob > threshold).float()
        
        # Compute metrics only on valid frames
        tp = 0
        fp = 0
        fn = 0
        
        for i in range(batch_size):
            length = lengths[i]
            pred = pred_binary[i, :length]
            label = labels[i, :length]
            
            tp += ((pred == 1) & (label == 1)).sum().item()
            fp += ((pred == 1) & (label == 0)).sum().item()
            fn += ((pred == 0) & (label == 1)).sum().item()
        
        # Compute precision, recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(
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
                'monitor': 'val_loss',
                'frequency': 1
            }
        }


if __name__ == "__main__":
    # Test model
    print("Testing DrumTranscriptionCRNN...")
    
    # Create model
    model = DrumTranscriptionCRNN(
        n_mels=128,
        n_classes=8,
        conv_filters=[32, 64, 128],
        hidden_size=128,
        num_gru_layers=3
    )
    
    print(f"Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    time_frames = 100
    dummy_input = torch.randn(batch_size, 1, 128, time_frames)
    
    output = model(dummy_input)
    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\nModel test passed!")
