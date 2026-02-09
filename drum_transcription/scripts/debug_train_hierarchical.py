#!/usr/bin/env python3
"""Debug training script - trains on small subset of data."""

import argparse
import yaml
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.data_module import EGMDDataModule
from src.models.hierarchical_crnn import HierarchicalDrumCRNN


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def limit_dataset_files(split_file_path: Path, limit_pct: float = 0.1) -> list:
    """Load and limit dataset files to a percentage."""
    with open(split_file_path, 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    # Limit to percentage
    limit_count = max(1, int(len(files) * limit_pct))
    limited_files = files[:limit_count]

    print(f"   Original: {len(files)} files")
    print(f"   Limited to: {len(limited_files)} files ({limit_pct*100:.0f}%)")

    return limited_files


class LimitedEGMDDataModule(EGMDDataModule):
    """Data module that limits dataset size for debugging."""

    def __init__(self, limit_pct: float = 0.1, *args, **kwargs):
        self.limit_pct = limit_pct
        super().__init__(*args, **kwargs)

    def _load_split_file(self, filename: str) -> list:
        """Load and limit split file."""
        split_path = self.splits_dir / filename

        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with open(split_path, 'r') as f:
            files = [line.strip() for line in f if line.strip()]

        # Limit to percentage
        limit_count = max(1, int(len(files) * self.limit_pct))
        limited_files = files[:limit_count]

        return limited_files


def main(args):
    """Main debug training function."""
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Print configuration summary
    print("\n" + "="*70)
    print("HIERARCHICAL DRUM TRANSCRIPTION - DEBUG TRAINING")
    print("="*70)
    print(f"Mode: Debug (subset training)")
    print(f"Subset size: {args.limit_pct*100:.0f}% of data")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Max epochs: {config['training']['num_epochs']}")
    print(f"Checkpoints: {config['checkpoint']['dirpath']}")
    print("="*70 + "\n")

    # Set random seed
    L.seed_everything(config['data']['random_seed'])

    # Create limited data module
    print("Creating limited data module...")
    data_module = LimitedEGMDDataModule(
        limit_pct=args.limit_pct,
        processed_root=config['data']['processed_root'],
        splits_dir=config['data']['splits_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        use_hdf5=config['data'].get('use_hdf5', True),
        augmentation_config=config.get('augmentation', None),
        use_hierarchical_labels=True
    )

    # Setup data to get counts
    data_module.setup('fit')
    print(f"Train dataset: {len(data_module.train_dataset)} samples")
    print(f"Val dataset: {len(data_module.val_dataset)} samples")

    # Create model
    print("\nCreating hierarchical model...")
    model = HierarchicalDrumCRNN(
        n_mels=config['model']['n_mels'],
        conv_filters=config['model']['conv_filters'],
        conv_kernel_size=config['model']['conv_kernel_size'],
        pool_size=config['model']['pool_size'],
        dropout_cnn=config['model']['dropout_cnn'],
        hidden_size=config['model']['hidden_size'],
        num_lstm_layers=config['model']['num_lstm_layers'],
        dropout_lstm=config['model']['dropout_lstm'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        scheduler_patience=config['training']['scheduler_patience'],
        scheduler_factor=config['training']['scheduler_factor'],
        branch_weights=config['model'].get('branch_weights', None)
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create callbacks
    callbacks = []

    # Checkpoint
    Path(config['checkpoint']['dirpath']).mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint']['dirpath'],
        filename=config['checkpoint']['filename'],
        monitor=config['checkpoint']['monitor'],
        mode=config['checkpoint']['mode'],
        save_top_k=config['checkpoint']['save_top_k'],
        save_last=config['checkpoint']['save_last'],
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # LR monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Logger
    logger = TensorBoardLogger(
        save_dir=config['logging']['save_dir'],
        name=config['logging']['experiment_name'],
        version=None
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['training']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        deterministic=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        # Limit batches for faster debugging if requested
        limit_train_batches=args.limit_batches if args.limit_batches > 0 else None,
        limit_val_batches=args.limit_batches if args.limit_batches > 0 else None
    )

    # Train
    print("\n" + "="*70)
    print("STARTING DEBUG TRAINING")
    print("="*70 + "\n")

    trainer.fit(model, data_module)

    # Summary
    print("\n" + "="*70)
    print("DEBUG TRAINING COMPLETED")
    print("="*70)
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    if trainer.checkpoint_callback.best_model_score is not None:
        print(f"Best {config['checkpoint']['monitor']}: {trainer.checkpoint_callback.best_model_score:.4f}")
    print("="*70 + "\n")

    # Print final metrics
    if hasattr(trainer, 'logged_metrics'):
        print("Final metrics:")
        for key, value in trainer.logged_metrics.items():
            if 'val_' in key:
                print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug train hierarchical model on subset")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/hierarchical_debug.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--limit-pct",
        type=float,
        default=0.1,
        help="Percentage of data to use (0.1 = 10%%)"
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=0,
        help="Limit number of batches per epoch (0 = no limit, for even faster debugging)"
    )

    args = parser.parse_args()
    main(args)
