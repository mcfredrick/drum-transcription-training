#!/usr/bin/env python3
"""Training script for hierarchical drum transcription model."""

import argparse
import yaml
from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.data_module import EGMDDataModule
from src.models.hierarchical_crnn import HierarchicalDrumCRNN


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_module(config: dict) -> EGMDDataModule:
    """Create data module from config."""
    data_module = EGMDDataModule(
        processed_root=config['data']['processed_root'],
        splits_dir=config['data']['splits_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        use_hdf5=config['data'].get('use_hdf5', True),
        augmentation_config=config.get('augmentation', None),
        use_hierarchical_labels=True  # IMPORTANT: Use hierarchical labels
    )
    return data_module


def create_model(config: dict) -> HierarchicalDrumCRNN:
    """Create hierarchical model from config."""
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
    return model


def create_callbacks(config: dict) -> list:
    """Create training callbacks."""
    callbacks = []

    # Model checkpoint
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

    # Early stopping
    if config['early_stopping']['enabled']:
        early_stop_callback = EarlyStopping(
            monitor=config['early_stopping']['monitor'],
            patience=config['early_stopping']['patience'],
            mode=config['early_stopping']['mode'],
            min_delta=config['early_stopping']['min_delta'],
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    return callbacks


def create_logger(config: dict) -> TensorBoardLogger:
    """Create logger."""
    logger = TensorBoardLogger(
        save_dir=config['logging']['save_dir'],
        name=config['logging']['experiment_name'],
        version=None  # Auto-increment version
    )
    return logger


def main(args):
    """Main training function."""
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Override config with command line arguments if provided
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.max_epochs is not None:
        config['training']['num_epochs'] = args.max_epochs

    # Print configuration summary
    print("\n" + "="*70)
    print("HIERARCHICAL DRUM TRANSCRIPTION - TRAINING")
    print("="*70)
    print(f"Model: {config['model']['name']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Max epochs: {config['training']['num_epochs']}")
    print(f"Precision: {config['training']['precision']}")
    print(f"Checkpoints: {config['checkpoint']['dirpath']}")
    print(f"Logs: {config['logging']['save_dir']}")
    print("="*70 + "\n")

    # Set random seed for reproducibility
    L.seed_everything(config['data']['random_seed'])

    # Create data module
    print("Creating data module...")
    data_module = create_data_module(config)

    # Create model
    print("Creating hierarchical model...")
    model = create_model(config)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")

    # Print branch weights
    print("\nBranch weights:")
    for branch, weight in config['model']['branch_weights'].items():
        print(f"  {branch:20s}: {weight:.1f}x")

    # Create callbacks and logger
    print("\nSetting up training...")
    callbacks = create_callbacks(config)
    logger = create_logger(config)

    # Create checkpoint directory if it doesn't exist
    Path(config['checkpoint']['dirpath']).mkdir(parents=True, exist_ok=True)

    # Create trainer
    trainer = L.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['training']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        deterministic=False,  # For performance
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train model
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")

    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.fit(model, data_module, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(model, data_module)

    # Print training summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best {config['checkpoint']['monitor']}: {trainer.checkpoint_callback.best_model_score:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hierarchical drum transcription model")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/hierarchical_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate from config"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max epochs from config"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    args = parser.parse_args()
    main(args)
