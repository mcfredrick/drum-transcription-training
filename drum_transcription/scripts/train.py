"""Main training script for drum transcription model."""

import argparse
from pathlib import Path
import sys

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.crnn import DrumTranscriptionCRNN
from src.data.data_module import EGMDDataModule
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description='Train drum transcription model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (overrides config)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--fast-dev-run',
        action='store_true',
        help='Run 1 batch for testing'
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Set experiment name
    experiment_name = args.experiment_name or config.logging.experiment_name
    
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}\n")
    
    # Set random seed for reproducibility
    L.seed_everything(config.data.random_seed, workers=True)
    
    # Initialize DataModule
    print("Initializing DataModule...")
    data_module = EGMDDataModule(
        processed_root=config.data.processed_root,
        splits_dir=config.data.splits_dir,
        batch_size=config.training.batch_size,
        num_workers=config.hardware.num_workers,
        use_hdf5=True,  # Use HDF5 for faster loading
        augmentation_config=config.augmentation.to_dict()
    )
    
    # Initialize model
    print("Initializing model...")
    model = DrumTranscriptionCRNN(
        n_mels=config.model.n_mels,
        n_classes=config.model.n_classes,
        conv_filters=config.model.conv_filters,
        conv_kernel_size=config.model.conv_kernel_size,
        pool_size=config.model.pool_size,
        dropout_cnn=config.model.dropout_cnn,
        hidden_size=config.model.hidden_size,
        num_gru_layers=config.model.num_gru_layers,
        dropout_gru=config.model.dropout_gru,
        bidirectional=config.model.bidirectional,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler_patience=config.training.scheduler_patience,
        scheduler_factor=config.training.scheduler_factor,
        class_weights=config.loss.class_weights
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor=config.checkpoint.monitor,
        dirpath=config.checkpoint.dirpath,
        filename=config.checkpoint.filename,
        save_top_k=config.checkpoint.save_top_k,
        mode=config.checkpoint.mode,
        save_last=config.checkpoint.save_last,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if config.early_stopping.enabled:
        early_stop_callback = EarlyStopping(
            monitor=config.early_stopping.monitor,
            patience=config.early_stopping.patience,
            mode=config.early_stopping.mode,
            min_delta=config.early_stopping.min_delta,
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Logger
    if config.logging.logger == 'wandb':
        logger = WandbLogger(
            project=config.logging.project_name,
            name=experiment_name,
            save_dir=config.logging.save_dir
        )
        # Log hyperparameters
        logger.experiment.config.update(config.to_dict())
    else:
        logger = TensorBoardLogger(
            save_dir=config.logging.save_dir,
            name=experiment_name
        )
    
    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = L.Trainer(
        max_epochs=config.training.num_epochs,
        accelerator=config.hardware.accelerator,
        devices=config.hardware.devices,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.logging.log_every_n_steps,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=args.resume
    )
    
    # Test on best model
    print("\n" + "="*60)
    print("Testing best model...")
    print("="*60 + "\n")
    
    trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path='best'
    )
    
    print("\nTraining complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best {config.checkpoint.monitor}: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
